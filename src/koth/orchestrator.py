"""Pod orchestrator: spin up ephemeral Lium pods for evaluation.

Each evaluation runs on a clean GPU machine. The orchestrator handles:
- Spinning up a pod via the Lium/Celium CLI
- Deploying the eval worker script via SSH
- Monitoring the eval via R2 polling
- Tearing down the pod when done
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PodConfig, R2Config, EvalConfig, BoundingBoxConfig
from .r2 import R2Client

logger = logging.getLogger(__name__)

EVAL_WORKER_SCRIPT = Path(__file__).parent / "eval_worker.py"


@dataclass
class PodInfo:
    pod_id: str
    name: str
    host: str
    port: int
    user: str = "root"


class PodOrchestrator:
    """Manages ephemeral Lium/Celium pods for eval."""

    def __init__(self, pod_cfg: PodConfig, r2_cfg: R2Config):
        self.pod_cfg = pod_cfg
        self.r2_cfg = r2_cfg

    def start_pod(self, name: str) -> PodInfo:
        """Start a new Lium pod for evaluation.

        Uses `lium start` CLI to provision a GPU machine.
        Falls back to returning connection details from `lium list` if
        a pod with that name already exists.
        """
        logger.info("Starting eval pod: %s", name)

        try:
            result = subprocess.run(
                [
                    "lium", "start",
                    "--name", name,
                    "--gpu-type", self.pod_cfg.gpu_type,
                ],
                capture_output=True, text=True, timeout=self.pod_cfg.startup_timeout_s,
            )
            logger.info("lium start stdout: %s", result.stdout[:500])
            if result.returncode != 0:
                logger.warning("lium start stderr: %s", result.stderr[:500])
        except FileNotFoundError:
            logger.warning("lium CLI not found, attempting rentcompute")
            return self._start_via_rentcompute(name)
        except subprocess.TimeoutExpired:
            logger.error("Pod startup timed out after %ds", self.pod_cfg.startup_timeout_s)
            raise

        return self._get_pod_info(name)

    def _start_via_rentcompute(self, name: str) -> PodInfo:
        result = subprocess.run(
            [
                "rentcompute", "start",
                "--name", name,
                "--gpu-type", self.pod_cfg.gpu_type,
            ],
            capture_output=True, text=True, timeout=self.pod_cfg.startup_timeout_s,
        )
        logger.info("rentcompute start: %s", result.stdout[:500])
        return self._get_pod_info(name)

    def _get_pod_info(self, name: str) -> PodInfo:
        """Get connection details for a running pod."""
        try:
            result = subprocess.run(
                ["lium", "list", "--json"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                pods = json.loads(result.stdout)
                for pod in pods:
                    if pod.get("name") == name or pod.get("pod_name") == name:
                        return PodInfo(
                            pod_id=pod.get("id", ""),
                            name=name,
                            host=pod.get("host", pod.get("ip", "")),
                            port=pod.get("port", pod.get("ssh_port", 22)),
                            user=pod.get("user", "root"),
                        )
        except Exception as e:
            logger.warning("Failed to get pod info via lium list: %s", e)

        raise RuntimeError(f"Could not get connection info for pod {name}")

    def stop_pod(self, pod: PodInfo) -> None:
        """Tear down a pod."""
        logger.info("Stopping pod %s (%s)", pod.name, pod.pod_id)
        try:
            subprocess.run(
                ["lium", "stop", pod.pod_id, "-y"],
                capture_output=True, text=True, timeout=60,
            )
        except FileNotFoundError:
            try:
                subprocess.run(
                    ["rentcompute", "stop", "--id", pod.pod_id, "-y"],
                    capture_output=True, text=True, timeout=60,
                )
            except Exception as e:
                logger.error("Failed to stop pod %s: %s", pod.pod_id, e)

    def deploy_and_run_eval(
        self,
        pod: PodInfo,
        challenge_id: str,
        king_repo: str,
        challenger_repo: str,
        eval_cfg: EvalConfig,
        bbox_cfg: BoundingBoxConfig,
        dataset_shard_key: str,
        hf_token: str = "",
    ) -> None:
        """Deploy eval worker script to pod and start evaluation.

        The eval worker writes results directly to R2, so the coordinator
        monitors by polling R2 rather than SSH stdout.
        """
        ssh_target = f"{pod.user}@{pod.host}"
        ssh_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=30",
            "-p", str(pod.port),
        ]
        ssh_key = Path(self.pod_cfg.ssh_key_path).expanduser()
        if ssh_key.exists():
            ssh_opts.extend(["-i", str(ssh_key)])

        # Build config JSON for the eval worker
        worker_config = {
            "challenge_id": challenge_id,
            "king_repo": king_repo,
            "challenger_repo": challenger_repo,
            "N": eval_cfg.N,
            "alpha": eval_cfg.alpha,
            "sequence_length": eval_cfg.sequence_length,
            "use_amp": eval_cfg.use_amp,
            "amp_dtype": eval_cfg.amp_dtype,
            "bbox_max_linf": bbox_cfg.max_linf,
            "bbox_max_l2_per_tensor": bbox_cfg.max_l2_per_tensor,
            "bbox_max_l2_global": bbox_cfg.max_l2_global,
            "bbox_frozen_prefixes": bbox_cfg.frozen_param_prefixes,
            "r2_endpoint_url": self.r2_cfg.endpoint_url,
            "r2_bucket_name": self.r2_cfg.bucket_name,
            "r2_access_key_id": self.r2_cfg.access_key_id,
            "r2_secret_access_key": self.r2_cfg.secret_access_key,
            "dataset_shard_key": dataset_shard_key,
            "hf_token": hf_token,
        }

        # Write config to temp file and SCP it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(worker_config, f)
            config_path = f.name

        try:
            # SCP eval worker script
            self._scp(ssh_opts, str(EVAL_WORKER_SCRIPT), ssh_target, "/tmp/eval_worker.py")
            # SCP config
            self._scp(ssh_opts, config_path, ssh_target, "/tmp/eval_config.json")

            # Run eval worker in background on the pod
            run_cmd = (
                "nohup bash -c '"
                "pip install -q safetensors huggingface-hub boto3 scipy numpy torch 2>/dev/null; "
                "python /tmp/eval_worker.py --config /tmp/eval_config.json"
                "' > /tmp/eval.log 2>&1 &"
            )
            self._ssh_exec(ssh_opts, ssh_target, run_cmd)
            logger.info("Eval worker launched on pod %s", pod.name)
        finally:
            Path(config_path).unlink(missing_ok=True)

    def _scp(self, ssh_opts: list[str], local_path: str, target: str, remote_path: str) -> None:
        port_idx = ssh_opts.index("-p")
        port = ssh_opts[port_idx + 1]
        scp_opts = [o for o in ssh_opts if o not in ("-p", port)]
        cmd = ["scp", "-P", port] + scp_opts + [local_path, f"{target}:{remote_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {result.stderr}")

    def _ssh_exec(self, ssh_opts: list[str], target: str, command: str) -> str:
        cmd = ["ssh"] + ssh_opts + [target, command]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("SSH command returned %d: %s", result.returncode, result.stderr[:300])
        return result.stdout


def poll_for_verdict(r2: R2Client, challenge_id: str, timeout_s: int = 1800, interval_s: int = 15) -> dict | None:
    """Poll R2 for the eval verdict, with progress tracking."""
    verdict_key = f"eval/{challenge_id}/verdict.json"
    outcomes_key = f"eval/{challenge_id}/outcomes.jsonl"

    start = time.time()
    while time.time() - start < timeout_s:
        verdict = r2.get_json(verdict_key)
        if verdict:
            logger.info("Verdict received for %s: %s", challenge_id, verdict.get("verdict"))
            return verdict

        outcomes = r2.get_jsonl(outcomes_key)
        if outcomes:
            last = outcomes[-1]
            logger.info(
                "Eval %s in progress: s=%s n=%s (%.1f%%)",
                challenge_id,
                last.get("s", "?"),
                last.get("n", "?"),
                100 * last.get("n", 0) / max(last.get("N", 1), 1),
            )

        time.sleep(interval_s)

    logger.error("Eval %s timed out after %ds", challenge_id, timeout_s)
    return None
