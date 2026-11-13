"""Horizon pipeline environment and shared bounded worker lifecycle."""

import json
import os
import subprocess
import sys
from pathlib import Path

from teutonic.evaluator import horizon_pipeline as scorer
from teutonic.evaluator.vllm_runtime import Worker


def worker_python():
    return os.environ.get("TEUTONIC_HORIZON_PYTHON", sys.executable)


def worker_env(gpu_ids):
    allowed = {"HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL", "TMPDIR",
               "LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CC", "CXX", "CUDA_HOME",
               "CUDA_PATH", "HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(scorer.deterministic_env())
    env["PATH"] = str(Path(worker_python()).absolute().parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    root = Path(os.environ.get("TEUTONIC_HORIZON_KERNEL_CACHE_DIR",
                              "/tmp/teutonic/kernel_cache/horizon-pipeline"))
    for name, suffix in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                         ("TORCH_EXTENSIONS_DIR", "extensions"), ("CUDA_CACHE_PATH", "cuda")):
        path = root / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    return env


def runtime_identity():
    result = subprocess.run([worker_python(), "-m", "teutonic.evaluator.horizon_pipeline", "--identity"],
                            env=worker_env([]), capture_output=True, text=True, check=True, timeout=60)
    value = json.loads(result.stdout)
    if value["code"] != scorer.code_identity():
        raise RuntimeError("Horizon worker and evaluator source differ")
    return value


class HorizonWorker(Worker):
    def __init__(self, *args, load_timeout=1200, **kwargs):
        super().__init__(*args, load_timeout=load_timeout, **kwargs)

    def _settings(self, role, gpu_ids, max_model_len):
        if role != "horizon" or len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or max_model_len is not None:
            raise ValueError("Horizon pipeline requires eight distinct GPUs and its pinned context")
        return scorer.settings()

    def _command(self, role, snapshot, write_fd, gpu_ids):
        return [worker_python(), "-m", "teutonic.evaluator.horizon_pipeline",
                "--snapshot", str(snapshot), "--response-fd", str(write_fd)]

    def _environment(self, role, gpu_ids):
        return worker_env(gpu_ids)
