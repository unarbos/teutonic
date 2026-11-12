"""Scorer accounting and real subprocess lifecycle checks without a GPU."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from teutonic.evaluator import vllm_runtime as runtime
from teutonic.evaluator import vllm_score as scorer


def test_actual_target_logprobs_exclude_first_and_include_embedded_and_final_eos():
    tokens = [9, 151645, 8, 151645]
    output = SimpleNamespace(prompt_token_ids=tokens, prompt_logprobs=[None] + [
        {token: SimpleNamespace(logprob=-nll), 0: SimpleNamespace(logprob=-999.)}
        for token, nll in zip(tokens[1:], [.1, .2, .3], strict=True)])
    llm = SimpleNamespace(generate=lambda rows, params, use_tqdm: [output])
    result = scorer.score_batch(llm, None, [{"id": "doc", "tokens": tokens}], "king")
    assert result == [{"id": "doc", "nll": math.fsum([.1, .2, .3]), "scored_tokens": 3,
                       "input_sha256": scorer.input_hash(tokens)}]
    output.prompt_logprobs[2] = {8: SimpleNamespace(logprob=float("nan"))}
    with pytest.raises(ValueError, match="invalid token NLL"):
        scorer.score_batch(llm, None, [{"id": "doc", "tokens": tokens}], "king")
    with pytest.raises(ValueError, match="context length or EOS"):
        scorer.score_batch(llm, None, [{"id": "doc", "tokens": [9, 8]}], "king")


def test_worker_environment_pins_determinism_and_excludes_credentials(monkeypatch):
    for key in ("TEUTONIC_EVALUATOR_R2_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "HF_TOKEN"):
        monkeypatch.setenv(key, "do-not-forward")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    env = runtime.worker_env("king", [4, 5, 6, 7])
    assert "do-not-forward" not in env.values()
    assert env["VLLM_BATCH_INVARIANT"] == "1"
    assert env["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert env["VLLM_USE_FUSED_MOE_GROUPED_TOPK"] == "0"
    assert env["VLLM_PLUGINS"] == "teutonic_mimo"
    assert env["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert runtime.worker_env("horizon", range(8))["VLLM_PLUGINS"] == ""


def fake_process(tmp_path, monkeypatch, mode="normal"):
    """Replace only the GPU executable; exercise actual pipes, polling and signals."""
    source = tmp_path / "fake_worker.py"
    source.write_text('''import argparse, json, os, signal, subprocess, sys, time
from teutonic.evaluator.vllm_score import input_hash, settings
p=argparse.ArgumentParser()
p.add_argument('--role'); p.add_argument('--snapshot'); p.add_argument('--response-fd',type=int)
p.add_argument('--tensor-parallel-size',type=int); p.add_argument('--max-model-len',type=int)
a=p.parse_args()
mode = MODE
if mode == 'load_failure':
    sys.exit(7)
if mode == 'orphan':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    open(a.snapshot, 'w').write(str(child.pid))
    sys.exit(9)
with os.fdopen(a.response_fd,'w',buffering=1) as out:
    out.write(json.dumps({'type':'ready','identity':{'fixture':1},'settings':settings(a.role,a.tensor_parallel_size,a.max_model_len),'load_seconds':.1})+'\\n')
    for line in sys.stdin:
        c=json.loads(line)
        if c['type']=='stop': break
        if mode == 'hang':
            time.sleep(60)
        rows=[{'id':r['id'], 'nll':2., 'scored_tokens':len(r['tokens'])-1,
               'input_sha256':input_hash(r['tokens'])} for r in c['rows']]
        if mode == 'wrong_input': rows[0]['input_sha256']='wrong'
        out.write(json.dumps({'type':'scores','batch':c['batch'],'rows':rows,'seconds':.2})+'\\n')
'''.replace('MODE', repr(mode)))
    popen = subprocess.Popen
    def start(command, **kwargs):
        return popen([sys.executable, str(source), *command[3:]], **kwargs)
    monkeypatch.setattr(runtime.subprocess, "Popen", start)


def make_worker(tmp_path, **kwargs):
    return runtime.Worker("king", tmp_path / "snapshot", [0], tmp_path,
                          {"fixture": 1}, lambda event: None, load_timeout=5, **kwargs)


def test_bounded_pipe_roundtrip_and_clean_shutdown(tmp_path, monkeypatch):
    fake_process(tmp_path, monkeypatch)
    with make_worker(tmp_path) as worker:
        # Larger than pipe capacity: catches blocking write/read deadlocks.
        tokens = [7] * 16383 + [151645]
        docs = [SimpleNamespace(doc_id=str(i)) for i in range(8)]
        rows = worker.score(docs, [tokens] * 8)
        assert len(rows) == 8 and all(row['scored_tokens'] == 16383 for row in rows)
        assert worker.report['documents'] == 8
    assert worker.process.poll() is not None
    assert worker.closed
    worker.close()  # Idempotent, including service shutdown after context cleanup.


@pytest.mark.parametrize("mode,error", [("load_failure", RuntimeError),
                                        ("wrong_input", ValueError), ("hang", TimeoutError)])
def test_failure_timeout_and_mismatched_result_close_owned_processes(tmp_path, monkeypatch, mode, error):
    fake_process(tmp_path, monkeypatch, mode)
    with pytest.raises(error), make_worker(tmp_path, score_timeout=.1) as worker:
        worker.score([SimpleNamespace(doc_id="a")], [[8, 151645]])
    assert worker.process.poll() is not None and worker.closed


def test_orphaned_child_is_reaped_after_engine_parent_exits(tmp_path, monkeypatch):
    import time
    fake_process(tmp_path, monkeypatch, "orphan")
    with pytest.raises(RuntimeError), make_worker(tmp_path) as worker:
        worker.score([SimpleNamespace(doc_id="a")], [[8, 151645]])
    pid = int((tmp_path / 'snapshot').read_text())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except FileNotFoundError:
            break
        if state == "Z":  # Exited; the host init process owns final reaping.
            break
        time.sleep(.01)
    else:
        os.kill(pid, 9)
        pytest.fail("worker group cleanup left a child running")


def test_runtime_code_mismatch_rejected(monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps({"code": {"old": "scorer"}})))
    with pytest.raises(RuntimeError, match="scorer code differ"):
        runtime.runtime_identity()


def test_gpu_release_retries_transient_nvml_timeout(monkeypatch):
    calls = []
    def check(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired('nvidia-smi', 15)
        return SimpleNamespace(stdout='0\n0\n')
    monkeypatch.setattr(runtime.subprocess, 'run', check)
    monkeypatch.setattr(runtime.time, 'sleep', lambda _: None)
    runtime.wait_for_gpu_release([0, 1])
    assert len(calls) == 2


def test_gpu_release_timeout_is_not_idle(monkeypatch):
    def check(*args, **kwargs):
        raise subprocess.TimeoutExpired('nvidia-smi', 15)
    monkeypatch.setattr(runtime.subprocess, 'run', check)
    with pytest.raises(RuntimeError, match='did not release'):
        runtime.wait_for_gpu_release([0], timeout=0)
