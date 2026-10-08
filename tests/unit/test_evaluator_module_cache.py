from __future__ import annotations

import fcntl
import importlib.util
import multiprocessing as mp
import sys
from contextlib import contextmanager, nullcontext
from queue import Empty
from types import ModuleType
from unittest.mock import patch

import pytest

from teutonic.evaluator.module_cache import transformers_module_cache_lock


@contextmanager
def _cache_lock(cache_dir):
    # Exercise the production lock without installing the GPU evaluator stack.
    utils = ModuleType("transformers.utils")
    utils.HF_MODULES_CACHE = str(cache_dir)
    with (
        patch.dict(sys.modules, {"transformers.utils": utils}),
        transformers_module_cache_lock(),
    ):
        yield


def _copy_model_code(cache_dir, truncated, finish_copy, locked):
    # shutil.copyfile opens/truncates its destination before writing bytes.
    with (
        _cache_lock(cache_dir) if locked else nullcontext(),
        (cache_dir / "modeling_mimo_v2.py").open("wb") as destination,
    ):
        truncated.set()
        if not finish_copy.wait(10):
            raise TimeoutError("test did not release the cache writer")
        destination.write(b"class MiMoV2ForCausalLM: pass\n")


def _import_model_code(cache_dir, importing, results, locked):
    importing.set()
    try:
        with _cache_lock(cache_dir) if locked else nullcontext():
            spec = importlib.util.spec_from_file_location(
                "modeling_mimo_v2", cache_dir / "modeling_mimo_v2.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            results.put(module.MiMoV2ForCausalLM.__name__)
    except AttributeError as exc:
        results.put(f"{type(exc).__name__}: {exc}")


@pytest.mark.parametrize("locked", [False, True])
def test_replica_import_during_cache_copy(tmp_path, locked):
    """Reproduce the missing class; the lock must wait for the complete file."""
    context = mp.get_context("spawn")
    truncated, finish_copy, importing = (context.Event() for _ in range(3))
    results = context.Queue()
    writer = context.Process(
        target=_copy_model_code, args=(tmp_path, truncated, finish_copy, locked)
    )
    reader = context.Process(
        target=_import_model_code, args=(tmp_path, importing, results, locked)
    )
    processes = []
    try:
        writer.start()
        processes.append(writer)
        assert truncated.wait(10)
        reader.start()
        processes.append(reader)
        assert importing.wait(10)
        if locked:
            with pytest.raises(Empty):
                results.get(timeout=0.25)
            finish_copy.set()
            assert results.get(timeout=10) == "MiMoV2ForCausalLM"
        else:
            assert results.get(timeout=10) == (
                "AttributeError: module 'modeling_mimo_v2' "
                "has no attribute 'MiMoV2ForCausalLM'"
            )
            finish_copy.set()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        finish_copy.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            process.close()
        results.close()
        results.join_thread()


def test_import_failure_releases_cache_lock(tmp_path):
    cache_dir = tmp_path / "custom-hf-modules"
    with pytest.raises(ValueError, match="invalid model"), _cache_lock(cache_dir):
        raise ValueError("invalid model")

    # A different open file description must be able to acquire the same lock.
    with (cache_dir / ".teutonic-import.lock").open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
