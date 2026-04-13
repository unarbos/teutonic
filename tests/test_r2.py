#!/usr/bin/env python3
"""R2 integration tests against live constantinople bucket.

Fetches credentials from Doppler, runs tests under teutonic/test/ prefix,
cleans up afterwards. Each test prints timing data.

Usage:
    source .venv/bin/activate
    python test_r2.py
"""

from __future__ import annotations

import asyncio
import copy
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from teutonic.hparams import HParams
from teutonic.model import LlamaConfig, TinyLlama
from teutonic.dataset.synthetic import SyntheticDataset
from teutonic.storage.r2 import R2Storage
from teutonic.submission import MinerSubmission

from neurons.miner import Miner
from neurons.validator import Validator


# ──────────────────────────────────────────────────────────────────────────
# Credential loading
# ──────────────────────────────────────────────────────────────────────────

def _doppler_get(key: str) -> str:
    result = subprocess.run(
        ["doppler", "secrets", "get", key, "--plain",
         "--project", "arbos", "--config", "dev", "--no-check-version"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"doppler failed for {key}: {result.stderr}")
    return result.stdout.strip()


def get_r2_storage(prefix: str = "teutonic/test/") -> R2Storage:
    return R2Storage(
        endpoint_url=_doppler_get("R2_URL"),
        access_key_id=_doppler_get("R2_ACCESS_KEY_ID"),
        secret_access_key=_doppler_get("R2_SECRET_ACCESS_KEY"),
        bucket_name=_doppler_get("R2_BUCKET_NAME"),
        prefix=prefix,
    )


# ──────────────────────────────────────────────────────────────────────────
# Test helpers
# ──────────────────────────────────────────────────────────────────────────

def make_model(seed: int = 0) -> TinyLlama:
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=512, hidden_dim=64, intermediate_dim=128,
                       n_layers=2, n_heads=2, seq_len=64)
    return TinyLlama(cfg)


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────

async def test_1_round_trip(r2: R2Storage) -> tuple[bool, str]:
    """Put a dict with tensors, get it back, assert equality."""
    t0 = time.perf_counter()
    data = {
        "uid": 42,
        "tensor": torch.randn(100, 100),
        "nested": {"a": torch.tensor([1, 2, 3])},
        "scalar": 3.14,
    }
    key = "_test/roundtrip"
    await r2.put(key, data)
    t_put = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = await r2.get(key)
    t_get = time.perf_counter() - t0

    if got is None:
        return False, "get returned None"
    if got["uid"] != 42:
        return False, f"uid mismatch: {got['uid']}"
    if not torch.equal(got["tensor"], data["tensor"]):
        return False, "tensor mismatch"
    if not torch.equal(got["nested"]["a"], data["nested"]["a"]):
        return False, "nested tensor mismatch"
    return True, f"put={t_put:.3f}s, get={t_get:.3f}s"


async def test_2_list_keys(r2: R2Storage) -> tuple[bool, str]:
    """Put 5 objects, list them, assert all found."""
    t0 = time.perf_counter()
    for i in range(5):
        await r2.put(f"_test/list/{i}", {"i": i})

    keys = await r2.list_keys("_test/list/")
    elapsed = time.perf_counter() - t0

    if len(keys) != 5:
        return False, f"Expected 5 keys, got {len(keys)}: {keys}"
    return True, f"found {len(keys)} keys in {elapsed:.3f}s"


async def test_3_missing_key(r2: R2Storage) -> tuple[bool, str]:
    """Get a non-existent key returns None."""
    t0 = time.perf_counter()
    result = await r2.get("_test/does_not_exist_xyz")
    elapsed = time.perf_counter() - t0
    if result is not None:
        return False, f"Expected None, got {type(result)}"
    return True, f"None returned in {elapsed:.3f}s"


async def test_4_overwrite(r2: R2Storage) -> tuple[bool, str]:
    """Put same key twice, get returns latest."""
    key = "_test/overwrite"
    await r2.put(key, {"version": 1})
    await r2.put(key, {"version": 2})
    got = await r2.get(key)
    if got is None or got["version"] != 2:
        return False, f"Expected version 2, got {got}"
    return True, "overwrite works"


async def test_5_concurrent_writes(r2: R2Storage) -> tuple[bool, str]:
    """Put 20 objects in parallel, assert all retrievable."""
    t0 = time.perf_counter()
    tasks = [r2.put(f"_test/concurrent/{i}", {"i": i}) for i in range(20)]
    await asyncio.gather(*tasks)
    t_put = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = await asyncio.gather(*[r2.get(f"_test/concurrent/{i}") for i in range(20)])
    t_get = time.perf_counter() - t0

    missing = [i for i, r in enumerate(results) if r is None]
    if missing:
        return False, f"Missing keys: {missing}"
    wrong = [i for i, r in enumerate(results) if r["i"] != i]
    if wrong:
        return False, f"Wrong values at: {wrong}"
    return True, f"20 writes in {t_put:.3f}s, 20 reads in {t_get:.3f}s"


async def test_6_large_payload(r2: R2Storage) -> tuple[bool, str]:
    """Put a 10MB tensor, measure round-trip time."""
    big_tensor = torch.randn(1024, 1024)  # ~4MB float32
    data = {"big": big_tensor, "label": "stress"}
    raw_size = big_tensor.nelement() * big_tensor.element_size()

    t0 = time.perf_counter()
    await r2.put("_test/large", data)
    t_put = time.perf_counter() - t0

    t0 = time.perf_counter()
    got = await r2.get("_test/large")
    t_get = time.perf_counter() - t0

    if got is None:
        return False, "get returned None"
    if not torch.equal(got["big"], big_tensor):
        return False, "tensor mismatch"
    mb = raw_size / (1024 * 1024)
    return True, f"{mb:.1f}MB tensor: put={t_put:.3f}s, get={t_get:.3f}s"


async def test_7_full_pipeline(r2: R2Storage) -> tuple[bool, str]:
    """Run 1 miner + 1 validator end-to-end over R2."""
    t0 = time.perf_counter()

    cfg = LlamaConfig(vocab_size=512, hidden_dim=64, intermediate_dim=128,
                       n_layers=2, n_heads=2, seq_len=64)
    hp = HParams(max_batches=4, micro_bs=2, topk=32, lr=1e-3, outer_lr=0.4)
    dataset = SyntheticDataset(size=2048, seq_len=64, vocab_size=512, seed=42)

    # Use a unique prefix per run to avoid collisions
    pipeline_r2 = R2Storage(
        endpoint_url=r2._endpoint,
        access_key_id=r2._access_key,
        secret_access_key=r2._secret_key,
        bucket_name=r2._bucket,
        prefix="teutonic/test/_pipeline/",
    )

    shared_init = make_model(seed=0).state_dict()

    miner_model = make_model()
    miner_model.load_state_dict(copy.deepcopy(shared_init))
    miner = Miner(uid=1, model=miner_model, dataset=dataset,
                  storage=pipeline_r2, hparams=hp, device="cpu")

    val_model = make_model()
    val_model.load_state_dict(copy.deepcopy(shared_init))
    validator = Validator(uid=0, model=val_model, dataset=dataset,
                          storage=pipeline_r2, hparams=hp, device="cpu")

    # Miner trains and uploads to R2
    await miner.train_window(0)

    # Validator discovers and evaluates from R2
    discovered = await validator.discover_miners(0)
    if 1 not in discovered:
        return False, f"Miner 1 not discovered: {discovered}"

    results = await validator.evaluate_window(0, [1])
    r = results[0]

    # Cleanup
    await pipeline_r2.delete_prefix("")
    await pipeline_r2.close()

    elapsed = time.perf_counter() - t0

    if r.final_score < 0.9:
        return False, f"score {r.final_score:.4f} < 0.9"

    return True, (f"score={r.final_score:.4f}, loss_err={r.loss_result.max_error:.6f}, "
                  f"probe={r.probe_score:.3f}, total={elapsed:.3f}s")


async def test_8_cleanup(r2: R2Storage) -> tuple[bool, str]:
    """Delete all test objects."""
    t0 = time.perf_counter()
    count = await r2.delete_prefix("_test/")
    elapsed = time.perf_counter() - t0
    return True, f"deleted {count} objects in {elapsed:.3f}s"


# ──────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_1_round_trip,
    test_2_list_keys,
    test_3_missing_key,
    test_4_overwrite,
    test_5_concurrent_writes,
    test_6_large_payload,
    test_7_full_pipeline,
    test_8_cleanup,
]


async def main() -> None:
    print("=" * 70)
    print("R2 Integration Tests (live constantinople bucket)")
    print("=" * 70)

    r2 = get_r2_storage()
    print(f"  Bucket: {r2._bucket}")
    print(f"  Prefix: {r2._prefix}")
    print(f"  Endpoint: {r2._endpoint}")
    print()

    passed = 0
    failed = 0
    errors = []

    for test in ALL_TESTS:
        name = test.__name__
        try:
            ok, msg = await test(r2)
        except Exception as exc:
            ok = False
            msg = f"EXCEPTION: {exc}\n{traceback.format_exc()}"

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if ok:
            passed += 1
        else:
            failed += 1
            errors.append(name)

    await r2.close()

    print()
    print(f"{passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
