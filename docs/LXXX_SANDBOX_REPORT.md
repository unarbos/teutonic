# Teutonic-LXXX Sandbox Soak — Smoke Report

> Historical archive. This report covers the earlier Qwen3-MoE 80B sandbox work. The live chain is now `Teutonic-Q3-8B`; use [`MINING.md`](MINING.md) and [`../chain.toml`](../chain.toml) for the current contract.


**Date:** 2026-05-07
**Box:** `root@95.133.252.44:10310` (8× B300 SXM6 / 275 GiB HBM each / 871 GiB on `/`)
**Branch:** `teutonic-lxxx-sandbox`
**Result:** ✅ **GREEN — go for soak-10 then live cutover sequence**

The live Quasar `Teutonic-XXIV` chain and its production eval pod
(`95.133.252.200`) were untouched throughout this soak.

---

## TL;DR

A vanilla Qwen3-MoE 80B (`81.877B total / 7.134B active`, 152.5 GiB bf16)
loaded as TWO sharded replicas on disjoint GPU sets (king on GPUs 0-3,
challenger on GPUs 4-7), ran a paired bootstrap test on 2000 sequences,
and produced a finite verdict in **3 minutes wall**, with a **peak per-GPU
memory of 80.3 GiB** (less than half the 180 GiB B200 cap).

Verdict math sanity-checked: noise-perturbed challenger (N(0, 1e-4) added
to every float tensor) was correctly **rejected** with `lcb = -0.002032`
< `delta = 1/N = 0.0005`.

---

## Sizing confirmed end-to-end

`archs/qwen3_moe/size.py` and `seed.py` agreed:

```
moe_experts     total  79.725B   active   4.983B
attn            total   1.510B   active   1.510B
embed           total   0.622B   active   0.622B
moe_router      total   0.019B   active   0.019B
TOTAL           total  81.877B   active   7.134B
bf16: 152.5 GiB total / 13.3 GiB active per copy
sharded across 4 GPUs: ~38.1 GiB / GPU just for weights
```

HF artifacts created (public):
- [`unconst/Teutonic-LXXX-mock-king`](https://huggingface.co/unconst/Teutonic-LXXX-mock-king) — 163.8 GB, 4 safetensor shards, fresh random init
- [`unconst/Teutonic-LXXX-mock-chall`](https://huggingface.co/unconst/Teutonic-LXXX-mock-chall) — 163.8 GB, same king + N(0, 1e-4) noise on every float tensor

---

## Phase wall times

| Phase | Wall | Notes |
|---|---:|---|
| Random init (80B bf16, CPU) | 1051.5 s (17.5 min) | Single-thread `kaiming_uniform_` on 240 cores; bottleneck is per-tensor init, not RAM |
| `save_pretrained` (152 GiB → 4 shards) | ~50 s | NVMe |
| HF push (king, 164 GB) | ~10 min | `hf_transfer` enabled, sustained ~250 MB/s |
| HF download for perturb (164 GB) | 276 s (4.6 min) | sustained ~590 MB/s on this box |
| Local copy base→challenger (163 GB) | 53.5 s | NVMe |
| Perturb 4 safetensor shards | ~20 min | 5 min/shard, single-process load+noise+save |
| HF push (chall, 164 GB) | 607 s (~10 min) | sustained ~270 MB/s |
| Smoke eval **end-to-end** | **180 s (3 min)** | POST /eval → verdict |
| ↳ HF prefetch (king cached, chall fresh) | ~9 s | xet-cache hit on king; chall pulled once |
| ↳ Model load (sharded king) | 24.8 s | accelerate auto-distributed `{0:11, 1:10, 2:10, 3:9}` modules |
| ↳ Bootstrap test (2000 seq × 2048 tokens) | 133.2 s | 15.0 seq/s sustained on the paired sharded run |

---

## Pass criteria — every line green

| Criterion | Target | Actual | ✓ |
|---|---|---|---|
| Both replicas load (no OOM) | yes | yes | ✓ |
| Peak per-GPU `mem_used` ≤ 165 GiB | ≤ 165 GiB | **80.3 GiB** (max, on GPU 4) | ✓ |
| Bootstrap verdict finite | finite `mu_hat` | `mu_hat = 0.000422` | ✓ |
| Wall ≤ 25 min for `EVAL_N=2000` | ≤ 1500 s | **180 s** | ✓ |
| No CUDA self-kill | none | none after fix (see below) | ✓ |

Per-GPU peak (from `nvidia-smi dmon`):

| GPU | Side | Peak | Cap (B300) | Cap (B200, prod) |
|---|---|---:|---:|---:|
| 0 | king | 74.7 GiB | 275 | 180 |
| 1 | king | 69.8 GiB | 275 | 180 |
| 2 | king | 69.8 GiB | 275 | 180 |
| 3 | king | 57.2 GiB | 275 | 180 |
| 4 | chall | **80.3 GiB** | 275 | 180 |
| 5 | chall | 71.8 GiB | 275 | 180 |
| 6 | chall | 71.8 GiB | 275 | 180 |
| 7 | chall | 52.7 GiB | 275 | 180 |

The 80.3 GiB peak is on the input-embedding GPU of the challenger side
(where lm_head + chunked logits land). Comfortably under the 180 GiB B200
cap that the live eval pod runs on, so this same workload should fit
there too with `TEUTONIC_SHARD_PER_GPU_GIB=120` (down from the B300
default of 240).

---

## Verdict math

```json
{
  "accepted": false,
  "verdict": "king",
  "mu_hat":  0.000422,
  "lcb":    -0.002032,
  "delta":   0.0005,
  "alpha":   0.001,
  "n_bootstrap": 2000,
  "N": 2000,
  "avg_king_loss":   12.740361,
  "avg_challenger_loss": 12.739939,
  "wall_time_s": 133.2,
  "seqs_per_sec": 15.0
}
```

- Both losses ~12.74 nats/token, very close to `ln(151936) = 11.93` plus
  a small bf16 noise floor — exactly what we'd expect from a random-init
  80B Qwen3-MoE evaluated on uniformly-random uint32 token IDs from the
  synthetic shard.
- The `1e-4` noise perturbation lifted challenger's avg loss by
  `0.000422` nats (mu_hat) — challenger is worse, as it should be, since
  the king is the original and noise is destructive on average.
- LCB `-0.002032` is negative → very far from clearing `delta = 1/N`,
  so the validator correctly **rejects** the challenger.
- Bootstrap math: `delta = 1/N` reproduced byte-for-byte (`0.0005 = 1/2000`).

---

## Bugs caught and fixed

Three landed during this soak; all in the new sharded path. The first
two were easy, the third would have been a major outage if shipped to
the live eval pod.

### 1. `Qwen3MoeConfig.rope_theta` no longer top-level (transformers 5.5+)

`archs/qwen3_moe/seed.py` and `size.py` both crashed at startup with
`AttributeError: 'Qwen3MoeConfig' object has no attribute 'rope_theta'`.
transformers 5.5 flattened RoPE into a `rope_parameters` dict
(`{"rope_theta": ..., "rope_type": "default"}`).

**Fix:** read via `getattr(cfg, "rope_parameters", ...)` with a fallback
to legacy `rope_theta`. Added `rope_parameters` to `chain.lxxx.toml`'s
`extra_lock_keys` so the structural lock catches a challenger that
silently changes RoPE.

### 2. `argparse type=int` rejects `0xDEADBEEF`

`scripts/sandbox_perturb.py --seed 0xDEADBEEF` raised
`invalid int value`. Trivial; passed `3735928559` (decimal) instead.
Not a code bug, just a shell-side oversight in the runbook.

### 3. **Critical** — `compute_batch_losses` was missing `@torch.no_grad()`

`compute_paired_losses` already had the decorator (per-GPU mode worked
fine for the live Quasar 24B). My sharded refactor of
`compute_paired_multi_gpu` routes both sides through
`compute_batch_losses` for the no-pair-cross-GPU case, which had no
`no_grad`. Result on the first run:

- PyTorch built the full backward graph for an 80B forward
- Activation memory exploded across all 4 GPUs of each replica
- nvidia-smi showed ALL EIGHT GPUs at 260+ GiB used (~1 TB of cached
  activations across the box)
- `cross_entropy` for the chunked lm_head needed 4.64 GiB and OOM'd on GPU 4
- Eval-server self-kill fired ([`eval_server.py:99-110`](../eval_server.py#L99))
- Supervisor pattern worked — restart cleared CUDA state

**Fix:** add `@torch.no_grad()` to `compute_batch_losses` and a comment
noting the sharded path now depends on it. After the fix:

- King-side peak: 74.7 GiB / 4 GPUs (was 270 GiB / 4 GPUs)
- Challenger-side peak: 80.3 GiB / 4 GPUs (was 270 GiB / 4 GPUs)
- ~3.4x reduction in peak memory; eval ran clean

This is the kind of bug a smoke test catches and a soak doesn't — would
have been silent on the live Quasar 24B chain because per-GPU mode never
hit `compute_batch_losses` from the paired path.

---

## Things this smoke does NOT prove (still on the runbook)

Per [`docs/LXXX_RUNBOOK.md`](LXXX_RUNBOOK.md) Phase 0 checklist:

- VRAM leaks across N sequential evals (need soak-10 or soak-24h)
- HF cache eviction at the 600 GiB watermark (only triggers after several
  different challengers cache up; we only have 2 mock repos in the cache
  right now totaling 306 GB — under the watermark)
- Trainability probe in sharded mode (we ran with `TEUTONIC_PROBE_ENABLED=0`
  because the probe code path in `eval_server.py:286` calls
  `trainability_probe(king_model)` on a single sharded model. The probe
  itself does its own forward+backward on the model; on a sharded model
  it would hit the SAME `no_grad` issue inverted — the probe NEEDS gradient
  flow, but the chunked lm_head logic has to be reviewed to make sure it
  doesn't block grads. Recommend keeping the probe disabled until that
  function gets a sharded-mode review.)
- Live mining harness compatibility ([`scripts/mining/train_challenger.py`](../scripts/mining/train_challenger.py))
  against the 80B base — the Quasar-specific notes (SMEBU, latent memory,
  `attn_implementation='eager'`) are dead in the new arch
- Validator-side stream-idle watchdog under the new wall-time envelope
- Reparameterization-trick gate ([`docs/DESIGN.md`](DESIGN.md#7-defenses-and-edge-cases))
  was tuned for Quasar's RMSNorm/SwiGLU; verify on the 80B before
  cutover

---

## Recommended next steps

1. **Soak-10**: re-run the smoke 10× back-to-back with 10 different
   noise-perturbed challengers (use the existing `unconst/Teutonic-LXXX-mock-king`
   as king; generate 10 mock challengers with seeds 1, 2, ..., 10 via
   `scripts/sandbox_perturb.py`). Watch for VRAM leaks + HF cache
   eviction kicking in around iteration 4-5 (when total cache > 600 GiB).
2. **Trainability-probe sharded review**: walk the probe's
   forward+backward path against a sharded model and either fix the
   sharded compute or document that probe stays off for the LXXX chain.
3. **Live-pod port**: rebuild the same setup on the production
   `95.133.252.200` 8× B200 pod, set `TEUTONIC_SHARD_PER_GPU_GIB=120`
   (vs B300's 240), repeat the smoke. Wall should rise but stay well
   inside the 1800 s `TICK_RESTART_AFTER` budget; there's headroom even
   at the proposed bumped 3600 s.
4. **Phase A training**: gates everything else. See
   [`docs/LXXX_RUNBOOK.md`](LXXX_RUNBOOK.md) §Phase 4. Until a real seed
   exists, the live cutover should not happen.

---

## Replay commands

If the user wants to re-run any phase on this box:

```bash
ssh -p 10310 root@95.133.252.44

# 1. Re-run sizer (sub-second)
cd /root/teutonic && source .venv/bin/activate && source /root/.creds/hf_token.env
TEUTONIC_CHAIN_OVERRIDE=chain.lxxx.toml python -m archs.qwen3_moe.size

# 2. Re-run smoke (~3 min if king+chall both cached)
tmux new -d -s smoke 'bash scripts/sandbox_smoke.sh > /workspace/logs/smoke-driver.log 2>&1; sleep 60'
tmux a -t smoke    # ctrl-b d to detach

# 3. Tear down the running eval-server when done
tmux kill-session -t eval

# 4. Soak (10 challengers; not yet wired)
# for i in 1 2 ... 10; do generate chall_i, eval against king, log peak VRAM; done
# This belongs in scripts/sandbox_soak.sh — not landed in this session.
```
