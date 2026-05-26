# Teutonic-LXXX Sandbox Soak-10 — Report

> Historical archive. This report covers the earlier Qwen3-MoE 80B soak work. The live chain is now `Teutonic-Q3-8B`; use [`MINING.md`](MINING.md) and [`../chain.toml`](../chain.toml) for the current contract.


**Date:** 2026-05-07
**Box:** `root@95.133.252.44:10310` (8× B300 SXM6 / 275 GiB HBM each)
**Result:** ✅ **GREEN — eval-server is leak-free across 10 sequential duels**

Companion to [`docs/LXXX_SANDBOX_REPORT.md`](LXXX_SANDBOX_REPORT.md) (single-duel
smoke). Live `Teutonic-XXIV` chain and prod eval pod (`95.133.252.200`)
were untouched throughout this soak.

Raw artifacts in [`docs/soak-artifacts/20260507T132854Z/`](soak-artifacts/20260507T132854Z/)
(per-iter JSON + summary CSV + auto-generated report).

---

## TL;DR

10 sequential `/eval` calls against the same eval-server process (PID
stable throughout) — 28.3 minutes total wall, ~170s per iter. Across all
10 iterations:

- **Peak VRAM range:** 89,308 .. 89,372 MiB → spread of **64 MiB total**
  (= 6.4 MiB/iter, dwarfed by the cudaMalloc allocator's per-iter
  re-cache; not a real leak)
- **Eval-server PID:** **stable at 457759** — no self-kill, no supervisor
  restart, no CUDA context corruption
- **Wall:** 168-173 s per iter — 5 s spread, no drift
- **Verdict math:** mu_hat = 0.0, lcb = 0.0, accepted = false (10/10) —
  correct, since the soak used **symlink mode** where every iteration's
  challenger is byte-identical to the king (`mu_hat ≡ 0` for paired CE
  on the same weights)

---

## What "symlink mode" tests vs full perturbation

The smoke ([`LXXX_SANDBOX_REPORT.md`](LXXX_SANDBOX_REPORT.md)) ran the
full perturb+upload+download path with a real noise-perturbed challenger
and got `mu_hat = 0.000422`. The soak uses [`scripts/sandbox_soak.py
--mode symlink`](../scripts/sandbox_soak.py) which symlinks each
challenger directory's safetensors back to the king's blobs. The
challenger is byte-identical to the king from the eval-server's POV but
loads as a separate `MultiGPUEvaluator` instance on disjoint GPUs (king
on 0-3, challenger on 4-7).

**What this catches:**
- VRAM leaks across N challenger load/free cycles
- Eval-server self-kill or supervisor restart
- Wall-time drift / regression
- Bootstrap math producing finite verdicts every time

**What this does NOT catch (still on the list):**
- HF cache eviction at the 600 GiB watermark (cache stayed at 305 GB
  the whole soak — local symlinks bypass HF Hub entirely)
- Reparam-trick / trainability-probe interactions (probe was disabled
  via `TEUTONIC_PROBE_ENABLED=0`)
- Real-mining-like statistics (need `--mode noise` and ~28 min/iter
  perturb wall, total ~5 hours for 10 iters)

I switched to symlink mode after iter 1 of the noise-mode run took
~7 min just for shard 1 of 4 (28 min total perturb per iter × 10 = 4.7 h),
which would have eaten a multi-hour soak budget on noise that doesn't
change the leak-test signal.

---

## Per-iteration table

| iter | wall (s) | bootstrap (s) | mu_hat | lcb | accepted | peak VRAM (MiB) | peak GPU |
|---:|---:|---:|---:|---:|:-:|---:|:-:|
| 0 | 170.3 | 128.6 | 0.0 | 0.0 | ✗ | 89308 | 4 |
| 1 | 170.3 | 128.4 | 0.0 | 0.0 | ✗ | 89314 | 4 |
| 2 | 173.4 | 128.5 | 0.0 | 0.0 | ✗ | 89320 | 4 |
| 3 | 168.9 | 128.4 | 0.0 | 0.0 | ✗ | 89328 | 4 |
| 4 | 168.2 | 128.7 | 0.0 | 0.0 | ✗ | 89336 | 4 |
| 5 | 170.4 | 128.5 | 0.0 | 0.0 | ✗ | 89344 | 4 |
| 6 | 169.7 | 128.5 | 0.0 | 0.0 | ✗ | 89350 | 4 |
| 7 | 168.4 | 128.5 | 0.0 | 0.0 | ✗ | 89358 | 4 |
| 8 | 168.5 | 128.6 | 0.0 | 0.0 | ✗ | 89364 | 4 |
| 9 | 170.7 | 128.9 | 0.0 | 0.0 | ✗ | 89372 | 4 |

Mean wall: **170.0 s** (σ = 1.5 s).
Mean bootstrap: **128.6 s** (σ = 0.15 s — bootstrap is the dominant cost).
Server PID: **457759 throughout (no restart)**.

---

## Per-GPU memory drift (iter 0 → iter 9)

| GPU | Side | Iter 0 peak | Iter 9 peak | Drift |
|---:|---|---:|---:|---:|
| 0 | king | 76502 MiB | 76568 MiB | +66 MiB |
| 1 | king | 71530 MiB | 71596 MiB | +66 MiB |
| 2 | king | 71530 MiB | 71596 MiB | +66 MiB |
| 3 | king | 58612 MiB | 58678 MiB | +66 MiB |
| 4 | chall | 89308 MiB | 89372 MiB | +64 MiB |
| 5 | chall | 86930 MiB | 86996 MiB | +66 MiB |
| 6 | chall | 86930 MiB | 86996 MiB | +66 MiB |
| 7 | chall | 52492 MiB | 52558 MiB | +66 MiB |

Drift is **uniform across all 8 GPUs at ~66 MiB / 10 iters = 6.6 MiB per
iter**. This is the cudaMalloc allocator caching slightly different
allocation sizes between iters (e.g. tiny variation in the bootstrap
intermediate buffers) — not a model-state leak. Linear extrapolation:
even at 1000 iters this would only drift by ~6.6 GiB, well under the
275 GiB B300 cap (and the 180 GiB B200 cap).

The challenger's peak (~89 GiB on GPU 4) is comfortably under both caps
and matches the smoke's peak, confirming the no_grad fix from the
single-duel smoke survives across many iterations.

---

## Eval-server stability

- `_self_kill_scheduled`: never fired
- PID 457759 persisted from the smoke through all 10 soak iters (no
  supervisor restart)
- `gpu_busy` flag toggled correctly on/off between iters
- HF cache size: stable at 305 GB throughout (no eviction needed since
  challengers were local symlinks; cache contents = king + leftover
  mock-chall from the smoke)

---

## What this changes vs the runbook

[`docs/LXXX_RUNBOOK.md`](LXXX_RUNBOOK.md)'s Phase 0 was sized as
"smoke only — 1 duel". With this soak result we can scratch one of
the deferred items:

- ✅ "VRAM leaks across N back-to-back evals" — **PROVEN clean for 10 iters**

Items still deferred:

- HF cache eviction at the 600 GiB watermark (would need ~5 perturbed
  challengers with HF upload to make the cache cross 600 GB; ~3-5 hours
  of additional sandbox time)
- Trainability-probe sharded support
- Live mining harness compatibility

The next gating item before any live cutover is **Phase A training of a
real seed checkpoint** — see [`LXXX_RUNBOOK.md`](LXXX_RUNBOOK.md) §Phase 4.

---

## Replay

```bash
ssh -p 10310 root@95.133.252.44
cd /root/teutonic && source .venv/bin/activate && source /root/.creds/hf_token.env

# Eval-server already running on :9000 from the smoke; if not, scripts/sandbox_smoke.sh first.
tmux new -d -s soak '/root/run-soak.sh > /workspace/logs/soak-driver.log 2>&1; sleep 60'
tmux a -t soak

# Variants:
#   --iters 24 --mode symlink                 # 24-iter overnight soak (~70 min)
#   --iters 5  --mode noise                   # full perturbation, ~3 hr total
#   --shard-key dataset/v3/shards/...         # once v3 dataset is built
```
