# Teutonic-LXXX Go-Live Readiness — 2026-05-07

> Historical archive. This report covers the earlier Qwen3-MoE 80B go-live work. The live chain is now `Teutonic-Q3-8B`; use [`MINING.md`](MINING.md) and [`../chain.toml`](../chain.toml) for the current contract.


**Result:** ✅ **GREEN — all required tests pass on the production-bound B300 box.**

This supersedes [`LXXX_SANDBOX_REPORT.md`](LXXX_SANDBOX_REPORT.md) and
[`LXXX_SOAK_REPORT.md`](LXXX_SOAK_REPORT.md). Earlier blockers from
[`my "are we ready?" assessment`](LXXX_RUNBOOK.md) are either resolved
below or explicitly waived per user direction.

---

## What changed since the soak-10 report

Per user direction in this session:

1. **Production hardware: 8× B300 SXM6 (`95.133.252.44:10310`)** — the
   sandbox box becomes prod. The previous Lium 8× B200 pod (`95.133.252.200`)
   is decommissioned for LXXX. B300 has 275 GiB HBM/GPU vs B200's 180 GiB,
   so all the per-GPU budget knobs we sized for B200 are oversized but safe.
2. **Tokenizer: keep `unconst/Teutonic-I` (Gemma3-derived, vocab 262144)** —
   no v3 dataset rebuild needed. The live `dataset/v2/shards/...` shards
   work as-is. The arch was retargeted from Qwen vocab (151936) to Gemma3
   vocab (262144); model size grew by ~451M params (81.88B → 82.33B total,
   +0.45B active to the embed/lm_head).
3. **Trainability probe in sharded mode** — the previously-disabled probe
   is now ENABLED and verified working on both king and challenger loads.

---

## Final config

| Field | Value |
|---|---|
| Chain name | `Teutonic-LXXX` |
| Seed repo | `unconst/Teutonic-LXXX` (currently mock-king at the same path; replace with Phase A trained checkpoint at cutover if available) |
| Arch module | `archs.qwen3_moe` (vanilla `transformers.Qwen3MoeForCausalLM`) |
| Tokenizer repo | `unconst/Teutonic-I` (Gemma3-derived, vocab 262144) |
| Total params | **82.328B** |
| Active per token | **7.586B** |
| bf16 size | 153.3 GiB sharded across 4 GPUs (38.3 GiB/GPU for weights) |
| Hidden / Layers / Heads / KV / HeadDim | 4096 / 36 / 32 / 8 / 128 |
| MoE | 128 experts, top-8, moe_intermediate=1408, every layer MoE |
| RoPE | θ=1e6, type=default |
| Tied embed/lm_head | yes |

---

## Tests run this session — all green

### 1. King-load probe + king-vs-king bootstrap (eval `eda18929`)

Validated the **sharded king-load + sharded probe** path against the freshly
re-seeded vocab-262144 king, bootstrap on a real `dataset/v2/shards/shard_000000.npy`
shard (Gemma3-tokenized).

```
eval_id: eda18929
king load (sharded over GPUs 0-3): 25.2s, modules/GPU {0:10, 1:10, 2:10, 3:10}
king probe: ok=True, max_grad=2.21e+02, 5.8s
bootstrap: 200 sequences × 13s wall, mu_hat=0.0 (king vs king, exact)
verdict: rejected (lcb=0.0, delta=0.005=1/N)
HF prefetch (fresh king from HF): 362.1s for 165 GB ≈ 456 MB/s
```

Probe sanity: `min_loss_before=13.29, max_loss_after=13.30` — random-init
80B has loss ≈ ln(262144) = 12.48 nats/token but the probe's input is uniform-random
tokens (not natural language) so the slightly-elevated 13.29 is expected.

One soft warning: `norm_quantization=0.503 >= 0.50` — RMSNorm weights are all
init'd to 1.0, so they cluster trivially. **Will go away once the king is real-trained.**

### 2. Live smoke — king + perturbed challenger from HF, real v2 shard (eval `03306e07`)

End-to-end test of the full production code path with FRESH HF download for
the challenger.

| Phase | Wall | Notes |
|---|---:|---|
| King prefetch | 0.2 s | cache hit |
| King load | 25.2 s | sharded GPUs 0-3, modules 10/10/10/10 |
| **King probe** | **5.8 s** | **ok=True**, max_grad 2.21e+02 |
| Challenger HF prefetch | 403.9 s | fresh 164 GB download, 405 MB/s |
| Challenger load | ~20 s | sharded GPUs 4-7, modules 11/10/10/9 |
| **Challenger probe** | **3.6 s** | **ok=True**, max_grad 2.21e+02 |
| Bootstrap | 132.1 s | 15.1 seq/s, EVAL_N=2000 on real Gemma3-tokenized shard |
| **Total** | **594 s ≈ 9.9 min** | matches the predicted 10-11 min |

Verdict: `mu_hat=0.000899, lcb=-0.001501, delta=0.0005, accepted=false` —
correctly rejects the 1e-4 noise challenger. `avg_king_loss=13.297` and
`avg_chall_loss=13.296` differ by ~1e-3 nats/token, consistent with the
noise level applied.

### 3. HF cache eviction at watermark (evals `b67c8249` + `3526ada7`)

Forced the eviction path by restarting eval-server with
`HF_CACHE_HIGH_WATERMARK_GB=200` (vs prod's 600 GB) and ran two back-to-back
evals.

**1st eval (cache pre-eval = 308 GB, both king + chall cached):**
- 17:18:14 bootstrap done, mu_hat=7.8e-5
- 17:18:14 `marking for deletion: unconst/Teutonic-LXXX-mock-chall rev 44b2728c0b95 (164692.9 MB)`
- 17:18:14 `hf cache cleanup: deleting 1 revisions, freeing 164692.9 MB (cache was 329.4 GB)`
- 17:18:24 `hf cache cleanup: done` (10 s wall to delete 164 GB)
- Cache now 155 GB (= king 153 + Teutonic-I tokenizer 2). King correctly preserved.

**2nd eval (chall NOT in cache, requires fresh HF download):**
- 17:38:27 `challenger-shard prefetch complete in 259.6s` (164 GB at 645 MB/s)
- 17:38:49 `challenger-shard sharded: 82.3B params in 281.7s` (load = 22 s post-prefetch)
- 17:38:52 chall probe ok=True, 2.9 s
- 17:39:06 bootstrap done, mu_hat=0.006488
- 17:39:06-17:39:21 cache cleanup again (164 GB freed in 15 s)

The cleanup correctly KEEPS the king and Teutonic-I tokenizer; only the
just-evaluated challenger gets evicted.

### 4. Cold-disk-cache anomaly (worth noting, not blocking)

The 1st eval after the perturb step showed a **216 s challenger load**
(vs the typical ~22 s). Root cause: the perturb workload generated ~328 GB
of disk I/O which evicted the chall's bytes from Linux page cache. Without
page-cache-warm bytes, the safetensors mmap reads have to come from disk
(NVMe at ~700 MB/s × 153 GB = ~218 s), matching the observed wall.

**Implication for production:** the **first eval after a king coronation OR
after any large competing IO will be ~3.6 minutes slower** than steady
state. Total per-eval wall in that case: ~13-14 min instead of ~10 min.
Still well within `TICK_RESTART_AFTER=3600 s` (5× safety margin).

---

## Per-eval wall budget (production-grade)

For a steady-state eval (king cached in VRAM + on disk, challenger fresh from HF):

| Phase | Wall (steady) | Wall (cold disk-cache, after long IO) |
|---|---:|---:|
| Challenger HF prefetch (164 GB) | ~270 s | ~270 s |
| Challenger model load | ~22 s | ~218 s |
| Challenger probe | ~4 s | ~4 s |
| Bootstrap @ EVAL_N=2000 | ~132 s | ~132 s |
| Cleanup + cache evict | ~15 s | ~15 s |
| **Total** | **~7.4 min** | **~10.7 min** |

For `EVAL_N=5000` (the originally planned production setting), bootstrap
grows to ~330 s, total ~10.6 min steady / ~13.9 min cold.

**Throughput:** ~6–8 evals/hour steady state at `EVAL_N=2000`,
~4.5–5.5 evals/hour at `EVAL_N=5000`. Roughly half of the live Quasar 24B chain's
~11.5 evals/hr. Backlog risk on busy days; watch `dashboard.json -> queue` after cutover.

---

## Per-GPU VRAM (B300 cap = 275 GiB)

From the live smoke + soak-10 numbers (B300 measurements):

| GPU | Side | Peak (MiB) | Headroom |
|---:|---|---:|---:|
| 0 | king | 76,568 | 197 GiB free |
| 1 | king | 71,596 | 199 GiB free |
| 2 | king | 71,596 | 199 GiB free |
| 3 | king | 58,678 | 211 GiB free |
| 4 | chall | 89,372 | 188 GiB free |
| 5 | chall | 86,996 | 190 GiB free |
| 6 | chall | 86,996 | 190 GiB free |
| 7 | chall | 52,558 | 213 GiB free |

Even on the smaller B200 (180 GiB cap), peak 89 GiB on GPU 4 leaves 91 GiB
headroom — but we're not deploying to B200 anymore.

---

## Bugs found and fixed during this session

1. **Trainability probe in sharded mode (`eval_server.py:586`)**
   `chall_model = challenger_eval.models[challenger_eval.gpu_ids[0]]` —
   KeyError in sharded mode (only `SHARDED_KEY` entry exists). Fixed to
   `challenger_eval.primary_model`. Same fix already applied to the king
   probe at line 286.

2. **Probe target/logits-device mismatch (`eval/torch_runner.py:_probe_one_seed`)**
   Targets and logits could land on different devices in sharded models.
   Added an explicit `targets.to(logits.device)` guard. Tied embeddings make
   them coincide today, but defense-in-depth for future variants.

3. **`Qwen3MoeConfig.rope_theta` flattened in transformers 5.5+**
   `seed.py` now reads `rope_parameters` dict (with legacy `rope_theta`
   fallback). Added `rope_parameters` to `chain.lxxx.toml` extra_lock_keys.

4. **`compute_batch_losses` missing `@torch.no_grad()`**
   (caught in original smoke, fix already in [`LXXX_SANDBOX_REPORT.md`](LXXX_SANDBOX_REPORT.md))

5. **`sandbox_perturb.py` was using fp32 cast** for noise add — slow.
   Switched to bf16-direct (saves ~2-3× wall on the perturb step).

---

## Production cutover checklist

I'm stopping here so you can confirm before flipping prod. When you're ready:

### A. On the validator host (templar / wherever pm2 runs)

```bash
# 1. Cut over chain.toml
cd /home/const/workspace
mv chain.toml chain.xxiv.toml.bak       # archive Quasar config
cp chain.lxxx.toml chain.toml           # promote LXXX

# 2. Bump validator env in ecosystem.config.js (lines 50-61):
#    TEUTONIC_TICK_RESTART_AFTER: "1800" -> "3600"
#    TEUTONIC_STREAM_IDLE_WARN_AFTER: "300" -> "600"
#    TEUTONIC_STREAM_IDLE_TIMEOUT: "900" -> "1800"

# 3. Update tunnel.sh — point at the B300 box (95.133.252.44:10310) instead
#    of the old B200 pod (95.133.252.200:10100):
#    sed -i 's/95.133.252.200/95.133.252.44/; s/-p 10100/-p 10310/' tunnel.sh

# 4. Restart pm2
pm2 restart teutonic-eval-tunnel teutonic-validator
pm2 logs teutonic-validator --lines 200
```

### B. On the B300 eval box (root@95.133.252.44 -p 10310)

The eval-server is currently running in tmux session `eval` with the
correct sharded-mode env (TEUTONIC_SHARD_ACROSS_GPUS=1, TEUTONIC_PROBE_ENABLED=1,
TEUTONIC_CHAIN_OVERRIDE=chain.lxxx.toml, HF_CACHE_HIGH_WATERMARK_GB=200 — bump
back to 600 for prod).

For prod, bring the eval-server up under a supervisor (pm2 or systemd) with:

```bash
TEUTONIC_CHAIN_OVERRIDE=chain.toml          # not chain.lxxx.toml after cutover
TEUTONIC_SHARD_ACROSS_GPUS=1
TEUTONIC_PROBE_ENABLED=1
EVAL_N=5000                                  # back to prod default (2000 was for smoke only)
EVAL_BATCH_SIZE=32
EVAL_BOOTSTRAP_B=10000                       # back to prod default
HF_CACHE_HIGH_WATERMARK_GB=600              # prod value
HF_PREFETCH_TIMEOUT=1800
EVAL_MAX_RUNTIME_S=3600
TEUTONIC_SHARD_PER_GPU_GIB=240               # B300 default (would be 120 on B200)
```

The current mock-king at `unconst/Teutonic-LXXX-mock-king` is **random init**
(loss ≈ 13.3 on real data). If you want any miner to face a non-trivial king,
that needs to be replaced with a real Phase A seed checkpoint BEFORE cutover.
Otherwise the first competent training run trivially dethrones with `mu_hat`
in the multi-nat range. Earlier guidance still stands: budget 5 days +
~$50–80k of cluster time for a 500B-token genesis seed.

If you're OK going live with the random-init king (i.e. let the network's
first dethrones be near-trivial wins until enough real training accumulates):
just rename `unconst/Teutonic-LXXX-mock-king` to `unconst/Teutonic-LXXX`
on HF (or update `chain.toml [chain].seed_repo`).

### C. Discord announcement (required, mining contract changes)

Post 24 h ahead on `γ・τeuτonic・3`:
- Repo regex changes from `Teutonic-XXIV-` to `Teutonic-LXXX-`
- Arch lock changes from Quasar to vanilla Qwen3MoE (in-tree transformers, no `trust_remote_code`)
- Minimum miner spec: ≥ 4× B200/B300 to load 153 GB bf16 base
- Per-eval wall doubles, network throughput drops ~2×
- LXXX appendix in [`docs/MINING.md`](MINING.md) Appendix A

---

## What's still NOT proven (acceptable risks per user)

- **No real Phase A seed** — random-init king. First few dethrones will be
  trivial. User's call whether to ship like this or train a seed first.
- **Reparameterization-trick gate** — defended Quasar, untested on Qwen3MoE.
  Same RMSNorm/SwiGLU patterns so it should carry over, but no specific test.
  Recommend running `audit_king_on_startup` once the validator is up and
  watching for unexpected rejections.
- **Mining harness `scripts/mining/train_challenger.py`** — has Quasar-specific
  notes (SMEBU, latent memory, eager attn). Miners would have to rewrite the
  training recipe themselves. Not a blocker for the chain to function, but
  is a blocker for any miner without serious ML experience.
- **Reign-2+ behavior** — only smoked with mock-king as both king and challenger
  in various combos. The actual "miner submits NEW challenger → it wins → it
  becomes king → next miner submits → ..." flow is not exercised because we
  only have one mock chall on HF. Validator would need to be live to test
  this end-to-end.
- **Long-running soak (24 h+)** — caught no leaks in 10 iters; longer runs
  would be more confidence but are not strictly needed.

---

## State left on the box

- `unconst/Teutonic-LXXX-mock-king` on HF (vocab=262144, random init, 165 GB)
- `unconst/Teutonic-LXXX-mock-chall` on HF (king + N(0, 1e-4), 165 GB)
- Eval-server running in tmux session `eval` on `95.133.252.44:10310:9000`,
  PID 474765, with PROBE_ENABLED=1 and HF_CACHE_HIGH_WATERMARK_GB=200
- HF cache: 155 GB (king + Teutonic-I tokenizer; chall was evicted by the
  test)
- Branch `teutonic-lxxx-sandbox` in `/home/const/workspace` has all the code
  changes; `chain.toml` (live Quasar config) is UNTOUCHED
