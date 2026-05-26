# Teutonic-LXXX Operator Runbook

> Historical archive. This runbook documents the earlier Qwen3-MoE 80B cutover plan. The live chain is now `Teutonic-Q3-8B`; follow [`MINING.md`](MINING.md) and [`../chain.toml`](../chain.toml) for the current flow.


Companion to [`/home/const/.cursor/plans/teutonic-lxxx-80b-qwen3moe_90e4f74e.plan.md`](../../.cursor/plans/teutonic-lxxx-80b-qwen3moe_90e4f74e.plan.md).
This file is the **gate-by-gate operator script** for cutting Bittensor SN3 from the live Quasar 24B chain (`Teutonic-XXIV`) to a vanilla Qwen3-MoE 80B chain (`Teutonic-LXXX`).

The plan is deliberately split: Phase 0 is the sandbox soak we run on the
spare 8×B300 box (`95.133.252.44:10310`) to prove the eval pipeline survives
80B sharded models. Phases 1+ touch the live network and are gated on the
sandbox going green.

---

## Phase 0 — Sandbox soak (this session)

Code: `teutonic-lxxx-sandbox` branch in `/home/const/workspace`.
Box: `root@95.133.252.44 -p 10310` (8× B300 SXM6 / 275 GiB / 871 GiB free /).
HF cache: `/workspace/hf-cache` (NOT `/root/.cache/huggingface` — `/root` is on a 212 GiB loop mount that won't fit two 80B safetensors).

State on the box:

- `/root/teutonic/` — rsynced repo, branch `teutonic-lxxx-sandbox`
- `/root/teutonic/.venv/` — uv venv, python 3.12, torch 2.11.0+cu130, transformers 5.8.0, accelerate 1.13
- `/root/.creds/hf_token.env` — exports `HF_TOKEN`, `HF_HOME=/workspace/hf-cache`, `HF_HUB_ENABLE_HF_TRANSFER=1`
- `/root/run-seed.sh` — wrapper to `python scripts/seed.py --push --no-probe --public` with the LXXX env knobs
- `/workspace/seed-out` — local model dir for the genesis seed (~152 GiB)
- `/workspace/sandbox-mock/` — perturb workdir (~304 GiB peak: base + challenger)
- `/workspace/logs/` — log files

HF repos used by the soak (all created `--public`):

- `unconst/Teutonic-LXXX-mock-king` — random-init 80B Qwen3MoE (genesis stand-in)
- `unconst/Teutonic-LXXX-mock-chall` — same king + N(0, 1e-4) noise (paired-CE control)

Soak commands (in order):

```bash
# 0. SSH in
ssh -o StrictHostKeyChecking=no -p 10310 root@95.133.252.44

# 1. Confirm env (idempotent)
cd /root/teutonic && source .venv/bin/activate && source /root/.creds/hf_token.env

# 2. Sizer dry-run (~5 s)
TEUTONIC_CHAIN_OVERRIDE=chain.lxxx.toml python -m archs.qwen3_moe.size

# 3. Genesis seed — random init + push
tmux new -d -s seed '/root/run-seed.sh; sleep 60'
tmux attach -t seed   # ctrl-b d to detach

# 4. Mock challenger — download king, perturb, push
tmux new -d -s perturb '
  cd /root/teutonic && source .venv/bin/activate && source /root/.creds/hf_token.env
  python scripts/sandbox_perturb.py \
    --base unconst/Teutonic-LXXX-mock-king \
    --upload-repo unconst/Teutonic-LXXX-mock-chall \
    --noise 1e-4 \
    2>&1 | tee /workspace/logs/perturb.log
  sleep 60
'

# 5. Eval-server in sharded mode
tmux new -d -s eval '
  cd /root/teutonic && source .venv/bin/activate && source /root/.creds/hf_token.env
  TEUTONIC_CHAIN_OVERRIDE=chain.lxxx.toml \
  TEUTONIC_SHARD_ACROSS_GPUS=1 \
  TEUTONIC_PROBE_ENABLED=0 \
  EVAL_N=2000 \
  EVAL_BATCH_SIZE=32 \
  HF_CACHE_HIGH_WATERMARK_GB=600 \
  HF_PREFETCH_TIMEOUT=1800 \
  uvicorn eval_server:app --host 127.0.0.1 --port 9000 \
    2>&1 | tee /workspace/logs/eval-server.log
'

# 6. Smoke /eval
curl -N -X POST http://127.0.0.1:9000/eval \
  -H 'content-type: application/json' \
  -d '{"king_repo":"unconst/Teutonic-LXXX-mock-king",
       "challenger_repo":"unconst/Teutonic-LXXX-mock-chall",
       "block_hash":"smoke","hotkey":"smoke",
       "shard_key":"dataset/v2/shards/shard_000000.npy",
       "eval_n":2000,"alpha":0.001,"seq_len":2048,
       "batch_size":32,"n_bootstrap":2000}' \
  | tee /workspace/logs/smoke-eval-sse.log

# 7. Per-GPU memory capture (run in second pane during step 6)
nvidia-smi dmon -s mu -i 0,1,2,3,4,5,6,7 -d 5 \
  | tee /workspace/logs/nvidia-smi-dmon.log
```

Pass criteria for sandbox-go (must all hold):

- Both replicas load (no OOM, no `RuntimeError: could not load model with any attention implementation`)
- Peak per-GPU `mem_used` ≤ 165 GiB on every GPU (15 GiB headroom under the 180 GiB cap we'd see on B200 — B300 has even more)
- Final SSE `verdict` event has finite `mu_hat` (not NaN/Inf) and a sensible `lcb`
- Wall ≤ 25 min for `EVAL_N=2000` (linear projection ≤ 60 min at full `EVAL_N=5000`)
- No `_self_kill_scheduled` event in `/workspace/logs/eval-server.log`

If all pass → mark `sandbox-report` complete and decide whether to also do
the 10× soak (recommend yes) before touching live network.

---

## Phase 1 — Live eval-box env diff (NOT in this session)

Apply on `root@95.133.252.200:10100` (live eval pod, 8×B200) ONLY after sandbox is green AND the v3 dataset + Phase A seed both exist:

```bash
# /etc/systemd or pm2 ecosystem on the eval box (whichever runs eval_server)
export TEUTONIC_SHARD_ACROSS_GPUS=1
export EVAL_N=5000                    # was 10000; cuts wall in half, doubles delta floor 1e-4 -> 2e-4
export HF_CACHE_HIGH_WATERMARK_GB=600 # was 200; fits king + 2 challengers @ ~160 GiB
export HF_PREFETCH_TIMEOUT=1800       # was 600; 160 GiB prefetch needs > 600 s
export TEUTONIC_CHAIN_OVERRIDE=chain.lxxx.toml   # only after live cutover; before that, leave unset

# Verify disk:  >= 1 TB free under HF_HOME
df -h ~/.cache/huggingface
# Verify B200 budget:  TEUTONIC_SHARD_PER_GPU_GIB=150 (default 240 is for B300)
export TEUTONIC_SHARD_PER_GPU_GIB=150

pm2 restart teutonic-eval
```

## Phase 2 — Live validator env diff (NOT in this session)

Apply on the templar host ONLY after Phase 1 is in place. Edit
[`ecosystem.config.js`](../ecosystem.config.js) lines 50-61:

```diff
-      TEUTONIC_TICK_RESTART_AFTER: "1800",
+      TEUTONIC_TICK_RESTART_AFTER: "3600",
       TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS: "20",
-      TEUTONIC_STREAM_IDLE_WARN_AFTER: "300",
-      TEUTONIC_STREAM_IDLE_TIMEOUT: "900",
+      TEUTONIC_STREAM_IDLE_WARN_AFTER: "600",
+      TEUTONIC_STREAM_IDLE_TIMEOUT: "1800",
```

Then:

```bash
pm2 restart teutonic-validator
pm2 logs teutonic-validator --lines 200 | grep -iE 'tick|stream|watchdog'
```

Watch `dashboard.json -> current_eval.stage_elapsed_s` for the first 24 h —
if any stage runs > 1500 s the watchdog will trip and the miner gets
re-queued.

## Phase 3 — Dataset v3 (separate compute job)

See [`scripts/tokenize_qwen.py`](../scripts/tokenize_qwen.py) for the script
skeleton. Required infrastructure:

- 64-core CPU box with ≥ 4 TB local SSD
- Hippius S3 credentials with write access to `s3://teutonic-sn3/dataset/v3/`
- HF read access to `uonlp/CulturaX` and `Qwen/Qwen3-30B-A3B`
- Wall: ~3-5 days for the full CulturaX dump (~700-900 B Qwen tokens)

```bash
python scripts/tokenize_qwen.py \
  --tokenizer Qwen/Qwen3-30B-A3B \
  --source uonlp/CulturaX \
  --dest s3://teutonic-sn3/dataset/v3/ \
  --shard-tokens 536870912 \
  --workers 60
```

## Phase 4 — Phase A genesis training (separate $50-80k compute job)

See plan §8. Required infrastructure:

- 4-node 8×B200 cluster (32 B200 total) for ~5.5 days
- ~2.8 TB Hippius dataset access
- Saved checkpoint pushed to `unconst/Teutonic-LXXX-pretrain` ON COMPLETION,
  then promoted to `unconst/Teutonic-LXXX` (the chain.lxxx.toml `seed_repo`)

Hyperparameters:

- transformers `Qwen3MoeForCausalLM` + FSDP2 / DeepSpeed ZeRO-3 + EP=4
- AdamW bf16, peak LR 3e-4, cosine to 3e-5 over 500 B tokens, warmup 1 B tokens
- Seq len 2048, micro-batch 4, global batch ~4 M tokens
- Gradient clip 1.0; `router_aux_loss_coef = 0.001`
- Activation checkpointing on every MoE layer
- Save every 10 B tokens; keep last 5 checkpoints + the best-val-loss checkpoint

## Phase 5 — Cutover (NOT in this session)

Order of operations after all of Phase 1-4 are in place:

1. Announce cutover window on `γ・τeuτonic・3` Discord 24 h ahead. Mention:
   - New repo regex: `^[^/]+/Teutonic-LXXX-.+$`
   - Quasar arch lock dies, Qwen3MoE arch lock takes over
   - Minimum miner spec: ≥ 4× B200 (or ≥ 2× B300) to load the 80B base
2. `git checkout main && git merge teutonic-lxxx-sandbox` on dev workspace
3. `mv chain.toml chain.xxiv.toml.bak && mv chain.lxxx.toml chain.toml`
   (or just `cp chain.lxxx.toml chain.toml` if you want both checked in)
4. Commit + push to GitHub
5. On the eval box: `git pull && pm2 restart teutonic-eval`
6. On the validator host: `git pull && pm2 restart teutonic-validator`
7. Watch `audit_king_on_startup` log line for the LXXX chain — should pick up
   `unconst/Teutonic-LXXX` as the live king
8. Monitor first 4 h: queue depth, stage_elapsed_s, accepted-rate

Rollback: revert step 3 + restart pm2 on validator. The live `Teutonic-XXIV`
king is unaffected throughout because we never deleted its state, only stopped
crowning under it.
