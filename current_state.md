# Teutonic LXXXI Go-Live — Handoff State

**Date:** 2026-05-21  
**Task:** Bring full validator online on Teutonic-LXXXI with raw Hippius dataset + Hippius Hub models  
**Plan reference:** `.cursor/plans/validator_lxxxi_go-live_02896c4e.plan.md` (do not edit)

---

## Executive summary

Go-live is **~80% complete**. GPU eval server is healthy and passed end-to-end smoke eval. Validator host has tunnel up and validator **started but crash-looping** on seed coronation because:

1. Seed king was **never uploaded to Hippius Hub** (401 — no `HIPPIUS_HUB_TOKEN` / console API token in Doppler).
2. Workaround: seed model pre-cached on GPU box at local digest; eval server `/hash` works (~75s).
3. Validator `_eval_server_has_digest()` uses **60s timeout** but `/hash` takes **~75s** → falls through to `ensure_ref_exists()` → Hippius registry 401 → crash.

**Immediate fix for next agent:** bump `_eval_server_has_digest` timeout to ≥120s (or skip `ensure_ref_exists` when eval-server `/hash` is pending), then `pm2 restart teutonic-validator`. Optionally add `HIPPIUS_HUB_TOKEN` to Doppler after obtaining from [console.hippius.com/dashboard/settings](https://console.hippius.com/dashboard/settings).

---

## Infrastructure map

| Component | Host | Status |
|---|---|---|
| **Validator + tunnel** | This machine (`/home/const/workspace`, templar) | Tunnel online; validator crash-looping |
| **Eval server** | `root@95.133.252.200:10099` (8× B200) | Running, healthy |
| **B300 box** | `95.133.253.18:10300` | **Not accessible** (no SSH key) |
| **Burn fallback** | This machine | **Stopped** (correct — mutually exclusive with validator) |

SSH to GPU box works with local `~/.ssh/id_ed25519`.

---

## PM2 state (validator host)

```
teutonic-burn          STOPPED
teutonic-eval-tunnel   ONLINE  (forwards localhost:9000 → GPU:9000)
teutonic-validator     ONLINE but restarting (↺ 1+); crashes on seed materialize
```

Useful commands:
```bash
pm2 ls
pm2 logs teutonic-validator --lines 100
curl -sf http://127.0.0.1:9000/health   # via tunnel
pm2 restart teutonic-validator
pm2 save   # not done yet
```

---

## GPU eval server

- **Process:** `uvicorn eval_server:app --host 127.0.0.1 --port 9000`
- **Code:** synced from workspace to `/root/teutonic/` via tar (May 21)
- **Creds:** `/root/.creds/teutonic_eval.env` (deployed by `scripts/setup_gpu_creds.sh`)
- **HF token:** `/root/.creds/hf_token.env`
- **Start script:** `bash scripts/gpu_start_eval.sh` (sources creds files)
- **Disk:** ~102 GB free on `/tmp` after 164 GB model cache — tight

### Seed king cache (workaround — not on Hippius Hub)

Because Hippius registry upload failed (401), seed is served from **local cache only**:

| Field | Value |
|---|---|
| Repo | `unconst/Teutonic-LXXXI-mock-king` |
| Digest | `sha256:0950c71f59e03211e0754d5cf484abd99c877b6c79f844457126a7f1fd1b69c8` |
| Source dir | `/tmp/teutonic-lxxxi` (153 GB, built May 19) |
| Cache path | `/tmp/teutonic/hippius_models/unconst--Teutonic-LXXXI-mock-king/snapshots/sha256-0950c71f59e03211e0754d5cf484abd99c877b6c79f844457126a7f1fd1b69c8/` |

Verify on GPU:
```bash
curl -sf "http://127.0.0.1:9000/hash?repo=unconst/Teutonic-LXXXI-mock-king&digest=sha256:0950c71f59e03211e0754d5cf484abd99c877b6c79f844457126a7f1fd1b69c8"
# Returns sha256:0950c71f... after ~75s
```

---

## Smoke eval — PASSED (attempt 3)

Log: `/workspace/logs/smoke-lxxxi-sse3.log` on GPU box

- `eval_id`: `c7a2a04d`
- `eval_n`: 200, `raw_hippius` dataset, Qwen3-30B-A3B tokenizer
- `mu_hat`: 0.0 (king == challenger, expected)
- Wall time: 15.6s after king load
- Parquet used: `hf-mirrors/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2025-05/000_00018.parquet`

Attempts 1–2 failed with `Max Retries Exceeded` on S3 parquet download (transient GPU↔Hippius issue). Fixed by:
- Pinning `TEUTONIC_RAW_DATASET_KEYS` to that parquet file
- Pre-caching parquet at `/tmp/teutonic_raw_dataset/`
- Bumping S3 `read_timeout` 300→600, retries 5→10 in `eval/torch_runner.py`

---

## Validator startup — BLOCKED

Last crash (`~/.pm2/logs/teutonic-validator-error.log`):
```
TEUTONIC_FORCE_SEED_KING=1: replacing king with chain seed
→ ensure_ref_exists(seed_ref)
→ hippius_hub snapshot_download 401 Unauthorized
   registry.hippius.com/.../unconst/Teutonic-LXXXI-mock-king/manifests/sha256:0950c71f...
```

Root cause chain:
1. `_eval_server_has_digest()` GET `/hash` timed out at 60s (hash compute takes ~75s on GPU)
2. Fell back to `ensure_ref_exists()` which tries Hippius Hub download
3. No `HIPPIUS_HUB_TOKEN` → 401

### Code change already made (validator.py)

Added `_eval_server_has_digest()` and conditional skip of `ensure_ref_exists` when eval-server cache hit — **but timeout too short**.

```python
# validator.py ~line 775 — needs timeout bump from 60.0 to 180.0
def _eval_server_has_digest(repo: str, digest: str) -> bool:
    ...
    timeout=httpx.Timeout(60.0)  # ← TOO SHORT; /hash takes ~75s
```

---

## Config changes made (workspace)

| File | Change |
|---|---|
| `chain.toml` | `seed_digest = "sha256:0950c71f..."` |
| `ecosystem.config.js` | Added `TEUTONIC_SEED_DIGEST`, `HF_TOKEN` (from Doppler prd), `TEUTONIC_RAW_DATASET_KEYS` |
| `ecosystem.config.js` | **`TEUTONIC_FORCE_SEED_KING: "1"` still set** — remove after successful coronation |
| `validator.py` | `_eval_server_has_digest()` + conditional seed verify |
| `eval/torch_runner.py` | S3 ds_client: read_timeout 600, retries 10 |
| `scripts/gpu_start_eval.sh` | Source creds files; export `HIPPIUS_HUB_TOKEN` from cache if present |
| `scripts/setup_gpu_creds.sh` | **NEW** — deploy creds + pre-seed king cache to GPU via SSH |

---

## Hippius auth — NOT RESOLVED

**Blocker for real Hub upload/download of challenger models.**

- Doppler `arbos/dev` has `HIPPIUS_ACCESS_KEY` / `HIPPIUS_SECRET_KEY` (S3) but **no `HIPPIUS_HUB_TOKEN`** or console API token.
- Seed push to Hippius failed May 19 and May 21 with `401 Unauthorized` on registry blob upload.
- Obtain token: [hippius-hub quickstart](https://github.com/thenervelab/hippius-hub#quickstart) → `hippius-hub login --hippius-token <token>` → `hippius-hub registry provision unconst --docker-login`
- Then: `python archs/qwen3_moe/seed.py --push --hippius --no-probe` on GPU (model already at `/tmp/teutonic-lxxxi`)
- Add token to Doppler as `HIPPIUS_HUB_TOKEN`.

S3 keys do **not** work for OCI registry (service token 200 but manifest/upload 401).

---

## Dataset

- FineWeb-Edu mirror on Hippius S3: **ready** (`_manifest.json` ~527 KB)
- Bucket: `teutonic-sn3`, prefix: `hf-mirrors/HuggingFaceFW/fineweb-edu/data`
- Both validator and eval server use `TEUTONIC_EVAL_DATASET_MODE=raw_hippius`
- Pinned eval parquet: `CC-MAIN-2025-05/000_00018.parquet` (~958 MB)

---

## R2 validator state

Before go-live: stale LXXX king (`ClarenceDan/Teutonic-LXXX-...`).  
At handoff: state may be inconsistent — validator crashed mid-coronation. Check:
```bash
source .venv/bin/activate
python -c "
from eval.torch_runner import R2
st = R2().get('state/validator_state.json')
print(st.get('king'))
"
```

---

## Todo checklist (from plan)

| ID | Task | Status |
|---|---|---|
| hippius-auth | Registry docker creds + Doppler | **Partial** — creds file on GPU; no Hub token |
| seed-push | Push seed, record digest | **Partial** — digest recorded; not on Hub (cache workaround) |
| gpu-sync | Sync code, restart eval | **Done** |
| smoke-eval | POST /eval smoke | **Done** (attempt 3) |
| start-validator | Stop burn, start tunnel+validator | **Partial** — started but crash-looping |
| post-cutover | Remove FORCE_SEED_KING, pm2 save | **Not done** |

---

## Recommended next steps (priority order)

1. **Fix validator startup**
   - In `validator.py`, change `_eval_server_has_digest` timeout to `180.0` (match `_seed_king_hash`)
   - Ensure tunnel is up: `curl http://127.0.0.1:9000/health`
   - `pm2 restart teutonic-validator`
   - Watch for: `seed ... confirmed on eval-server cache`, `seed king unconst/Teutonic-LXXXI-mock-king@sha256:...`, `validator running`

2. **Confirm coronation**
   - Check R2 state king fields: `model_repo`, `king_digest`, `king_hash`
   - Confirm weights set on chain

3. **Post-cutover cleanup**
   - Remove `TEUTONIC_FORCE_SEED_KING: "1"` from `ecosystem.config.js`
   - `pm2 restart teutonic-validator && pm2 save`

4. **Hippius Hub (production blocker for miner evals)**
   - Get console API token → provision `unconst` namespace → upload seed
   - Add `HIPPIUS_HUB_TOKEN` to Doppler `arbos/dev`
   - Re-run `scripts/setup_gpu_creds.sh` or manually export on GPU

5. **Monitor first live eval**
   - Miners must use v2 Hippius reveals: `v2|{hash16}|{repo}|sha256:{digest}`
   - Legacy HF reveals are dropped
   - First challenger eval needs working Hippius Hub download (not just cache)

---

## Key paths & commands

```bash
# Validator host
cd /home/const/workspace && source .venv/bin/activate
pm2 ls
pm2 logs teutonic-validator --lines 200

# GPU box
ssh root@95.133.252.200 -p 10099
tail -f /workspace/logs/eval-server-restart.log
bash scripts/setup_gpu_creds.sh   # from validator host

# Health checks
curl http://127.0.0.1:9000/health                    # via tunnel
curl "http://127.0.0.1:9000/hash?repo=unconst/Teutonic-LXXXI-mock-king&digest=sha256:0950c71f59e03211e0754d5cf484abd99c877b6c79f844457126a7f1fd1b69c8"
```

---

## Git / uncommitted changes

All go-live changes are **local uncommitted** modifications to:
- `chain.toml`, `ecosystem.config.js`, `validator.py`, `eval/torch_runner.py`
- `scripts/gpu_start_eval.sh`, `scripts/setup_gpu_creds.sh` (new)

Do not commit unless user asks.
