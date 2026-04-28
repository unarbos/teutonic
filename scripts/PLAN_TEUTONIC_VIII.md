# Teutonic-VIII Cutover Plan

> Target: replace the live Teutonic-III king (Gemma4 ~3.3B) with a freshly
> initialized Qwen3 ~8B king ("Teutonic-VIII"). Same playbook as the
> Teutonic-I → Teutonic-III cutover. This document is the contract followed
> by the validator restart and the announcement post.

## TL;DR

- **New model family:** Qwen3 (`Qwen3ForCausalLM`, transformers 5.5.3,
  `model_type="qwen3"`)
- **Size:** **8.02B params**, 36 layers, hidden 4096, vocab 262144 (tied embed)
- **Repo:** `unconst/Teutonic-VIII` (private during build, public at cutover)
- **Tokenizer:** reused from `unconst/Teutonic-I` (vocab 262144 — same tokens
  as existing dataset shards; no re-tokenization needed)
- **Eval pipeline:** unchanged. `Qwen3ForCausalLM.model(...)` returns a
  `BaseModelOutputWithPast` (has `last_hidden_state`); `model.lm_head` is
  `nn.Linear`. Drop-in compatible with `eval_torch.compute_paired_losses`.
- **No carryover from Teutonic-III history.** The R2 dashboard, queue, and
  seen-hotkeys state are wiped (with the old dashboard archived for posterity);
  the validator restarts with `unconst/Teutonic-VIII` as seed king.

## Why Qwen3, not "Gemma5" or scaled Gemma4

User picked an SOTA-ish ~8B architecture. Two real candidates were Llama-3.1
8B and Qwen3 8B; both are clean RMSNorm + RoPE + SwiGLU + GQA decoder-only.
Qwen3 wins on:

- **Per-head q/k norm** (`q_norm`, `k_norm` of shape `(head_dim,)` applied
  pre-RoPE). Tighter attention numerics. Same trick Gemma3 used; familiar to
  the existing reparam-trick guard pattern (see DESIGN.md §7).
- **Distinct `model_type`** (`"qwen3"`) cleanly partitions the family from
  Teutonic-III's `"gemma4_text"`, so `validate_challenger_config` rejects any
  cross-family submissions automatically.
- **No per-layer-input embeddings**. Gemma4 inflates param count by
  `vocab × hidden_per_layer × num_layers`; on a 262k vocab that's 1.2B+ at
  18 layers, growing linearly. Qwen3 keeps all 8B in the actual transformer
  rather than spending ~25% on an embedding shenanigan.

## Why hidden=4096, layers=36, ffn=12288

Aiming for **~8.0B total** with the existing 262144 tokenizer.

| component                                          | params |
|----------------------------------------------------|-------:|
| `model.embed_tokens` (262144 × 4096, tied to lm_head) | 1.074B |
| 36 × decoder layers (attn + SwiGLU MLP + norms)       | ~6.946B |
| `model.norm` (final RMSNorm)                          | ~negl. |
| **TOTAL**                                             | **8.020B** |

Verified locally via `from_config` + `sum(p.numel())`. Shape mirrors a
Llama-3.1-style 8B but uses Qwen3's per-head q/k norm.

`tie_word_embeddings=True`, so `lm_head.weight` shares storage with
`embed_tokens.weight` (counted once).

## Compatibility verification (Phase A — MUST be green before cutover)

The validator tickling random repos in the wild is unforgiving — we verify
the eval server can load and probe the new arch BEFORE turning over.

1. **Local CPU build & save** of seed model via
   `scripts/seed_teutonic_viii.py --no-probe --no-push`; confirms
   `Qwen3Config + AutoModelForCausalLM.from_config` works in our
   transformers 5.5.3 (already verified — 8.020B).
2. **Push private** to `unconst/Teutonic-VIII`.
3. **scp seed script + smoke script** to the live Targon B200 box
   (`wrk-0638a6gucc7t`).
4. On Targon, run:
   - `seed_teutonic_viii.py --no-push` (rebuild locally there + run
     `trainability_probe`); confirms the probe returns `ok=True` with
     finite delta (untrained model has high but stable loss; one SGD step
     of LR 1e-5 should not blow it up).
   - `smoke_eval_teutonic_viii.py` (king == challenger): full pipeline
     including `MultiGPUEvaluator`, paired bootstrap on a real shard.
     Required:
     - both models load with `flash_attention_2` or `sdpa`
     - probe returns `ok=True`
     - paired loss diff is exactly 0 (same model, identical seeds)
     - no OOM at batch=128 seq=2048 (lower than Teutonic-III's 256 because
       8B is 2.4× larger; if 256 fits we keep it)
     - 8B eval throughput projection ≥ Teutonic-III's ~38 seq/s at parallel
       split / 2 (since each forward is ~2.4× the FLOPs)

5. Only if all 4 above pass → proceed to cutover.

## Edge cases & guardrails

| risk | mitigation |
|------|-----------|
| Validator caches king config; old Teutonic-III still in `_king_config` | `state.set_king` already nulls the cache. Restart guarantees a clean process. |
| Eval server caches king MultiGPUEvaluator | `_ensure_king` reloads on `repo` change; restart of `eval-tunnel` is not enough — but the eval server itself runs persistently on Targon. Cutover step hits `/eval` with `king_repo=unconst/Teutonic-VIII` which triggers reload. Probe will run on the new king. |
| Eval server's HF cache has old Teutonic-III revisions | `_cleanup_hf_cache` runs after every eval; harmless. |
| Live state on R2 has Teutonic-III king pinned | wipe `king/current.json`, `state/queue.json`, `state/seen_hotkeys.json`, `state/validator_state.json`, `state/dashboard_history.json`, and `eval/*` objects. Validator startup with empty state will set seed king to `SEED_REPO`. |
| Old `dashboard.json` shows Teutonic-III history | archive to `dashboard-teutonic-iii-final.json` (Hippius and R2) — link from announcement. |
| Existing miners still submitting `Teutonic-III-*` repos | `REPO_PATTERN` is now `^[^/]+/Teutonic-VIII-.+$`; old repos rejected at scan time. They never enter the queue. |
| Existing miners' on-chain commit format | unchanged: `<king_hash[:16]>:<hf_repo>:<model_hash>`. Old commits with king_hash matching old Teutonic-III will be filtered by `check_stale` once a new king is crowned, and by `REPO_PATTERN` even before that. |
| Hippius / R2 dataset shards format | unchanged. Tokenizer is identical (re-used from Teutonic-I). The shards are still valid because token IDs are the same vocab. |
| `validate_challenger_config` config-match | the field set checked is {vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, model_type}. Qwen3 sets `model_type="qwen3"` (vs Gemma4's `gemma4_text`), so a Teutonic-III-shaped challenger gets rejected at config-match step. Good. |
| Trainability probe LR (1e-5) on Qwen3 | The probe is arch-agnostic — single SGD step on random tokens, snapshot/restore of all params. Same code path as Teutonic-III. Qwen3's q/k norm + RoPE flow through normally. |
| `chunk_size` in `compute_paired_losses` (256) for vocab 262144 | unchanged from Teutonic-III — already validated. |
| `ecosystem.config.js` SEED_REPO | flip from `unconst/Teutonic-III` to `unconst/Teutonic-VIII`. |
| Discord announcement timing | Discord notifications are currently disabled in production (per 2026-04-26 operator decision). Post the cutover announcement manually if/when re-enabled. |
| First eval after cutover loads Qwen3 8B on the eval server, which has never seen it | The server uses `AutoModelForCausalLM.from_pretrained` and tries `flash_attention_2 → sdpa → eager`. Qwen3 supports `sdpa`; will succeed there even if FA2 is not yet wired for it. |
| Eval server might OOM on Qwen3 8B | 8B × 2 bytes (bf16) = 16 GB per model. King + challenger across 4-GPU-each split = 64 GB just for weights, plus activations at batch=256 seq=2048. B200 has 180 GB per GPU (1.46 TB/box). Fine, but we'll smoke test before we trust it. |
| Fresh-init Teutonic-VIII has high uniform loss (~ln(262144) ≈ 12.5) | Identical situation to Teutonic-III seed. The probe accepts (delta is tiny because gradient on random init is not destabilizing); the bootstrap test trivially shows challenger == king (no advantage). First real challenger to actually train will dethrone it. |
| Mining harness (`scripts/mining/*`) hardcodes `Teutonic-III` repo names | Update `UPLOAD_REPO` defaults and the `train_challenger` references; not a release blocker because miners use their own names, but our internal harness is stale otherwise. |
| Existing 5-king chain (`whiskeyman, sniper918, tom6979, tom6979, whiskeyman`) | Wiped by the cutover. The Teutonic-VIII chain restarts from the seed king (validator's own hotkey) — same behavior as Teutonic-III seed. |

## Execution order (only after Phase A passes)

1. `pm2 stop teutonic-validator teutonic-eval-tunnel`
2. R2 + Hippius: copy `dashboard.json` → `dashboard-teutonic-iii-final.json`
3. R2: append to `state/history.jsonl`:
   `{"event":"family_transition","from":"Teutonic-III","to":"Teutonic-VIII","arch":"Qwen3","params_b":8.02,"timestamp":"..."}`
4. R2: delete `king/current.json`, `state/queue.json`,
   `state/seen_hotkeys.json`, `state/validator_state.json`,
   `state/dashboard_history.json`, all `eval/*` objects
5. HF: `unconst/Teutonic-VIII` → public; capture commit SHA
6. git: commit `teutonic-viii` branch (this plan + scripts + code swap),
   merge to main, push
7. `pm2 start teutonic-eval-tunnel teutonic-validator` (and `pm2 save`)
8. Verify validator log: `loaded state: king=none ...` →
   `seed king unconst/Teutonic-VIII at revision <sha>` → `validator running`
9. Hit `/health` on eval server through tunnel; confirm reachable
10. Monitor 1 full poll cycle (30 s); ensure no exceptions
11. Watch dashboard for first real challenger and first successful eval

## Rollback

If Phase A (compatibility) fails:
- abort cutover; do NOT touch live R2 state
- the live Teutonic-III validator is unaffected (still running)
- iterate on the seed config / probe thresholds / arch tweaks

If post-cutover the validator fails to crown a non-trivial king within
24h (everyone's losses are identical to seed):
- not a rollback condition — that's just "no one trained yet"

If post-cutover the eval server consistently crashes on Teutonic-VIII load:
- restore old state from R2 versioning (Hippius supports object versions)
  and revert Teutonic-VIII repo to private
- but: we won't get there because we tested first.
