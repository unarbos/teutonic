# Teutonic-III Cutover Plan

> Target: replace the live Teutonic-I king (Gemma3 1B-shape) with a fresh
> ~3.3B Gemma4 king ("Teutonic-III"). This document is the contract
> followed by the validator restart and the Discord announcement.

## TL;DR

- **New model family:** Gemma4 (`Gemma4ForCausalLM`, transformers 5.5.3)
- **Size:** **3.29B params**, 18 layers, hidden 2304, vocab 262144 (tied embed)
- **Repo:** `unconst/Teutonic-III` (private during build, public at cutover)
- **Tokenizer:** reused from `unconst/Teutonic-I` (vocab 262144 — same tokens
  as existing dataset shards; no re-tokenization needed)
- **Eval pipeline:** unchanged. `model.model(...)` returns
  `BaseModelOutputWithPast` (has `last_hidden_state`); `model.lm_head` is
  `nn.Linear`. Verified compatible with `eval_torch.compute_paired_losses`.
- **All Teutonic-II artifacts are replaced.** Teutonic-II was never made
  live; the in-flight branch is pivoted directly to Teutonic-III.

## Why Gemma4, not Gemma3-4B

The previous plan (Teutonic-II) was a Gemma3 ~2.6B model (gemma-3-4b-pt
shape but with random init). User explicitly asked to base the new arch off
**Gemma 4** this time. `Gemma4TextConfig` is in transformers 5.5.3 and is a
real, distinct architecture (per-layer-input embeddings, mixed
sliding/full attention with separate RoPE configs, partial rotary on full
layers, double-wide MLP option, MoE option). We use the dense, non-MoE
variant.

## Why 18 layers (not the default 30)

Gemma4 defaults: 30 layers, hidden 2304, with `embed_tokens_per_layer` of
shape `[262144, hidden_per_layer*num_layers]` = a huge per-layer-input
embedding table that scales linearly in `num_hidden_layers`. With 30
layers the model is **5.08B**; user asked for "3B param models". Reducing
to 18 layers (still respecting that the last layer must be
`full_attention`, which transformers auto-fixes) lands at **3.29B params**
(0.60B tied embed + 1.21B per-layer-input embed + ~1.48B layer params).
Dropping further (e.g. 16 layers) would also work but 18 keeps a healthy
ratio of attention layers and matches the user's "3B" target precisely.

## Param breakdown (verified locally)

| component                                            | params |
|------------------------------------------------------|-------:|
| `model.embed_tokens` (262144 × 2304, tied to lm_head)| 604M   |
| `model.embed_tokens_per_layer` (262144 × 4608)       | 1208M  |
| `model.per_layer_model_projection` (4608 × 2304)     |  11M   |
| 18 × decoder layers (attention + SwiGLU MLP + norms) | 1465M  |
| **TOTAL**                                            |**3.29B**|

`tie_word_embeddings=True`, so `lm_head.weight` shares storage with
`embed_tokens.weight` (counted once).

## Compatibility verification (pre-cutover, MUST be green)

The validator tickling random repos in the wild is unforgiving — we verify
the eval server can load and probe the new arch BEFORE turning over.

1. **Local CPU build & save** of seed model via
   `scripts/seed_teutonic_iii.py` (`--no-probe`); confirms
   `Gemma4TextConfig + AutoModelForCausalLM.from_config` works in our
   transformers 5.5.3.
2. **Push private** to `unconst/Teutonic-III`.
3. **scp seed script + smoke script** to Targon GPU box.
4. On Targon, run:
   - `seed_teutonic_iii.py --no-push` (rebuild locally there + run
     `trainability_probe`); confirms the probe returns `ok=True` with
     finite delta (untrained model has high but stable loss; one SGD step
     of LR 1e-5 should not blow it up).
   - `smoke_eval_teutonic_iii.py` (king == challenger): full pipeline
     including `MultiGPUEvaluator`, paired bootstrap on a real shard.
     Required:
     - both models load with `flash_attention_2` or `sdpa`
     - probe returns `ok=True`
     - paired loss diff is exactly 0 (same model, identical seeds)
     - no OOM at batch=256 seq=2048
     - throughput estimate ≥ Teutonic-II's ~38 seq/s at parallel split

5. Only if all 4 above pass → proceed to cutover.

## Edge cases & guardrails

| risk | mitigation |
|------|-----------|
| Validator caches king config; old Teutonic-I still in `_king_config` | `state.set_king` already nulls the cache. Restart guarantees a clean process. |
| Eval server caches king MultiGPUEvaluator | `_ensure_king` reloads on `repo` change; restart of `eval-tunnel` is not enough — but the eval server itself is not on this machine and runs persistently on Targon. Cutover step hits `/eval` with `king_repo=unconst/Teutonic-III` which triggers reload. Probe will run on the new king. |
| Eval server's HF cache has old Teutonic-I revisions | `_cleanup_hf_cache` runs after every eval; harmless. |
| Live state on R2 has Teutonic-I king pinned | wipe `king/current.json`, `state/queue.json`, `state/seen_hotkeys.json`, `state/validator_state.json`, `state/dashboard_history.json`, and `eval/*` objects. Validator startup with empty state will set seed king to `SEED_REPO`. |
| Old `dashboard.json` shows Teutonic-I history | archive to `dashboard-teutonic-i-final.json` (same R2 bucket, served from Hippius dashboard bucket too) — link from announcement |
| Existing miners still submitting `Teutonic-I-*` repos | `REPO_PATTERN` is now `^[^/]+/Teutonic-III-.+$`; old repos will be rejected at scan time. They never enter the queue. |
| Existing miners' on-chain commit format | unchanged: `<king_hash>:<hf_repo>:<model_hash>`. Old commits with king_hash matching old Teutonic-I will be filtered by `check_stale` once a new king is crowned, and by `REPO_PATTERN` even before that. |
| Hippius / R2 dataset shards format | unchanged. Tokenizer is identical (re-used from Teutonic-I). The shards are still valid because token IDs are the same vocab. |
| Per-layer-input embeddings might trip `validate_challenger_config` | the field set checked is {vocab_size, hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, model_type}. Gemma4 sets `model_type="gemma4_text"` (vs Gemma3's `gemma3_text`), so a Teutonic-I-shaped challenger gets rejected at config-match step. Good. |
| Trainability probe LR (1e-5) on Gemma4 | The probe is arch-agnostic — single SGD step on random tokens, snapshot/restore of all params. Same code path as Teutonic-II. Gemma4's per-layer embeddings get one SGD step each — same as any other param. |
| `chunk_size` in `compute_paired_losses` (256) for vocab 262144 | unchanged from Teutonic-II — already validated. |
| Gemma4 sliding-window attention defaults to 512 | Smaller than seq_len 2048 — but the model's own attention masking handles this correctly. We're only computing forward/backward, no caching. |
| ecosystem.config.js has dead `TEUTONIC_KING_REPO` env var | rename to `TEUTONIC_SEED_REPO` so the validator actually picks it up. |
| Discord announcement timing | post AFTER first successful 30s poll cycle confirms validator is healthy; embed includes new dashboard URL and link to the archived old one. |
| First eval after cutover will load Teutonic-III on the eval server, which has never seen Gemma4 | The server uses `AutoModelForCausalLM.from_pretrained` and tries `flash_attention_2 → sdpa → eager`. Gemma4 supports `sdpa`; will succeed there even if FA2 is not yet wired for it. |
| Eval server might OOM on Gemma4's bigger embed table | Per-layer embed table is 1.2B params × 2 bytes (bf16) = 2.4 GB. King + challenger × 2 GPUs each = 9.6 GB just for embeds. B200 has 188 GB. Fine. |
| Fresh-init Teutonic-III has high uniform loss (~ln(262144) ≈ 12.5) | Identical situation to Teutonic-II. The probe accepts (delta is tiny because gradient on random init is not destabilizing); the bootstrap test trivially shows challenger == king (no advantage). First real challenger to actually train will dethrone it. |
| Validator's `set_king` on cold start uses `wallet.hotkey.ss58_address` as the seed king's hotkey | This means the validator's own hotkey is "the king" until first dethrone. Same behavior as before. Weights get set to validator's own hotkey only when no real king exists yet — a known seed-period quirk. |

## Execution order (only after Phase A passes)

1. `pm2 stop teutonic-validator teutonic-eval-tunnel`
2. R2: copy `dashboard.json` → `dashboard-teutonic-i-final.json`
   (Hippius bucket too, since dashboard is served from there)
3. R2: delete `king/current.json`, `state/queue.json`,
   `state/seen_hotkeys.json`, `state/validator_state.json`,
   `state/dashboard_history.json`, all `eval/*` objects
4. R2: append to `state/history.jsonl`:
   `{"event":"family_transition","from":"Teutonic-I","to":"Teutonic-III","arch":"Gemma4","params_b":3.29,"timestamp":"..."}`
5. HF: `unconst/Teutonic-III` → public; capture commit SHA
6. git: commit working-tree changes (validator/miner/ecosystem +
   seed/smoke scripts) on a fresh `teutonic-iii` branch; merge to main;
   push
7. `pm2 start teutonic-eval-tunnel teutonic-validator` (and `pm2 save`)
8. Verify validator log: `loaded state: king=none ...` →
   `seed king unconst/Teutonic-III at revision <sha>` → `validator running`
9. Hit `/health` on eval server through tunnel; confirm reachable
10. Monitor 1 full poll cycle (30 s); ensure no exceptions
11. Post Discord announcement to SN3 channel
12. Monitor 30 min for first real challenger and first successful eval

## Rollback

If Phase A (compatibility) fails:
- abort cutover; do NOT touch live R2 state
- the live Teutonic-I validator is unaffected (still running)
- iterate on the seed config / probe thresholds / arch tweaks

If post-cutover the validator fails to crown a non-trivial king within
24h (everyone's losses are identical to seed):
- not a rollback condition — that's just "no one trained yet"

If post-cutover the eval server consistently crashes on Teutonic-III load:
- restore old state from R2 versioning (Hippius supports object versions)
  and revert Teutonic-III repo to private
- but: we won't get there because we tested first.
