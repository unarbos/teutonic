# Teutonic — Optimal Design

> Proposal doc only. This describes a possible future redesign, not the live production flow. The current chain contract is `Teutonic-Q3-8B`; see [`DESIGN.md`](DESIGN.md), [`MINING.md`](MINING.md), and [`../chain.toml`](../chain.toml).


> Companion to [`DESIGN.md`](DESIGN.md). DESIGN.md describes the mechanism *as currently shipped*. This document describes the mechanism we will ship next: the same king-of-the-hill shape with the three mechanism-level corrections that the current design gets wrong, the dead code removed, and the operationally-confused pieces collapsed. No new frameworks. No imported methods. No composition pivot. Just the correct version of what's already there.

---

## §1. The mechanism, in one sentence

**One canonical king. Miners submit full challenger checkpoints. The validator runs a paired per-token cross-entropy duel on a public+private holdout. If the challenger's bootstrap LCB on the per-token NLL improvement clears `δ_t = c · king_loss_t`, the challenger replaces the king. Winner takes 100% of emission until dethroned.**

That's it. Read every other section as a sharpening of one piece of that sentence.

---

## §2. Why this and not something else

Three structural choices we are deliberately keeping from the current design:

1. **Single canonical king, not a composition DAG.** The protocol's job is to identify the best model anyone can train, not to merge contributions. Selection is the right primitive when the objective is a single scalar that admits a clean head-to-head test.
2. **Full checkpoint upload, not deltas.** The miner uploads what they trained. The protocol does not prescribe training algorithm, format, or compression. A 150 GB upload is a 150 GB upload — the miner pays once, the validator pays once, and nothing about how the Δ was produced becomes the protocol's concern.
3. **Winner-takes-all, not rolling windows.** High variance is a feature: it means selection pressure is strong and the king represents the genuine frontier. Smoothing payouts across recent kings is compensation for variance, not a fix.

Three places where the current design is wrong:

1. **Effect floor `δ` is a constant in a log-loss.** As the king matures, 0.0025 nats becomes a progressively larger relative improvement. Eventually unwinnable.
2. **Holdout is fully public.** A miner who trains on the entire public corpus has overfitted whatever shard is sampled. Shard randomization defeats shard-specific overfit but not whole-corpus overfit.
3. **The validator's verdicts are not publicly replayable.** External auditors cannot independently verify a given duel's verdict from public inputs. There is no proof beyond "the operator's box said so."

§§3–11 fix these three and keep everything else.

---

## §3. On-chain object

A single canonical king reference. Two fields:

```
king := (hf_repo: string, digest: 64-hex sha256)
```

Persisted on-chain via the validator's `set_commitment`. Recoverable from R2/Hippius state for crash recovery — same as the current `state/king/current.json`.

No king lineage chain. No `previous_king` field. The king is whoever is king *now*; the audit trail of past kings lives in `state/history.jsonl`, not in the validator's state machine. (Current `_reconcile_chain_from_history` becomes unnecessary because there's no chain to reconcile.)

On king disappearance (HF/Hippius 404 on the pinned digest for the current king): emit `king_lost` event, halt new dethrones, halt new emissions until operator intervention. **No automatic revert.** Reverting to a possibly-stale prior king masks the failure and risks crowning a previously-removed checkpoint. Halting forces a clear operator decision.

---

## §4. Reveal-commit format (v3)

```
v3|king_digest|challenger_repo|challenger_digest|author_hotkey
```

| Field | Length | Meaning |
|---|---|---|
| `v3` | 2 | Format version. v2 reveals dropped at scan with a one-time warn per hotkey (existing pattern). |
| `king_digest` | 64 hex | The full 64-hex sha256 of the king the challenger was trained against. **Enforced**: if `king_digest != current king at scoring time`, drop as `stale_parent`. Closes the "decorative king_hash" contradiction in the current v2 format. |
| `challenger_repo` | string | Hippius repo identifier. Must match `chain.toml::repo_pattern` and embed the first 8 ss58 chars of the author coldkey (existing anti-impersonation). |
| `challenger_digest` | 64 hex | sha256 OCI digest of the challenger safetensors. Binding cryptographic commitment, locked at 3-block commit-reveal. |
| `author_hotkey` | 48 char ss58 | The submitter. |

Total ~200 chars — fits within `set_reveal_commitment` budget.

**Stale-parent enforcement.** A challenger's reveal is dropped if the embedded `king_digest` no longer equals the current king at the moment the validator processes the challenger. Two consequences:

- Miners cannot grind a challenger against an old king hoping the eval framework still applies it cleanly.
- A flurry of in-flight challengers after a dethrone are all dropped at once, forcing miners to retrain against the new king. This is the right shape — the system advances in steps, not by stockpiled stale work.

---

## §5. Validator loop

Single async process, one outstanding eval at a time. Tick structure largely matches current `validator.py:main`:

```
every POLL_INTERVAL seconds:
    refresh_uid_map()
    fetch_market_data()
    check_king_alive()
    new_reveals = scan_reveals()
    enqueue(new_reveals filtered by:
        - hotkey not previously seen
        - challenger_repo not previously failed
        - king_digest matches current king
        - coldkey-prefix in repo name matches author)
    while queue:
        challenge = queue.pop(0)
        process_challenge(challenge)        # see §6
        maybe_set_weights()                 # see §9
    maybe_set_weights()
    flush_state()
    sleep(POLL_INTERVAL)
```

Continuous, not round-based. Each challenger is evaluated as it surfaces. Discrete rounds would add complexity without value here (no composition to amortize per-round).

---

## §6. Scoring

Paired per-token cross-entropy duel between king and challenger on a sampled holdout. Math unchanged from current `eval/torch_runner.py:compute_paired_losses` + bootstrap LCB. What changes is the holdout and the threshold.

### §6.1 Holdout (§-corrected: was public-only)

For each duel, the validator constructs:

```
holdout = concat(public_seqs, private_seqs)
```

Both at `seq_len = 2048`, `EVAL_N/2` sequences each.

**Public component.** Deterministically sampled from a chain-pinned public corpus (currently the FineWeb-Edu Hippius mirror; selection of which corpus is `chain.toml`-config) via:

```
public_seed = blake2b(block_hash_at_reveal || hotkey || b"public", digest_size=8)
public_indices = PCG64(public_seed).choice(n_public_sequences, EVAL_N/2, replace=False)
```

Pinned: `(public_corpus_digest, public_seed, public_indices)` all reproducible externally.

**Private component.** Validator-local. Sampled from a private pool maintained off-chain by the validator. Sources: recent CommonCrawl partitions past the model's training cutoff; recent GitHub post-cutoff; fresh ArXiv; any source where we can credibly assert "the miner cannot have trained on this." Rotated periodically (suggested: every 10000 blocks ≈ 33 hours). Never published, never logged with token content, only its digest and a count.

```
private_indices = local_rng.choice(n_private_sequences, EVAL_N/2, replace=False)
```

The validator commits `(private_pool_digest, n_private_sequences_sampled)` to the audit log per duel — enough for an auditor to verify a private pool of a given digest existed at the time, without revealing the data.

**Why 50/50.** A pure-private holdout can't be replayed externally (defeats §10). A pure-public holdout is gameable by whole-corpus training (the current bug). The 50/50 split keeps half the verdict reproducible and uses the other half to defeat whole-corpus overfit.

### §6.2 Per-sequence loss difference

For each sequence `i` in the concatenated holdout, the same tokens go through both models. Per-sequence per-token mean CE losses `king_loss_i, chall_loss_i`. Paired difference:

```
d_i = king_loss_i - chall_loss_i             (positive = challenger better)
mu_hat = mean(d_i)
```

Eval implementation: chunked `lm_head` for large-vocab models (current `torch_runner.py` pattern, keep). bf16 native, no fp32 upcast.

### §6.3 Bootstrap LCB

```
boot_seed = blake2b(block_hash || hotkey || b"boot", digest_size=8)
boot_rng = PCG64(boot_seed)
boot_means = [d[boot_rng.integers(0, N, size=N)].mean() for _ in range(B)]
lcb = quantile(boot_means, alpha)
```

`B = 10000` resamples, `alpha = 0.001` (one-sided 99.9%). Deterministic given `(block_hash, hotkey)`.

### §6.4 Adaptive effect floor (§-corrected: was constant 0.0025)

```
δ_t = c · king_loss_t
```

with `c = 0.001` (0.1% relative NLL reduction).

`king_loss_t` is the EMA over the last `K = 10` successful duels of the eval server's `avg_king_loss`. Initialized from the first 3 duels post-king-coronation (or from a one-shot king-only eval if the king has just been crowned and no challengers have been evaluated yet).

Why this and not constant: 0.0025 nats was tuned when `king_loss ≈ 6` (random init). At maturity (`king_loss ≈ 3`), the same 0.0025 represents a 0.08% relative improvement instead of 0.04%. As the king matures further, the constant floor either becomes unwinnable (mature king regime) or admits trivial wins (early king regime). `c · king_loss` keeps the relative bar constant at every age.

Why not EMA-of-improvements (the earlier proposal): an EMA over realized accepted improvements can be cartel-manipulated by colluding miners admitting only tiny deltas. `c · king_loss_t` has no such handle — `king_loss` is observed directly per duel and cannot be lowered by miner behavior except by genuine improvement, which raises the bar (the right direction).

Why c=0.001: ~0.1% relative is the smallest improvement we want to pay for. Genuine fine-tuning at frontier model scale routinely produces this; numerical noise routinely doesn't. Tune later if necessary; doesn't bind any other piece of the design.

### §6.5 Acceptance rule

```
accepted = (lcb > δ_t)
```

That's the entire verdict. No Pareto panel, no multi-metric weighted sum, no separate "trainability score." A single number out.

---

## §7. Sanity defenses (kept from current)

The reparam-trick / weight-symmetry attacks documented in `DESIGN.md` §7 are real. The patchwork defenses (`NORM_SANITY_*` caps, `TEUTONIC_PROJ_MIN_MEAN_ABS`, `TEUTONIC_LAYER_SCALE_RATIO_MAX`, the trainability probe at king-load and challenger-load) **stay as-is.** They work. The structural alternative would be composition, which we are deliberately not pursuing.

What changes:

- **Audit-king-on-startup chain walking is removed.** Per §3, there is no king lineage chain to walk. If the king fails the startup probe, halt and require operator intervention.
- **Reparam-defense documentation moves into the validator's source.** Each check is annotated with the symmetry class it defends against, so future operators understand why removing one breaks the defense.

---

## §8. Submission format

Full checkpoint upload to Hippius, content-addressed. **No prescribed compression, no LoRA mandate, no delta format.** The miner uploads what they trained.

Constraints:

- `safetensors` only — no pickle, no `.bin`, no `.pt`.
- `*.py` files rejected — vendored modeling code only.
- `auto_map` rejected in `config.json` — no `trust_remote_code` path.
- `config.json` must match king's `config.json` on the structural keys plus `chain.toml::extra_lock_keys` (current `validate_challenger_config` logic).
- Total safetensors size cap at `1.05 × king_safetensors_size_gib`. Stops misbehavior where a miner uploads a 10× model and runs the validator out of disk.

That's the whole format. Miners who want to upload faster can fine-tune with LoRA and merge before push; miners who want to fine-tune sparsely can; miners who want full SGD can. **The protocol does not care how the Δ was produced.**

---

## §9. Payout — burn-only

```
weights = {BURN_UID: 1.0}    # BURN_UID = 0 by default
```

**The validator never emits to miners.** Every weight write is 100% to the burn UID. Eval + duel + dashboard + audit continue to run for observability — the canonical king artifact is maintained so the operator can flip to emission at any time — but no chain emission flows to any miner under the validator's current policy.

Rationale (operator decision, 2026-05-21): we are not ready to start paying miners. The mechanism is being soak-tested with a real model and live chain, but the economic side is paused. Burning emission rather than paying it out is the cleanest expression of "no payout policy active yet" — the alpha goes back to the subnet-owner burn slot.

When the operator decides to enable emission, set `TEUTONIC_EMIT_TO_KING=1` and re-wire `maybe_set_weights` to write `{king_hotkey: 1.0}`. Until then, the code path is intentionally not present — the env var alone won't enable emission; it just refuses to start in burn mode. Two-step gate is deliberate.

Weight setting via **Bittensor's commit-reveal weights extrinsic** with the default reveal interval (≥ 5 tempos). This is mandatory even for burn writes — it defeats weight-copying by would-be parallel validators (the documented Yuma weight-copy problem). Asserted at startup; validator refuses to run if SN3 doesn't have CR enabled.

Liquid alpha (Yuma3) enabled with default settings — improves bond dynamics for any honest validator who joins later.

---

## §10. Reproducible audit trail

Every accepted/rejected duel pins a complete verdict record:

```
{
  "round_id": "eval-NNNN",
  "block_hash_at_reveal": "0x...",
  "hotkey": "5...",
  "king_repo": "...",
  "king_digest": "0x...",
  "challenger_repo": "...",
  "challenger_digest": "0x...",
  "public_corpus_digest": "0x...",
  "public_seed": "0x...",                # 8-byte hex
  "public_indices_digest": "0x...",      # sha256 of indices array
  "private_pool_digest": "0x...",        # validator-local pool digest
  "n_private_seqs": 1000,                # count, no content
  "boot_seed": "0x...",
  "delta_threshold": 0.00033,            # δ_t at evaluation time
  "king_loss_ema": 3.31,                 # current king_loss EMA
  "mu_hat": 0.00451,
  "lcb": 0.00112,
  "accepted": true,
  "eval_code_digest": "0x...",           # sha256 of eval/torch_runner.py
  "validator_hotkey": "5...",
  "validator_signature": "0x...",        # validator's signed hash of this record
}
```

Written to `state/audit/<round_id>.json` on R2 + Hippius mirror. Periodic digest of the entire audit log committed on-chain via `set_commitment` (cheap; one extrinsic per 100 records).

Any external party with the same king bytes + challenger bytes + public corpus snapshot can recompute `mu_hat` and `lcb` deterministically (modulo bf16 cross-hardware noise, bounded to ~1e-4 nats). If the recomputed values disagree with the pinned values, the validator is publicly exposed.

This is the "trust the operator → audit the operator" pivot. Doesn't require multi-validator consensus, doesn't require ZK, doesn't require novel crypto. Just publish the inputs.

---

## §11. Migration from the current code

Surgical, not a rewrite. Estimated diff size: ~600 lines changed, ~400 deleted, ~300 added.

### §11.1 Files touched

| File | Changes |
|---|---|
| `validator.py` | Replace `TOPK_WEIGHTS = [0.2]*5`, `KING_CHAIN_DEPTH`, `aggregate_chain_weights`, `recent_king_chain`, `_truncate_king_chain`, `_normalize_weights`, `topk_for_weight_set`, `score_window`, `recompute_topk`, `_reconcile_chain_from_history` with winner-takes-all `weights = {king_hotkey: 1.0}`. Replace `set_weights` direct call with commit-reveal extrinsic. Replace v2 reveal parsing with v3 (enforce `king_digest == current_king.digest`, drop stale). Replace `check_king_alive` auto-revert with halt-and-notify. Wire eval-result pinning to `audit/` path. Wire `c · king_loss` δ computation via shared `delta_threshold(king_loss_ema)` helper. |
| `eval_server.py` | Add private-pool sampler endpoint + holdout merge. Accept `public_seed, n_public, n_private` per request. Return `(mu_hat, lcb, king_loss, public_digest, private_pool_digest, n_private)` to validator. Remove the `_preload` dead code. Tighten `EVAL_N_CAP` to a real cap (`20000`). |
| `eval/torch_runner.py` | Accept already-prepared holdout tensor rather than sampling internally. Replace `eval()` on .npy header with `np.lib.format.read_array_header_1_0` or `ast.literal_eval`. Keep all current reparam/trainability checks. |
| `eval/raw_dataset.py` | Add `sample_private_pool()` helper that loads from a configured local-only directory (env: `TEUTONIC_PRIVATE_POOL_DIR`). |
| `miner.py` | Emit v3 reveal payload. Drop noise-perturbation reference (it never dethroned anything; mention LoRA + real fine-tuning in the docstring). |
| `model_store.py` | Add `parse_reveal_v3` / `build_reveal_v3`. Keep v2 parsing for one-time legacy-drop warn path. |
| `hourly_burn_weights.py` | Delete. The "mutually exclusive with validator" footgun goes away because the validator's commit-reveal weight extrinsic never wedges silently — it always sets *some* weight. If the validator process dies, pm2 restarts it. No fallback burner. |
| `burn_weights.py` | Delete. |
| `scripts/cascade_dethrone_now.py`, `restore_lost_reign.py`, `sandbox_*.py` | Delete. King-era ops scripts no longer needed (the halt-and-notify replaces them). |
| `docs/SCORING_PLAN.md` | Delete. Exponential 16-king decay never implemented, never will be. |

### §11.2 Cleanup catalogue

| Item | Action |
|---|---|
| `EVAL_DELTA = 0.0025` constant | Becomes `delta_threshold(king_loss_ema)`. |
| `_eval_lock` 409 dance + 30s backoff × 20 retries | Keep as-is. Single validator, single eval lock; this is fine. |
| `R2.append_jsonl` GET+PUT per event | Switch to `append_jsonl_batch` (already exists in code, never wired). |
| `eval()` on shard bytes | `ast.literal_eval`. One-line fix. |
| `EVAL_N_CAP = 999_999` | `EVAL_N_CAP = 20_000`. |
| `_fire_preload` no-op endpoint | Delete validator-side call + eval-server endpoint. |
| `_normalize_weights` (dead) | Delete. |
| `topk_for_weight_set`, `score_window`, `recompute_topk` (DEPRECATED) | Delete. |
| `TOPK_WEIGHTS`, `KING_CHAIN_DEPTH`, `aggregate_chain_weights`, `recent_king_chain`, `_truncate_king_chain` | Delete. |
| `_reconcile_chain_from_history` | Delete (no chain to reconcile). |
| `audit_incumbent_king` chain-walking on failure | Replace with halt-and-notify. |
| `notify_king_dethroned_untrainable` with chain-walk message | Replace with `notify_king_lost` (operator intervention needed). |

### §11.3 Validation plan

Before cutover:
1. **Smoke**: `scripts/smoke_eval.py` updated for v3 reveal + private holdout. Run king-vs-king (should produce `mu_hat ≈ 0, lcb < 0, accepted = false`).
2. **Sandbox soak**: run 10-iteration soak (existing `scripts/sandbox_soak.py` pattern). Verify VRAM stable, audit records written, commit-reveal weight extrinsic round-trips.
3. **Single live duel**: run against one perturbed challenger. Verify rejection on the correct grounds (LCB), audit record pinned, weight extrinsic accepts.
4. **Cutover**: pm2 restart. Existing R2 state for current king carries through (`king/current.json` schema unchanged at the king fields used in the new design).

Rollback path: revert `validator.py` + `eval_server.py` + `miner.py` + `model_store.py` to prior commit. R2 state schema is forward-compatible (new audit log lives in a new R2 prefix; old `state/dashboard_history.json` still readable).

---

## §12. What we are explicitly not doing

Listed so future operators know these were considered and rejected, not overlooked:

- **Composition / model merging / pool-of-contributions.** Selection is the right primitive here; composition is a different system. Future operators may explore it, but it is not in scope.
- **DiLoCo, SparseLoCo, IOTA, Pluralis, model soups, TIES, DARE, Fisher merging.** External frameworks. We have our own research. No need to import.
- **Pareto eval panels (MMLU + ARC + ...)**. The duel is a single scalar (held-out CE). Adding panel objectives expands the attack surface without buying a better signal — pretraining loss is what we are paying to reduce.
- **Multi-class emission (W + D + S + V).** One class: weight contributions, paid to the current king. Validators paid via native Bittensor validator emission.
- **Architecture transfer rounds.** Chain reset on arch upgrade (current operational pattern) is fine.
- **Decentralized governance over chain.toml.** We are the operator. Operator-controlled chain.toml stays.
- **Multi-validator consensus protocols beyond Bittensor's commit-reveal weights.** The reproducible audit trail (§10) is the trust mechanism; explicit consensus would be over-engineering.
- **Recipe attestation / proof-of-learning for training procedures.** Miners train however they want. The protocol verifies the eval, not the procedure.
- **Eval cost amortization via discrete rounds.** Continuous, one-eval-at-a-time is current and works at our throughput.
- **Distillation tracks, dataset contribution tracks, exploratory-branch tracks.** Out of scope for the king-of-the-hill mechanism.

---

## §13. The shape, restated

**Selection of one canonical king via a paired CE duel scored on a public+private holdout, with the bar set at `c · king_loss`, the verdict published with all inputs for external replay, and the winner paid 100% of emission until dethroned.**

Everything in §3–§11 is the surgical implementation of that sentence against the existing codebase. Everything in §12 is what we are deliberately not building.
