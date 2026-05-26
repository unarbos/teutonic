# Teutonic — Design

This file describes the current production flow for the live
`Teutonic-Q3-8B` chain. The source of truth is the code plus
[`chain.toml`](../chain.toml); this document is a human-readable map of that
behavior.

## 1. TL;DR

Teutonic is Bittensor subnet 3. At any moment one checkpoint is the **king**.
Miners submit a **challenger** by uploading it to Hippius Hub and revealing
`v4|repo|digest|author_hotkey` on-chain. The validator pins that immutable
digest, checks the challenger against the king's locked config, and sends both
models to the eval server for a paired next-token-loss duel. If the
challenger's bootstrap lower confidence bound clears `delta = 0.0025`
nats/token, it becomes the new king.

Rewards are not winner-take-all in the current implementation. The validator
splits weights equally across the current king plus up to four previous
distinct kings that are still registered.

## 2. Active chain contract

The live chain is configured in [`chain.toml`](../chain.toml):

- `chain.name = "Teutonic-Q3-8B"`
- `chain.seed_repo = "teutonic/teutonic-q3-8b-genesis"`
- `arch.module = "archs.qwen3"`
- `seed.tokenizer_repo = "Qwen/Qwen3-8B"`
- `seed.repo_backend = "hippius"`

That means the live architecture is dense `Qwen3ForCausalLM`. Historical
Quasar and Qwen3-MoE docs in this repo are archive material, not the current
submission contract.

## 3. End-to-end flow

1. A miner discovers the current king repo + digest from the dashboard.
2. The miner trains or perturbs a challenger starting from that king.
3. The miner uploads the challenger to Hippius Hub and gets an immutable
   digest, usually `sha256:<64hex>`.
4. The miner submits `v4|<repo>|<digest>|<author_hotkey>` with
   `set_reveal_commitment(..., blocks_until_reveal=3)`.
5. The validator polls revealed commitments, keeps the latest valid one per
   hotkey, and rejects legacy `v3` reveals.
6. The validator verifies repo naming, coldkey-prefix ownership, digest
   existence, config lock, and file hygiene.
7. The validator dispatches the duel to the eval server.
8. The eval server scores king and challenger on the same token sequences and
   streams progress back over SSE.
9. If the challenger wins, the validator crowns it, updates dashboard/history,
   and refreshes miner weights.

## 4. Submission contract

The live reveal format is:

```text
v4|<challenger_repo>|<challenger_digest>|<author_hotkey>
```

Important rules:

- `challenger_repo` must match the active `repo_pattern`, which defaults to
  `^[^/]+/Teutonic-Q3-8B-.+$`.
- The full repo id must contain the first 8 characters of the miner's coldkey
  SS58, case-insensitive.
- `challenger_digest` must match `sha256:<64hex>` or `hf:<40hex>`.
- The reference miner publishes to Hippius; `hf:` remains supported by the
  shared model-store layer for chains seeded from HF.
- Legacy king-bound reveals are dropped at intake.

The point of the digest is TOCTOU safety: the validator evaluates exactly the
snapshot the miner committed to, not whatever later appears at the same tag.

## 5. Config lock and file hygiene

`validate_challenger_config` enforces:

- exact `architectures` match
- generic structural-key match:
  `vocab_size`, `hidden_size`, `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, `head_dim`,
  `intermediate_size`, `model_type`, `tie_word_embeddings`,
  `rope_theta`, `max_position_embeddings`, `max_seq_len`
- active `[arch].extra_lock_keys` from [`chain.toml`](../chain.toml):
  `rope_theta`, `rope_scaling`, `tie_word_embeddings`,
  `max_position_embeddings`
- no `auto_map` in `config.json`
- no `*.py` files in the uploaded repo
- canonical `safetensors` layout
- total safetensors size under the operator cap

Because the live arch is `archs.qwen3`, challengers must load with upstream
`transformers` after `chain_config.load_arch()`. No custom modeling code.

## 6. Evaluation contract

The production evaluator uses paired next-token cross-entropy:

- `EVAL_N` defaults to `5000`
- `SEQ_LEN = 2048`
- `alpha = 0.001`
- `delta = 0.0025`
- bootstrap samples default to `10000`
- public/private split defaults to 50/50 unless operators override it

Sequence selection is keyed by the reveal block hash and miner hotkey:

```text
seed = blake2b(block_hash_at_reveal || hotkey)
```

The validator chooses the shard using that seed, then the eval server scores
both models on the same sequences. If `lcb > delta`, the challenger is
accepted.

In normal production setup the public corpus is the FineWeb-Edu Hippius mirror
in `raw_hippius` mode, tokenized at eval time with the configured tokenizer.

## 7. Reward routing

The live code keeps a rolling king chain:

- current king first
- then up to four prior distinct kings
- equal weight across all currently registered members of that set
- fall back to the configured burn UID if none are registered

Examples:

- 1 registered king: `100%`
- 2 registered kings: `50% / 50%`
- 5 registered kings: `20%` each

This is the production behavior in `maybe_set_weights`, even though some
proposal docs in the repo discuss alternative scoring or payout schemes.

## 8. Safety checks and edge cases

Key guards in the live flow:

- self-challenges are skipped
- duplicate/previously failed submissions are skipped
- malformed or missing digests are failed before eval
- coldkey-prefix ownership is enforced against the metagraph
- trainability and weight-sanity checks run on the eval side before the full
  duel
- if the eval server is busy, the validator backs off and retries
- dashboard/history/Discord writes are best-effort and do not break the main
  loop

## 9. Operator notes

Two production details matter when reading the code:

- `chain.toml` is the chain contract. Swapping the live chain should usually be
  a config change plus arch shim selection, not a validator rewrite.
- Archived docs under `docs/LXXX_*`, `docs/SCORING_PLAN.md`, and
  `docs/OPTIMAL_DESIGN.md` describe earlier or proposed systems. They are
  useful context, but they are not the live miner contract.
