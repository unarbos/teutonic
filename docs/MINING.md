# Mining `Teutonic-Q3-8B`

This is the live mining contract for the current chain. The source of truth is
[`chain.toml`](../chain.toml); if that file changes, the live contract changes
with it.

## 1. Current chain

At time of writing the active chain is:

- `chain.name = "Teutonic-Q3-8B"`
- `chain.seed_repo = "teutonic/teutonic-q3-8b-genesis"`
- `arch.module = "archs.qwen3"`
- `seed.tokenizer_repo = "Qwen/Qwen3-8B"`
- `seed.repo_backend = "hippius"`

This is a dense `Qwen3ForCausalLM` chain, not the older Quasar or Qwen3-MoE
flow. Historical LXXX/Quasar docs remain in `docs/` as archive material only.

## 2. The live flow in one paragraph

You start from the current king, train or perturb a challenger, upload that
checkpoint to Hippius Hub, then submit a `v4` reveal on Bittensor SN3. The
validator reads the reveal, pins the challenger's immutable digest, checks repo
hygiene and config lock, samples a holdout stream keyed by reveal block hash +
hotkey, and asks the GPU eval server to run a paired cross-entropy duel against
the current king. The challenger wins only if its one-sided bootstrap lower
confidence bound clears `delta = 0.0025` nats/token.

## 3. What must match exactly

The king's `config.json` is the source of truth. The validator enforces:

- `architectures` must match the king.
- Generic structural keys must match:
  `vocab_size`, `hidden_size`, `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, `head_dim`,
  `intermediate_size`, `model_type`, `tie_word_embeddings`,
  `rope_theta`, `max_position_embeddings`, `max_seq_len`.
- Extra lock keys from [`chain.toml`](../chain.toml) must match:
  `rope_theta`, `rope_scaling`, `tie_word_embeddings`,
  `max_position_embeddings`.
- `config.json` must not contain `auto_map`.
- The repo must not ship any `*.py` files.
- The repo must contain canonical `safetensors` output:
  either `model.safetensors` or a sharded
  `model.safetensors.index.json` + `model-00001-of-000NN.safetensors` layout.
- Total `*.safetensors` size must stay under the validator's size cap
  (`TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB`, default `200`).

Because the live arch is `archs.qwen3`, challengers must load through plain
`transformers` after `chain_config.load_arch()`. No `trust_remote_code`.

## 4. Repo naming and anti-impersonation

Your challenger repo must satisfy both of these:

- It matches the chain regex. With the default auto-derived rule that means
  `^[^/]+/Teutonic-Q3-8B-.+$`.
- It contains the first 8 characters of your coldkey SS58 somewhere in the
  full repo id, case-insensitive.

Examples for coldkey prefix `5DhAqMpd`:

- `myorg/Teutonic-Q3-8B-5DhAqMpd-v1`
- `5DhAqMpd/Teutonic-Q3-8B-lora-03`
- `myorg/Teutonic-Q3-8B-v1` is rejected

The validator checks the prefix against the chain metagraph. If your hotkey is
too fresh for that mapping to exist yet, the validator skips the coldkey check
until a later tick instead of hard-failing you.

## 5. Reveal format

The live wire format is:

```text
v4|<challenger_repo>|<challenger_digest>|<author_hotkey>
```

Notes:

- `challenger_digest` is the binding commitment. It is usually a Hippius OCI
  digest `sha256:<64hex>`.
- `hf:<40hex>` digests are still supported by the shared model-store code, but
  the reference miner uploads challengers to Hippius Hub.
- Legacy `v3|king_digest|...` reveals are rejected at intake.
- The reference miner submits with `blocks_until_reveal=3`.

## 6. Quick start

Set up a local environment:

```bash
uv venv
. .venv/bin/activate
uv pip install -e .
```

For a simple pipeline test, run the reference miner:

```bash
export HIPPIUS_HUB_TOKEN=...
export BT_WALLET_NAME=mywallet

python miner.py \
  --hotkey default \
  --suffix 5DhAqMpd-noise-01 \
  --noise 1e-4
```

`miner.py` will:

1. Read the current king repo + digest from the live dashboard.
2. Materialize the king at its immutable digest.
3. Create a challenger by perturbing every floating-point tensor.
4. Run a local config sanity check.
5. Upload to Hippius Hub.
6. Submit a `v4` reveal.

This is only a pipeline test. Noise almost never beats a mature king.

## 7. Real training

The network does not prescribe a training recipe. The important part is that
you train against the current king and end with a standalone checkpoint that
passes the submission gates above.

Useful starting points in this repo:

- [`scripts/mining/train_challenger.py`](../scripts/mining/train_challenger.py)
  for an end-to-end training/eval loop.
- [`scripts/training_bundle/README.md`](../scripts/training_bundle/README.md)
  for a token-id / LoRA-oriented starter path.

For the live Qwen3 dense chain, common LoRA target modules are the standard
Qwen blocks such as `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`.

## 8. How evaluation works

The current validator/eval-server contract is:

- `TEUTONIC_EVAL_N` defaults to `5000`.
- Public/private split defaults to `2500 / 2500` on the validator unless
  operators override it.
- `SEQ_LEN = 2048`.
- `alpha = 0.001`.
- `delta_threshold = 0.0025`.
- Production eval commonly runs in `raw_hippius` mode, meaning the public
  corpus is the FineWeb-Edu Hippius mirror tokenized at eval time with the
  configured tokenizer.
- The shard/seed comes from `blake2b(block_hash_at_reveal || hotkey)`.

Acceptance is `lcb > delta`, where `lcb` is the one-sided bootstrap lower
confidence bound of the paired loss difference.

## 9. How rewards are routed

The live validator does not pay only the current king. It distributes weight
equally across the current king plus up to four previous distinct kings still
registered in the metagraph. If some prior kings are no longer registered, the
survivors are renormalized. If no king hotkey is usable, weights fall back to
the configured burn UID.

## 10. What the validator will reject

Common failure modes:

- `architecture mismatch` or `<key> mismatch`: your config drifted.
- `auto_map present in config.json`: custom modeling code is not allowed.
- `repo ships *.py files`: upload weights/config/tokenizer only.
- `missing model.safetensors.index.json`: your sharded save is incomplete.
- `oversized: ... > 200 GB cap`: you uploaded extra state or wrong precision.
- `coldkey_required`: your repo name does not embed the required coldkey prefix.
- `digest_not_found` or `digest_malformed`: the reveal did not bind a real
  immutable snapshot.

## 11. Useful links

- Live dashboard: <https://teutonic.ai>
- Live JSON: <https://us-east-1.hippius.com/teutonic-sn3/dashboard.json>
- Active seed config: [`chain.toml`](../chain.toml)
- Current mechanism: [`DESIGN.md`](DESIGN.md)
- Source: <https://github.com/unarbos/teutonic>

## 12. FAQ

**Can I submit a different architecture?**
No. The validator locks the active model family and structural config.

**Can I submit quantized weights?**
Evaluation loads in bf16. If your export dequantizes cleanly into the same
effective weights and passes the file-layout gates it can work, but plain
bf16 `safetensors` is the least surprising path.

**Can I replace a bad submission immediately?**
Usually no. The validator de-dupes by hotkey within a reign. Wait for the next
king or use a different registered miner key.
