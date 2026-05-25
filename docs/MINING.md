# Mining `Teutonic-Q3-4B`

This is the live mining guide for the chain declared in
[`chain.toml`](../chain.toml).

As of `2026-05-25`, the active flow is:

- Chain name: `Teutonic-Q3-4B`
- Genesis king repo: `teutonic/teutonic-q3-4b-genesis`
- Genesis source family: `Qwen/Qwen3-4B`
- Tokenizer: `Qwen/Qwen3-4B`
- Architecture: vanilla `Qwen3ForCausalLM` via [`archs/qwen3`](../archs/qwen3/)
- Seed backend: Hippius Hub (`sha256:...` digest pinned in [`chain.toml`](../chain.toml))
- Eval data: raw FineWeb-Edu from the Hippius mirror, tokenized at eval time

If you remember the old Quasar or Qwen3-MoE docs, ignore them. The active
chain is now dense Qwen3 4B, with Hippius-backed model storage and `v4`
reveal commitments.

## 0. The current flow

1. Pull the current king from Hippius Hub by the immutable digest in
   [`chain.toml`](../chain.toml).
2. Train or perturb a challenger that keeps the exact same Qwen3-4B
   architecture and config lock.
3. Upload the challenger to Hippius Hub.
4. Commit a `v4|<repo>|<digest>|<author_hotkey>` reveal on-chain.
5. The validator checks repo hygiene and config match, then runs a paired
   cross-entropy duel on raw FineWeb-Edu tokenized with `Qwen/Qwen3-4B`.

Acceptance is still the same mechanism: the challenger's bootstrap LCB on
per-token NLL improvement must clear `delta = 0.0025` nats/token.

## 1. What must match

Your challenger must stay load-compatible with the active king.

- `model_type` must remain `qwen3`.
- The validator compares the king and challenger on the generic structural
  keys in [`validator.py`](../validator.py): `vocab_size`, `hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
  `head_dim`, `intermediate_size`, `model_type`, `tie_word_embeddings`,
  `rope_theta`, `max_position_embeddings`, and `max_seq_len`.
- It also enforces the active arch extras from [`chain.toml`](../chain.toml):
  `rope_theta`, `rope_scaling`, `tie_word_embeddings`, and
  `max_position_embeddings`.
- The king is the vanilla `Qwen/Qwen3-4B` shape described at the top of
  [`chain.toml`](../chain.toml): vocab `151936`, `36` layers, hidden size
  `2560`, GQA `32/8`, RoPE theta `1e6`.

Repo hygiene rules are strict:

- No `auto_map` in `config.json`
- No `*.py` files in the uploaded repo
- `safetensors` only, in standard Transformers layout
- The model must load through `AutoModelForCausalLM.from_pretrained(...)`
  after `chain_config.load_arch()`

## 2. Repo naming and coldkey gate

Your uploaded repo must match the chain repo pattern. For the current chain
that means:

```text
^[^/]+/Teutonic-Q3-4B-.+$
```

It must also contain the first 8 characters of your coldkey ss58 somewhere in
the full repo id, case-insensitive. Example for coldkey prefix `5DhAqMpd`:

- `myteam/Teutonic-Q3-4B-5DhAqMpd-v1`
- `5DhAqMpd/Teutonic-Q3-4B-lora01`

Without that prefix the validator records `coldkey_required` and skips the
submission.

## 3. Environment

Python `3.11+` is enough for the repo itself.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If you want the optional dev tools:

```bash
uv pip install -e ".[dev]"
```

For a quick local sanity check that the active arch resolves cleanly:

```bash
python -c "
import chain_config
chain_config.load_arch()
from transformers import Qwen3ForCausalLM
print(chain_config.NAME, chain_config.SEED_REPO, Qwen3ForCausalLM.__name__)
"
```

## 4. Quick start: noise miner

[`miner.py`](../miner.py) is still the easiest end-to-end smoke test. It:

1. Discovers the live king from the dashboard
2. Downloads it from Hippius by immutable digest
3. Adds low-amplitude Gaussian noise
4. Uploads the challenger to Hippius Hub
5. Broadcasts a `v4` reveal

Example:

```bash
export HIPPIUS_HUB_TOKEN=...
export BT_WALLET_NAME=teutonic
export BT_WALLET_HOTKEY=default

python miner.py \
  --hotkey default \
  --suffix 5DhAqMpd-noise-01 \
  --noise 1e-4
```

That almost certainly will not dethrone a trained king, but it validates the
current storage and submission path.

## 5. Current operator seed flow

If you are refreshing or reproducing the genesis seed, the operator flow is:

```bash
HIPPIUS_HUB_TOKEN=... HF_TOKEN=... python -m archs.qwen3.seed \
  --source-repo Qwen/Qwen3-4B \
  --target-repo teutonic/teutonic-q3-4b-genesis \
  --repo-backend hippius
```

That script downloads `Qwen/Qwen3-4B`, strips anything the validator would
reject, uploads to Hippius, and prints the `seed_repo` / `seed_digest` values
to paste into [`chain.toml`](../chain.toml).

## 6. Real challengers

The protocol does not require a specific training recipe. What matters is the
final artifact you upload.

- Start from the live king or another dense `Qwen3-4B` checkpoint that you can
  safely reconcile to the king's config lock.
- Train however you want: LoRA, full fine-tune, distillation, curriculum over
  FineWeb-Edu, or your own data mix.
- Before upload, make sure the final model still passes the config/code-hygiene
  checks above.
- Upload to Hippius Hub, not HuggingFace, for the live challenger path.

Useful scripts:

- [`scripts/mining/submit_external_model.py`](../scripts/mining/submit_external_model.py)
  stages a HuggingFace model, patches safe config fields, uploads it to
  Hippius, and submits the reveal.
- [`scripts/mining/submit_challenger.py`](../scripts/mining/submit_challenger.py)
  broadcasts a reveal for a pre-uploaded challenger from a verdict JSON.
- [`scripts/mining/`](../scripts/mining/) and
  [`scripts/training_bundle/`](../scripts/training_bundle/) are starter
  training/orchestration helpers, but the contract is the final Hippius upload
  plus `v4` reveal, not any specific harness.

## 7. Live eval path

The live validator is no longer describing the old tokenized Quasar-era setup.
Today it:

- uses `TEUTONIC_EVAL_DATASET_MODE=raw_hippius`
- reads raw FineWeb-Edu parquet files from the Hippius mirror
- tokenizes them with `Qwen/Qwen3-4B`
- evaluates challengers against the reigning king with the paired bootstrap CE
  test

So if you are optimizing for real dethrones, optimize for the dense Qwen3-4B
tokenizer/domain mix, not the old Teutonic-I / Quasar assumptions.

## 8. What rejections look like

Common failure classes:

- `coldkey_required`: repo id does not contain your coldkey prefix
- `config_rejected`: structural config mismatch, `auto_map`, `*.py`, or bad
  safetensors layout
- `digest_not_found`: you committed a digest the validator could not resolve on
  Hippius
- `verdict: "king"`: the challenger loaded and ran, but did not beat the king

## 9. Pointers

- Live contract source: [`chain.toml`](../chain.toml)
- Validator enforcement: [`validator.py`](../validator.py)
- Reference miner: [`miner.py`](../miner.py)
- Design doc: [`docs/DESIGN.md`](DESIGN.md)
- Public dashboard: <https://teutonic.ai>
- Public JSON: <https://teutonic.ai/dashboard.json>
