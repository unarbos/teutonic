# Mining the active king

This guide tells you how to:

1. Set up your environment.
2. Build / train a challenger.
3. Submit it on-chain.

If you only want to play with random noise, jump to **Quick start (noise
miner)**. If you want to actually dethrone the king, see **Real training**.

> The active chain (king name, seed repo, repo-name regex, vendored
> architecture) is declared in [`chain.toml`](../chain.toml) at the repo
> root. Throughout this guide `<chain.name>` and `<seed_repo>` mean
> whatever `[chain].name` and `[chain].seed_repo` are set to. At the time
> of writing the active chain is **Teutonic-XXIV** (`unconst/Teutonic-XXIV`),
> a freshly-initialised SILX-AI Quasar hybrid MoE: ~8B active per token,
> ~24B total parameters, RoPE θ=1e6, vocab 262144. Arch-specific numbers
> below assume Quasar; check `chain.toml` if it has been swapped.

---

## 0. The mechanism in one paragraph

The validator pulls every reveal commitment from chain, downloads the
challenger from HF, runs a paired cross-entropy test against the king on a
random Hippius shard, and crowns the challenger if its bootstrap LCB on the
per-token NLL improvement clears `delta = 0.0025` (fixed nats/token effect
floor; see `EVAL_DELTA` in `eval/torch_runner.py`). Winner takes 100% of
SN3 emission until dethroned. Full mechanism in [`DESIGN.md`](DESIGN.md).

The architecture lock is enforced by `validate_challenger_config` in
[`validator.py`](../validator.py): your challenger's `config.json`
must match the king on every key in the generic structural set (vocab,
dims, RoPE, …) plus `[arch].extra_lock_keys` from `chain.toml` (for the
current Quasar chain that includes MoE shape, looped depth, latent-memory
shape, …) and **must not** ship any `*.py` files or set `auto_map`.
Vendored modeling code only.

---

## 1. Environment

You need Python 3.12, CUDA 12.8, PyTorch 2.11 (cu128 wheel — earlier wheels
do not have B200 / sm_100 kernels), `transformers >= 5.5`, and our
flash-linear-attention fork that ships the Quasar/GLA layers.

```bash
python3.12 -m venv .venv
. .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install transformers accelerate safetensors huggingface_hub bittensor numpy

# The Quasar attention layers + GLA cache live in this fork pinned by SILX:
pip install "flash-linear-attention @ git+https://github.com/SILX-LABS/quasar-flash-linear-attention.git@84ad1cc5a7428609d7e0e56d4041a775cd19b7bb"
```

Clone the validator/miner code so you have the vendored arch package
locally:

```bash
git clone https://github.com/unarbos/teutonic
cd teutonic
```

Sanity-check the model loads without `trust_remote_code`:

```bash
python -c "
import sys; sys.path.insert(0, '.')
import chain_config
chain_config.load_arch()  # registers the active arch with HF Auto*
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(chain_config.SEED_REPO, torch_dtype='bfloat16', device_map={'': 'cuda:0'})
print('loaded', sum(p.numel() for p in m.parameters())/1e9, 'B params')
"
```

`attn_implementation` will fall back to `eager` for Quasar layers — that's
expected (FA2 / SDPA upstream do not yet support QuasarForCausalLM).

---

## 2. Architecture you must match exactly

The king's `config.json` is the source of truth; the lock keys are the
union of the generic structural set in `validator.py` plus
`[arch].extra_lock_keys` in `chain.toml`. For the current Quasar king the
locked values are:

| field | value |
|---|---|
| `model_type` | `quasar` |
| `architectures` | `["QuasarForCausalLM"]` |
| `vocab_size` | `262144` (Teutonic-I tokenizer) |
| `d_model` / `hidden_size` | `4096` |
| `n_layers` / `num_hidden_layers` | `32` |
| `n_heads` / `num_attention_heads` | `32` |
| `head_dim` | `128` |
| `d_ff` / `intermediate_size` | `11008` |
| `quasar_layers` / `gated_layers` | `4` / `2` (cycle of 6) |
| `dense_input_layers` | `4` |
| `moe_type` | `bigmac` |
| `num_routed_experts` / `top_k` | `56` / `8` |
| `routed_expert_size` (effective) | `1024` |
| `shared_expert_size` | `2048` |
| `bigmac_r` | `0.25` (DCCA bottleneck) |
| `memory_slots` / `memory_dim` | `128` / `128` |
| `num_loops` | `1` |
| `tie_word_embeddings` | `true` |
| `rope_theta` | `1_000_000` |
| `max_seq_len` / `max_position_embeddings` | `16384` |

If any of these drift, the validator rejects with `"<key> mismatch"`.

If your repo contains `*.py` or your config has `auto_map`, the validator
rejects with `"repo ships *.py files"` or `"auto_map present in
config.json"`. The vendored `archs/<chain.toml [arch].module>` package in
this repo is the only path the network accepts — your weights must load
via plain `AutoModelForCausalLM.from_pretrained(...)` after
`chain_config.load_arch()`.

---

## 3. Repo naming and anti-impersonation

Your HF repo MUST match the chain's `repo_pattern`, which defaults to
`^[^/]+/<chain.name>-.+$` (today: `^[^/]+/Teutonic-XXIV-.+$`). It must
also embed the first 8 ss58 chars of your coldkey somewhere in the full
repo id (case insensitive, in either the namespace or the model
basename). Examples for coldkey `5DhAqMpdABCDEFG…` against the current
chain:

- ✅ `myaccount/Teutonic-XXIV-5DhAqMpd-v3`
- ✅ `5DhAqMpd/Teutonic-XXIV-noise01`
- ❌ `myaccount/Teutonic-XXIV-v3` (no coldkey prefix)

This is the anti-impersonation gate added 2026-04-29.

---

## 4. Quick start — noise miner

For testing only; will almost never dethrone but verifies your end-to-end
pipeline works.

```bash
. .venv/bin/activate
export HF_TOKEN=hf_...                 # write access to your HF org
export BT_WALLET_NAME=mywallet         # registered on SN3
export BT_WALLET_HOTKEY=default

python miner.py \
    --hotkey default \
    --suffix 5DhAqMpd-noise-01 \
    --noise 1e-4
```

Under the hood `miner.py`:

1. Pulls the king at its pinned commit SHA.
2. Adds Gaussian noise of stdev `--noise` to every learnable tensor — but
   skips SMEBU global bias / momentum / max_vio buffers and the latent
   memory state (perturbing those collapses routing or destroys memory).
3. Runs the same `validate_local_config` checks the validator runs.
4. Uploads to `<seed_namespace>/<chain.name>-<suffix>` (today:
   `unconst/Teutonic-XXIV-5DhAqMpd-noise-01`).
5. Submits the on-chain reveal commitment.

You can watch the validator pick it up at
[`https://teutonic.ai/dashboard.json`](https://teutonic.ai/dashboard.json).
The dashboard payload's `chain.name` field tells you which king is active.

---

## 5. Real training

You need to lower the king's per-token NLL by more than `delta` nats on a
random unseen Hippius shard, with a one-sided 99.9% bootstrap LCB > delta.
At chain genesis the king is uniform over its vocabulary (for the current
Quasar king, ln(262144) ≈ 12.48), so the first real training run will
dethrone.

A reasonable starting point uses
[`scripts/mining/train_challenger.py`](../scripts/mining/train_challenger.py)
which:

1. Reads the king repo + revision from the live dashboard.
2. Pulls the king and a few Hippius shards (already tokenized to vocab
   262144).
3. Trains a LoRA adapter (default targets cover Quasar's `q/k/v/o_proj`,
   `ffn.gate/up/down`, `w_down_proj/w_up_proj`).
4. Merges LoRA into the base weights → standalone candidate.
5. Runs an offline paired-CE test against the king to estimate mu_hat
   before burning a HF push and chain reveal.

```bash
torchrun --nproc-per-node=8 scripts/mining/train_challenger.py \
    --upload-repo myaccount/Teutonic-XXIV-5DhAqMpd-v1 \
    --noise-only false \
    --max-iters 3
```

Notes specific to Quasar:

- `nn.Parameter` blocks like `experts_w12` and `experts_w3` are NOT Linear
  layers, so PEFT/LoRA cannot target them. Train them with full SGD if you
  want to move the routed experts.
- The SMEBU bias buffer (`model.all_moe_bias`) is updated by the model
  itself only when `model.training=True`. Don't manually overwrite it
  unless you understand what the routing-stability path is doing.
- Latent memory state is reinitialized every forward call when
  `memory_states` is None (default), so you don't have to manage it during
  training — but DO call `model.eval()` before paired evaluation so SMEBU
  doesn't shift bias mid-test.
- `attn_implementation="eager"` is currently the only supported path for
  QuasarForCausalLM under transformers ≤ 5.7. SDPA / FA2 will be wired up
  later.

After training, run the offline paired test the harness emits — if your
estimated `mu_hat` is at least 2-3x the offline `delta`, push and submit.
Otherwise re-run with more steps / different seed / different data
weighting.

---

## 6. What the validator will tell you

Verdicts you might see in `dashboard.json` under `history[*]`:

- `accepted: true, verdict: "challenger"` — you are king.
- `verdict: "king"` — you didn't beat the king (LCB ≤ delta).
- `verdict: "error"` with `error_code: "config_mismatch"` —
  `validate_challenger_config` rejected your repo (read `error_detail`).
- `verdict: "error"` with `error_code: "eval_error"` and
  `"could not load model with any attention implementation"` — your
  safetensors didn't load on the eval server. Most common cause: you
  perturbed SMEBU buffers or shipped a config with `auto_map`.
- `rejection_reason: "untrainable:seed0(...):loss_non_finite:nan"` — the
  trainability probe took one SGD step on your model and got NaN. Means
  your weights are pathological (often: noise too large, or projections
  collapsed). Lower `--noise` or check the reparam-trick guard.

---

## 7. Useful links

- King model: see [`chain.toml`](../chain.toml) `[chain].seed_repo` (today:
  <https://huggingface.co/unconst/Teutonic-XXIV>).
- Live dashboard: <https://teutonic.ai>
- Live JSON: <https://teutonic.ai/dashboard.json> (the active chain name
  is published in the top-level `chain` field).
- Source: <https://github.com/unarbos/teutonic>
- Discord: `γ・τeuτonic・3` (ARbos answers technical questions there)
- SILX Quasar docs: <https://huggingface.co/silx-ai/Quasar-3B-A1B-Preview>

---

## 8. FAQ

**Q: Can I just upload my own MoE / dense / Mamba checkpoint?**
A: No. The validator pins `model_type` and the active arch's full dim
set. Cross-architecture submissions are rejected at config-match.

**Q: Why isn't FlashAttention-2 used?**
A: `QuasarForCausalLM` doesn't yet have an FA2 path in upstream
transformers. The eval server falls back to `eager`. This roughly doubles
forward wall vs FA2 but is otherwise correct.

**Q: Can I train and submit a quantized challenger?**
A: Eval loads in bf16. Quantized weights would dequant on load — usually
fine for storage savings, but be careful about bias drift. `safetensors`
only, no pickle.

**Q: How do I reset my submission if I made a mistake?**
A: You can't dethrone yourself. Wait for the next reign and submit again.
The validator de-dupes per-hotkey within a reign.

---

## Appendix A — Teutonic-LXXX (LIVE since 2026-05-07)

> This appendix is the live mining contract for `Teutonic-LXXX` (vanilla
> Qwen3-MoE 80 B total / 7.6 B active). The chain is **live** as of
> 2026-05-07 18:30 UTC, block 8133379. Genesis seed is
> [`unconst/Teutonic-LXXX-mock-king`](https://huggingface.co/unconst/Teutonic-LXXX-mock-king),
> a freshly random-init checkpoint at loss ≈ 13.3 nats/token on real
> CulturaX data — first competent training run dethrones easily.
>
> Active config lives in [`chain.toml`](../chain.toml). The chain switched
> from `Teutonic-XXIV` (Quasar 24 B); the previous Quasar config is
> archived at `chain.xxiv.toml.bak` (gitignored, operator-local).

### A.1 Architecture you must match exactly (LXXX)

`config.json` lock = generic structural keys + the LXXX `extra_lock_keys`
in [`chain.toml`](../chain.toml):

| field | value |
|---|---|
| `model_type` | `qwen3_moe` |
| `architectures` | `["Qwen3MoeForCausalLM"]` |
| `vocab_size` | `262144` (Teutonic-I / Gemma3-derived tokenizer — same as Quasar) |
| `hidden_size` | `4096` |
| `num_hidden_layers` | `36` |
| `num_attention_heads` / `num_key_value_heads` | `32` / `8` (GQA 4:1) |
| `head_dim` | `128` |
| `intermediate_size` | `11008` (used by any future dense layers; current king is fully MoE) |
| `num_experts` / `num_experts_per_tok` | `128` / `8` |
| `moe_intermediate_size` | `1408` |
| `decoder_sparse_step` | `1` (every layer is MoE) |
| `norm_topk_prob` | `true` |
| `router_aux_loss_coef` | `0.001` |
| `mlp_only_layers` | `[]` |
| `tie_word_embeddings` | `true` |
| `rope_parameters` | `{"rope_theta": 1000000.0, "rope_type": "default"}` |
| `max_position_embeddings` | `16384` |

Total: 82.328 B params / 7.586 B active per token / 153 GiB bf16 on disk.

Vanilla `Qwen3MoeForCausalLM` ships in `transformers ≥ 4.51`; no
`trust_remote_code`, no `auto_map`, no `*.py` files in the repo. Same
defenses as the Quasar chain.

### A.2 Minimum miner spec (LXXX)

The base king is ~153 GiB bf16 on disk. Per-iteration compute:

- ≥ 4× B200 (180 GiB) or ≥ 2× B300 (275 GiB) just to load the base in bf16
- ≥ 256 GiB host RAM for safetensors I/O during perturbation / save
- ≥ 1 TB free local SSD for king + challenger + a few HF cache copies
- HF account with write quota for ~165 GiB challenger pushes (each)
- Patience: a single push to HF takes 20-60 min at typical 50-150 MB/s

Reference noise-perturb script (mirrors [`miner.py:187-204`](../miner.py#L187-L204)
but standalone, no on-chain reveal): [`scripts/sandbox_perturb.py`](../scripts/sandbox_perturb.py).
For real training, build your own LoRA / full-finetune around
`Qwen3MoeForCausalLM` — note the experts are stored as
`model.layers.{l}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}`
(`nn.Linear`, LoRA-targetable), not a single fused parameter like
Quasar's BigMac.

### A.3 What dies, what stays the same

Stays the same vs Quasar chain:
- Bootstrap LCB acceptance rule with fixed `delta = 0.0025` nats/token
- Per-submission shard randomization via `blake2b(block_hash || hotkey)`
- Coldkey-prefix repo gate (8-char ss58 prefix in repo namespace OR basename)
- 5-king rolling payout
- **Tokenizer + dataset unchanged**: `unconst/Teutonic-I` (Gemma3-derived,
  vocab 262144). Live `dataset/v2/shards/...` on Hippius are still the
  eval source — no v3 dataset rebuild was needed.

Changes from Quasar:
- Repo regex switches from `^[^/]+/Teutonic-XXIV-.+$` to `^[^/]+/Teutonic-LXXX-.+$`
- Quasar-specific notes are obsolete: no SMEBU buffers, no latent memory,
  no `attn_implementation='eager'` requirement (FA2 / SDPA work natively
  for `Qwen3MoeForCausalLM`). LoRA target modules are
  `q_proj/k_proj/v_proj/o_proj` + `gate_proj/up_proj/down_proj` (per-expert).
- Per-eval wall grows from ~5 min to ~10 min steady (~14 min cold-page-cache),
  network throughput drops from ~11.5 evals/hour to ~5–6 evals/hour.
  Validator's `TEUTONIC_TICK_RESTART_AFTER` grew from 1800 s to 3600 s to match.

### A.4 On-chain reveal commitment (REVISED 2026-05-22)

**Wire format** (`|`-delimited, ~160 chars total):

```
v4|{challenger_repo}|{challenger_digest}|{author_hotkey}
```

| field | length | meaning |
|---|---|---|
| `v4` | 2 | Format version. |
| `challenger_repo` | ~30–80 | your Hippius repo, must match `chain.toml::repo_pattern` and contain your 8-char coldkey prefix anywhere (substring, case-insensitive). |
| `challenger_digest` | 71 or 43 chars | **binding commitment**. Immutable snapshot digest returned by Hippius upload: `sha256:<64hex>` or `hf:<40hex>`. |
| `author_hotkey` | 47–48 chars | submitter hotkey ss58, cross-checked against the chain reveal key. |

**What changed and why**:

The protocol no longer accepts a king-bound submission payload. Miners submit
the challenger they want evaluated, full stop. The validator already scores
that challenger against the current king at evaluation time, so forcing the
miner to also pin a `king_digest` in the on-chain payload only added racey,
failure-prone complexity.

**Legacy reveals that still include `king_digest` are dropped** at
`scan_reveals`. They are not enqueued, and any old queued entry that still
has the legacy field is failed as `legacy_reveal_version`.
