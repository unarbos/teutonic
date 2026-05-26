# Teutonic

Holy Pretraining Incentives — Bittensor SN3 (`netuid 3`).

Website: <https://teutonic.ai>

Teutonic is a king-of-the-hill pretraining subnet. Miners train a challenger
checkpoint, upload it to Hippius Hub, and submit a `v4` reveal on-chain. A
validator pins the challenger's immutable digest, runs a paired-bootstrap
cross-entropy duel against the reigning king on a held-out stream, and either
accepts (challenger becomes the new king) or rejects.

The active king (chain name, seed repo, vendored architecture, config lock
keys) is declared in [`chain.toml`](chain.toml). At time of writing that is
`Teutonic-Q3-8B` with seed repo `teutonic/teutonic-q3-8b-genesis`,
architecture module `archs.qwen3`, and tokenizer `Qwen/Qwen3-8B`. See
[`docs/MINING.md`](docs/MINING.md) for the live mining contract and
[`docs/DESIGN.md`](docs/DESIGN.md) for the current end-to-end flow.

## Repo layout

| Path | What it is |
| --- | --- |
| [`validator.py`](validator.py) | Single-file king-of-the-hill validator. Polls chain, validates challenger submissions, dispatches duels to the eval server, manages king lifecycle on Hippius, persists state to R2, and refreshes miner weights. |
| [`miner.py`](miner.py) | Reference miner: downloads the pinned king, perturbs weights, uploads to Hippius Hub, and commits a `v4|repo|digest|author_hotkey` reveal on-chain. |
| [`eval_server.py`](eval_server.py) | Persistent FastAPI service wrapping the eval pipeline. Caches the king across duels. SSE-streams progress to the validator. |
| [`eval/`](eval/) | Eval runners: [`torch_runner.py`](eval/torch_runner.py) (multi-GPU PyTorch paired-bootstrap CE), [`vllm_runner.py`](eval/vllm_runner.py) (vLLM evaluator), [`vllm_server.py`](eval/vllm_server.py) (vLLM-backed alternative eval server, not yet in production). |
| [`chain.toml`](chain.toml), [`chain_config.py`](chain_config.py) | Single source of truth for the active king (name, seed repo, repo pattern, active arch module, arch-specific config-lock keys). All other code reads from here. |
| [`archs/`](archs/) | Architecture shims, one subdir per arch. The live chain uses [`archs/qwen3/`](archs/qwen3/); historical Quasar and Qwen3-MoE shims remain for archived docs and experiments. |
| [`scripts/`](scripts/) | Operator + miner tooling: bot, dashboard, mining harness, dataset reshard, Cloudflare publish, chain-agnostic [`seed.py`](scripts/seed.py) / [`smoke_eval.py`](scripts/smoke_eval.py). |
| [`docs/`](docs/) | Current design and mining docs for `Teutonic-Q3-8B`, plus archived Quasar/LXXX migration notes. |
| [`website/`](website/) | Public dashboard assets (`index.html`, favicons). The validator uploads `index.html` to Hippius on every restart. |
| [`ecosystem.config.js`](ecosystem.config.js) | PM2 process manifest for the eval-tunnel + validator. Reads secrets via Doppler. |
| [`tunnel.sh`](tunnel.sh) | SSH port-forward to the GPU box hosting the eval server. |

## Setup

Python 3.11+. We use [`uv`](https://github.com/astral-sh/uv) for everything.

```bash
uv venv
source .venv/bin/activate
uv pip install -e .            # base
uv pip install -e ".[dev]"     # base + ruff
```

Secrets are read at runtime from Doppler (project `arbos`). `ecosystem.config.js`
shows every variable the validator expects.

## Running

### Validator + tunnel (production)

PM2 manages both the SSH tunnel to the GPU box and the validator loop:

```bash
pm2 start ecosystem.config.js
pm2 logs teutonic-validator
```

### Eval server (GPU box)

On the GPU machine that the tunnel forwards to:

```bash
uvicorn eval_server:app --host 127.0.0.1 --port 9000
```

`eval_server.py` lazy-loads the king from its pinned digest (`sha256:` on
Hippius or `hf:` if a chain is seeded from HF), then keeps it warm across
duels. Model cache lives under `/tmp/teutonic/hippius_models` by default, and
the underlying HF cache is used only for `hf:` digests.

### Mining

Don't follow the README alone for mining. Read the live recipe at
<https://teutonic.ai/mining> (also at [`docs/MINING.md`](docs/MINING.md)) and
use [`scripts/mining/`](scripts/mining/) as a working harness.

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md) — current live flow: reveal format,
  Hippius pinning, config lock, eval contract, and payout behavior.
- [`docs/MINING.md`](docs/MINING.md) — live mining contract for
  `Teutonic-Q3-8B`.
- [`docs/SCORING_PLAN.md`](docs/SCORING_PLAN.md) and
  [`docs/OPTIMAL_DESIGN.md`](docs/OPTIMAL_DESIGN.md) — proposal docs, not the
  current production path.

## License

MIT.
