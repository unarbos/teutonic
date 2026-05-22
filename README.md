# Teutonic

Holy Pretraining Incentives — Bittensor SN3 (`netuid 3`).

Website: <https://teutonic.ai>

Teutonic is a king-of-the-hill pretraining subnet. Miners train a challenger
LLM and submit it on-chain. A validator pulls the challenger from HuggingFace,
runs a paired-bootstrap cross-entropy duel against the reigning king on a
held-out tokenized stream, and either accepts (challenger becomes the new king)
or rejects. Winner takes 100% of SN3 emission until dethroned.

The active king (chain name, seed HF repo, vendored architecture, config
lock keys) is declared in [`chain.toml`](chain.toml). At time of writing
that's `unconst/Teutonic-XXIV` — a freshly-initialised SILX-AI Quasar
hybrid MoE (~8B active, ~24B total). See [`docs/MINING.md`](docs/MINING.md)
for the live mining recipe and [`docs/DESIGN.md`](docs/DESIGN.md) for the
full mechanism.

## Repo layout

| Path | What it is |
| --- | --- |
| [`validator.py`](validator.py) | Single-file king-of-the-hill validator. Polls chain, dispatches duels to the eval server, manages king lifecycle on HF, persists state to R2. |
| [`miner.py`](miner.py) | Reference miner: clones the king, perturbs weights, uploads to HF, commits a `v4` reveal on-chain. |
| [`eval_server.py`](eval_server.py) | Persistent FastAPI service wrapping the eval pipeline. Caches the king across duels. SSE-streams progress to the validator. |
| [`eval/`](eval/) | Eval runners: [`torch_runner.py`](eval/torch_runner.py) (multi-GPU PyTorch paired-bootstrap CE), [`vllm_runner.py`](eval/vllm_runner.py) (vLLM evaluator), [`vllm_server.py`](eval/vllm_server.py) (vLLM-backed alternative eval server, not yet in production). |
| [`chain.toml`](chain.toml), [`chain_config.py`](chain_config.py) | Single source of truth for the active king (name, seed repo, repo pattern, vendored arch module, arch-specific config-lock keys). All other code reads from here. |
| [`archs/`](archs/) | Vendored architectures, one subdir per arch (currently [`archs/quasar/`](archs/quasar/)). Each self-registers with HF Auto* on import so checkpoints load without `trust_remote_code`. The active arch is selected by `chain.toml -> [arch].module`. |
| [`scripts/`](scripts/) | Operator + miner tooling: bot, dashboard, mining harness, dataset reshard, Cloudflare publish, chain-agnostic [`seed.py`](scripts/seed.py) / [`smoke_eval.py`](scripts/smoke_eval.py). |
| [`docs/`](docs/) | Design doc, scoring plan, current-chain mining guide. |
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

`eval_server.py` will lazy-load the king from HF, then sit on it across duels.
HF cache lives under `~/.cache/huggingface/hub/` and is watermark-cleaned by
the server itself.

### Mining

Don't follow the README — it's deliberately thin so it can't go stale. Read
the live recipe at <https://teutonic.ai/mining> (also at
[`docs/MINING.md`](docs/MINING.md)) and use [`scripts/mining/`](scripts/mining/)
as a working harness.

## Docs

- [`docs/DESIGN.md`](docs/DESIGN.md) — full mechanism: how the duel scoring,
  config-lock, trainability probe, dethrone rule, and emission incentive fit
  together.
- [`docs/SCORING_PLAN.md`](docs/SCORING_PLAN.md) — exponential dethrone
  scoring rollout plan.
- [`docs/MINING.md`](docs/MINING.md) — active chain's mining contract and
  step-by-step recipe (chain identity comes from [`chain.toml`](chain.toml)).

## License

MIT.
