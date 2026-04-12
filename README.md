# Teutonic

Backward-pass verification for decentralized training on [Bittensor](https://bittensor.com/) Subnet 3.

Miners train a shared model and submit Top-K compressed gradients along with a loss ledger and gradient probes. Validators replay spot-checked micro-batches, compare gradient probes via cosine similarity, and slash dishonest submissions.

## Requirements

- Python >= 3.11
- PyTorch >= 2.2

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For R2 storage support (Cloudflare R2 backend):

```bash
pip install -e ".[r2]"
```

## Run the local test

The local harness simulates 2 honest miners, 3 cheating miners (random losses, fake gradients, wrong data), and 1 validator over multiple windows on CPU:

```bash
source .venv/bin/activate
python run_local.py
```

You should see honest miners score high while each cheating strategy gets slashed and scored near zero.

## Project layout

```
src/teutonic/       Core library (compression, verification, sampling, protocols)
neurons/            Reference miner, validator, and trainer implementations
tests/              Integration and stress tests
run_local.py        Local simulation harness
```
