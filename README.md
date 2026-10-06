# [Teutonic](https://teutonic.ai/)

[Teutonic](https://teutonic.ai/) is a king-of-the-hill pretraining system for Bittensor subnet 3.

Miners submit immutable model checkpoints. The validator verifies each
submission and sends the challenger and current king to a remote GPU evaluator
for paired cross-entropy scoring. A successful challenger becomes the new king,
and the validator updates subnet weights and publishes the resulting state.

## Miner CLI

Each hotkey can submit one model. Teutonic requires an Ed25519 hotkey because
the validator encrypts its temporary upload credentials to that hotkey.

### Install

Use Python 3.11 or newer. `btcli` must also be available on `PATH` if the
hotkey still needs to be registered on the subnet.

```bash
git clone https://github.com/unarbos/teutonic.git
cd teutonic

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[miner]'
```

Set the wallet and target subnet values used in the examples:

```bash
WALLET_PATH="$HOME/.bittensor/wallets"
WALLET_NAME=your-coldkey-wallet
HOTKEY_NAME=your-hotkey
NETWORK=finney
NETUID=3
```

Create a new Ed25519 hotkey if needed:

```bash
btcli wallet new-hotkey \
  --wallet-name "$WALLET_NAME" \
  --hotkey "$HOTKEY_NAME" \
  --wallet-path "$WALLET_PATH" \
  --crypto-type ed25519
```

Save the common wallet location:

```bash
teutonic-miner configure --wallet-path "$WALLET_PATH"
```

### Register and activate the mailbox

This command validates the Ed25519 hotkey, registers it on the subnet if
necessary, waits for its finalized UID, and commits its signed mailbox
activation. Subnet registration can spend TAO.

```bash
teutonic-miner register \
  --wallet-name "$WALLET_NAME" \
  --hotkey-name "$HOTKEY_NAME" \
  --network "$NETWORK" \
  --netuid "$NETUID"
```

Add `--check-only` to fail instead of paying for a missing subnet registration.
The active chain generation is read automatically from `chain.toml`.

Registration creates local state under
`.teutonic-miner/<hotkey-address>/registration.json` and selects the hotkey as
active. Inspect the saved state with:

```bash
teutonic-miner list
teutonic-miner status --hotkey "$HOTKEY_NAME"
```

### Retrieve upload authorization

Poll the public mailbox and decrypt the temporary, prefix-scoped R2
credentials:

```bash
teutonic-miner auth --hotkey "$HOTKEY_NAME"
```

The public mailbox URL is built in. The encrypted result is written locally as
`upload-auth.json` with mode `0600`. Never share or commit this file.

### Upload and submit a model

The model directory must contain a complete checkpoint. It must not contain
symlinks or a file named `manifest.json`; the CLI creates and signs that
manifest itself.

```bash
MODEL_DIR=/absolute/path/to/model
MODEL_NAME=your-model-name

teutonic-miner upload \
  --hotkey "$HOTKEY_NAME" \
  --name "$MODEL_NAME" \
  "$MODEL_DIR"
```

After the upload finishes, commit the model as ready:

```bash
teutonic-miner ready --hotkey "$HOTKEY_NAME"
```

`ready` is the irreversible submission point. Once it finalizes, that
hotkey's one submission is consumed, its R2 upload authority is revoked, the
public mailbox credential is removed, and validator processing continues
asynchronously. If evaluation rejects the checkpoint because it has reached
the allowed evaluation reuse limit, the access controller automatically
deletes that checkpoint's private R2 prefix.

For an already saved hotkey, the complete check, registration validation,
authorization, upload, and ready flow can be run as one command:

```bash
teutonic-miner submit \
  --hotkey "$HOTKEY_NAME" \
  --name "$MODEL_NAME" \
  "$MODEL_DIR"
```

### Multiple hotkeys

List saved hotkeys and select the default used when `--hotkey` is omitted:

```bash
teutonic-miner list
teutonic-miner use your-other-hotkey
teutonic-miner status
```

## Competitions

Miners can select `main`, `math`, `code`, or `text` with
`teutonic-miner ready --competition math` (also supported by `submit`). Omission
selects MAIN. One ready submission per hotkey is shared across all competitions.
Specialists use 30,000 planned samples with a 70/15/15 category-group mix;
early stopping remains enabled. PostgreSQL controls active manifests and thresholds.
See [deployment and configuration](DEPLOY.md#main-math-code-and-text-competitions)
for manifest publication, `scripts/configure_evaluation.py`, and gradual rewards.
