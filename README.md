# [Teutonic](https://teutonic.ai/)

[Teutonic](https://teutonic.ai/) is a king-of-the-hill pretraining system for Bittensor subnet 3.

Miners submit immutable model checkpoints. The validator verifies each
submission and sends the challenger and current king to a remote GPU evaluator
for paired cross-entropy scoring. A successful challenger becomes the new king,
and the validator updates subnet weights and publishes the resulting state.

The evaluator update samples complete UTF-8 documents deterministically from
packed text and byte-offset indexes. Each model scores the same text with its
native tokenizer. Reward measures improvement in the byte-weighted loss deficit
against the pinned Horizon reference; ordinary token loss remains available for
historical charts. The default sampling budget is 35M original MiMo tokens.

Horizon uses a deterministic eight-GPU layer pipeline, followed by four MiMo king
and four challenger vLLM replicas. Large MiMo documents use two-GPU groups.
Horizon preloads at startup and after each duel, remaining on the GPUs while idle;
it unloads before the MiMo phase. GPU readiness is exposed as `horizon_gpu` in health.
See [deployment and GPU validation](DEPLOY.md); production cutover is pending.

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

Credentials last seven days. While the registration remains eligible, the
controller renews them one day before expiry and replaces already-expired
credentials after an outage. The same hotkey and upload prefix are retained;
the expired secret and session token cannot be reused.

The current CLI discovers and decrypts the latest generation automatically.
`auth` refreshes the local file, and `upload` and `submit` retrieve current
credentials before uploading. Update older CLI installations to use automatic
discovery; `auth --generation N` remains available to select an explicit generation.
Renewal never restores upload access after a finalized ready submission,
deregistration, or upload quota revocation.

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

For the new rollout, miners can select `MAIN`, `MIX01`, `MIX06` or `MIX07` with
`teutonic-miner ready --competition MIX01` (also supported by `submit`). Omission
selects MAIN. One ready submission per hotkey is shared across all competitions.
The allocation is MAIN **40%**, MIX01/MIX06/MIX07 **20% each**. MIX02/03/05 retire
at cutover and MIX04 stays retired; models and historical results are retained.
Each competition uses its existing category weights and a **35M original MiMo-token**
whole-document budget. Acceptance uses a byte-weighted bootstrap LCB for reduction
in loss above the fixed Horizon reference, in bits/byte. PostgreSQL holds the live
configuration; repository defaults take effect only after approved migration.
Each MIX challenges its own current king. Historical starting kings remain pinned
and are published as `starting_king` in `competitions.json`.

