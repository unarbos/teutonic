# Teutonic deployment and competition startup

This runbook describes the production two-host deployment and the order used to
start a new competition. Run commands from the repository root unless stated
otherwise.

## Host layout

| Host | Service | PM2 name or command | Purpose |
| --- | --- | --- | --- |
| Validator/control host | PostgreSQL 16 | `docker compose up -d postgres` | Durable control-plane state |
| Validator/control host | Access controller | `teutonic-access-controller` | Chain scanning, scoped R2 credentials, upload verification |
| Validator/control host | Evaluator SSH tunnel | `teutonic-eval-tunnel` | Loopback-only connection to the GPU evaluator |
| Validator/control host | Validator scheduler | `teutonic-validator` | Evaluation queue and verdict persistence |
| Validator/control host | Promotion worker | `teutonic-promotion-worker` | Private-to-public model promotion and crowning |
| Validator/control host | Weight publisher | `teutonic-weight-publisher` | Durable on-chain weight publication |
| Validator/control host | Dashboard publisher | `teutonic-dashboard-view` | Sanitized `dashboard.json` publication |
| Remote GPU host | Evaluator | `teutonic-evaluator` | Model download and paired-loss evaluation |
| Remote GPU host | Model cleanup | `teutonic-model-cache-cleanup` | Scheduled GPU model-cache cleanup |
| Remote GPU host | Shard cleanup | `teutonic-shard-cache-cleanup` | Scheduled dataset-cache cleanup |

Do not run the cleanup processes on the validator host. Do not run PostgreSQL
or the control-plane services on the GPU host.

## Shared prerequisites

Both hosts need:

- The same Teutonic source revision.
- Python 3.11 or newer.
- Node.js and PM2.
- `rclone` available on `PATH`.
- A dedicated Unix account with access only to the files it needs.

Install PM2 once per host:

```bash
npm install --global pm2
```

Keep the repository, environment files, wallet, SSH keys, and PM2 home owned by
the dedicated service account. Never commit `.env` or `.gpu.env`.

## R2 layout

Create these three buckets before starting a competition:

| Bucket | Visibility | Use |
| --- | --- | --- |
| `teutonic-private-models-enam` | Private | One isolated upload prefix per miner registration |
| `teutonic-models-enam` | Public | Genesis and promoted immutable models |
| `teutonic-dash-enam` | Public read | Dashboard state and encrypted credential mailboxes |

The validator-host R2 key needs the object permissions used by seed upload,
verification, promotion, cleanup, and dashboard publication. The Cloudflare API
token must be able to create and revoke the scoped parent tokens used by the
access controller.

The GPU uses a separate read-only R2 key that can read both model buckets. It
does not receive the Cloudflare management token or access-controller secrets.

## Configure the active chain

Edit `chain.toml` before creating the database competition. Important fields
are:

- `[chain].name`: stable chain/model family name.
- `[chain].seed_repo`: Hugging Face repository containing the genesis model.
- `[chain].repo_pattern`: allowed challenger naming pattern.
- `[arch].module`: architecture registration module.
- `[seed].tokenizer_repo`: tokenizer used by the evaluator.
- `[seed].seed_digest`: exact `hf:<commit>` revision.
- `[seed].genesis_hotkey`: hotkey that owns genesis and is registered on the
  target subnet.

The default chain generation is derived from the chain name and seed digest:

```bash
.venv/bin/python -c 'import chain_config; print(chain_config.CHAIN_GENERATION)'
```

Set `TEUTONIC_CHAIN_GENERATION` on the validator host to exactly that output.
For a deliberate reset that reuses the same name and seed, set an explicit,
new generation in `chain.toml`:

```toml
[chain]
generation = "mimo-v2.5-pro-reset-2"
```

Never reuse a generation for a different genesis model or policy namespace.

## Validator/control host setup

Create the environment and install the validator dependency group:

```bash
cd /srv/teutonic
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[validator]'

cp example.env .env
chmod 600 .env
```

Fill every required value in `.env`. At minimum, verify these groups:

- Chain scope: network, netuid, chain generation, and competition name.
- Weight wallet: wallet path, coldkey wallet name, and validator hotkey name.
- PostgreSQL: one database name, user, password, port, and matching URL.
- GPU tunnel: SSH host, user, key, known-hosts file, and loopback ports.
- Evaluator policy: matching code, dataset, tokenizer, and evaluator versions.
- R2: account, API token, S3 keys, endpoint, region, and all bucket names.
- Dashboard: dashboard R2 endpoint/key and public mailbox base URL.

Generate the two stable access-controller keys once:

```bash
openssl rand -hex 32
openssl rand -hex 32
```

Store the first output as `TEUTONIC_CONTROLLER_SECRET_KEY` and the second as
`TEUTONIC_MAILBOX_SIGNING_KEY`. Back them up securely. Replacing the controller
secret prevents recovery of encrypted durable token state.

The current PM2 ecosystem loads the complete `.env` into every process in
`ecosystem.config.js`. Protect the service account and `~/.pm2`; PM2 metadata
and `dump.pm2` can contain environment values.

## GPU host setup

Use the same checkout revision as the validator host:

```bash
cd /srv/teutonic
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[evaluator]'

cp example.gpu.env .gpu.env
chmod 600 .gpu.env
```

Fill `.gpu.env`, paying particular attention to:

- Loopback listener `127.0.0.1:9000`.
- Read-only evaluator R2 credentials.
- Dataset bundle and tokenizer identities.
- The same evaluator/policy/code versions configured on the validator host.
- The same `TEUTONIC_EVAL_BATCH_SIZE` configured on both hosts (default `40`),
  validated for the GPU model and sequence length in use.

Start the evaluator and GPU cleanup schedules:

```bash
pm2 start ecosystem.eval.config.js
pm2 status
curl --fail http://127.0.0.1:9000/health
```

The evaluator must be healthy locally before starting the validator-host SSH
tunnel.

## Start PostgreSQL

On the validator/control host, Compose automatically reads `.env`:

```bash
docker compose up -d postgres
docker compose ps
docker compose exec postgres sh -lc 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

PostgreSQL is bound to `127.0.0.1:${POSTGRES_PORT}` and is not exposed publicly.
The complete start-state schema is installed automatically only when the
`postgres-data` volume is empty. Existing volumes are never migrated
automatically; apply any documented additive setup scripts explicitly.

Do not run `docker compose down -v` during ordinary deployment or competition
startup. It destroys the PostgreSQL volume. A new chain generation can coexist
with previous generations in the same database.

### Back up PostgreSQL

Install the PostgreSQL 16 client on the validator host, load `.env` into the
shell, and use the verified backup helper before upgrades or a new generation:

```bash
mkdir -p /srv/teutonic-backups
export TEUTONIC_BACKUP_FILE="/srv/teutonic-backups/control-plane-$(date +%Y%m%d-%H%M%S).dump"
scripts/db/backup_control_plane.sh
```

The helper creates a custom-format dump, validates it with `pg_restore`, and
writes a SHA-256 checksum beside it. Copy both files to protected off-host
storage.

### Configure evaluation early stopping

Existing PostgreSQL volumes need the one-time additive schema setup before a
validator running this revision is started:

```bash
psql "$TEUTONIC_DATABASE_URL" \
  --no-psqlrc \
  --set=ON_ERROR_STOP=1 \
  --file=scripts/db/add_evaluation_early_stopping.sql

psql "$TEUTONIC_DATABASE_URL" \
  --no-psqlrc \
  --set=ON_ERROR_STOP=1 \
  --file=scripts/db/add_safetensors_reuse_limit_error.sql
```

The first setup creates one competition-scoped policy row with early stopping
enabled, a `0.4` minimum fraction, `0.95` observed-advantage quantile, zero
margin, and a 100-sequence check interval. The second exposes the evaluator's
safetensors reuse-limit code through the dashboard view. Both are idempotent;
the policy setup preserves an existing row.

Inspect or update the current competition through the validated helper:

```bash
.venv/bin/python scripts/configure_early_stopping.py \
  --enabled \
  --min-fraction 0.4 \
  --advantage-quantile 0.95 \
  --margin 0.0 \
  --check-interval 100
```

Arguments that are omitted retain their current PostgreSQL values. The
validator reads the row before claiming each new evaluation, so a change does
not require a validator restart and never changes an evaluation already in
progress. Every claim snapshots the complete policy in its signed request and
durable `evaluations.thresholds` record.

Early stopping is rejection-only: it can retain the king when the configured
futility projection is below `delta_threshold - margin`, but it never crowns a
challenger before the full evaluation. The observed advantage quantile is a
heuristic for unseen sequences, not a mathematical bound; lower fractions and
lower quantiles stop more aggressively.

## Publish genesis and start the competition

The seed bootstrap reads `chain.toml`, downloads the exact Hugging Face commit,
verifies the model inventory, uploads it to the public model bucket, resolves
the genesis hotkey in the finalized metagraph, and creates the PostgreSQL
competition and first king reign.

Load the validator environment into the current shell. Keep `.env`
shell-compatible and quote values containing spaces.

```bash
set -a
. ./.env
set +a
```

For a large seed, publish and verify R2 first without changing PostgreSQL:

```bash
.venv/bin/python scripts/bootstrap_seed.py \
  --local-dir /srv/teutonic-data/seed-cache \
  --upload-only
```

Then start the competition:

```bash
.venv/bin/python scripts/bootstrap_seed.py \
  --local-dir /srv/teutonic-data/seed-cache
```

The default transfer profile uses 16 concurrent files, 16 multipart streams,
and 64 MiB parts. The operation is resumable and idempotent. Re-running it with
the same configuration verifies the existing public artifact and competition;
a conflicting genesis fails closed.

The genesis hotkey from `chain.toml` must already exist in the target subnet's
finalized metagraph. A private or gated Hugging Face seed additionally requires
`HF_TOKEN` in the bootstrap environment.

Confirm the competition exists:

```bash
docker compose exec postgres sh -lc \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "SELECT netuid, chain_generation, name, current_reign_id FROM control_plane.competitions;"'
```

## Start the validator-host PM2 services

Start in dependency order after PostgreSQL, genesis, and the GPU evaluator are
ready:

```bash
pm2 start ecosystem.config.js --only teutonic-access-controller
pm2 start ecosystem.config.js --only teutonic-eval-tunnel
pm2 start ecosystem.config.js --only teutonic-validator
pm2 start ecosystem.config.js --only teutonic-promotion-worker
pm2 start ecosystem.config.js --only teutonic-weight-publisher
pm2 start ecosystem.config.js --only teutonic-dashboard-view
```

Verify the evaluator through the tunnel:

```bash
curl --fail http://127.0.0.1:9000/health
pm2 status
```

Keep `TEUTONIC_WEIGHT_PUBLISHER_MODE=dry_run` until the wallet, finalized UID
mapping, generated plans, and logs are verified. To enable live weight
extrinsics, change it to `active` in `.env` and reload only that process:

```bash
pm2 restart ecosystem.config.js \
  --only teutonic-weight-publisher \
  --update-env
```

## Logs and health checks

View all control-plane logs:

```bash
pm2 logs --lines 100
```

View one process:

```bash
pm2 logs teutonic-access-controller --lines 100
pm2 logs teutonic-validator --lines 100
pm2 logs teutonic-promotion-worker --lines 100
pm2 logs teutonic-weight-publisher --lines 100
pm2 logs teutonic-dashboard-view --lines 100
```

On the GPU host:

```bash
pm2 logs teutonic-evaluator --lines 100
pm2 status
```

Verify public dashboard publication using the configured public dashboard/mailbox
base URL:

```bash
curl --fail "$TEUTONIC_MAILBOX_PUBLIC_BASE_URL/dashboard.json"
```

## Survive host reboots

After all processes are healthy, configure PM2 startup under the dedicated
service account:

```bash
pm2 startup
pm2 save
```

Run the system command printed by `pm2 startup`. Secure `~/.pm2/dump.pm2`
because PM2 can persist process environments there. PostgreSQL already uses
`restart: unless-stopped` in Compose; ensure Docker itself starts at boot.

## Credential renewal update

This update needs no database migration or GPU evaluator changes. Deploy the
updated control-host checkout and restart only the access controller:

```bash
pm2 restart ecosystem.config.js --only teutonic-access-controller --update-env
pm2 logs teutonic-access-controller --lines 100 --nostream
pm2 save
```

The controller checks its configured subnet and chain generation every 60 seconds.
It schedules seven-day replacement credentials when an active registration's
published credentials have at most one day left, including those already expired.
Only active parent tokens for hotkeys whose submission eligibility is unconsumed
can renew. Revoked authority is never restored. Pending publication jobs are
retried without creating duplicate generations; expired jobs are completed and
replaced on a subsequent scan.

Each publication retains its immutable generation object and updates the encrypted
`mailbox/v1/<registration_id>/latest.bin` alias with cache storage disabled.
Revocation deletes both generation objects and the alias. No plaintext credentials
are published. Ensure any custom mailbox cache rules respect `Cache-Control: no-store`.

Miners should update their CLI. `auth` discovers and decrypts the latest generation;
`upload` and `submit` retrieve it before uploading. An explicit `--generation N`
still pins a generation. Old registrations without an alias fall back to generation
one until their first renewal. Existing exported credentials or external S3 clients
must be refreshed separately; running uploads do not swap credentials mid-transfer.
Publish the updated website documentation through the normal website deployment.

## Start another competition generation

1. Back up PostgreSQL.
2. Stop the six validator-host PM2 processes so no old-generation work runs.
3. Update both hosts to the same source revision.
4. Update `chain.toml` and choose a new chain generation.
5. Update `.env`, `.gpu.env`, and all evaluator policy/version identities.
6. Restart the GPU evaluator and verify its health.
7. Run the seed bootstrap without `--upload-only` to create the new competition.
8. Restart the validator-host processes in dependency order.
9. Keep the weight publisher in `dry_run` until the new generation is verified.

Old database records and content-addressed public models do not need to be
deleted. Never point a new competition at a previous generation identifier.

## Local test with a mock evaluator

The mock evaluator is test-only and must never be used in production. On a
single local host, use the control `.env` and do not start the SSH tunnel:

```bash
TEUTONIC_MOCK_ENV_FILE=.env \
  pm2 start tests/ecosystem.mock.config.js --only teutonic-mock-evaluator

pm2 start ecosystem.config.js --only \
  teutonic-access-controller,teutonic-validator,teutonic-promotion-worker,teutonic-weight-publisher,teutonic-dashboard-view
```

Verify it on the validator's configured evaluator port:

```bash
curl --fail http://127.0.0.1:9000/health
pm2 logs teutonic-mock-evaluator --lines 100
```

Do not add the mock evaluator to either production ecosystem file.

## MAIN, MATH, CODE, and TEXT competitions

PostgreSQL is the runtime source of truth for evaluation manifests, their category
weights, sample counts, and thresholds. `chain.toml` supplies initialization defaults
only. Editing it or restarting a service does not overwrite active database policy.
`TEUTONIC_COMPETITION` continues to name the existing MAIN database competition;
it must not be changed to a specialist name on the validator or weight publisher.
One validator, one promotion worker, and one weight publisher serve all four competitions.

Before this rollout, drain evaluations and promotions, then stop the validator,
promotion worker, weight publisher, and dashboard publisher. Back up PostgreSQL and
apply `scripts/db/add_split_competitions.sql` with `psql -v ON_ERROR_STOP=1`. The
migration preserves current kings and payouts. Deploy all control-plane services
together: scheduler and promotion-worker locks are global to the subnet/generation.
Stop the old validator and promotion worker before starting their updated versions.

Build the specialist inventories from the existing public source manifests:

```bash
python scripts/publish_split_manifests.py --output-dir artifacts/split-manifests
```

With dataset-bucket credentials exported as `TEUTONIC_R2_ENDPOINT`,
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (optional `R2_SESSION_TOKEN`), publish:

```bash
python scripts/publish_split_manifests.py \
  --output-dir artifacts/split-manifests --bucket datasets --publish
```

The three public discovery keys are `splits/math/manifest.json`,
`splits/code/manifest.json`, and `splits/text/manifest.json`. Content-addressed copies
are also stored under `splits/<name>/versions/<sha256>.json`. The files reference
existing shards; no token files are copied. Category percentages are explicit:
CODE reasoning is 31.4%, TEXT txt360-qa is 26.5%, and every group totals 100%.

Initialize the missing specialist configurations without replacing MAIN's active
settings. Run administrative commands with the configured database owner credentials
and the existing `TEUTONIC_NETUID`, `TEUTONIC_CHAIN_GENERATION`, and
`TEUTONIC_COMPETITION` environment values:

```bash
python scripts/configure_evaluation.py --all --initialize --dry-run
python scripts/configure_evaluation.py --all --initialize
```

Each specialist plans 30,000 sequences: 21,000 from its own manifest and 4,500
from each other manifest. Category percentages apply inside all three allocations.
Integer quotas use deterministic largest-remainder rounding. The combined sample
mix produces one paired-bootstrap verdict. Each competition has its own threshold,
initially 0.003; specialist early stopping is enabled by default. MAIN's existing
sampling and early-stopping configuration is retained.

Administrative updates are explicit, versioned, and atomic (`--all` changes all
four in one transaction). Unspecified settings retain their active database values:

```bash
# Update only MATH's threshold; performs no manifest downloads.
python scripts/configure_evaluation.py --competition math --delta-threshold 0.004 --dry-run
python scripts/configure_evaluation.py --competition math --delta-threshold 0.004

# Update MAIN's threshold independently.
python scripts/configure_evaluation.py --competition main --delta-threshold 0.003

# Refresh R2 inventories deliberately, retaining each competition's threshold.
python scripts/configure_evaluation.py --all --refresh-manifests --dry-run
python scripts/configure_evaluation.py --all --refresh-manifests

# Change one inventory URL in one specialist competition's pinned mix.
python scripts/configure_evaluation.py --competition code \
  --manifest math=https://example.org/splits/math/manifest.json --dry-run
```

`--n` and `--shards-per-dataset` are also supported. Manifest updates validate
hashes, category coverage, proportions, and sample capacity before activation.
Newly claimed evaluations read the active policy; existing attempts and retries
retain their recorded samples, threshold, versions, and evaluated king.

Miners select a competition with `teutonic-miner ready --competition math` or
`teutonic-miner submit ... --competition math`. Omission selects MAIN, including
all existing commitments. The compact commitment suffix is `:math`, `:code`, or
`:text`; the server preserves legacy MAIN commitments. Selection is immutable.
The one-ready-submission-per-hotkey constraint remains global, as does the existing
three-completed-evaluations limit for identical safetensors weights.

The global queue follows finalized ready-commit order. A specialist without a king
faces the current MAIN king on the specialist mix. A win creates only that split's
king; a loss leaves it unfilled. Pending promotion/crowning blocks the next duel.
The promotion worker handles MAIN and all three specialists, including recovery
of copied winners awaiting a crown. An identical checkpoint may be resubmitted
under a new hotkey only by its original coldkey, recorded at the original ready
block. Another coldkey (or unverifiable ownership) is rejected before evaluation
and checked again before verdict acceptance and crowning. The check also compares
safetensors hashes regardless of filenames or changes to other inventory files.
An allowed resubmission can reuse the verified public artifact while preserving
its original provenance; the new winning evaluation determines the crowned
upload and hotkey. Evaluation and submission limits still apply. This guard runs
on the control plane and needs no GPU evaluator update or new database migration.

Rewards transition only as first specialist winners appear:

| Established specialist kings | MAIN shares, newest to oldest | Each established specialist |
|---|---|---|
| 0 | 20%, 20%, 20%, 20%, 20% | — |
| 1 | 25%, 20%, 20%, 20% | 15% |
| 2 | 30%, 20%, 20% | 15% |
| 3 | 40%, 15% | 15% |

A new MAIN winner shifts MAIN history and removes the oldest paid MAIN recipient.
A split's first winner replaces the oldest paid MAIN recipient; a later split winner
replaces that split's recipient. Failed evaluations do not move the transition.
The migration itself changes no shares. Rewards are combined into one publication
anchored to the latest crown in any competition, with the existing 101-block cadence.
Hotkey-to-UID remapping retains relative shares; unavailable recipients are omitted
and remaining shares normalized, with the existing burn fallback if none remain.

The dashboard publisher retains MAIN's `dashboard.json` and `datasets/manifest.json`
and adds `competitions/<math|code|text>/dashboard.json` plus each competition's
`datasets/manifest.json`. `competitions.json` lists configured competitions. Deploy
the updated website assets to show all four competitions together. MAIN retains
the primary dashboard and benchmarks; specialist panels show their own kings,
configurations and results. One global queue combines all competition entries,
labels each competition, and preserves the published global queue positions.
One current-evaluation panel shows the active duel and its competition.
Restart the updated services after configuration, and check all four public dataset
summaries and the combined weight plan before enabling new submissions.
