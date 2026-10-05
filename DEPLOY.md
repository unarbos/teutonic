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
