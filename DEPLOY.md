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

### Document-masked evaluation

The evaluator scores only isolated document fragments (`v1`). Sampling still uses
the same fixed windows, shard quotas, source proportions, and seeds. EOS 151645
ends a fragment; window boundaries also end fragments. Positions reset at every
fragment, with no synthetic BOS/EOS or context restored from outside the window.
EOS targets are scored, but each fragment's first token is excluded. Windows with
no scored targets fail explicitly rather than being silently replaced or assigned
a loss. Full and sliding attention retain their existing attention sinks.

Before deploying this revision, configure **both** validator and evaluator with:

```dotenv
TEUTONIC_EVALUATION_POLICY_VERSION=category-mix-token-bootstrap-v2
```

Both services reject the old policy identity; this is not a switch that can enable
unmasked scoring. Drain outstanding evaluations before a coordinated restart so
requests created under the old identity are not retried against the new evaluator.
Update `TEUTONIC_EVALUATOR_CODE_VERSION` consistently on both services as usual.
Window-only masked evaluation does not require dataset or threshold changes.
The optional long-document extension below adds a database column and a new
dataset configuration identity.

FA4 receives explicit cumulative fragment lengths through a scoped attention
adapter, including for batches with multiple rows. Eager uses explicit causal
document masks. Checkpoint Python files remain immutable. There is no automatic
fallback to unmasked attention or another backend.

The evaluator loads each checkpoint on the first GPU in its four-replica group,
then broadcasts all parameters and buffers to the other three GPUs with NCCL.
The king and challenger groups load independently. Receivers allocate BF16 model
storage without reading checkpoint shards; transfers use chunks of at most 256 MiB.
Workers wait on CPU queues until their group is prepared, and loading communicators
are released before scoring. Unchanged cached models skip loading and broadcast.
A preparation or broadcast failure closes the entire worker pool rather than
scoring with incomplete weights. Progress events report per-worker preparation,
broadcast transfer, and total readiness times in `load_timing`.

Model loading and each scoring worker enable strict PyTorch deterministic
algorithms with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic cuDNN, and
cuDNN benchmarking disabled. Unsupported nondeterministic operations fail rather
than produce a warning and continue. Worker audit metadata records these settings;
kernel caches include the execution settings in their identity. This controls
run-to-run numerical variation on the same software and hardware stack; it does
not promise identical floating-point results across GPU types, library versions,
or different batch shapes. No random seed replaces these execution settings.

The verdict, per-source scores, progress, and futility projection weight each
window's mean loss by its number of scored targets. Bootstrap resamples original
paired windows and recomputes the sum/count ratio in each draw. The futility rule
retains its observed-window advantage quantile heuristic, applying that advantage
to the exact remaining token count. Audit output uses `columnar-masked-v1` with
`scored_tokens` alongside each pair of window losses; verdicts also record the
scoring policy, EOS ID, total scored tokens, and aggregation/bootstrap units.

### Additional long-document evaluation

The `long-documents-2k-8k-v1` extension adds whole-document **input tokens** from
complete documents of **2049–8192 tokens** to the window sample. The category release
uses a 30,000,000-token budget and the same explicit category weights for windows
and documents in every competition. Within each category, document counts follow the index's relative
populations in 2049–4096 and 4097–8192 (inclusive boundaries), normalized
over those two eligible buckets. Documents of 8193 tokens or more are excluded.
Every eligible nonempty bucket gets
at least one complete document. Quotas are approximate because documents are
never truncated; this minimum can exceed a small category's token allocation.
The audit reports both planned and actual allocations.

The index is read from the public datasets bucket's `doc_index/<dataset>/` prefix.
Configuration fetches `doc_index/manifest.json` once and stores its URL, exact
UTF-8 contents and SHA-256. The inventory lists all index files with their
strong ETag and size, binary/token formats, categories and original dataset
manifest hashes. Inventory v1 requires per-file SHA-256; v2 also permits ETag/size
pins without a whole-file checksum. Requests and audits retain this pin; replay never refetches
the mutable inventory. Changing it requires an explicit configuration refresh.
Index range reads use `If-Match` and cache validated lookup pages (document
lengths and fragment locations) under `TEUTONIC_DOC_INDEX_CACHE_DIR`
(default: `$TEUTONIC_SHARD_CACHE_DIR/doc_index`).

The scheduled shard cleanup also expires document-index pages, cached manifest
JSON, and partial downloads. A custom `TEUTONIC_DOC_INDEX_CACHE_DIR` is cleaned
separately with the same age limits. Evaluations lock both cache locations so
cleanup cannot remove pages or manifests while an evaluation uses them.

For persistent full document indexes on the evaluator, add these settings to
`.gpu.env` (adjust the storage path for that host):

```dotenv
TEUTONIC_DOC_INDEX_STORE_DIR=/path/to/persistent/doc_index
TEUTONIC_DOC_INDEX_PREFETCH=1
TEUTONIC_DOC_INDEX_PREFETCH_MANIFEST_URL=https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev/doc_index/manifest-bb93856956e924c6139bebf73601c72cd58007ce916e6095b9929dd1b1d450c7.json
TEUTONIC_DOC_INDEX_PREFETCH_MANIFEST_SHA256=bb93856956e924c6139bebf73601c72cd58007ce916e6095b9929dd1b1d450c7
TEUTONIC_DOC_INDEX_DOWNLOAD_WORKERS=4
```

The release inventory contains 81 unique files, about 302.92 GB (282.11 GiB).
Use a persistent disk with that much free space **in addition to** headroom for
model downloads, shards and audit records. Keep the store outside disposable
cache directories; a path inside the model cache is rejected. The evaluator
creates the directory, which must be writable by its service account.

At process startup, background workers fetch the small SHA-256-pinned inventory
and download missing indexes. `/health` becomes available without waiting for
the inventory or files. Existing completed files are checked using their exact
descriptor, size and recorded filesystem identity/timestamps. No startup scan
or whole-index hashing occurs. A same-size file copied in manually is not trusted:
the store requires its own completion receipt, written only after an ETag-pinned
transfer finishes. This detects truncation and ordinary file changes, but is not
a cryptographic check against silent disk corruption; use trusted local storage.

An accepted evaluation pauses background downloads. Workers close their HTTP
responses at the next 256 KiB stream boundary and save a durable resume offset;
an already-running network read can take time to return (10-second inactivity
timeout). Evaluation admission does not wait for this. Complete files use local
reads; missing or partial files use the existing pinned HTTP ranges. The same
samples and scores are produced. Downloads resume after evaluation completion
or failure, and partial files resume after a process restart. A crash can discard
uncheckpointed bytes; transfers without a valid resume record start again.
Each range is limited to 8 MiB per worker. Newly requested index
pins are queued for prefetch too. Download failures are retried in the background
and do not block evaluation; inspect logs and free space if errors persist.

The download pool defaults to four workers (configurable from 1 to 16), separate
from the 128 index-sampling workers but using the same shared HTTP connection
pool and dataset R2 credentials. Full indexes and partial downloads are retained
across restarts and excluded from shard cleanup, including when nested in a
shard cache. They are not automatically expired. Stop the evaluator before
manually deleting the store. Set `TEUTONIC_DOC_INDEX_PREFETCH=0` to keep reading
completed local files without background downloads; unset the store variable to
use only the existing HTTP range/page-cache path.

After deploying the code and updating `.gpu.env`, drain any active evaluation
and restart the evaluator and cleanup schedules with their updated environment:

```bash
pm2 restart ecosystem.eval.config.js --update-env
curl --fail http://127.0.0.1:9000/health
pm2 logs teutonic-evaluator --lines 100 --nostream
```

`/health.document_index_store` reports whether prefetch is enabled/paused,
inventory readiness, ready/total files, bytes in completed files, active downloads
and cumulative retry errors. The evaluator can serve evaluations before all files
are ready. Continue to keep the evaluator code-version identity aligned with the
validator for the deployed commit.

Token shards use HTTP range downloads without disk caching;
`TEUTONIC_PRETOKENIZED_CACHE_DIR` is unused. Document-index caching remains enabled.
SHA-256 metadata is checked when available; whole-file hashes are not verified.
Transient download failures are retried automatically.

Evaluator token and document-index range reads for
`https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev` use the authenticated
R2 S3 endpoint when `TEUTONIC_EVALUATOR_R2_ENDPOINT`,
`TEUTONIC_EVALUATOR_R2_ACCESS_KEY_ID` and `TEUTONIC_EVALUATOR_R2_SECRET_ACCESS_KEY`
are configured. These credentials need read access to the `datasets` bucket;
`TEUTONIC_EVALUATOR_R2_BUCKET` does not override that dataset bucket. Optional
`TEUTONIC_EVALUATOR_R2_SESSION_TOKEN` and `TEUTONIC_EVALUATOR_R2_REGION` (default
`auto`) are supported. With no R2 configuration, reads remain public; incomplete
configuration or denied access fails rather than silently falling back. Other
origins remain public. Canonical URLs, cache identities and object pins stay unchanged.

Window sampling downloads four shard files concurrently, with up to eight range
workers per file. Document fragments use sixteen shard workers; index planning
uses up to 128 workers, with one task per category (currently at most 25).
All range reads share a pool capped at 128 connections. Window indices
are selected before concurrent I/O, preserving deterministic selection and output order.

Deploy this update to both hosts; no migration or new dependencies are required.
Drain evaluations and persist verdicts before stopping the validator. Restart the
evaluator, check `/health`, then resume the validator. Keep code versions aligned;
resolve pending retries before changing the version. PM2's stop timeout does not
drain running evaluations.

Build the inventory locally (this command performs no uploads):

```bash
python scripts/build_document_index_manifest.py \
  --config artifacts/long-document-eval/main.json \
  --manifest-url https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev/doc_index/manifest.json \
  --output artifacts/long-document-eval/manifest.json --metadata-only
```

The builder accepts the earlier prepared MAIN configuration to enumerate files.
Use `--metadata-only` for HEAD-only ETag/size pins, or `--index-root /path/to/doc_index`
to hash local originals. Remote hashing is disabled. Existing verified checksums are
preserved when object and source pins match. New ETag-only entries use inventory v2;
legacy v1 inventories still require SHA-256 for every entry.
Publish the generated bytes unchanged to the manifest URL as a separate action.
Before publication, `configure_evaluation.py --doc-index-manifest-file PATH`
and `prepare_long_document_evaluation.py --index-manifest-file PATH` use those
same local bytes. Their corresponding manifest-URL and manifest-SHA256 options
select the public location and optionally require an expected hash.

Documents are reconstructed in fragment order across rows/shards. Incomplete
source documents are skipped deterministically and counted. The selection cap is
8192 tokens, including terminal EOS; larger source documents are excluded and
counted, never cropped. The checkpoint must support at least 8192 positions.
Out-of-vocabulary tokens, missing shards, invalid fragments and missing
terminal EOS fail evaluation.

FA4 packs unequal-length samples with explicit attention boundaries and reset
positions. Original window boundaries remain isolated even when a window ends
without EOS. Scheduling limits batch token count and sum of squared lengths.
The current 8192-token selection cap uses the normal single-GPU path for every
sample. Sequence-parallel groups are never initialized under this policy.
The tested two-GPU path above 128,974 tokens remains implemented for a future
sampling policy; it is inactive with the current selection range.
Grouped MoE releases intermediate
projection buffers before allocating weighted outputs, avoiding overlapping
large temporary tensors for long documents. GPU memory fit still requires
validation on the real checkpoints. Every replica preflights the longest sample before
ordinary scoring (diagnostic losses do not enter the verdict). An OOM in a
multi-sample single-GPU batch retries the same samples
in smaller batches. An OOM in a single document's LM-head projection retries
with a smaller projection chunk, down to 64 positions. An OOM in the single
document's forward pass, or with the minimum projection chunk, fails evaluation
explicitly. Nothing is truncated or skipped to fit. Preflight memory peaks and
OOM retries are recorded in worker metadata. King and challenger score identical samples.

All scored tokens contribute to the combined token-weighted verdict. Bootstrap
resamples paired original windows or whole documents, never their individual
fragments. `component_scores` separates windows from long documents, and
`long_document_scores` reports bucket, category and category/bucket losses.
`columnar-masked-documents-v2` records aligned document rows, lengths, categories,
fragment locations and scored-token counts. For this first version, early
stopping is disabled when long documents are enabled, so the full sample is scored.

Apply the additive migration before deploying the new validator/configuration
tools, then update the validator/evaluator code identities together:

```bash
psql "$TEUTONIC_DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/add_evaluation_long_documents.sql
```

After the category cutover and GPU validation, budgets can be changed using the configuration tool:

```bash
python scripts/configure_evaluation.py --competition main --long-document-tokens 6000000 --dry-run
python scripts/configure_evaluation.py --competition main --long-document-tokens 6000000
```

For current releases, perform the [category cutover](#main-and-mix01mix06-category-competitions)
first. Then use `--competition MIX01` through `MIX06`, or `--all` for all seven. The current policy fixes `--max-document-tokens` at
8192; zero and larger caps are rejected. Existing configurations remain window-only
until enabled. `--long-document-tokens 0` disables the additional sample. Budget
and threshold edits preserve index pins; `--refresh-doc-index` explicitly selects
new versions. Rebuilding an index requires the corresponding tokenization snapshot.
The former `long-documents-v1` configuration is rejected rather than silently
reinterpreted. Applying `--long-document-tokens 6000000` upgrades that configuration
to the new bounded policy while preserving its pinned index and source manifests.

For an isolated test, prepare a configuration and optional CPU-only sample plan
without touching PostgreSQL, services or model weights:

```bash
python scripts/prepare_long_document_evaluation.py --competition main \
  --output artifacts/long-document-eval/main.json --plan-context-limit 1048576
```

The standalone preparation tool uses `--bundle-dir` for a prepared immutable bundle,
or the pinned `chain.toml` manifest URLs when omitted, and validates the explicit `chain.toml` category weights. For a replay, use a cached evaluation with matching sources and
proportions. In an isolated checkout with all writable caches redirected to its
test directory, use real cached checkpoints on all eight GPUs:

```bash
python scripts/replay_cached_evaluation.py --record /path/to/cached-main-record.json \
  --model-cache /path/to/read-only/immutable-r2 --batch-size 40 \
  --long-document-config artifacts/long-document-eval/main.json \
  --verify-packing --output artifacts/long-document-eval/main-gpu.json
```

This explicitly adds a new sample and scores it fully; its comparison against
the saved verdict therefore includes the dataset change. `--verify-packing`
compares real-model losses in mixed batches against one-sample batches, covering
both current length buckets and an ordinary window. `--prepare-only` samples and
downloads data without loading weights or using GPUs. Replay never promotes
models or publishes evaluation artifacts to R2.

GPU validation is required before rollout. Use real cached checkpoints and
cached evaluation records through the normal eight-worker topology. The replay
checks the reconstructed window identities against saved per-sample provenance
before loading the models. Run it from an isolated checkout with its own caches
and output paths; do not start the evaluator HTTP service for a replay.

```bash
python -m scripts.replay_cached_evaluation \
  --record /path/to/cached/evaluation.json \
  --model-cache /path/to/model/cache \
  --batch-size 40 --gpu-ids 0,1,2,3,4,5,6,7 \
  --attn-implementation flash_attention_4 \
  --output /tmp/masked-replay.json
```

Replay always uses masked scoring, even when its input record was produced under
the old policy; both policy identities are included in the report. Its differences
from an old record measure the method change, not exact replay error. Compare
several batch sizes and record throughput, memory, paired advantages, and verdicts.
The experimental report's slowdown is not an assumed property of this implementation.

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

## MAIN and MIX01–MIX06 category competitions

PostgreSQL remains the runtime source of truth. `chain.toml` defines the category
registry and seven explicit category-weight tables under `[competitions.*]`.
Every competition uses the same sampler: **20,000 windows plus 30,000,000 whole-document
input tokens**, with the existing 8,192-token document cap. Percentages are sampling
targets. The verdict remains the paired, scored-token-weighted bootstrap over the
whole mix, with **LCB strictly greater than the threshold**. There is no additional
largest-source or category gate.

| Competition | Category allocation | Final incentive | LCB threshold |
|---|---|---:|---:|
| MAIN | Original 24 categories, proportional to token inventory | 25% | 0.0015 |
| MIX01 | Former MATH categories 45%, former CODE categories 45%, DCLM 10% | 12.5% | 0.0015 |
| MIX02 | Former CODE category set 50%; remaining PretBeh categories 50%; token-proportional inside both sets | 12.5% | 0.0015 |
| MIX03 | Former MATH categories 60%, former CODE categories 40% | 12.5% | 0.0015 |
| MIX04 | TXT360-V2 categories 70%, PretBeh categories 30%; token-proportional inside each | 12.5% | 0.0015 |
| MIX05 | DCLM 40%, PretBeh categories 60%; token-proportional inside PretBeh | 12.5% | 0.0015 |
| MIX06 | All 25 categories including DCLM, proportional to token inventory | 12.5% | 0.0015 |

The former MATH/CODE labels above explain the migration only. Published manifests
contain unique flat category lists, original dataset identities, and source hashes.
MIX01/MIX03 retain the former groups' internal weights. MIX02's CODE set includes
`reasoning`, `other`, and `data-science` once; the other half contains
`format-rewrites`, `games`, `general`, and `planning`. No overlapping weight is added,
and MIX02 has no special sampling path. DCLM is the single category `dclm-baseline-1.0`.

### Kings, commitments, and replacement rewards

Keep `TEUTONIC_COMPETITION` set to the existing MAIN database name on every control
plane service. One validator, promotion worker, and weight publisher serve all seven.
MIXes start without kings. A kingless MIX challenges the **latest MAIN king** using
that MIX's categories and threshold. If MAIN changes before coronation, the old
comparison is requeued against the latest MAIN without consuming a failure retry.
After its first win, a MIX uses its own latest king.

The cutover snapshots all currently rewarded positions and their durable shares;
cutover itself changes no payouts. Legacy recipients are ordered by crowned block
(oldest first; bootstrap recipients without reigns come first). Each **first** MIX
winner gets 12.5% and removes one oldest legacy position immediately. That position
receives zero; its share is not split with the challenger. The remaining budget is
normalized over the MAIN position and surviving legacy positions, retaining their
relative snapshot shares. MAIN is protected from these evictions. After the legacy
roster is exhausted, the remaining budget belongs to MAIN. With six MIX kings,
MAIN receives 25% and each MIX receives 12.5%.

A later winner in an established MIX only replaces that MIX's recipient. A new MAIN
winner takes the MAIN position immediately; the outgoing MAIN model is not added to
the legacy roster. Positions held by the same hotkey aggregate after allocation;
removing one position does not remove that hotkey's other entitlements. Repeated
coronation attempts cannot evict twice. The persisted transition and weight
publication update in the same transaction. Existing UID mapping, deregistration
normalization, burn fallback and 101-block publication cadence are retained.

Miners use `teutonic-miner ready --competition MIX02` or
`teutonic-miner submit ... --competition MIX02`. MAIN is the default. Uppercase names
normalize to lowercase wire keys. The wire format recognizes MIX01–MIX99, but this
release enables only MIX01–MIX06. New ready signals must target a configured, active
competition. Historical math/code/text commitments and results remain readable;
those competitions stop accepting new submissions after cutover. One submission
per hotkey and existing model ownership/reuse limits remain global.

### 1. Prepare artifacts while the existing services keep running

For the already prepared release, the immutable mix manifests and document-index
inventory are published; `artifacts/category-mixes/publication.json` records their
pins. Copy the reviewed `main.json`, `mix01.json` through `mix06.json`, `bundle.json`,
`document-index.json`, and `publication.json` to the same relative directory on the
administration host. These artifacts are ignored by Git. Skip rebuilding and
republishing them, and continue at "Load the final URL and hash" below. Changing
only the LCB threshold does not require new dataset or index manifests.

Use a separate checkout for this preparation so running Python services cannot
load partially upgraded code. Administrative commands use database-owner credentials,
not the validator or dashboard login. Export the existing deployment environment:

```bash
set -a
source .env
set +a
.venv/bin/python scripts/prepare_competition_mixes.py \
  --output-dir artifacts/category-mixes
```

This reads public manifests and creates `main.json`, `mix01.json` … `mix06.json`,
`bundle.json` and `index-sources.json` locally. No token shards are copied. The bundle
contains the exact version URLs that the configuration command will pin. Release
defaults in `chain.toml` also point to the published immutable URLs. Reuse the
reviewed bundle for all following steps; rebuilding it can select newer inputs.

Publish those prepared files without changing any discovery URL:

```bash
.venv/bin/python scripts/publish_category_artifacts.py \
  --bundle-dir artifacts/category-mixes --bucket datasets
```

The publisher validates every category inventory against `chain.toml`, writes only
`splits/<competition>/versions/<sha256>.json`, and reads each object back to verify
its checksum. Existing objects are never overwritten. It records `publication.json`.
Do not run the legacy `publish_split_manifests.py --publish` for this rollout.

Build the expanded document-index inventory from all six original dataset sources.
DCLM's binary files already exist under `doc_index/dclm-baseline-1.0/`; the missing
step is adding their verified descriptors to the inventory. Reuse verified entries
from the current manifest only when source pins, ETags, and sizes still match:

```bash
curl --fail --show-error --location --user-agent 'teutonic-eval/1.0' \
  'https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev/doc_index/manifest.json' \
  --output artifacts/category-mixes/previous-index.json
.venv/bin/python scripts/build_document_index_manifest.py \
  --sources artifacts/category-mixes/index-sources.json \
  --base-manifest artifacts/category-mixes/previous-index.json \
  --manifest-url 'https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev/doc_index/manifest-category-build.json' \
  --output artifacts/category-mixes/document-index.json \
  --workers 4 --metadata-only
.venv/bin/python scripts/publish_category_artifacts.py \
  --bundle-dir artifacts/category-mixes --bucket datasets \
  --index-manifest-file artifacts/category-mixes/document-index.json
```

The metadata-only build makes HEAD requests; it does **not download index bodies**.
It records DCLM's ETags and sizes and preserves existing verified SHA-256 values
for unchanged entries. Inventory v2 explicitly permits these ETag/size descriptors;
the evaluator checks `If-Match`, returned ETag, size and exact content range on every
range read. DCLM's entries do not claim whole-file SHA-256 verification. The inventory
JSON itself remains SHA-256 pinned. Old v1 manifests retain their stricter schema.
If exact originals are local, `--index-root /path/to/doc_index` can be used instead
of `--metadata-only`, without downloading data. No index binaries are regenerated.
The final inventory is published as `doc_index/manifest-<sha256>.json`; relative file
paths still resolve beside it. **Leave `doc_index/manifest.json` untouched** while
old configurations run. Use the versioned URL for cutover; it supplies the updated
inventory without breaking existing pins. The original stable URL is not required
by the new configurations and can remain unchanged.

Load the final URL and hash from the publication receipt:

```bash
INDEX_URL=$(.venv/bin/python -c 'import json; print(json.load(open("artifacts/category-mixes/publication.json"))["document_index"]["url"])')
INDEX_SHA=$(.venv/bin/python -c 'import json; print(json.load(open("artifacts/category-mixes/publication.json"))["document_index"]["sha256"])')
```

Before downtime, prepare CPU-only configs and document selection plans for all seven.
This uses separate local caches, reads public indexes, and does not touch PostgreSQL:

```bash
for competition in main mix01 mix02 mix03 mix04 mix05 mix06; do
  .venv/bin/python scripts/prepare_long_document_evaluation.py \
    --competition "$competition" --bundle-dir artifacts/category-mixes \
    --token-budget 30000000 --index-manifest-url "$INDEX_URL" \
    --index-manifest-file artifacts/category-mixes/document-index.json \
    --index-manifest-sha256 "$INDEX_SHA" --plan-context-limit 8192 \
    --index-cache artifacts/category-mixes/index-cache \
    --output "artifacts/category-mixes/$competition.documents.json" || break
done
```

Check every plan succeeds, includes its declared categories, and meets the existing
whole-document budget tolerance. Full model scoring still needs the evaluator GPUs;
local CPU tests do not establish GPU memory usage or evaluation duration.

### 2. Drain, back up, and rehearse the cutover

Announce the retirement of math/code/text and pause new submissions. Let all existing
upload verification and snapshot jobs finish before stopping the access controller,
then let the old validator and promotion worker finish the remaining queue.
The cutover refuses while submissions are pending, evaluating, or awaiting promotion.
When drained, stop the remaining control-plane services:

```bash
# Confirm existing verification/snapshot jobs have finished, then stop intake.
pm2 stop teutonic-access-controller
# Wait for the remaining evaluations and promotions to drain.
pm2 stop teutonic-validator teutonic-promotion-worker teutonic-weight-publisher teutonic-dashboard-view
mkdir -p /srv/teutonic-backups
export TEUTONIC_BACKUP_FILE="/srv/teutonic-backups/pre-category-mixes-$(date +%Y%m%d-%H%M%S).dump"
scripts/db/backup_control_plane.sh
psql "$TEUTONIC_DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/add_category_competitions.sql
```

This migration assumes the existing split-competition and long-document migrations
are already installed. Large flat shard lists use a reversible compressed database
representation to avoid PostgreSQL's JSONB container limit. Public manifests and
config hashes remain unchanged; the new readers verify the original manifest hash. It adds the active flag and reward-transition state, expands
identifier constraints, and updates the dashboard view and access-controller read
grant. It preserves kings, configs and payouts. It is safe to apply again.

Run the complete configuration transaction in dry-run mode, using the reviewed local
bundle and the final published index pin:

```bash
.venv/bin/python scripts/configure_evaluation.py \
  --all --initialize --replace-definition --cutover \
  --bundle-dir artifacts/category-mixes --n 20000 --long-document-tokens 30000000 \
  --delta-threshold 0.0015 \
  --doc-index-manifest-url "$INDEX_URL" \
  --doc-index-manifest-file artifacts/category-mixes/document-index.json \
  --doc-index-manifest-sha256 "$INDEX_SHA" --dry-run \
  > artifacts/category-mixes/cutover-preview.jsonl
```

Review all seven category lists and thresholds, the MAIN recipient, legacy eviction
order, and initial reward shares. A dry run makes no persistent changes. A missing
MAIN king, absent durable reward shares, running control-plane services, existing MIX kings, pending submissions,
unmatched source/index pins, or insufficient shard capacity blocks cutover explicitly.

### 3. Activate and deploy matching services

Install the same reviewed code release on validator and evaluator hosts. Set both
`.env` and `.gpu.env` to:

```dotenv
TEUTONIC_EVALUATION_POLICY_VERSION=category-mix-token-bootstrap-v2
TEUTONIC_EVALUATOR_CODE_VERSION=<same-release-identity-on-both-hosts>
```

Re-run the preview command above **without `--dry-run`**. This atomically snapshots
rewards, installs seven versioned configs, activates MAIN/MIX01–MIX06, and retires
math/code/text. Existing MAIN and all historical reigns remain in place. MIXes begin
kingless. Do not repeat `--cutover` after success; it deliberately refuses to reset
a transition already in progress.

On the GPU host, restart the evaluator with the updated environment and check its
`/health` reports the matching policy/code versions:

```bash
pm2 restart ecosystem.eval.config.js --only teutonic-evaluator,teutonic-model-cache-cleanup,teutonic-shard-cache-cleanup --update-env
```

On the validator host, start the updated services and publish the dashboard:

```bash
pm2 restart ecosystem.config.js --only teutonic-validator,teutonic-promotion-worker,teutonic-weight-publisher,teutonic-dashboard-view --update-env
```

Deploy the updated `website/` assets through the existing static-site deployment.
Check `competitions.json`, MAIN's `datasets/manifest.json`, and every
`competitions/mixNN/datasets/manifest.json`: seven active competitions, 20k/30M
budgets, correct thresholds, all MIXes kingless, and unchanged initial weights.
The page shows the six-dataset inventory (including DCLM), category percentages,
king/fallback state, and retired history. The reward transition stays in the
backend and is not displayed as a separate section. Upload `dashboard-v1.js`,
`app.js`, `style.css`, and the updated `llms.txt` before `index.html`; the HTML uses
new asset cache versions. Publish the
updated dashboard JSON feeds before uploading the page, including dataset sources'
`dataset` field and `competitions/mix06/datasets/manifest.json`. Do not upload local
preview JSON or screenshots. Then restart intake and announce the new miner names:

```bash
pm2 restart ecosystem.config.js --only teutonic-access-controller --update-env
```

Keep the existing weight-publisher mode; do not accidentally switch a live publisher
to dry-run or a dry-run publisher to live. The first MIX crown should create one new
reign, one persisted eviction (if legacy recipients remain), and one global weight
publication. Check the outgoing legacy position is gone and the winner has 12.5%.

### 4. Subsequent updates and rollback

After cutover, unspecified settings retain their database values. For example:

```bash
.venv/bin/python scripts/configure_evaluation.py --competition MIX02 --delta-threshold .0015 --dry-run
.venv/bin/python scripts/configure_evaluation.py --all --n 20000 --long-document-tokens 30000000 --dry-run
```

Budget-only changes preserve source and index pins. Category changes require a new
reviewed bundle and `--replace-definition`; refreshing a URL cannot silently change
category weights. All seven may be updated atomically using `--all`.

Before activation, rollback means keeping the old code/environment and active
configs; the additive schema and unused immutable R2 objects can stay. A failed
cutover transaction rolls back as a whole. After successful activation, **before
reopening intake or any new MIX evaluation/crown**, keep services stopped, restore
the verified pre-cutover database backup with the existing restore procedure, restore
the old code and policy environment on both hosts, then restart the old services.
After new work has been accepted or a new crown has changed published weights, do
not reset the transition or blindly restore the old backup: that would discard new
accepted history. Stop intake and reconcile the recorded crowns/publications before
choosing a forward fix or a reviewed recovery. Do not delete token/index objects or
historical competition rows as part of rollback.

### Explicit category percentages

These are the release defaults in `chain.toml`. Percentages are rounded here to
four decimal places; manifests retain full precision. Each category appears once.

#### MAIN

| Category | Sampling target |
|---|---:|
| `3efforts-pretrain` | 0.5702% |
| `code-no-thinking` | 3.2027% |
| `code-no-thinking-hard` | 2.4974% |
| `code-task-synthesis` | 2.0233% |
| `code-thinking-hard` | 2.0786% |
| `code-thinking-v0` | 3.6276% |
| `code-thinking-v1` | 3.3843% |
| `codeio` | 0.1709% |
| `data-science` | 3.1064% |
| `format-rewrites` | 2.6048% |
| `games` | 1.1680% |
| `general` | 4.7384% |
| `instruction-following` | 0.3203% |
| `math-dialogue` | 13.3791% |
| `math-rewrite` | 4.5334% |
| `math-thinking-oss` | 3.3751% |
| `math-thinking-qwen` | 5.6152% |
| `other` | 3.5973% |
| `planning` | 6.4962% |
| `reasoning` | 10.7884% |
| `socratic-math-dialogue` | 1.0922% |
| `txt360-qa` | 8.8330% |
| `web-high-medium` | 6.2307% |
| `web-high-nltk-qa` | 6.5664% |

#### MIX01

| Category | Sampling target |
|---|---:|
| `3efforts-pretrain` | 0.8100% |
| `code-no-thinking` | 4.1850% |
| `code-no-thinking-hard` | 3.2400% |
| `code-task-synthesis` | 2.6550% |
| `code-thinking-hard` | 2.7000% |
| `code-thinking-v0` | 4.7250% |
| `code-thinking-v1` | 4.4100% |
| `codeio` | 0.2250% |
| `data-science` | 4.0500% |
| `dclm-baseline-1.0` | 10.0000% |
| `format-rewrites` | 3.6450% |
| `games` | 1.6200% |
| `math-dialogue` | 18.5850% |
| `math-rewrite` | 6.3000% |
| `math-thinking-oss` | 4.6800% |
| `math-thinking-qwen` | 7.8300% |
| `other` | 4.6800% |
| `reasoning` | 14.1300% |
| `socratic-math-dialogue` | 1.5300% |

#### MIX02

| Category | Sampling target |
|---|---:|
| `code-no-thinking` | 4.6447% |
| `code-no-thinking-hard` | 3.6219% |
| `code-task-synthesis` | 2.9343% |
| `code-thinking-hard` | 3.0145% |
| `code-thinking-v0` | 5.2609% |
| `code-thinking-v1` | 4.9081% |
| `codeio` | 0.2478% |
| `data-science` | 4.5050% |
| `format-rewrites` | 8.6783% |
| `games` | 3.8915% |
| `general` | 15.7868% |
| `other` | 5.2170% |
| `planning` | 21.6435% |
| `reasoning` | 15.6458% |

#### MIX03

| Category | Sampling target |
|---|---:|
| `3efforts-pretrain` | 1.0800% |
| `code-no-thinking` | 3.7200% |
| `code-no-thinking-hard` | 2.8800% |
| `code-task-synthesis` | 2.3600% |
| `code-thinking-hard` | 2.4000% |
| `code-thinking-v0` | 4.2000% |
| `code-thinking-v1` | 3.9200% |
| `codeio` | 0.2000% |
| `data-science` | 3.6000% |
| `format-rewrites` | 4.8600% |
| `games` | 2.1600% |
| `math-dialogue` | 24.7800% |
| `math-rewrite` | 8.4000% |
| `math-thinking-oss` | 6.2400% |
| `math-thinking-qwen` | 10.4400% |
| `other` | 4.1600% |
| `reasoning` | 12.5600% |
| `socratic-math-dialogue` | 2.0400% |

#### MIX04

| Category | Sampling target |
|---|---:|
| `data-science` | 2.8675% |
| `format-rewrites` | 2.4044% |
| `games` | 1.0782% |
| `general` | 4.3739% |
| `other` | 3.3207% |
| `planning` | 5.9966% |
| `reasoning` | 9.9587% |
| `txt360-qa` | 28.5857% |
| `web-high-medium` | 20.1639% |
| `web-high-nltk-qa` | 21.2504% |

#### MIX05

| Category | Sampling target |
|---|---:|
| `data-science` | 5.7349% |
| `dclm-baseline-1.0` | 40.0000% |
| `format-rewrites` | 4.8089% |
| `games` | 2.1564% |
| `general` | 8.7479% |
| `other` | 6.6413% |
| `planning` | 11.9933% |
| `reasoning` | 19.9173% |

#### MIX06

| Category | Sampling target |
|---|---:|
| `3efforts-pretrain` | 0.4531% |
| `code-no-thinking` | 2.5451% |
| `code-no-thinking-hard` | 1.9846% |
| `code-task-synthesis` | 1.6078% |
| `code-thinking-hard` | 1.6518% |
| `code-thinking-v0` | 2.8827% |
| `code-thinking-v1` | 2.6894% |
| `codeio` | 0.1358% |
| `data-science` | 2.4685% |
| `dclm-baseline-1.0` | 20.5338% |
| `format-rewrites` | 2.0699% |
| `games` | 0.9282% |
| `general` | 3.7654% |
| `instruction-following` | 0.2545% |
| `math-dialogue` | 10.6319% |
| `math-rewrite` | 3.6025% |
| `math-thinking-oss` | 2.6821% |
| `math-thinking-qwen` | 4.4622% |
| `other` | 2.8586% |
| `planning` | 5.1623% |
| `reasoning` | 8.5731% |
| `socratic-math-dialogue` | 0.8679% |
| `txt360-qa` | 7.0193% |
| `web-high-medium` | 4.9513% |
| `web-high-nltk-qa` | 5.2181% |
