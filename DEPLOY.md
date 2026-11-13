# Teutonic deployment and competition startup

This runbook describes the production two-host deployment and the order used to
start a new competition. Run commands from the repository root unless stated
otherwise.

## Prepared 13 November update: 60M and active-document scoring

The current local update selects **60M original MiMo tokens** for MAIN, MIX01,
MIX06 and MIX07, with threshold **0.00075 bits/byte**. The live configurations
read on 13 November still use 35M. Nothing in this preparation changes them.

Use [`deployment/packed-text-60m-20261113/`](deployment/packed-text-60m-20261113/README.md)
for this update. It contains four immutable 60M manifests, exact previous config
versions and guarded budget-only SQL. Publish and verify these manifests after
approval, then back up the database and apply the SQL with claimers stopped and
active work drained. Preserve current dataset/catalog pins and category weights.
Deploy matching validator/evaluator code with policy
`horizon-deficit-active-bpb` on both hosts. The policy excludes documents where
both deficits are zero from reward/LCB, while retaining every sample's diagnostics.
The 60M sampling budget is measured before filtering; there is no replacement
sampling. No schema migration is needed; the budget requires configuration row
updates. The original 12 November 35M cutover below is historical, not the SQL to
apply for this update.

## Host layout

| Host | Service | PM2 name or command | Purpose |
| --- | --- | --- | --- |
| Validator/control host | PostgreSQL 16 | `docker compose up -d postgres` | Durable control-plane state |
| Validator/control host | Access controller | `teutonic-access-controller` | Chain scanning, scoped R2 credentials, upload verification |
| Validator/control host | Evaluator SSH tunnel | `teutonic-eval-tunnel` | Loopback-only connection to the GPU evaluator |
| Validator/control host | Validator scheduler | `teutonic-validator` | Evaluation queue and verdict persistence |
| Validator/control host | Promotion worker | `teutonic-promotion-worker` | Private-to-public model promotion and crowning |
| Validator/control host | Weight publisher | `teutonic-weight-publisher` | Durable on-chain weight publication |
| Validator/control host | Live queue/progress publisher | `teutonic-dashboard-live` | Independent `dashboard-live.json` every 5 seconds |
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

Provision these buckets before starting a packed-text competition:

| Bucket | Visibility | Use |
| --- | --- | --- |
| `teutonic-private-models-enam` | Private | One isolated upload prefix per miner registration |
| `teutonic-models-enam` | Public | Genesis and promoted immutable models |
| `teutonic-dash-enam` | Public read | Dashboard state and encrypted credential mailboxes |
| `dataset` | Private | Packed UTF-8 documents, indexes, sampling catalog and split manifests |

The validator-host R2 key needs the object permissions used by seed upload,
verification, promotion, cleanup, and dashboard publication. The Cloudflare API
token must be able to create and revoke the scoped parent tokens used by the
access controller.

The GPU uses a separate read-only R2 key that can read both model buckets and `dataset`. It
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

### Packed-text evaluation with Horizon

The prepared update uses `teutonic-evaluator-v3` and policy `horizon-deficit-active-bpb`.
Set `TEUTONIC_EVALUATION_POLICY_VERSION=horizon-deficit-active-bpb` on both
validator and GPU hosts when deploying this update, after the current evaluation
has finished. The validator and evaluator reject the previous policy ID for new
evaluations; historical `horizon-deficit-bpb` results remain readable. This policy
change requires no SQL migration, new dataset or inference-cache invalidation.
Deploy the validator and evaluator together after the new manifests and PostgreSQL
configuration are ready. This local implementation does **not** activate itself
against the previous configuration: the validator rejects legacy active settings.
The final sampling catalog and four 35M competition manifests were published and
hash-verified on 12 November 2026. Their small manifests, exact publication pins,
release scope and staged SQL are in
[`deployment/packed-text-20261112/`](deployment/packed-text-20261112/).
The 192 MB sampling catalog remains in R2; it is not checked into the repository.
Those artifacts describe the initial 35M release; use the prepared 60M update above for the next deployment.

The corrected 7M MIX06 full flow passed on 12 November 2026: 2,563 documents
across all 25 categories, four king and four challenger TP1 replicas, and successful
startup/preload reuse and post-evaluation Horizon reload. The verdict took 21m 55s.
The final Horizon minimum-padding repeatability check also passed: 19 cases and
512 document-loss comparisons matched the saved deterministic baseline exactly,
including short and mixed-length groups of 1/2/3/8, repeats and reversed order.
It used the final worker source with one fresh load. Run this focused check with
`scripts/verify_horizon_pipeline_worker.py --small-batches-only` and the saved
fixture/baseline arguments. The full harness additionally covers groups of 16 and
large throughput cases. The 35M runtime remains an estimate.

Keep the current Teutonic II 110B king/challenger architecture. Horizon is pinned to
`IFM/K2-Horizon-375B-A23B` commit `7d22cc0ac26a91003d19482efa71070244239b11`.
Startup downloads its verified checkpoint into a persistent cache outside the
king/challenger cleanup scope. There is no old `doc_index` prefetch job.
After verification, the evaluator automatically loads Horizon onto all eight GPUs
in the background and keeps it resident while idle. A duel reuses this worker for
reference scoring, then unloads it before starting the MiMo replicas—even if all
reference losses were cached. After evaluation cleanup, Horizon reloads in the
background for the next duel. The verdict does not wait for this reload; a new duel
arriving before loading finishes waits when it reaches reference scoring.
`/health` reports GPU readiness under `horizon_gpu`, separately from checkpoint
verification under `horizon_download`. Horizon worker loading has a 20-minute
timeout, including startup and post-evaluation preloads. Failed preloads are reported there and
retried when a duel needs the reference. Shutdown prevents further preloads and
closes the resident worker. Current and preceding load logs live under
`TEUTONIC_HORIZON_CACHE_DIR/gpu-worker/`. This lifecycle passed the corrected 7M
GPU test, including a post-evaluation reload of 6m 50s. The final small-batch
padding repeatability check subsequently passed with exact document losses.
Reserve at least 758 GB for Horizon weights plus space for the two Teutonic snapshots,
text scratch space, and normal download overhead. Use a persistent NVMe path:

```dotenv
TEUTONIC_HORIZON_CACHE_DIR=/path/to/persistent/horizon
TEUTONIC_EVALUATION_POLICY_VERSION=horizon-deficit-active-bpb
TEUTONIC_EVALUATOR_VERSION=document-evaluator-v3
TEUTONIC_TOKENIZER_BACKEND=gigatoken
TEUTONIC_REFERENCE_TOKENIZER_BACKEND=gigatoken
TEUTONIC_TOKENIZER_THREADS=8
TEUTONIC_VLLM_PYTHON=/absolute/path/to/.venv-vllm/bin/python
TEUTONIC_VLLM_CACHE_DIR=/path/to/persistent/vllm-kernels
TEUTONIC_HORIZON_PYTHON=/absolute/path/to/.venv-horizon/bin/python
TEUTONIC_HORIZON_KERNEL_CACHE_DIR=/path/to/persistent/horizon-pipeline-kernels
```

Keep `TEUTONIC_EVALUATOR_CODE_VERSION` aligned on both hosts. Install the evaluator
extra and the locally patched Gigatoken wheel described in
[`patches/gigatoken-horizon/README.md`](patches/gigatoken-horizon/README.md).
Token-ID comparisons run in the verification script/tests, not during evaluation.
An explicit `reference` tokenizer backend remains available for diagnostics.

Install the MiMo GPU worker in a **separate Python 3.12 environment**, using the tested
vLLM 0.33.0 / Torch 2.14.0 CUDA 13.0 / Transformers 5.20.2 stack. Do not combine
the `evaluator` and `vllm-worker` extras in one environment. The control process
retains its existing Transformers/tokenizer environment. On the target B300 host:

```bash
sudo apt-get install --no-install-recommends build-essential python3.12-dev
python3.12 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install --upgrade pip
.venv-vllm/bin/python -m pip install 'torch==2.14.0' --index-url https://download.pytorch.org/whl/cu130
.venv-vllm/bin/python -m pip install -e '.[vllm-worker]'
.venv-vllm/bin/python -m pip check
.venv-vllm/bin/python -m teutonic.evaluator.vllm_score --identity
```

Triton compiles a CUDA helper during engine initialization and needs the development
headers matching the worker's Python interpreter (`Python.h`). Test environments
may supply private headers through `CPATH` or reuse compiled helpers; neither proves
that a fresh production environment can compile. Check with an empty cache before
accepting submissions; this does not load a model:

```bash
.venv-vllm/bin/python - <<'PY'
import os
import subprocess
import sys
import tempfile

with tempfile.TemporaryDirectory(prefix="teutonic-triton-preflight-") as cache:
    subprocess.run(
        [sys.executable, "-c",
         "from triton.backends.nvidia.driver import CudaUtils; CudaUtils(); print('Triton compilation OK')"],
        env={**os.environ, "TRITON_CACHE_DIR": cache},
        check=True,
        timeout=90,
    )
PY
```

The extra installs the `teutonic_mimo` vLLM plugin for shared experts; do not use
the old test-only plugin pointing at `scripts.mimo_vllm_adapter`. Locked checkpoint
configs/tokenizers remain unchanged. Runtime preflight rejects an incorrect plugin
entry point or dependency version. Save the resolved package inventory with the
release. The tested B300 driver is
620.70.06. The integrated four-king/four-challenger TP1 layout passed the 7M MIX06
full-flow test; that run did not exercise the long-document TP2 fallback.

MiMo workers enforce `VLLM_BATCH_INVARIANT=1`, `VLLM_USE_V2_MODEL_RUNNER=0`, eager
execution, seed 42, BF16, no prefix caching and native prompt-token scoring.
MiMo additionally uses `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0` and
`TRITON_ATTN_DIFFKV`. These settings are fixed in the scorer, not optional service
flags. No production tokenizer-parity or repeatability reruns are performed.

Horizon uses a separate **layer-pipeline worker**, derived from
`artifacts/s3_models_doc_eval/horizon/eval_horizon_fast.py`, with Torch 2.15.0
CUDA 13.2, Transformers 5.15.1 and Accelerate 1.15.0. Provision a separate
`.venv-horizon` with that tested CUDA build, then install `.[horizon-worker]` there.
Do not combine it with `.[vllm-worker]`. Before starting the evaluator, check:

```bash
.venv-horizon/bin/python -m pip check
.venv-horizon/bin/python -m teutonic.evaluator.horizon_pipeline --identity
```

Horizon enforces `torch.use_deterministic_algorithms(True)`, seed 42,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `PYTORCH_ALLOC_CONF=expandable_segments:True`.
It runs BF16 grouped expert GEMMs and SDPA, computes FP32 token cross-entropy,
and sums each document with CPU `math.fsum`. Reference cache identities include
the Horizon worker's code, dependency versions and fixed scoring settings.
Existing vLLM reference entries are preserved but cannot be reused by this backend.
Its acceptance script is `scripts/verify_horizon_pipeline_worker.py`; comparisons
against saved losses run only in validation, never inside live evaluations.

Dataset reads require authenticated R2 access to **`dataset`**, under
`teutonic-text/`. Configure `TEUTONIC_EVALUATOR_R2_ENDPOINT`,
`TEUTONIC_EVALUATOR_R2_ACCESS_KEY_ID`, and `TEUTONIC_EVALUATOR_R2_SECRET_ACCESS_KEY`.
Optional session-token credentials are supported. The evaluator never reads the
old `datasets` bucket. Preserve that bucket and all existing objects unchanged.

For the historical initial 12 November release, the already-published artifacts and staged
[`cutover.sql`](deployment/packed-text-20261112/cutover.sql). It targets netuid 3,
generation `teutonic-II-110B-hf-8f971f030f5d801f812540007b396058f69cc08a`,
MAIN/MIX01/MIX06/MIX07, the expected previous 70M budget, and a new 35M budget with
threshold 0.00075 bits/byte. The SQL preserves category weights and source mappings.
The receipt in that directory records every remote key/hash. Local full-bundle
and audit copies are in `artifacts/packed-evaluation-20261112/`.

The live database preflight is still pending: the configured local endpoint was
unavailable during preparation. On the control host, confirm this release scope,
active configuration IDs, category weights/source mappings, current budgets,
early-stopping policies, crowned kings and reward state. Save the configuration
IDs/policies with the database backup for rollback. Run the read-only competition
retirement preview below. The migration and retirement guards remain mandatory;
passing the disposable-database rehearsal does not establish live compatibility.

For a future dataset release, the completed export manifest is the dataset
inventory. Prepare an exact JSON pin
with `bucket`, `key`, and `sha256`; do not use a changing bucket listing. Then prepare
local sampling and competition manifests, explicitly selecting active competitions:

```bash
.venv/bin/python scripts/prepare_competition_mixes.py \
  --dataset-pin /path/to/completed-dataset-pin.json \
  --competition main --competition mix01 --competition mix06 --competition mix07 \
  --output-dir artifacts/packed-evaluation-bundle
.venv/bin/python scripts/configure_evaluation.py \
  --bundle-dir artifacts/packed-evaluation-bundle \
  --netuid 3 --chain-generation YOUR_GENERATION --delta-bpb 0.00075 \
  --previous-token-budget 70000000 \
  --output artifacts/packed-evaluation-cutover.sql
```

Both commands write local files only. The first reads dataset metadata/ranges;
the second has no database connection. Review `pending-publication.json` and the
SQL before the separately approved publication/cutover. SQL guards preserve active
category weights, source mappings, and competition identities, and explicitly
guard the historical budget change from 70M to 35M. Repository defaults now select
60,000,000 original MiMo input tokens; use the new budget-only bundle above to update
the live 35M configuration. The old
`n=0` and `long_documents` database columns are compatibility storage, not a legacy
runtime path. `chain.toml` category definitions supply initialization data; the
active PostgreSQL configuration is authoritative and SQL rejects mismatches.

Validate the bundle against R2 without uploading:

```bash
.venv/bin/python scripts/publish_packed_evaluation.py \
  --bundle-dir artifacts/packed-evaluation-bundle --dry-run
```

**Publication is enabled by default.** After approval, run the same command
without `--dry-run` (or with the explicit, optional `--publish`). The publisher
validates all manifests first, publishes the sampling catalog before the mixes,
and verifies immutable content. It refuses to overwrite a different object.
Local `validation.json` and `publication.json` distinguish a read-only check from
verified publication. Neither mode changes PostgreSQL.

After GPU acceptance and explicit deployment approval:

1. Drain active evaluations and pending retries. Stop access, validator, promotion,
   weight-publisher and dashboard-publisher services; take and verify the DB backup.
2. Save active configuration IDs and early-stopping settings. Review the read-only
   retirement preview and verify the release's published hashes.
3. With `psql -v ON_ERROR_STOP=1`, apply `scripts/db/allow_packed_text_manifests.sql`,
   then `deployment/packed-text-20261112/cutover.sql`. These files are prepared,
   **not applied**. No new R2 publication is needed for this release.
4. Apply the reviewed retirement command below, saving its before-state snapshot.
   Keep services stopped if either configuration or retirement validation fails.
5. Deploy matching source/version identities on both hosts, install the patched
   Gigatoken wheel and the separate MiMo/Horizon runtimes, and verify evaluator
   health. Deploy the website assets from the same release.
6. Restart the validator for finalized UID reconciliation, then the weight
   publisher, remaining control services and dashboard publisher. Verify the four
   active competitions, 35M budgets, and MAIN 40% / MIX01, MIX06, MIX07 20% each.

Do not proceed past a failed step. Each database operation is separately guarded;
the complete multi-service cutover is not one atomic transaction. Do not run the historical commands under
`scripts/legacy/` against the migrated configuration. Existing R2 objects, database
history, kings, and reward positions remain intact.
Resolve pending evaluation retries as well: the staged transaction refuses claimed,
evaluating, or retryable-failure evaluations. It disables early stopping for each
migrated competition and retains the previous configuration rows.

Before cutover, save each competition's active configuration ID and early-stopping
policy, and retain the previous code/environment. Before accepting new work,
rollback can restore those exact IDs and policies with services stopped, together
with the matching previous evaluator and validator release. After new evaluations
or promotions, review their effects before rollback; never erase history or undo
kings automatically. A synthetic PostgreSQL rehearsal covers migration,
idempotence, guard failures, and restoring the old configuration; the final live
configuration snapshot still needs review.

The rollout retires MIX02, MIX03 and MIX05 in addition to already retired MIX04.
MAIN receives 40%; MIX01, MIX06 and MIX07 receive 20% each. Definitions, categories,
models and historical results remain available. Category sampling weights do not change.

Prepare the reward/activation cutover using the control-host database environment:

```bash
.venv/bin/python scripts/retire_mix_competitions.py \
  --output artifacts/four-competition-preview.json
```

This command defaults to a read-only transaction. After approval, stop and drain
access, validator, promotion and weight-publisher services, then rerun with
`--apply --output artifacts/four-competition-before-cutover.json`. The output path
must be new; it saves previous competition/publication rows before any mutations.
The transaction requires crowned MAIN/MIX01/MIX06/MIX07 recipients, no remaining
legacy reward positions, matching durable policy and no outstanding weight attempt.
It deactivates MIX02/03/05 and updates both the versioned reward policy and current
publication payload atomically. Existing weight-attempt history stays intact.
It does not upload to R2 or submit weights to chain. Repeating it is idempotent.

Deploy the new policy code before restarting services. Restart the validator first
and verify its finalized-metagraph UID reconciliation before restarting the weight
publisher. The existing 101-block cadence remains. The dashboard publication then
reflects active flags and the updated allocation; a hotkey holding multiple active
positions receives their combined shares. Retain the saved snapshot for reviewed
rollback with services stopped; do not use historical MIX activation commands below.

The sampling catalog supplies eligible row intervals in each length-sorted index
partition. Selection is uniform over documents across unequal partitions and
reproducible from the pinned manifests and seed. Use the selected 60M original-MiMo-token
budget, preserving category weights and the 40/60 split between lengths 2–4096 and
4097–16384 (approximately 14M and 21M tokens).
Fetch selected UTF-8 byte ranges and verify their byte counts/hashes. Category index
reads and text fetches are concurrent across objects. Reads to the same object are
serialized through body consumption, with bounded retry/backoff for throttling and
transient failures. Authentication and ETag/hash/range failures fail the evaluation.
Bounded preparation queues overlap inference.
A temporary disk cache reuses text between the reference and duel phases.

Tokenize raw text with no chat template or BOS and append the model's native EOS.
Score targets after the first token, including EOS. Original MiMo counts govern
sampling only; actual context lengths are checked separately. Never truncate or
replace an over-limit document. Each document remains isolated even if its text
contains an embedded special token. Canonical UTF-8 bytes are the bits/byte denominator.

Checkpoint downloads overlap dataset preparation and Horizon scoring. The evaluator
starts a CPU/network-only background task for the king and challenger R2 snapshots
(king then challenger; each transfer retains its existing file concurrency).
Horizon uses its own tokenizer and the pinned text sample, so it does not wait for
MiMo weights. MiMo GPU loading still waits for both verified snapshots, all existing
config/tokenizer/duplicate/reuse checks, and Horizon GPU cleanup.

Download failures are checked between reference batches. On a reference failure,
the evaluator joins the in-flight download before releasing cache locks or running
cleanup; it skips starting the next role's download. Completed reference scores
remain reusable under their existing cache identity. Sampling, tokenization, loss
calculation and the reward threshold are unchanged.

Result dataset metadata reports `model_download_seconds` (the overlapping task),
`model_download_wait_seconds` (download time still remaining after Horizon),
`model_validation_seconds`, and `model_preparation_wait_seconds` (the full gate
before MiMo). These overlap with `reference_phase_seconds` and must not all be
summed as sequential phases. Progress events include per-role `model_downloads`
states. Hardware throughput and total wall-time savings require a GPU acceptance
run; local tests verify ordering, cached-reference behavior and failure cleanup.

Reference cache misses use the deterministic Horizon layer pipeline across eight
GPUs, with layer counts `[8,8,8,8,8,9,9,3]`. Its micro-batches have a 131,072 padded
token budget and at most 128 documents; queues hold at most two micro-batches.
Requests contain at most 1,024 documents and 4,194,304 native tokens. The text
prefetch bound can produce smaller requests. Documents are stably sorted by
length within each request, scored with full causal context, and returned in
sampling order. A long document remains intact even above the micro-batch budget;
supported production lengths still require GPU memory qualification. No KV cache
is allocated for generation. After that process
group exits and memory is released, king and challenger run as four independent single-GPU replicas each,
on GPUs 0–3 and 4–7, with 16 GiB KV cache per GPU. A deterministic token-balanced
dispatcher sends identical subsets to matching king/challenger replicas. Documents
above the 128,974-token single-GPU boundary (subject to the new vLLM GPU acceptance checks) use two TP2 groups per side,
with 32 GiB KV per GPU and a context sized for the complete document. Ordinary
documents always use TP1. Switching layouts unloads the previous workers first.
This is tensor-parallel long-document scoring, replacing the old FA4 sequence-parallel
mechanism; GPU memory/determinism validation is required for each supported length.
Each phase streams bounded batches of up to 128 documents while preparing the next
batches. Both MiMo engines receive identical token IDs. Native EOS is included in
the scored targets; the ignored one-token generation is not part of NLL.

Every completed evaluation saves aligned per-document arrays in
`verdict.sample_results` in the evaluator JSON record and in the validator database's
`control_plane.evaluations.verdict_summary.sample_results`. The `documents` array
contains document IDs, source locations, text hashes and UTF-8 byte lengths; array
index `i` refers to that same document in every score column. `reference` means
Horizon, identified by `reference_model.repository` and its pinned revision.

| Per-sample fields | Meaning |
| --- | --- |
| `king_nll`, `challenger_nll`, `reference_nll` | Summed negative log-likelihood in nats |
| `king_token_loss`, `challenger_token_loss`, `reference_token_loss` | NLL divided by that model's scored-token count, in nats/token |
| `king_bits_per_byte`, `challenger_bits_per_byte`, `reference_bits_per_byte` | NLL divided by `ln(2) * documents[i].byte_length` |
| `scored_tokens` | Shared MiMo scored-token count for king and challenger |
| `reference_scored_tokens` | Horizon's own scored-token count, also recovered on reference-cache hits |

These diagnostics use the existing inference results and do not change scoring or
require a database migration. Existing NLL fields remain compatible with category
aggregation. Older records are unchanged; failed/incomplete evaluations do not
produce a complete sample-results record. Raw document text is not stored here.

The reward and bootstrap exclude a document when both king and challenger losses
are at or below Horizon's loss. Their two capped deficits are then both zero.
The reward denominator is the total UTF-8 bytes of the remaining documents, and
bootstrap draws come only from that population. Equal positive deficits remain
included, as do regressions. If no documents qualify, mean reward and LCB are zero
and the king stays. The threshold remains 0.00075 bits/byte.

All selected documents still receive inference and remain saved. The verdict
records `n_scoring_sequences`, `n_skipped_sequences`, `scoring_bytes` and
`skipped_bytes`; `sample_results.scoring_included` marks each document's eligibility.
Raw average losses and historical token-loss charts continue to cover the whole
sample. New category rewards use each category's eligible-byte denominator, while
old records without the new policy marker retain their original denominator.

During evaluation, the progress bar measures two document-scoring stages: Horizon
occupies 0–50%, and paired king/challenger scoring occupies 50–100%. This is work
completion, not a time estimate. The displayed document counter belongs to the
current stage. Cached Horizon results count immediately; identical-text documents
count individually even when one reference inference serves multiple samples.
Downloads, model loading and validation do not advance scoring progress.

At the first completed batch crossing each 10% checkpoint of paired scoring, the
evaluator refreshes provisional `mu_hat` and LCB using the same byte-weighted
Horizon-deficit bootstrap, seed, alpha and bootstrap count as the final verdict.
Only fully paired documents enter that checkpoint. The last LCB remains visible
through worker events and heartbeats, and the 100% checkpoint becomes the final
verdict. Partial checkpoints are informational and never trigger early stopping
or promotion. Progress uses the existing PostgreSQL `progress_summary` fields
and dashboard views; no SQL migration is needed. Horizon-only scoring cannot
produce a paired LCB, so the website waits for the first paired checkpoint.

Cache reference losses by checkpoint, tokenizer, text, policy, scorer/adapter code,
inference settings and dependency versions. Older Transformers cache entries cannot
match the vLLM identity. A fully cached sample skips the reference GPU phase.
MiMo engines currently reload for each duel. Workers have bounded loading/scoring
timeouts and owned process-group cleanup; GPU memory must be released before the
next phase. Failed-worker logs are retained beneath `TEUTONIC_HORIZON_CACHE_DIR` /
`failed-workers/`; temporary corpus text is removed on success or failure.

Reward is byte-weighted reduction in `max(model_bpb - horizon_bpb, 0)`.
Promotion requires the paired document-bootstrap lower confidence bound to exceed
an explicitly chosen threshold in **bits/byte**. There are no raw-loss retention
vetoes and no early stopping. A regression that remains better than Horizon is neutral.
The old numeric 0.0015 nats/token threshold is not an equivalent calibration.

The integrated real-pair 7M test passed on **8×B300** with all 25 MIX06 categories.
Final Horizon padding repeatability also passed. The prepared next budget is 60M;
the active configurations read on 13 November remain at 35M until the approved
budget-only cutover. A full 60M runtime has not been measured for this update.
The replay command reads R2/Hugging Face and
writes local results; it never promotes a model or changes R2:

```bash
.venv/bin/python scripts/replay_cached_evaluation.py \
  --mix artifacts/packed-evaluation-bundle/main.json \
  --king-snapshot /path/to/verified/king \
  --challenger-snapshot /path/to/verified/king \
  --self-duel --token-budget 100000 \
  --output artifacts/packed-self-duel.json
```

The mix's sampling object must already be published and hash-verified for that replay.
Before the completed dataset is published, model-only GPU checks can run against
local snapshots without any R2 access:

```bash
.venv/bin/python scripts/verify_horizon_gpu.py \
  --snapshot /path/to/verified/horizon --report artifacts/horizon-gpu.json \
  --throughput-lengths 512 4096
.venv/bin/python scripts/verify_mimo_gpu.py \
  --snapshot /path/to/verified/king --report artifacts/mimo-self-duel.json
```

The Horizon check compares the production chunked NLL function with a full
forward pass. The MiMo check loads all eight replicas and compares identical
checkpoints. Their inputs are synthetic, so their timings are not a production
corpus benchmark. Neither script publishes data or promotes a model.

While the full UTF-8 export is still running, exercise the complete two-phase
orchestrator with a local packed-text fixture and real models:

```bash
TEUTONIC_ATTN_IMPLEMENTATION=flash_attention_4 \
  .venv/bin/python scripts/verify_packed_duel_gpu.py \
  --mimo-snapshot /path/to/verified/king \
  --horizon-snapshot /path/to/verified/horizon \
  --output-dir artifacts/packed-fixture-gpu
```

Use a new output directory for each run. This script replaces only dataset
transport with local byte-range reads and uses the existing verified Horizon
snapshot directly. It makes no R2 requests or uploads. It runs real Horizon and
MiMo scoring, then repeats the same sample to verify reference-cache reuse,
identical scores, zero self-duel reward, text-cache cleanup, and streamed batches.
This validates orchestration before export completion; it does not replace the
later authenticated R2 replay or real-corpus performance measurement.

Run with legacy `datasets` access denied. Verify K2 summed NLL against a trusted forward,
self-duel zero reward, GPU memory, phase cleanup, deterministic sample replay, and
cold/warm reference-cache behavior. Benchmark loading, tokenizer throughput, inference,
and input stalls separately; historical Teutonic-only times do not predict K2 latency.
Inspect `/health.horizon_download` for checkpoint preparation progress and failures.

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
pm2 start ecosystem.config.js --only teutonic-dashboard-live
```

The live publisher uses the same database and dashboard R2 credentials as the
history publisher. **No SQL migration or new database grants are required.** It
reads the existing sanitized queue/progress views and writes only
`dashboard-live.json`, with a five-second cache lifetime. It has a separate
advisory lock and process; historical queries and market APIs cannot block it.
Individual SQL statements time out after three seconds, and projections taking
more than five seconds are discarded. Failed refreshes retain the last good
object and its original timestamp.

For an existing deployment, start `teutonic-dashboard-live` first and verify that
`dashboard-live.json` refreshes and contains the expected competitions and queue.
Then publish `website/app.js`, `website/dashboard-v1.js`, and `website/index.html`
together. No evaluator/validator restart is needed. The website polls live status
every five seconds, marks snapshots older than 30 seconds as last-known data,
and keeps history/charts on the existing dashboard snapshots. Until its first
live snapshot arrives it falls back to the old snapshots; after that, queue and
progress use the single global live snapshot to avoid mixing duel transitions.

Check both `generated_at` and the publisher's `projection_and_upload_seconds` log.
The expected visible delay is seconds, with a 15–30 second rollout acceptance
target under normal load; measure this on the host rather than assuming the
five-second cadence guarantees it. Historical pages can still refresh slowly;
this change removes them from the live status path without changing their SQL.
Rollback by restoring the three website assets together and stopping only
`teutonic-dashboard-live`.

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
pm2 logs teutonic-dashboard-live --lines 100
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
  teutonic-access-controller,teutonic-validator,teutonic-promotion-worker,teutonic-weight-publisher,teutonic-dashboard-view,teutonic-dashboard-live
```

Verify it on the validator's configured evaluator port:

```bash
curl --fail http://127.0.0.1:9000/health
pm2 logs teutonic-mock-evaluator --lines 100
```

Do not add the mock evaluator to either production ecosystem file.

## MAIN and MIX category competitions

The rollout commands below are historical. For the current evaluator use the packed-text
section above; legacy token-inventory tools were moved to `scripts/legacy/`.


PostgreSQL remains the runtime source of truth. `chain.toml` defines the category
registry and seven explicit category-weight tables under `[competitions.*]`.
The active configuration and initialization defaults use **zero windows and a
70,000,000 whole-document input-token budget**. The current document policy caps
lengths at 16,384 tokens and uses the 40/60 token split. Runtime budgets are set
through `configure_evaluation.py`; repository defaults apply to initialization or
explicit definition replacement. Percentages are sampling targets. The verdict
remains the paired, scored-token-weighted bootstrap over the whole mix, with
**LCB strictly greater than the threshold**. There is no additional
largest-source or category gate.

| Competition | Category allocation | Final incentive | LCB threshold |
|---|---|---:|---:|
| MAIN | Original 24 categories, proportional to token inventory | 25% | 0.0015 |
| MIX01 | Former MATH categories 45%, former CODE categories 45%, DCLM 10% | 12.5% | 0.0015 |
| MIX02 | Former CODE category set 50%; remaining PretBeh categories 50%; token-proportional inside both sets | 12.5% | 0.0015 |
| MIX03 | Former MATH categories 60%, former CODE categories 40% | 12.5% | 0.0015 |
| MIX04 (retired at MIX07 activation) | Historical TXT360-V2 70%, PretBeh 30% | Retains 12.5% until MIX07's first crown | — |
| MIX05 | DCLM 40%, PretBeh categories 60%; token-proportional inside PretBeh | 12.5% | 0.0015 |
| MIX06 | All 25 categories including DCLM, proportional to token inventory | 12.5% | 0.0015 |
| MIX07 | 100% `dclm-baseline-1.0` | Takes MIX04's 12.5% on first crown | 0.0015 |

### Replace MIX04 with MIX07

MIX07 starts from **MIX05's current king at cutover**, pinned once in
`starting_reign_id`. Later MIX05 promotions and configuration reruns do not move
that starting point. MIX07 initially has no king or reward recipient of its own.
MIX04 becomes inactive for submissions/evaluations but keeps its existing king,
history and 12.5% reward slot. The first successful MIX07 crown transfers only that
slot to its winner, atomically with the normal reward publication. MAIN, MIX05 and
all other reward positions are unchanged. A failed evaluation or promotion does
not transfer the slot. This also works before the older legacy roster is exhausted:
MIX07 replaces MIX04, rather than evicting another legacy recipient.

Prepare and publish only the new DCLM mix before downtime, using the dataset
publication credentials. Existing manifests are not rewritten:

```bash
.venv/bin/python scripts/prepare_competition_mixes.py \
  --competition MIX07 --output-dir artifacts/mix07
.venv/bin/python scripts/publish_category_artifacts.py \
  --bundle-dir artifacts/mix07 --bucket datasets
```

The new immutable URL/hash lives in `artifacts/mix07/bundle.json`; MIX07 deliberately
has no invented published URL in `chain.toml`. Keep the bundle for activation.
The cutover reuses MIX05's pinned document index and validates the new manifest's
DCLM source against it. If the source manifest has changed, prepare from the pinned
source snapshot with `--source-dir` instead of repinning evaluation data implicitly.

Pause intake and drain evaluations, retries and promotions. Stop the access
controller only after outstanding verification/snapshot jobs finish; then let the
remaining evaluations/promotions drain. Stop these services before cutover:

```bash
pm2 stop teutonic-access-controller teutonic-validator teutonic-promotion-worker teutonic-weight-publisher
```

Deploy the updated control-plane code and, for the 16k document policy, the GPU
evaluator code. Apply `allow_document_only_evaluation.sql` if not already applied.
Use the loaded deployment environment with database-owner credentials:

```bash
.venv/bin/python scripts/configure_evaluation.py \
  --competition MIX07 --initialize --replace-mix04 \
  --bundle-dir artifacts/mix07 --n 0 --long-document-tokens 70000000 --dry-run
.venv/bin/python scripts/configure_evaluation.py \
  --competition MIX07 --initialize --replace-mix04 \
  --bundle-dir artifacts/mix07 --n 0 --long-document-tokens 70000000
```

The dry run prints the pinned MIX05 reign and performs no writes. Both runs require
the services stopped and all pending work drained. The script requires a rewarded
MIX04 king and an actual current MIX05 king; it never substitutes MIX05's starting
baseline. Config creation, MIX04 retirement, MIX07 activation and reward-transition
metadata commit together. Rerunning preserves the original pin and reward state,
including after MIX07 has crowned a king. Cutover changes no published weights.

After MIX07 exists, the earlier `--all --n 0 --long-document-tokens 70000000`
command updates the seven active definitions (MAIN, MIX01–03, MIX05–07), preserving
their pinned indexes and thresholds. MIX04's old configuration stays archived.
Do not rerun the historical `--cutover`: it bootstraps MIX01–06 from legacy
competitions and cannot replace an initialized reward transition.

Check GPU `/health` and matching deployment versions, then restart the control
services and dashboard publisher:

```bash
pm2 restart ecosystem.config.js --only teutonic-access-controller,teutonic-validator,teutonic-promotion-worker,teutonic-weight-publisher,teutonic-dashboard-view --update-env
```

Verify `competitions.json` shows MIX04 inactive, MIX07 active with the printed
starting reign, zero windows and a 70M document budget. MIX04 must still receive
its reward share until MIX07's first crown. Deploy the updated website assets so
the baseline label, retired reward recipient and MIX07 benchmarks display correctly.
Deploy the updated Cloudflare proxy with `scripts/cloudflare/deploy.sh` so MIX07's
benchmark routes resolve to the benchmark bucket.

The former MATH/CODE labels above explain the migration only. Published manifests
contain unique flat category lists, original dataset identities, and source hashes.
MIX01/MIX03 retain the former groups' internal weights. MIX02's CODE set includes
`reasoning`, `other`, and `data-science` once; the other half contains
`format-rewrites`, `games`, `general`, and `planning`. No overlapping weight is added,
and MIX02 has no special sampling path. DCLM is the single category `dclm-baseline-1.0`.

### Kings, commitments, and replacement rewards

Keep `TEUTONIC_COMPETITION` set to the existing MAIN database name on every control
plane service. One validator, promotion worker, and weight publisher serve all seven.
MIXes start without kings. Original MIX01–06 challenge a **fixed starting MAIN king**
using that MIX's categories and threshold. `starting_reign_id` is pinned once at
initialization; later MAIN promotions do not change claims, retries, or coronation
checks for that MIX. MIX07 instead pins MIX05's king as described above.
After its first win, a MIX uses its own current king.
Configuration refreshes preserve the pin. `competitions.json` includes a
`starting_king` object with the pinned reign ID, model digest, public model reference,
name, UID, and hotkey; a starting king does not count as a MIX king or receive MIX rewards.

For an existing deployment that follows the latest MAIN, use the following update.
Let active evaluations finish and their promotions commit, then stop the validator
and promotion worker before pulling the new code. This prevents a MAIN promotion
between selecting the fixed starting point and switching the workers to the new rule.

```bash
pm2 stop teutonic-validator teutonic-promotion-worker
git pull --ff-only
docker compose exec -T postgres sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-psqlrc -v ON_ERROR_STOP=1' \
  < scripts/db/pin_mix_starting_kings.sql
pm2 restart ecosystem.config.js --only teutonic-validator,teutonic-promotion-worker,teutonic-dashboard-view --update-env
```

Only restart after the migration reports `COMMIT`. The migration prints the pins
for active kingless MIXes and captures the MAIN king **at migration time**. Existing
MIX kings and reward state are unchanged. Re-running the migration does not move
existing pins. If the migration times out waiting for a lock, retry it before
restarting. No new environment variables or evaluator restart are required for
this baseline-selection change. Publish the updated `website/llms.txt` with the
normal website documentation deployment so miners see the new rule.

The cutover snapshots all currently rewarded positions and their durable shares;
cutover itself changes no payouts. Legacy recipients are ordered by crowned block
(oldest first; bootstrap recipients without reigns come first). Each **first** original MIX01–06
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
release enables MIX01–03 and MIX05–07 after replacement. New ready signals must target a configured, active
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
    --token-budget 70000000 --index-manifest-url "$INDEX_URL" \
    --index-manifest-file artifacts/category-mixes/document-index.json \
    --index-manifest-sha256 "$INDEX_SHA" --plan-context-limit 16384 \
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
psql "$TEUTONIC_DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/allow_document_only_evaluation.sql
psql "$TEUTONIC_DATABASE_URL" -v ON_ERROR_STOP=1 -f scripts/db/pin_mix_starting_kings.sql
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
  --bundle-dir artifacts/category-mixes --n 0 --long-document-tokens 70000000 \
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
`competitions/mixNN/datasets/manifest.json`: seven active competitions, zero windows
and 70M document-token budgets, correct thresholds, all MIXes kingless, and
unchanged initial weights.
The page shows the six-dataset inventory (including DCLM), category percentages,
king/fallback state, and retired history. The reward transition stays in the
backend and is not displayed as a separate section. Upload `dashboard-v1.js`,
`app.js`, `style.css`, and the updated
`llms.txt` before `index.html`; the HTML uses
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
.venv/bin/python scripts/configure_evaluation.py --all --n 0 --long-document-tokens 70000000 --dry-run
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


## CPU release verification

The test image runs pytest for both unittest classes and pytest functions; the
unit and integration suites run in separate processes to avoid same-name module
collisions. The database in `tests/compose.yaml` is disposable and uses tmpfs.
Do not point these tests at a production database.

```bash
docker compose -f tests/compose.yaml build control-plane-tests
docker compose -f tests/compose.yaml up --abort-on-container-exit --exit-code-from control-plane-tests postgres control-plane-tests
node --test tests/contracts/*.test.js
docker compose -f tests/compose.yaml down
```

The packed-cutover integration test rehearses the actual staged release SQL
and published mix bytes, checks 70M-to-35M migration, preserves old configurations,
reapplies idempotently, restores the old IDs/policies, and verifies that a mismatch
in the last competition rolls back preceding changes. Retirement tests separately
check the 40/20/20/20 policy and retained history. No GPU inference is involved.
