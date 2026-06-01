# King Benchmark Service

Authenticated two-process flow for evaluating every Teutonic king without putting Hippius S3/Doppler secrets on the GPU host.

## Shape

- Controller runs locally under PM2, fetches `dashboard.json`, chooses the newest king first, then older unevaluated kings sequentially.
- Worker runs on the configured remote benchmark host (`$TEUTONIC_KING_BENCH_WORKER_SSH`, HTTP worker URL from `$TEUTONIC_KING_BENCH_WORKER_URL`) and only receives `Authorization: Bearer <token>` jobs.
- Worker posts progress/results back to the controller. Controller writes local JSON and uploads S3 JSON.
- Default benchmarks are `MMLU,MMLU-Pro,BBH,ARC-C,TruthfulQA,WinoGrande`.

## S3 JSON

The all-kings service intentionally writes under a separate prefix so it does not overwrite the existing dashboard-six files:

- Latest state: `s3://teutonic-sn3/king-benchmark-daily/all-kings/latest.json`
- All kings index: `s3://teutonic-sn3/king-benchmark-daily/all-kings/index.json`
- Append-only history: `s3://teutonic-sn3/king-benchmark-daily/all-kings/history.jsonl`
- Per-king result: `s3://teutonic-sn3/king-benchmark-daily/kings/<king_id>/results.json`

## GPU Use

The fixed H100 worker defaults to `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `TEUTONIC_KING_BENCH_DEVICE=auto`, and `TEUTONIC_KING_BENCH_MODEL_ARGS_EXTRA=device_map=auto`. This avoids forcing `lm-eval` onto `cuda:0` and lets the HuggingFace backend shard the model across all visible GPUs.

## Model Resolution

Before a king is labeled `missing`, the evaluator tries multiple sources in order:

- Hippius model-index API metadata.
- Hippius registry by pinned digest, even if the API detail endpoint is missing.
- Docker/OCI extraction from `registry.hippius.com/<repo>@sha256:...` when the Python Hippius downloader cannot fetch the artifact.
- Hugging Face fallback by exact model-name search, plus optional `TEUTONIC_KING_BENCH_HF_FALLBACK_REPOS` / `TEUTONIC_KING_BENCH_HF_FALLBACK_ORGS` overrides.

The HF owner is not hardcoded; for example, `shallowtensr/...` is discovered by exact model-name search when it exists.

## Caching

The worker keeps reusable caches under `/root/teutonic/cache` by default:

- HuggingFace model/dataset cache: `/root/teutonic/cache/huggingface`
- lm-eval preprocessed request cache: `/root/teutonic/cache/lm-harness/requests`
- Optional per-model response cache for interrupted reruns: `/root/teutonic/cache/lm-harness/responses`

Request caching is enabled with `TEUTONIC_KING_BENCH_CACHE_REQUESTS=true`. Do not share one global tokenized cache across different kings unless the tokenizer is identical; tokenizer-dependent artifacts must stay per model/revision.

## Start Later

Create one token and put it in ignored env files on both machines:

```bash
python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
```

Controller, on this machine:

```bash
export TEUTONIC_KING_BENCH_WORKER_TOKEN='...'
export TEUTONIC_KING_BENCH_CONTROLLER_TOKEN="$TEUTONIC_KING_BENCH_WORKER_TOKEN"
export TEUTONIC_KING_BENCH_WORKER_SSH='root@YOUR_WORKER_HOST'
export TEUTONIC_KING_BENCH_WORKER_URL='http://YOUR_WORKER_HOST:32000'
pm2 start scripts/king_benchmark_service/ecosystem.controller.config.js
```

Worker, on the remote benchmark host:

```bash
cd /root/teutonic/king-benchmark-worker
# Optional local-only secrets file, chmod 600, never committed:
# printf 'HF_TOKEN=...\n' > .worker.secrets.env
export TEUTONIC_KING_BENCH_WORKER_TOKEN='...'
export TEUTONIC_KING_BENCH_CONTROLLER_TOKEN="$TEUTONIC_KING_BENCH_WORKER_TOKEN"
export TEUTONIC_KING_BENCH_CONTROLLER_URL='http://YOUR_CONTROLLER_HOST_OR_TUNNEL:32100'
pm2 start ecosystem.worker.config.js
```

## Check Status

```bash
curl -H "Authorization: Bearer $TEUTONIC_KING_BENCH_WORKER_TOKEN" "$TEUTONIC_KING_BENCH_WORKER_URL/status"
curl -H "Authorization: Bearer $TEUTONIC_KING_BENCH_CONTROLLER_TOKEN" http://127.0.0.1:32100/queue
pm2 logs teutonic-king-bench-controller --lines 100
pm2 logs teutonic-king-bench-worker --lines 100
```

## Manual Dispatch

If auto-dispatch is disabled with `TEUTONIC_KING_BENCH_AUTO_DISPATCH=0`, dispatch exactly one next missing king with:

```bash
curl -X POST -H "Authorization: Bearer $TEUTONIC_KING_BENCH_CONTROLLER_TOKEN" http://127.0.0.1:32100/dispatch-next
```
