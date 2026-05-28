# Daily King Benchmark Automation

This service runs the default six-benchmark dashboard panel for the current Teutonic king on Lium. It fetches `dashboard.json`, builds a pinned `model_repo@sha256:...` input, rents one pod per benchmark, copies logs/results back, and writes stable dashboard JSON.

## Start

```bash
bash scripts/king_benchmark_daily/start.sh
```

For a safe no-rent check:

```bash
python3 scripts/king_benchmark_daily/runner.py --run-once --dry-run
```

Run one full benchmark through the automated rent/eval/delete path without publishing to S3:

```bash
python3 scripts/king_benchmark_daily/runner.py --run-once --benchmarks HumanEval --no-upload-hippius --state-file /tmp/king-bench-humaneval-state.json
```

## Default Benchmarks

Default daily benchmarks:

```text
MMLU,MMLU-Pro,BBH,ARC-C,TruthfulQA,WinoGrande
```

You can still override with `--benchmarks` or `TEUTONIC_KING_BENCH_BENCHMARKS` for one-off/full-panel runs.

## Outputs

Latest dashboard-ready JSON is sanitized, written locally, and uploaded to Hippius S3. It does not include local filesystem artifact paths.

```text
runs/king-benchmark-daily/latest.json
s3://teutonic-sn3/king-benchmark-daily/latest.json
https://s3.hippius.com/teutonic-sn3/king-benchmark-daily/latest.json
```

Private per-run JSON and artifacts for operators:

```text
runs/king-benchmark-daily/runs/<run_id>/run.json
runs/king-benchmark-daily/runs/<run_id>/benchmarks/<benchmark>/...
```

History and rental registry:

```text
runs/king-benchmark-daily/history.jsonl
s3://teutonic-sn3/king-benchmark-daily/history.jsonl
runs/lium-rentals/registry.json
```

## PM2 Status

```bash
pm2 status teutonic-king-bench-lium
pm2 logs teutonic-king-bench-lium --lines 120
cat runs/king-benchmark-daily/latest.json
curl -s https://s3.hippius.com/teutonic-sn3/king-benchmark-daily/latest.json
```

## Defaults

Runs daily at 22:00 GMT+2 by default. By default it deletes successful pods after artifact copyback and deletes failed pods after attempting to copy logs/artifacts, to avoid leaked GPU spend.

Key env overrides:

```bash
TEUTONIC_KING_BENCH_INTERVAL_HOURS=24
TEUTONIC_KING_BENCH_DAILY_TIME=22:00
TEUTONIC_KING_BENCH_SCHEDULE_UTC_OFFSET=+02:00
TEUTONIC_KING_BENCH_TTL=36h
TEUTONIC_KING_BENCH_BENCHMARKS=MMLU,MMLU-Pro,BBH,ARC-C,TruthfulQA,WinoGrande
TEUTONIC_KING_BENCH_RENT_TIMEOUT_S=180
TEUTONIC_KING_BENCH_KEEP_ON_SUCCESS=0
TEUTONIC_KING_BENCH_DELETE_ON_FAILURE=1
TEUTONIC_KING_BENCH_RESULTS_ROOT=/home/const/workspace/runs/king-benchmark-daily
TEUTONIC_KING_BENCH_UPLOAD_HIPPIUS=1
TEUTONIC_KING_BENCH_LATEST_KEY=king-benchmark-daily/latest.json
TEUTONIC_KING_BENCH_HISTORY_KEY=king-benchmark-daily/history.jsonl
```
