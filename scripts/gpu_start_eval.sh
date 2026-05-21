#!/usr/bin/env bash
# Start eval_server on the B200 GPU box (port 10099).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export TEUTONIC_SHARD_ACROSS_GPUS="${TEUTONIC_SHARD_ACROSS_GPUS:-1}"
export TEUTONIC_SHARD_PER_GPU_GIB="${TEUTONIC_SHARD_PER_GPU_GIB:-150}"
export TEUTONIC_PROBE_ENABLED="${TEUTONIC_PROBE_ENABLED:-0}"
export EVAL_N="${EVAL_N:-5000}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
export EVAL_BOOTSTRAP_B="${EVAL_BOOTSTRAP_B:-5000}"
export TEUTONIC_MODEL_CACHE_DIR="${TEUTONIC_MODEL_CACHE_DIR:-/tmp/teutonic/hippius_models}"
export TEUTONIC_EVAL_DATASET_MODE="${TEUTONIC_EVAL_DATASET_MODE:-raw_hippius}"
export TEUTONIC_RAW_DATASET_PREFIX="${TEUTONIC_RAW_DATASET_PREFIX:-hf-mirrors/HuggingFaceFW/fineweb-edu/data}"
export TEUTONIC_RAW_DATASET_MANIFEST="${TEUTONIC_RAW_DATASET_MANIFEST:-hf-mirrors/HuggingFaceFW/fineweb-edu/data/_manifest.json}"
export TEUTONIC_RAW_TOKENIZER_REPO="${TEUTONIC_RAW_TOKENIZER_REPO:-Qwen/Qwen3-30B-A3B}"
export TEUTONIC_DS_ENDPOINT="${TEUTONIC_DS_ENDPOINT:-https://s3.hippius.com}"
export TEUTONIC_DS_BUCKET="${TEUTONIC_DS_BUCKET:-teutonic-sn3}"

mkdir -p /workspace/logs
exec uvicorn eval_server:app --host 127.0.0.1 --port 9000 \
  2>&1 | tee -a /workspace/logs/eval-server.log
