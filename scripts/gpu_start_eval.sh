#!/usr/bin/env bash
# Start eval_server on the B200 GPU box (port 10099).
# Q3-4B chain: 8 GiB bf16 — per-GPU replicas (4 king + 4 challenger),
# no sharding needed. Batch size tuned for 180 GiB B200 headroom.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

export TEUTONIC_SHARD_ACROSS_GPUS="${TEUTONIC_SHARD_ACROSS_GPUS:-0}"
export TEUTONIC_PROBE_ENABLED="${TEUTONIC_PROBE_ENABLED:-1}"
export EVAL_N_CAP="${EVAL_N_CAP:-30000}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
export EVAL_BOOTSTRAP_B="${EVAL_BOOTSTRAP_B:-10000}"
export TEUTONIC_MODEL_CACHE_DIR="${TEUTONIC_MODEL_CACHE_DIR:-/tmp/teutonic/hippius_models}"
export TEUTONIC_EVAL_DATASET_MODE="${TEUTONIC_EVAL_DATASET_MODE:-raw_hippius}"
export TEUTONIC_RAW_DATASET_PREFIX="${TEUTONIC_RAW_DATASET_PREFIX:-hf-mirrors/HuggingFaceFW/fineweb-edu/data}"
export TEUTONIC_RAW_DATASET_MANIFEST="${TEUTONIC_RAW_DATASET_MANIFEST:-hf-mirrors/HuggingFaceFW/fineweb-edu/data/_manifest.json}"
export TEUTONIC_RAW_TOKENIZER_REPO="${TEUTONIC_RAW_TOKENIZER_REPO:-Qwen/Qwen3-4B}"
export TEUTONIC_DS_ENDPOINT="${TEUTONIC_DS_ENDPOINT:-https://s3.hippius.com}"
export TEUTONIC_DS_BUCKET="${TEUTONIC_DS_BUCKET:-teutonic-sn3}"

mkdir -p /workspace/logs
exec uvicorn eval_server:app --host 127.0.0.1 --port 9000 \
  2>&1 | tee -a /workspace/logs/eval-server.log
