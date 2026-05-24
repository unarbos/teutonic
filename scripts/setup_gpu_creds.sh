#!/usr/bin/env bash
# Push eval-server credentials to the GPU box and pre-seed the LXXXI king cache.
set -euo pipefail

GPU_HOST="${GPU_HOST:-95.133.252.200}"
GPU_PORT="${GPU_PORT:-10099}"
GPU_USER="${GPU_USER:-root}"
SEED_DIGEST="${SEED_DIGEST:-sha256:0950c71f59e03211e0754d5cf484abd99c877b6c79f844457126a7f1fd1b69c8}"
SEED_SRC="${SEED_SRC:-/tmp/teutonic-lxxxi}"
CACHE_ROOT="${TEUTONIC_MODEL_CACHE_DIR:-/tmp/teutonic/hippius_models}"

SSH=(ssh -p "$GPU_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${GPU_USER}@${GPU_HOST}")

doppler_val() {
  doppler secrets get "$1" --plain -p arbos -c dev
}

HF_TOKEN="$(doppler secrets get HF_TOKEN --plain -p arbos -c prd 2>/dev/null || doppler secrets get HUGGINGFACE_API_KEY --plain -p arbos -c dev)"

CREDS_FILE="$(mktemp)"
trap 'rm -f "$CREDS_FILE"' EXIT
cat >"$CREDS_FILE" <<EOF
export HF_TOKEN='${HF_TOKEN}'
export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}'
export TEUTONIC_R2_ENDPOINT='$(doppler_val R2_URL)'
export TEUTONIC_R2_BUCKET='$(doppler_val R2_BUCKET_NAME)'
export TEUTONIC_R2_ACCESS_KEY='$(doppler_val R2_ACCESS_KEY_ID)'
export TEUTONIC_R2_SECRET_KEY='$(doppler_val R2_SECRET_ACCESS_KEY)'
export TEUTONIC_HIPPIUS_ACCESS_KEY='$(doppler_val HIPPIUS_ACCESS_KEY)'
export TEUTONIC_HIPPIUS_SECRET_KEY='$(doppler_val HIPPIUS_SECRET_KEY)'
export TEUTONIC_DS_ENDPOINT='https://s3.hippius.com'
export TEUTONIC_DS_BUCKET='teutonic-sn3'
export TEUTONIC_DS_ACCESS_KEY='$(doppler_val HIPPIUS_ACCESS_KEY)'
export TEUTONIC_DS_SECRET_KEY='$(doppler_val HIPPIUS_SECRET_KEY)'
export TEUTONIC_SHARD_ACROSS_GPUS='1'
export TEUTONIC_SHARD_PER_GPU_GIB='150'
export TEUTONIC_PROBE_ENABLED='0'
export EVAL_N='5000'
export EVAL_BATCH_SIZE='32'
export EVAL_BOOTSTRAP_B='5000'
export TEUTONIC_MODEL_CACHE_DIR='${CACHE_ROOT}'
export TEUTONIC_EVAL_DATASET_MODE='raw_hippius'
export TEUTONIC_RAW_DATASET_PREFIX='hf-mirrors/HuggingFaceFW/fineweb-edu/data'
export TEUTONIC_RAW_DATASET_MANIFEST='hf-mirrors/HuggingFaceFW/fineweb-edu/data/_manifest.json'
export TEUTONIC_RAW_TOKENIZER_REPO='Qwen/Qwen3-30B-A3B'
export TEUTONIC_RAW_DATASET_KEYS='hf-mirrors/HuggingFaceFW/fineweb-edu/data/CC-MAIN-2025-05/000_00018.parquet'
export HIPPIUS_PREFETCH_TIMEOUT='1800'
EOF

echo "=== installing creds on GPU box ==="
"${SSH[@]}" "mkdir -p /root/.creds"
scp -P "$GPU_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$CREDS_FILE" "${GPU_USER}@${GPU_HOST}:/root/.creds/teutonic_eval.env"

echo "=== pre-seeding king cache (${SEED_DIGEST}) ==="
DIGEST_KEY="${SEED_DIGEST/:/-}"
CACHE_DEST="${CACHE_ROOT}/unconst--Teutonic-LXXXI-mock-king/snapshots/${DIGEST_KEY}"
"${SSH[@]}" "mkdir -p '${CACHE_DEST}' && cp -a '${SEED_SRC}/.' '${CACHE_DEST}/'"

echo "=== done: creds + cache on ${GPU_HOST} ==="
