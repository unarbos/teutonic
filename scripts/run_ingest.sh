#!/bin/bash
set -euo pipefail

export HF_TOKEN=$(doppler secrets get HF_TOKEN --plain -p arbos -c prd)
export TEUTONIC_DS_ENDPOINT="https://s3.hippius.com"
export TEUTONIC_DS_BUCKET="teutonic-sn3"
export TEUTONIC_DS_ACCESS_KEY=$(doppler secrets get HIPPIUS_ACCESS_KEY --plain -p arbos -c dev)
export TEUTONIC_DS_SECRET_KEY=$(doppler secrets get HIPPIUS_SECRET_KEY --plain -p arbos -c dev)

exec /home/const/workspace/.venv/bin/python \
  /home/const/workspace/teutonic/scripts/ingest_hf.py \
  --dataset uonlp/CulturaX \
  --langs en \
  --workers 16 \
  "$@"
