#!/usr/bin/env bash
set -euo pipefail

cd "${TEUTONIC_WORKER_ROOT:-/root/teutonic/king-benchmark-worker}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/teutonic/cache/uv}"

if ! command -v uv >/dev/null 2>&1; then
  if command -v pipx >/dev/null 2>&1; then
    pipx install uv || pipx upgrade uv
  else
    python3 -m pip install --user --break-system-packages --upgrade uv
  fi
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --clear --python 3.12 .venv
. .venv/bin/activate
uv pip install --upgrade pip wheel setuptools
for attempt in 1 2 3 4 5; do
  if uv pip install --upgrade "lm-eval[api]" accelerate datasets evaluate hf_transfer huggingface_hub hippius-hub httpx nvidia-ml-py sentencepiece protobuf transformers; then
    break
  fi
  if [ "$attempt" = "5" ]; then
    exit 1
  fi
  sleep $((attempt * 15))
done

if python - <<'INNERPY'
try:
    import hippius_hub, httpx, lm_eval, torch
    print(f"bootstrap imports ok; torch={torch.__version__}")
except Exception:
    raise SystemExit(1)
INNERPY
then
  :
else
  for attempt in 1 2 3 4 5; do
    if uv pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124; then
      break
    fi
    if [ "$attempt" = "5" ]; then
      exit 1
    fi
    sleep $((attempt * 15))
  done
fi
