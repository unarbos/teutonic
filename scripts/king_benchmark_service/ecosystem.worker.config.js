// PM2 worker for the fixed GPU host. This host only receives a scoped API token.
// It does not need Hippius S3 or Doppler credentials.

const fs = require("fs");
const path = require("path");
const workerRoot = process.env.TEUTONIC_WORKER_ROOT || "/root/teutonic/king-benchmark-worker";
const cacheRoot = process.env.TEUTONIC_KING_BENCH_CACHE_ROOT || "/root/teutonic/cache";
const venvPython = path.join(workerRoot, ".venv", "bin", "python");
const accelerateBin = path.join(workerRoot, ".venv", "bin", "accelerate");

function envOr(key, fallback) {
  return process.env[key] != null && process.env[key] !== ""
    ? process.env[key]
    : fallback;
}

function loadLocalSecrets(filePath) {
  try {
    const raw = fs.readFileSync(filePath, "utf8");
    for (const line of raw.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const idx = trimmed.indexOf("=");
      if (idx <= 0) continue;
      const key = trimmed.slice(0, idx).trim();
      const value = trimmed.slice(idx + 1).trim();
      if (process.env[key] == null || process.env[key] === "") {
        process.env[key] = value;
      }
    }
  } catch (_) {
    // Optional local-only secrets file. Do not commit it.
  }
}

loadLocalSecrets(path.join(workerRoot, ".worker.secrets.env"));

module.exports = {
  apps: [{
    name: "teutonic-king-bench-worker",
    script: path.join(workerRoot, "worker.py"),
    interpreter: envOr("TEUTONIC_KING_BENCH_WORKER_PYTHON", venvPython),
    cwd: workerRoot,
    autorestart: true,
    restart_delay: 30000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    out_file: path.join(workerRoot, "logs/worker.out.log"),
    error_file: path.join(workerRoot, "logs/worker.err.log"),
    env: {
      PYTHONUNBUFFERED: "1",
      TOKENIZERS_PARALLELISM: "false",
      CUDA_VISIBLE_DEVICES: envOr("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"),
      HF_HUB_ENABLE_HF_TRANSFER: "1",
      HF_TOKEN: envOr("HF_TOKEN", ""),
      HUGGING_FACE_HUB_TOKEN: envOr("HUGGING_FACE_HUB_TOKEN", envOr("HF_TOKEN", "")),
      HF_XET_HIGH_PERFORMANCE: envOr("HF_XET_HIGH_PERFORMANCE", "1"),
      LD_LIBRARY_PATH: envOr(
        "LD_LIBRARY_PATH",
        [
          path.join(workerRoot, ".venv", "lib", "python3.12", "site-packages", "nvidia", "cuda_runtime", "lib"),
          path.join(workerRoot, ".venv", "lib", "python3.12", "site-packages", "nvidia", "cublas", "lib"),
          path.join(workerRoot, ".venv", "lib", "python3.12", "site-packages", "nvidia", "cudnn", "lib"),
          path.join(workerRoot, ".venv", "lib", "python3.12", "site-packages", "nvidia", "cu13", "lib"),
        ].join(":"),
      ),
      XDG_CACHE_HOME: envOr("XDG_CACHE_HOME", path.join(cacheRoot, "xdg")),
      HF_HOME: envOr("HF_HOME", path.join(cacheRoot, "huggingface")),
      HF_DATASETS_CACHE: envOr("HF_DATASETS_CACHE", path.join(cacheRoot, "huggingface", "datasets")),
      TRANSFORMERS_CACHE: envOr("TRANSFORMERS_CACHE", path.join(cacheRoot, "huggingface", "transformers")),
      LM_HARNESS_CACHE_PATH: envOr("LM_HARNESS_CACHE_PATH", path.join(cacheRoot, "lm-harness", "requests")),
      TEUTONIC_KING_BENCH_CACHE_REQUESTS: envOr("TEUTONIC_KING_BENCH_CACHE_REQUESTS", "true"),
      TEUTONIC_KING_BENCH_RESPONSE_CACHE_ROOT: envOr("TEUTONIC_KING_BENCH_RESPONSE_CACHE_ROOT", path.join(cacheRoot, "lm-harness", "responses")),
      TEUTONIC_WORKER_HOST: envOr("TEUTONIC_WORKER_HOST", "0.0.0.0"),
      TEUTONIC_WORKER_PORT: envOr("TEUTONIC_WORKER_PORT", "32000"),
      TEUTONIC_WORKER_ROOT: workerRoot,
      TEUTONIC_KING_BENCH_WORKER_TOKEN: envOr("TEUTONIC_KING_BENCH_WORKER_TOKEN", ""),
      TEUTONIC_KING_BENCH_CONTROLLER_URL: envOr("TEUTONIC_KING_BENCH_CONTROLLER_URL", "http://127.0.0.1:32100"),
      TEUTONIC_KING_BENCH_CONTROLLER_TOKEN: envOr("TEUTONIC_KING_BENCH_CONTROLLER_TOKEN", envOr("TEUTONIC_KING_BENCH_WORKER_TOKEN", "")),
      TEUTONIC_KING_BENCH_WORKER_MODE: envOr("TEUTONIC_KING_BENCH_WORKER_MODE", "accelerate-data-parallel"),
      TEUTONIC_KING_BENCH_BATCH_SIZE: envOr("TEUTONIC_KING_BENCH_BATCH_SIZE", "auto"),
      TEUTONIC_KING_BENCH_BATCH_SIZE_OVERRIDES: envOr("TEUTONIC_KING_BENCH_BATCH_SIZE_OVERRIDES", "MMLU-Pro=8,mmlu_pro=8"),
      TEUTONIC_KING_BENCH_DEVICE: envOr("TEUTONIC_KING_BENCH_DEVICE", "auto"),
      TEUTONIC_KING_BENCH_MODEL_FAMILY: envOr("TEUTONIC_KING_BENCH_MODEL_FAMILY", "quasar"),
      TEUTONIC_KING_BENCH_TRUST_REMOTE_CODE: envOr("TEUTONIC_KING_BENCH_TRUST_REMOTE_CODE", "true"),
      QUASAR_TOKENIZER_MODEL: envOr("QUASAR_TOKENIZER_MODEL", "silx-ai/Quasar-10B"),
      TEUTONIC_KING_BENCH_TOKENIZER_REPO: envOr("TEUTONIC_KING_BENCH_TOKENIZER_REPO", ""),
      TEUTONIC_KING_BENCH_MODEL_ARGS_EXTRA: envOr("TEUTONIC_KING_BENCH_MODEL_ARGS_EXTRA", ""),
      TEUTONIC_KING_BENCH_ACCELERATE_DEVICE: envOr("TEUTONIC_KING_BENCH_ACCELERATE_DEVICE", "cuda"),
      TEUTONIC_KING_BENCH_ACCELERATE_MODEL_ARGS_EXTRA: envOr("TEUTONIC_KING_BENCH_ACCELERATE_MODEL_ARGS_EXTRA", ""),
      TEUTONIC_KING_BENCH_LM_EVAL_BIN: envOr(
        "TEUTONIC_KING_BENCH_LM_EVAL_BIN",
        `${accelerateBin} launch --multi_gpu --num_processes 8 --gpu_ids 0,1,2,3,4,5,6,7 --mixed_precision bf16 --num_cpu_threads_per_process 8 --enable_cpu_affinity --main_process_port 29600 -m lm_eval`,
      ),
      TEUTONIC_KING_BENCH_SINGLE_GPU_LM_EVAL_BIN: envOr("TEUTONIC_KING_BENCH_SINGLE_GPU_LM_EVAL_BIN", path.join(workerRoot, ".venv", "bin", "lm-eval")),
      TEUTONIC_KING_BENCH_EVAL_PYTHON: envOr("TEUTONIC_KING_BENCH_EVAL_PYTHON", envOr("TEUTONIC_KING_BENCH_WORKER_PYTHON", venvPython)),
    },
  }],
};
