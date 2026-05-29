// PM2 controller for the authenticated all-kings benchmark service.
// Runs locally so S3/Doppler credentials never need to live on the GPU worker.

const path = require("path");
const repoRoot = path.resolve(__dirname, "..", "..");

function envOr(key, fallback) {
  return process.env[key] != null && process.env[key] !== ""
    ? process.env[key]
    : fallback;
}

function requiredEnv(key) {
  if (process.env[key] == null || process.env[key] === "") {
    throw new Error(`${key} is required`);
  }
  return process.env[key];
}

module.exports = {
  apps: [{
    name: "teutonic-king-bench-tunnel",
    script: "ssh",
    args: [
      "-N",
      "-o", "ExitOnForwardFailure=yes",
      "-o", "ServerAliveInterval=30",
      "-o", "ServerAliveCountMax=3",
      "-i", envOr("TEUTONIC_KING_BENCH_SSH_KEY", "/home/const/.ssh/id_ed25519"),
      "-p", envOr("TEUTONIC_KING_BENCH_WORKER_SSH_PORT", "32298"),
      "-R", envOr("TEUTONIC_KING_BENCH_REVERSE_TUNNEL", "32100:127.0.0.1:32100"),
      requiredEnv("TEUTONIC_KING_BENCH_WORKER_SSH"),
    ],
    cwd: repoRoot,
    autorestart: true,
    restart_delay: 5000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    out_file: path.join(repoRoot, "logs/king-bench-tunnel.out.log"),
    error_file: path.join(repoRoot, "logs/king-bench-tunnel.err.log"),
  }, {
    name: "teutonic-king-bench-controller",
    script: "scripts/king_benchmark_service/controller.py",
    interpreter: envOr("TEUTONIC_KING_BENCH_SERVICE_PYTHON", "python3"),
    cwd: repoRoot,
    autorestart: true,
    restart_delay: 30000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    out_file: path.join(repoRoot, "logs/king-bench-controller.out.log"),
    error_file: path.join(repoRoot, "logs/king-bench-controller.err.log"),
    env: {
      PYTHONUNBUFFERED: "1",
      TOKENIZERS_PARALLELISM: "false",
      TEUTONIC_CONTROLLER_HOST: envOr("TEUTONIC_CONTROLLER_HOST", "0.0.0.0"),
      TEUTONIC_CONTROLLER_PORT: envOr("TEUTONIC_CONTROLLER_PORT", "32100"),
      TEUTONIC_KING_BENCH_WORKER_URL: requiredEnv("TEUTONIC_KING_BENCH_WORKER_URL"),
      TEUTONIC_KING_BENCH_WORKER_TOKEN: envOr("TEUTONIC_KING_BENCH_WORKER_TOKEN", ""),
      TEUTONIC_KING_BENCH_CONTROLLER_TOKEN: envOr("TEUTONIC_KING_BENCH_CONTROLLER_TOKEN", envOr("TEUTONIC_KING_BENCH_WORKER_TOKEN", "")),
      TEUTONIC_KING_BENCH_BENCHMARKS: envOr("TEUTONIC_KING_BENCH_BENCHMARKS", "MMLU,MMLU-Pro,BBH,ARC-C,TruthfulQA,WinoGrande"),
      TEUTONIC_KING_BENCH_SERVICE_ROOT: envOr("TEUTONIC_KING_BENCH_SERVICE_ROOT", path.join(repoRoot, "runs/king-benchmark-service")),
      TEUTONIC_KING_BENCH_SERVICE_LATEST_KEY: envOr("TEUTONIC_KING_BENCH_SERVICE_LATEST_KEY", "king-benchmark-daily/all-kings/latest.json"),
      TEUTONIC_KING_BENCH_SERVICE_INDEX_KEY: envOr("TEUTONIC_KING_BENCH_SERVICE_INDEX_KEY", "king-benchmark-daily/all-kings/index.json"),
      TEUTONIC_KING_BENCH_SERVICE_HISTORY_KEY: envOr("TEUTONIC_KING_BENCH_SERVICE_HISTORY_KEY", "king-benchmark-daily/all-kings/history.jsonl"),
      TEUTONIC_HIPPIUS_ENDPOINT: envOr("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com"),
      TEUTONIC_HIPPIUS_BUCKET: envOr("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3"),
      TEUTONIC_KING_BENCH_DTYPE: envOr("TEUTONIC_KING_BENCH_DTYPE", "bfloat16"),
      TEUTONIC_KING_BENCH_DEVICE: envOr("TEUTONIC_KING_BENCH_DEVICE", "auto"),
      TEUTONIC_KING_BENCH_MODEL_ARGS_EXTRA: envOr("TEUTONIC_KING_BENCH_MODEL_ARGS_EXTRA", ""),
      TEUTONIC_WORKER_PROGRESS_INTERVAL_S: envOr("TEUTONIC_WORKER_PROGRESS_INTERVAL_S", "60"),
      TEUTONIC_KING_BENCH_AUTO_DISPATCH: envOr("TEUTONIC_KING_BENCH_AUTO_DISPATCH", "1"),
      TEUTONIC_KING_BENCH_AUTO_DISPATCH_INTERVAL_S: envOr("TEUTONIC_KING_BENCH_AUTO_DISPATCH_INTERVAL_S", "60"),
    },
  }],
};
