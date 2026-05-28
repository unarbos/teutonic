// PM2 entry for the daily Teutonic king benchmark automation.
//
// Starts scripts/king_benchmark_daily/runner.py, which every 24h fetches the
// current king from dashboard.json, rents 11 Lium pods, runs one benchmark per
// pod, copies results/logs back, and writes stable JSON under runs/.

const path = require("path");

const repoRoot = path.resolve(__dirname, "..", "..");

function envOr(key, fallback) {
  return process.env[key] != null && process.env[key] !== ""
    ? process.env[key]
    : fallback;
}

module.exports = {
  apps: [{
    name: "teutonic-king-bench-lium",
    script: "scripts/king_benchmark_daily/runner.py",
    interpreter: envOr("TEUTONIC_KING_BENCH_PYTHON", "python3"),
    cwd: repoRoot,
    autorestart: true,
    restart_delay: 30000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    out_file: path.join(repoRoot, "logs/king-bench-lium.out.log"),
    error_file: path.join(repoRoot, "logs/king-bench-lium.err.log"),
    env: {
      PYTHONUNBUFFERED: "1",
      TOKENIZERS_PARALLELISM: "false",

      TEUTONIC_KING_BENCH_INTERVAL_HOURS: envOr("TEUTONIC_KING_BENCH_INTERVAL_HOURS", "24"),
      TEUTONIC_KING_BENCH_DAILY_TIME: envOr("TEUTONIC_KING_BENCH_DAILY_TIME", "00:00"),
      TEUTONIC_KING_BENCH_SCHEDULE_UTC_OFFSET: envOr("TEUTONIC_KING_BENCH_SCHEDULE_UTC_OFFSET", "+02:00"),
      TEUTONIC_KING_BENCH_POLL_SECS: envOr("TEUTONIC_KING_BENCH_POLL_SECS", "300"),
      TEUTONIC_KING_BENCH_RESULTS_ROOT: envOr(
        "TEUTONIC_KING_BENCH_RESULTS_ROOT",
        path.join(repoRoot, "runs/king-benchmark-daily"),
      ),
      TEUTONIC_KING_BENCH_REGISTRY: envOr(
        "TEUTONIC_KING_BENCH_REGISTRY",
        path.join(repoRoot, "runs/lium-rentals/registry.json"),
      ),
      TEUTONIC_KING_BENCH_REMOTE_BASE: envOr(
        "TEUTONIC_KING_BENCH_REMOTE_BASE",
        "/root/king-benchmark-evals-daily",
      ),
      TEUTONIC_KING_BENCH_TTL: envOr("TEUTONIC_KING_BENCH_TTL", "36h"),
      TEUTONIC_KING_BENCH_BENCHMARKS: envOr("TEUTONIC_KING_BENCH_BENCHMARKS", "MMLU,MMLU-Pro,BBH,ARC-C,TruthfulQA,WinoGrande"),
      TEUTONIC_KING_BENCH_RENT_TIMEOUT_S: envOr("TEUTONIC_KING_BENCH_RENT_TIMEOUT_S", "180"),
      TEUTONIC_KING_BENCH_UPLOAD_HIPPIUS: envOr("TEUTONIC_KING_BENCH_UPLOAD_HIPPIUS", "1"),
      TEUTONIC_KING_BENCH_LATEST_KEY: envOr("TEUTONIC_KING_BENCH_LATEST_KEY", "king-benchmark-daily/latest.json"),
      TEUTONIC_KING_BENCH_HISTORY_KEY: envOr("TEUTONIC_KING_BENCH_HISTORY_KEY", "king-benchmark-daily/history.jsonl"),
      TEUTONIC_HIPPIUS_ENDPOINT: envOr("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com"),
      TEUTONIC_HIPPIUS_BUCKET: envOr("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3"),
      TEUTONIC_KING_BENCH_DOPPLER_PROJECT: envOr("TEUTONIC_KING_BENCH_DOPPLER_PROJECT", "arbos"),
      TEUTONIC_KING_BENCH_DOPPLER_CONFIG: envOr("TEUTONIC_KING_BENCH_DOPPLER_CONFIG", "dev"),

      // Default automation behavior: avoid cost leaks. The runner copies logs
      // and artifacts first, then deletes success and failure pods it rented.
      TEUTONIC_KING_BENCH_KEEP_ON_SUCCESS: envOr("TEUTONIC_KING_BENCH_KEEP_ON_SUCCESS", "0"),
      TEUTONIC_KING_BENCH_DELETE_ON_FAILURE: envOr("TEUTONIC_KING_BENCH_DELETE_ON_FAILURE", "1"),
    },
  }],
};
