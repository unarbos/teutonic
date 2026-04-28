// pm2 entry for the Teutonic benchmark runner.
//
// Lives separately from the validator's root ecosystem.config.js so it can
// run on a different host (the GPU box at wrk-8pr7na60ovvz) without
// dragging in the validator/eval-tunnel processes.
//
// Bring-up:
//
//     bash scripts/benchmark/start.sh   # sources ~/.teutonic-bench.env
//
// or, if doppler is available locally:
//
//     doppler run --project arbos --config dev -- \
//         pm2 start scripts/benchmark/ecosystem.config.js && pm2 save
//
// The script reads creds from process.env so either flow works.

const path = require("path");

const repoRoot = path.resolve(__dirname, "..", "..");

function envOr(key, fallback) {
  return process.env[key] != null && process.env[key] !== ""
    ? process.env[key]
    : fallback;
}

module.exports = {
  apps: [{
    name: "teutonic-bench",
    script: "scripts/benchmark/runner.py",
    interpreter: envOr("TEUTONIC_BENCH_PYTHON", "/root/.venv/bin/python"),
    cwd: repoRoot,
    autorestart: true,
    restart_delay: 30000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    out_file: path.join(repoRoot, "logs/bench.out.log"),
    error_file: path.join(repoRoot, "logs/bench.err.log"),
    env: {
      TEUTONIC_HIPPIUS_ENDPOINT: envOr("TEUTONIC_HIPPIUS_ENDPOINT", "https://s3.hippius.com"),
      TEUTONIC_HIPPIUS_BUCKET: envOr("TEUTONIC_HIPPIUS_BUCKET", "teutonic-sn3"),
      TEUTONIC_HIPPIUS_ACCESS_KEY: envOr("TEUTONIC_HIPPIUS_ACCESS_KEY",
        envOr("HIPPIUS_ACCESS_KEY", "")),
      TEUTONIC_HIPPIUS_SECRET_KEY: envOr("TEUTONIC_HIPPIUS_SECRET_KEY",
        envOr("HIPPIUS_SECRET_KEY", "")),
      HF_TOKEN: envOr("HF_TOKEN", ""),

      // Tunables. Defaults match Covenant-72B Table 1 (zero-shot).
      TEUTONIC_BENCH_TASKS: envOr("TEUTONIC_BENCH_TASKS",
        "arc_challenge,arc_easy,piqa,openbookqa,hellaswag,winogrande,mmlu"),
      TEUTONIC_BENCH_NUM_FEWSHOT: envOr("TEUTONIC_BENCH_NUM_FEWSHOT", "0"),
      TEUTONIC_BENCH_BATCH_SIZE: envOr("TEUTONIC_BENCH_BATCH_SIZE", "auto"),
      TEUTONIC_BENCH_POLL_SECS: envOr("TEUTONIC_BENCH_POLL_SECS", "60"),
      TEUTONIC_BENCH_FAIL_COOLDOWN: envOr("TEUTONIC_BENCH_FAIL_COOLDOWN", "3600"),
      TEUTONIC_BENCH_HISTORY_MAX: envOr("TEUTONIC_BENCH_HISTORY_MAX", "200"),
      TEUTONIC_BENCH_LIMIT: envOr("TEUTONIC_BENCH_LIMIT", ""),

      HF_HOME: envOr("HF_HOME", "/root/.cache/huggingface"),
      TOKENIZERS_PARALLELISM: "false",
      PYTHONUNBUFFERED: "1",
    },
  }],
};
