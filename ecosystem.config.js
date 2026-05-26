const { execSync } = require("child_process");

function doppler(key) {
  return execSync(`doppler secrets get ${key} --plain -p arbos -c dev`, { encoding: "utf8" }).trim();
}

function dopplerPrd(key) {
  return execSync(`doppler secrets get ${key} --plain -p arbos -c prd`, { encoding: "utf8" }).trim();
}

module.exports = {
  apps: [{
    name: "teutonic-eval-tunnel",
    script: "./tunnel.sh",
    cwd: "/home/const/workspace",
    autorestart: true,
    restart_delay: 5000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }, {
    name: "teutonic-validator",
    script: "validator.py",
    args: "",
    interpreter: "/home/const/workspace/.venv/bin/python",
    cwd: "/home/const/workspace",
    env: {
      TEUTONIC_EVAL_SERVER: "http://localhost:9000",
      // Active chain (name, seed_repo, repo_pattern, arch) is read from
      // chain.toml at the repo root. Override here only for short-lived
      // experiments — the static file is the source of truth.
      TEUTONIC_EVAL_DATASET_MODE: "raw_hippius",
      TEUTONIC_RAW_DATASET_PREFIX: "hf-mirrors/HuggingFaceFW/fineweb-edu/data",
      TEUTONIC_RAW_DATASET_MANIFEST: "hf-mirrors/HuggingFaceFW/fineweb-edu/data/_manifest.json",
      TEUTONIC_RAW_TOKENIZER_REPO: "Qwen/Qwen3-8B",
      ///TEUTONIC_FORCE_SEED_KING: "1", 22.05.2026 16:07 override
      // First-deploy degraded mode: no private holdout pool yet, so use
      // public-only eval. Once /var/teutonic/private_pool is populated, bump
      // TEUTONIC_EVAL_N_PRIVATE to 2500 and TEUTONIC_EVAL_N_PUBLIC to 2500.
      TEUTONIC_EVAL_N: "10000",
      TEUTONIC_EVAL_N_PUBLIC: "10000",
      TEUTONIC_EVAL_N_PRIVATE: "0",
      TEUTONIC_NETUID: "3",
      TEUTONIC_NETWORK: "finney",
      BT_WALLET_NAME: "teutonic",
      BT_WALLET_HOTKEY: "default",
      TEUTONIC_R2_ENDPOINT: doppler("R2_URL"),
      TEUTONIC_R2_BUCKET: doppler("R2_BUCKET_NAME"),
      TEUTONIC_R2_ACCESS_KEY: doppler("R2_ACCESS_KEY_ID"),
      TEUTONIC_R2_SECRET_KEY: doppler("R2_SECRET_ACCESS_KEY"),
      TEUTONIC_HIPPIUS_ACCESS_KEY: doppler("HIPPIUS_ACCESS_KEY"),
      TEUTONIC_HIPPIUS_SECRET_KEY: doppler("HIPPIUS_SECRET_KEY"),
      TEUTONIC_DS_ENDPOINT: "https://s3.hippius.com",
      TEUTONIC_DS_BUCKET: "teutonic-sn3",
      TEUTONIC_DS_ACCESS_KEY: doppler("HIPPIUS_ACCESS_KEY"),
      TEUTONIC_DS_SECRET_KEY: doppler("HIPPIUS_SECRET_KEY"),
      TMC_API_KEY: doppler("TMC_API_KEY"),
      DISCORD_BOT_TOKEN: doppler("DISCORD_BOT_TOKEN"),
      DISCORD_CHANNEL_ID: doppler("DISCORD_CHANNEL_ID"),
      // Hard wall-clock cap per model: validator's `_bounded_eval` aborts
      // and records `eval_hard_timeout` at this mark — no retry, next entry
      // runs. Aligned with eval-server's HF_PREFETCH_TIMEOUT=1800s so a
      // stalled HF CDN burns exactly one 30-min budget before being skipped.
      // Pre-2026-05-13 value was 3600s (1h envelope, 3x retries on transient
      // errors = up to 90 min wasted per stuck CDN — observed live with the
      // jenny08311 v5.13 case 2026-05-12 and ClarenceDan A5518/A5519 cases
      // 2026-05-13). Steady-state eval is ~10 min so 1800s still gives ~3x
      // headroom for legitimate slow downloads on the new B200 pod.
      TEUTONIC_TICK_RESTART_AFTER: "1800",
      TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS: "20",
      // Stream-idle watchdog envelope must accommodate the multi-minute
      // challenger Hippius prefetch + sharded model load. The eval_server emits SSE
      // heartbeat events during these phases so this rarely actually fires
      // in normal operation. Pre-LXXX values were 300/900s.
      TEUTONIC_STREAM_IDLE_WARN_AFTER: "600",
      TEUTONIC_STREAM_IDLE_TIMEOUT: "1800",
      // Qwen3-4B is ~8 GB bf16; Hippius prefetch should complete in 1-2 min.
      // 600s (10 min) is generous; bump if you swap to a larger king.
      TEUTONIC_KING_HASH_TIMEOUT_S: "1200",
    },
    max_restarts: 10,
    restart_delay: 5000,
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }],
};