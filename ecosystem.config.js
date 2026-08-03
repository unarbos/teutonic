const { execSync } = require("child_process");

function doppler(key) {
  return execSync(`doppler secrets get ${key} --plain -p arbos -c dev`, { encoding: "utf8" }).trim();
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
  },
  {
    name: "teutonic-validator",
    script: "validator.py",
    args: "",
    interpreter: "/home/const/workspace/.venv-fix/bin/python",
    cwd: "/home/const/workspace",
    env: {
      VIRTUAL_ENV: "/home/const/workspace/.venv-fix",
      PATH: "/home/const/workspace/.venv-fix/bin:" + (process.env.PATH || ""),
      TEUTONIC_EVAL_SERVER: "http://localhost:9000",
      // Active chain (name, seed_repo, repo_pattern, arch) is read from
      // chain.toml at the repo root. Override here only for short-lived
      // experiments — the static file is the source of truth.
      TEUTONIC_EVAL_N: "25000",
      TEUTONIC_NETUID: "3",
      TEUTONIC_MIN_SUBMISSION_BLOCK: "8440350",
      TEUTONIC_NETWORK: "finney",
      BT_WALLET_NAME: "teutonic",
      BT_WALLET_HOTKEY: "default",
      TEUTONIC_R2_ENDPOINT: doppler("R2_URL"),
      TEUTONIC_R2_BUCKET: doppler("R2_BUCKET_NAME"),
      TEUTONIC_R2_ACCESS_KEY: doppler("R2_ACCESS_KEY_ID"),
      TEUTONIC_R2_SECRET_KEY: doppler("R2_SECRET_ACCESS_KEY"),
      TEUTONIC_HIPPIUS_ACCESS_KEY: doppler("HIPPIUS_ACCESS_KEY"),
      TEUTONIC_HIPPIUS_SECRET_KEY: doppler("HIPPIUS_SECRET_KEY"),
      TMC_API_KEY: doppler("TMC_API_KEY"),
      TEUTONIC_S3_CONNECT_TIMEOUT: "10",
      TEUTONIC_S3_READ_TIMEOUT: "900",
      TEUTONIC_S3_MAX_ATTEMPTS: "1",
      TEUTONIC_TICK_RESTART_AFTER: "3000",
      TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS: "20",
      TEUTONIC_STREAM_IDLE_WARN_AFTER: "600",
      TEUTONIC_STREAM_IDLE_TIMEOUT: "2400",
      TEUTONIC_KING_HASH_TIMEOUT_S: "1200",
    },
    max_restarts: 10,
    restart_delay: 5000,
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }],
};
