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
      HF_TOKEN: dopplerPrd("HF_TOKEN"),
      HF_HUB_ENABLE_HF_TRANSFER: "1",
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
      // Each eval takes ~250s of bootstrap + setup + busy-wait for the
      // eval-server lock. A single tick can legitimately take 7-15 minutes
      // when the server is saturated; default 600s was tripping the watchdog
      // on every successful eval.
      TEUTONIC_TICK_RESTART_AFTER: "1800",
      TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS: "20",
      // Defense-in-depth for the eval-stream idle watchdog. The eval_server
      // now emits SSE heartbeat events during king/challenger load so this
      // should never trip in normal operation; the wider envelope is just
      // insurance against slow CDN downloads of an uncached challenger.
      TEUTONIC_STREAM_IDLE_WARN_AFTER: "300",
      TEUTONIC_STREAM_IDLE_TIMEOUT: "900",
    },
    max_restarts: 10,
    restart_delay: 5000,
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }],
};
