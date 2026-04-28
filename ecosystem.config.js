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
    script: "/home/const/workspace/teutonic/tunnel.sh",
    cwd: "/home/const/workspace/teutonic",
    autorestart: true,
    restart_delay: 5000,
    max_restarts: 1000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }, {
    name: "teutonic-validator",
    script: "validator.py",
    // --no-seen: only evaluate genuinely new (unseen) hotkeys; idle when
    // queue is empty rather than replenishing with re-eval candidates.
    // Set 2026-04-26 per operator request.
    args: "--no-seen",
    interpreter: "/home/const/workspace/.venv/bin/python",
    cwd: "/home/const/workspace/teutonic",
    env: {
      TEUTONIC_EVAL_SERVER: "http://localhost:9000",
      TEUTONIC_SEED_REPO: "unconst/Teutonic-VIII",
      HF_TOKEN: dopplerPrd("HF_TOKEN"),
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
      // Discord notifications disabled per operator request. Re-enable by
      // restoring DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID from Doppler.
      DISCORD_BOT_TOKEN: "",
      DISCORD_CHANNEL_ID: "",
      // Disable hf-xet downloader (it has aborted the validator twice during
      // dethrone-target king downloads on 2026-04-26). validator.py also sets
      // this defensively before importing huggingface_hub, but having it in
      // the env makes it explicit for any subprocess we spawn too.
      HF_HUB_DISABLE_XET: "1",
      // Each Teutonic-VIII eval is ~2.4x larger than Teutonic-III's ~250s,
      // i.e. ~600s of bootstrap + setup + busy-wait for the eval-server lock.
      // A single tick can legitimately take 10-20 minutes when the server is
      // saturated; bumped from 1800 -> 2700 to give 8B evals headroom.
      TEUTONIC_TICK_RESTART_AFTER: "2700",
      TEUTONIC_MAX_CONSECUTIVE_TICK_ERRORS: "20",
    },
    // Bumped from 10 → 1000 after 2026-04-26 incident: PM2 gave up on the
    // validator at restart #15 and the subnet ran without a validator for
    // ~21 minutes. This counter resets only when the process stays up for
    // min_uptime (default 1s), which is generous; the real safety net is
    // the rest of our error handling.
    max_restarts: 1000,
    restart_delay: 5000,
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }],
};
