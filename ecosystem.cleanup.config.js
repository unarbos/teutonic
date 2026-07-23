// PM2 config for Teutonic maintenance jobs.
// Short maintenance jobs use cron_restart + autorestart:false.
// The Hugging Face upload worker stays alive so cron ticks cannot interrupt
// large in-flight uploads.
//
// Start:  pm2 start ecosystem.cleanup.config.js
// Reload: pm2 reload ecosystem.cleanup.config.js

module.exports = {
  apps: [{
    // Keeps .current_king up-to-date every 5 minutes
    name: "teutonic-king-ref",
    script: "scripts/write_king_ref.py",
    interpreter: "/root/teutonic/.venv/bin/python",
    cwd: "/root/teutonic",
    cron_restart: "*/5 * * * *",
    autorestart: false,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: { PYTHONUNBUFFERED: "1" },
  }, {
    // Uploads king/non-king snapshots to HF. Runs continuously so a large upload
    // is never killed by the next 5-minute schedule tick.
    name: "teutonic-upload-king",
    script: "scripts/upload_king_to_hf.py",
    args: "--loop",
    interpreter: "/root/teutonic/.venv/bin/python",
    cwd: "/root/teutonic",
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: { PYTHONUNBUFFERED: "1" },
  }, {
    // Purges stale model snapshots every 30 minutes
    name: "teutonic-model-cache-cleanup",
    script: "scripts/cleanup_model_cache.py",
    interpreter: "/root/teutonic/.venv/bin/python",
    cwd: "/root/teutonic",
    cron_restart: "*/30 * * * *",
    autorestart: false,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: { PYTHONUNBUFFERED: "1" },
  }, {
    // Purges fineweb-edu shard cache files older than 3 hours, every 30 minutes
    name: "teutonic-shard-cache-cleanup",
    script: "scripts/cleanup_shard_cache.py",
    args: "--max-age-hours 3",
    interpreter: "/root/teutonic/.venv/bin/python",
    cwd: "/root/teutonic",
    cron_restart: "*/30 * * * *",
    autorestart: false,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: { PYTHONUNBUFFERED: "1" },
  }],
};
