// PM2 config for periodic Teutonic maintenance jobs.
// All jobs use cron_restart + autorestart:false so PM2 fires them on schedule
// and does NOT restart them immediately after they exit.
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
    // Uploads king to HF only when it has changed (checks every 5 minutes)
    name: "teutonic-upload-king",
    script: "scripts/upload_king_to_hf.py",
    interpreter: "/root/teutonic/.venv/bin/python",
    cwd: "/root/teutonic",
    cron_restart: "*/5 * * * *",
    autorestart: false,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: { PYTHONUNBUFFERED: "1" },
  }, {
    // Purges stale model snapshots and shard files every 30 minutes
    name: "teutonic-model-cache-cleanup",
    script: "scripts/cleanup_model_cache.py",
    interpreter: "/root/teutonic/.venv/bin/python",
    cwd: "/root/teutonic",
    cron_restart: "*/30 * * * *",
    autorestart: false,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: { PYTHONUNBUFFERED: "1" },
  }],
};
