// PM2 config for the model cache cleanup job.
// Runs cleanup_model_cache.py every 30 minutes via cron_restart.
// autorestart: false ensures PM2 does not restart the script immediately
// after it exits — it waits for the next cron tick instead.
//
// Start:  pm2 start ecosystem.cleanup.config.js
// Reload: pm2 reload ecosystem.cleanup.config.js

module.exports = {
  apps: [{
    name: "teutonic-king-ref",
    script: "scripts/write_king_ref.py",
    interpreter: "/root/teutonic/.venv/bin/python",
    cwd: "/root/teutonic",
    cron_restart: "*/5 * * * *",
    autorestart: false,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: {
      PYTHONUNBUFFERED: "1",
    },
  }, {
    name: "teutonic-model-cache-cleanup",
    script: "scripts/cleanup_model_cache.py",
    interpreter: "/root/teutonic/.venv/bin/python",
    cwd: "/root/teutonic",
    cron_restart: "*/30 * * * *",
    autorestart: false,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: {
      PYTHONUNBUFFERED: "1",
    },
  }],
};
