// PM2 entry for the eval server host.
//
// Keeps eval-server supervision separate from the validator/tunnel config in
// ecosystem.config.js. The launcher script owns env loading and uvicorn args.

module.exports = {
  apps: [{
    name: "teutonic-eval-quasar",
    script: "/root/start_eval_quasar.sh",
    interpreter: "/bin/bash",
    cwd: "/root",
    exec_mode: "fork",
    autorestart: true,
    restart_delay: 5000,
    max_restarts: 1000,
    kill_timeout: 10000,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    env: {
      PYTHONUNBUFFERED: "1",
    },
  }],
};
