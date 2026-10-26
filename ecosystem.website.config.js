"use strict";

module.exports = {
  apps: [{
    name: "teutonic-website",
    cwd: __dirname,
    script: "scripts/serve_website_preview.py",
    interpreter: process.env.TEUTONIC_WEBSITE_PYTHON || "/usr/bin/python3",
    args: [
      "--host", process.env.TEUTONIC_WEBSITE_HOST || "0.0.0.0",
      "--port", process.env.TEUTONIC_WEBSITE_PORT || "18765",
    ],
    exec_mode: "fork",
    instances: 1,
    autorestart: true,
    restart_delay: 2000,
    max_memory_restart: "512M",
    kill_timeout: 5000,
    time: true,
    env: { PYTHONUNBUFFERED: "1", PYTHONDONTWRITEBYTECODE: "1" },
  }],
};
