"use strict";

const path = require("path");
const { loadProcessEnv, resolveExecutable } = require("./pm2.env");

const root = __dirname;
const envFile = process.env.TEUTONIC_ENV_FILE || path.join(root, ".env");
const fileEnv = loadProcessEnv(path.resolve(root, envFile));
const python = resolveExecutable(fileEnv.TEUTONIC_PYTHON || ".venv/bin/python", root);
const pythonBin = python.includes("/") ? path.dirname(python) : "";
const commonEnv = {
  ...fileEnv,
  PYTHONUNBUFFERED: fileEnv.PYTHONUNBUFFERED || "1",
  PYTHONPATH: [root, fileEnv.PYTHONPATH].filter(Boolean).join(path.delimiter),
  ...(pythonBin ? { PATH: `${pythonBin}:${fileEnv.PATH || ""}` } : {}),
};
const longRunning = {
  interpreter: python,
  cwd: root,
  exec_mode: "fork",
  autorestart: true,
  restart_delay: 5000,
  max_restarts: 1000,
  kill_timeout: 10000,
  log_date_format: "YYYY-MM-DD HH:mm:ss",
};

module.exports = {
  apps: [
    {
      ...longRunning,
      name: "teutonic-access-controller",
      script: "scripts/access_controller.py",
      kill_timeout: 30000,
      env: commonEnv,
    },
    {
      name: "teutonic-eval-tunnel",
      script: "scripts/evaluator_tunnel.sh",
      interpreter: "/bin/bash",
      cwd: root,
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 1000,
      kill_timeout: 5000,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      env: commonEnv,
    },
    {
      ...longRunning,
      name: "teutonic-validator",
      script: "scripts/validator_service.py",
      kill_timeout: 30000,
      env: commonEnv,
    },
    {
      ...longRunning,
      name: "teutonic-promotion-worker",
      script: "scripts/promotion_worker.py",
      kill_timeout: 30000,
      env: commonEnv,
    },
    {
      ...longRunning,
      name: "teutonic-weight-publisher",
      script: "scripts/weight_publisher.py",
      env: {
        ...commonEnv,
        TEUTONIC_WEIGHT_PUBLISHER_MODE:
          fileEnv.TEUTONIC_WEIGHT_PUBLISHER_MODE || "dry_run",
      },
    },
    {
      ...longRunning,
      name: "teutonic-dashboard-live",
      script: "scripts/dashboard_live.py",
      env: commonEnv,
    },
    {
      ...longRunning,
      name: "teutonic-dashboard-view",
      script: "scripts/dashboard_view.py",
      env: commonEnv,
    },
  ],
};
