const path = require("path");

const CWD = __dirname;
const PYTHON = path.join(CWD, ".venv", "bin", "python");
const LOGS = path.join(CWD, "logs");
const USE_WANDB = process.env.TEUTONIC_WANDB === "1";
const WANDB_PROJECT = process.env.TEUTONIC_WANDB_PROJECT || "teutonic";

const wandbArgs = USE_WANDB
  ? ` --use-wandb --wandb-project ${WANDB_PROJECT}`
  : "";

module.exports = {
  apps: [
    {
      name: "teutonic-test",
      script: "run_local.py",
      cwd: CWD,
      interpreter: PYTHON,
      args: "--log-level INFO" + wandbArgs,
      autorestart: false,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: path.join(LOGS, "test-err.log"),
      out_file: path.join(LOGS, "test-out.log"),
    },
    {
      name: "teutonic-test-json",
      script: "run_local.py",
      cwd: CWD,
      interpreter: PYTHON,
      args: "--json --log-level INFO --log-file " + path.join(LOGS, "teutonic.jsonl") + wandbArgs,
      autorestart: false,
      merge_logs: true,
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: path.join(LOGS, "test-json-err.log"),
      out_file: path.join(LOGS, "test-json-out.log"),
    },
  ],
};
