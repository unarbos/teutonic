const { execSync } = require("child_process");

function doppler(key) {
  return execSync(`doppler secrets get ${key} --plain -p arbos -c dev`, { encoding: "utf8" }).trim();
}

function dopplerPrd(key) {
  return execSync(`doppler secrets get ${key} --plain -p arbos -c prd`, { encoding: "utf8" }).trim();
}

module.exports = {
  apps: [{
    name: "teutonic-validator",
    script: "validator.py",
    args: "",
    interpreter: "/home/const/workspace/.venv/bin/python",
    cwd: "/home/const/workspace/teutonic",
    env: {
      TEUTONIC_EVAL_SERVER: "http://localhost:9000",
      TEUTONIC_KING_REPO: "unconst/Teutonic-I",
      HF_TOKEN: dopplerPrd("HF_TOKEN"),
      TEUTONIC_NETUID: "3",
      TEUTONIC_NETWORK: "finney",
      BT_WALLET_NAME: "teutonic",
      BT_WALLET_HOTKEY: "default",
      TEUTONIC_R2_ENDPOINT: doppler("R2_URL"),
      TEUTONIC_R2_BUCKET: doppler("R2_BUCKET_NAME"),
      TEUTONIC_R2_ACCESS_KEY: doppler("R2_ACCESS_KEY_ID"),
      TEUTONIC_R2_SECRET_KEY: doppler("R2_SECRET_ACCESS_KEY"),
      TMC_API_KEY: doppler("TMC_API_KEY"),
    },
    max_restarts: 10,
    restart_delay: 5000,
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }],
};
