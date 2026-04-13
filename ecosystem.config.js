module.exports = {
  apps: [{
    name: "teutonic-validator",
    script: "validator.py",
    interpreter: "/home/const/workspace/.venv/bin/python",
    cwd: "/home/const/workspace/teutonic",
    env: {
      TEUTONIC_SSH_KING: process.env.TEUTONIC_SSH_KING || "",
      TEUTONIC_SSH_CHALLENGER: process.env.TEUTONIC_SSH_CHALLENGER || "",
      TEUTONIC_KING_REPO: "unconst/Teutonic-I",
      HF_TOKEN: process.env.HF_TOKEN || "",
      TEUTONIC_NETUID: "3",
      TEUTONIC_NETWORK: "finney",
      BT_WALLET_NAME: "teutonic",
      BT_WALLET_HOTKEY: "default",
    },
    max_restarts: 10,
    restart_delay: 5000,
    autorestart: true,
    log_date_format: "YYYY-MM-DD HH:mm:ss",
  }],
};
