#!/usr/bin/env bash
# Teutonic mining pipeline — local orchestrator (runs on templar).
#
# 1. Rsync the mining scripts + training_bundle to the secondary B200 box.
# 2. Bootstrap a venv on the box if missing.
# 3. Launch train_challenger.py inside a tmux session so it survives ssh hiccups.
# 4. Poll for the verdict.json.
# 5. Pull verdict + submit reveal locally.
# 6. Write a status report to .arbos/outbox/.
#
# Usage:
#   ./run_pipeline.sh start [--hotkey h0] [--upload-repo unconst/Teutonic-VIII-h0]
#   ./run_pipeline.sh tail
#   ./run_pipeline.sh status
#   ./run_pipeline.sh fetch        # pull verdict only
#   ./run_pipeline.sh submit       # submit reveal locally using fetched verdict
set -euo pipefail

cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
REMOTE_HOST="wrk-nlapfgb9asmx@ssh.deployments.targon.com"
REMOTE_DIR="/root/teutonic-mining"
SESSION="teutonic-miner"
VERDICT_REMOTE="$REMOTE_DIR/work/verdict.json"
VERDICT_LOCAL="$ROOT/reports/teutonic-mining/verdict.json"
LOG_REMOTE="$REMOTE_DIR/work/train.log"
HOTKEY="${HOTKEY:-h0}"
UPLOAD_REPO="${UPLOAD_REPO:-unconst/Teutonic-VIII-h0}"

ssh_run() { ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no "$REMOTE_HOST" "$@"; }

# rsync doesn't work because Targon's ssh proxy prints a "Connecting to container..."
# banner that breaks rsync's protocol. Use tar-over-ssh instead.
push_tree() {
  local src="$1" dst="$2"
  ssh_run "mkdir -p $dst" >/dev/null
  tar -C "$src" -czf - . | ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
    "tar -C $dst -xzf -"
}
push_file() {
  local src="$1" dst="$2"
  scp -q -o StrictHostKeyChecking=no "$src" "$REMOTE_HOST":"$dst"
}
pull_file() {
  local src="$1" dst="$2"
  scp -q -o StrictHostKeyChecking=no "$REMOTE_HOST":"$src" "$dst"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  start)
    echo "[pipeline] syncing scripts to $REMOTE_HOST..."
    ssh_run "mkdir -p $REMOTE_DIR/bundle $REMOTE_DIR/work" >/dev/null
    push_file teutonic/scripts/mining/train_challenger.py "$REMOTE_DIR/train_challenger.py"
    push_file teutonic/scripts/mining/requirements.txt "$REMOTE_DIR/requirements.txt"
    push_tree teutonic/scripts/training_bundle "$REMOTE_DIR/bundle"

    push_file teutonic/scripts/mining/_remote_run.sh "$REMOTE_DIR/_remote_run.sh"
    ssh_run "chmod +x $REMOTE_DIR/_remote_run.sh"

    echo "[pipeline] writing HF_TOKEN to remote..."
    HF_TOKEN_VAL="$(doppler secrets get HF_TOKEN --plain -p arbos -c prd)"
    printf '%s' "$HF_TOKEN_VAL" | ssh -o StrictHostKeyChecking=no "$REMOTE_HOST" \
      "cat > $REMOTE_DIR/.hf_token && chmod 600 $REMOTE_DIR/.hf_token"

    echo "[pipeline] bootstrapping venv on remote..."
    ssh_run "bash -lc 'set -e; cd $REMOTE_DIR; \
      if [ ! -d venv ]; then python3 -m venv venv; fi; \
      ./venv/bin/pip -q install --upgrade pip; \
      ./venv/bin/pip -q install -r requirements.txt'"

    echo "[pipeline] launching tmux session $SESSION..."
    ssh_run "tmux kill-session -t $SESSION 2>/dev/null || true; \
      tmux new-session -d -s $SESSION 'cd $REMOTE_DIR && UPLOAD_REPO=$UPLOAD_REPO ./_remote_run.sh 2>&1 | tee $LOG_REMOTE'; \
      sleep 1; tmux ls"
    echo "[pipeline] started. tail with: $0 tail"
    ;;

  tail)
    ssh_run "tail -f $LOG_REMOTE"
    ;;

  status)
    ssh_run "ls -la $REMOTE_DIR/work/ 2>/dev/null; echo ---; tail -n 40 $LOG_REMOTE 2>/dev/null"
    ;;

  fetch)
    mkdir -p "$(dirname "$VERDICT_LOCAL")"
    pull_file "$VERDICT_REMOTE" "$VERDICT_LOCAL"
    cat "$VERDICT_LOCAL" | python3 -m json.tool
    ;;

  submit)
    if [ ! -f "$VERDICT_LOCAL" ]; then
      echo "no local verdict at $VERDICT_LOCAL — run '$0 fetch' first" >&2
      exit 1
    fi
    source "$ROOT/.venv/bin/activate"
    python "$ROOT/teutonic/scripts/mining/submit_challenger.py" \
      --verdict "$VERDICT_LOCAL" --hotkey "$HOTKEY"
    ;;

  stop)
    ssh_run "tmux kill-session -t $SESSION 2>/dev/null || true; echo stopped"
    ;;

  *)
    echo "usage: $0 {start|tail|status|fetch|submit|stop}" >&2
    exit 2
    ;;
esac
