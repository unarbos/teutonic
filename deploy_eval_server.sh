#!/usr/bin/env bash
set -euo pipefail
# Deploy eval server code + dependencies to the GPU pod.
# Usage: ./deploy_eval_server.sh [host] [port]

HOST="${1:-93.120.231.186}"
PORT="${2:-32298}"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
SCP="scp -o StrictHostKeyChecking=no -P $PORT"

echo "==> deploying code to $HOST:$PORT"
$SCP eval_server.py "root@$HOST:/root/workspace/eval_server.py"
$SCP eval/torch_runner.py "root@$HOST:/root/workspace/eval/torch_runner.py"
$SCP eval/raw_dataset.py "root@$HOST:/root/workspace/eval/raw_dataset.py"
$SCP model_store.py "root@$HOST:/root/workspace/model_store.py"
$SCP chain_config.py "root@$HOST:/root/workspace/chain_config.py"

echo "==> installing dependencies"
$SSH "/root/workspace/.venv/bin/pip install -q liger-kernel"

echo "==> restarting eval server"
$SSH "supervisorctl reread && supervisorctl update && supervisorctl restart eval_server"

echo "==> verifying"
sleep 3
$SSH "curl -s http://localhost:9000/health" | python3 -m json.tool
echo "done"
