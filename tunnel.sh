#!/usr/bin/env bash
# Forward the validator host's port 9000 to the remote 8xB200 eval service.
exec ssh -N \
  -L 9000:localhost:9000 \
  -p 10099 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  root@95.133.252.200
