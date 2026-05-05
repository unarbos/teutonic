#!/usr/bin/env bash
# Eval-server SSH tunnel. Targets the Lium 8xB200 pod `teutonic-eval`
# (HUID `lunar-hawk-92`, IP 95.133.252.200, SSH on port 10100).
# Migrated from Targon `wrk-0638a6gucc7t` on 2026-05-05 after Targon went down.
exec ssh -N \
  -L 9000:localhost:9000 \
  -p 10100 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  root@95.133.252.200
