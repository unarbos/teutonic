#!/usr/bin/env bash
exec ssh -N \
  -L 9000:localhost:9000 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  wrk-wbqzbod2yype@ssh.deployments.targon.com
