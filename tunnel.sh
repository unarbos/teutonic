#!/usr/bin/env bash
# Eval-server SSH tunnel. Targets 8xH100 pod.
# Previous pods: B200 95.133.252.200:10099, B200 95.133.252.33:10299,
#                B300 95.133.252.44:10310, Lium B200 95.133.252.200:10100.
exec ssh -N   -L 9000:localhost:9000   -p 32298   -o ServerAliveInterval=30   -o ServerAliveCountMax=3   -o ExitOnForwardFailure=yes   -o StrictHostKeyChecking=no   -o UserKnownHostsFile=/dev/null   root@93.120.231.186
