#!/usr/bin/env bash
# Boot Xvfb on :99 then hand off to the user's command.
set -euo pipefail

Xvfb :99 -screen 0 1280x720x24 -nolisten tcp &
XVFB_PID=$!
export DISPLAY=:99

# Wait until Xvfb is actually listening.
for i in {1..30}; do
    if xdpyinfo -display :99 >/dev/null 2>&1; then break; fi
    sleep 0.2
done

echo "[fly-entrypoint] Xvfb up on :99 (pid=$XVFB_PID)"
echo "[fly-entrypoint] exec: $*"
exec "$@"
