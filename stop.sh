#!/usr/bin/env bash
# Stop the Agent Bouncer — Benchmark Studio server started by ./start.sh.
set -uo pipefail
cd "$(dirname "$0")"

PIDFILE=".studio.pid"
stopped=0

# 1) graceful stop via the recorded PID
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 0.2; done
    kill -9 "$PID" 2>/dev/null || true
    stopped=1
  fi
  rm -f "$PIDFILE"
fi

# 2) fallback: kill any stray uvicorn serving the studio app
if pkill -f 'uvicorn agent_bouncer.serving.api' 2>/dev/null; then
  stopped=1
fi

if [ "$stopped" = "1" ]; then
  echo "✓ Benchmark Studio stopped."
else
  echo "• Benchmark Studio was not running."
fi
