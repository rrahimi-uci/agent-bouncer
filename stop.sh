#!/usr/bin/env bash
# Stop the Agent Bouncer Workbench server started by ./start.sh.
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

# 3) reap any orphaned training/eval/dataset subprocesses the server spawned. The server kills
#    these itself on graceful shutdown (SIGTERM); this is a safety net for the SIGKILL path so
#    nothing heavy keeps running in the background after ./stop.sh.
CHILD_PATTERNS=(
  'scripts/train/run_training.py' 'scripts/eval/run_testing.py' 'scripts/eval/run_eval_only.py'
  'scripts/eval/run_benchmarks.py' 'scripts/eval/eval_added_guard.py'
  'scripts/report/compute_curves.py' 'scripts/data/build_dataset.py'
)
killed_children=0
for pat in "${CHILD_PATTERNS[@]}"; do
  if pkill -f "$pat" 2>/dev/null; then killed_children=1; stopped=1; fi
done
if [ "$killed_children" = "1" ]; then
  sleep 1
  for pat in "${CHILD_PATTERNS[@]}"; do
    pkill -9 -f "$pat" 2>/dev/null || true   # SIGKILL anything that ignored SIGTERM
  done
  echo "  ↳ also stopped in-flight training/eval jobs."
fi

if [ "$stopped" = "1" ]; then
  echo "✓ Agent Bouncer Workbench stopped."
else
  echo "• Agent Bouncer Workbench was not running."
fi
