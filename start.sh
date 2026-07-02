#!/usr/bin/env bash
# Start Agent Bouncer — Benchmark Studio (dashboard + /screen API).
#
#   ./start.sh                 # http://127.0.0.1:8000, opens your browser
#   ./start.sh 8080            # custom port
#   PORT=9000 HOST=0.0.0.0 OPEN=0 ./start.sh
#
# Runs in the background; logs to outputs/logs/studio.log. Stop with ./stop.sh.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${HOST:-127.0.0.1}"
PORT="${1:-${PORT:-8000}}"
PIDFILE=".studio.pid"
LOG="outputs/logs/studio.log"
URL="http://${HOST}:${PORT}"
mkdir -p outputs/logs

# --- pick a Python interpreter (prefer the project venv) --------------------
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PY" ]; then
  echo "✗ No Python found. Create the env first:"
  echo "    python3 -m venv .venv && .venv/bin/pip install -e '.[dev,eval,serve]'"
  exit 1
fi

# --- already running? -------------------------------------------------------
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "✓ Benchmark Studio already running (PID $(cat "$PIDFILE")) → ${URL}"
  exit 0
fi
rm -f "$PIDFILE"

# --- ensure serve deps (fastapi + uvicorn) ----------------------------------
if ! "$PY" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "• Installing serve deps (fastapi, uvicorn)…"
  "$PY" -m pip install -q fastapi uvicorn
fi
# soft warning: running the pipeline from the UI needs the eval/train extras
if ! "$PY" -c "import torch, transformers, datasets" >/dev/null 2>&1; then
  echo "• Note: viewing works now, but running the pipeline needs:  pip install -e '.[eval]'"
fi

# --- launch -----------------------------------------------------------------
echo "• Starting Benchmark Studio on ${URL} …"
TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  "$PY" -m uvicorn agent_bouncer.serving.api:app --host "$HOST" --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

# --- wait for readiness -----------------------------------------------------
for _ in $(seq 1 40); do
  if curl -fsS "${URL}/health" >/dev/null 2>&1; then
    echo "✓ Ready → ${URL}"
    if [ "${OPEN:-1}" = "1" ]; then
      { command -v open >/dev/null 2>&1 && open "$URL"; } \
        || { command -v xdg-open >/dev/null 2>&1 && xdg-open "$URL"; } || true
    fi
    echo "  logs:  tail -f ${LOG}"
    echo "  stop:  ./stop.sh"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "✗ Server exited during startup — last log lines:"; tail -n 20 "$LOG"; rm -f "$PIDFILE"; exit 1
  fi
  sleep 0.5
done

echo "✗ Not ready after 20s. Check ${LOG}"
exit 1
