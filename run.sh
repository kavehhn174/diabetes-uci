#!/usr/bin/env bash
#
# Runs the whole stack together for local development:
#   - backend   (Express API on :3001, which auto-starts the Python
#                Bayesian Network prediction microservice on :5001)
#   - frontend  (Vite dev server on :5173, proxies /api -> backend)
#
# Usage:
#   ./run.sh                 # start everything
#   ./run.sh --no-install     # skip dependency auto-install checks
#
# Press Ctrl+C to stop everything.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"
MODEL_DIR="$BACKEND_DIR/model"
LOG_DIR="$ROOT_DIR/logs"

BACKEND_PORT=3001
MODEL_PORT=5001
FRONTEND_PORT=5173

SKIP_INSTALL=false
for arg in "$@"; do
  case "$arg" in
    --no-install) SKIP_INSTALL=true ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
  esac
done

# --- colored logging helpers -------------------------------------------------
color() { printf "\033[%sm%s\033[0m" "$1" "$2"; }
info()  { echo "$(color '1;34' '[run.sh]') $1"; }
ok()    { echo "$(color '1;32' '[run.sh]') $1"; }
warn()  { echo "$(color '1;33' '[run.sh]') $1"; }
err()   { echo "$(color '1;31' '[run.sh]') $1" >&2; }

mkdir -p "$LOG_DIR"

# --- sanity checks ------------------------------------------------------------
command -v node >/dev/null 2>&1 || { err "node is not installed / not on PATH."; exit 1; }
command -v npm  >/dev/null 2>&1 || { err "npm is not installed / not on PATH."; exit 1; }

NODE_MAJOR="$(node -e 'console.log(process.versions.node.split(".")[0])')"
if [[ "$NODE_MAJOR" -lt 18 ]]; then
  warn "Node $NODE_MAJOR detected; Node 18+ is recommended (global fetch is used)."
fi

# --- free up ports from any leftover processes from a previous run -----------
kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    warn "Port $port is in use (pid(s): $pids) - stopping previous instance."
    kill $pids 2>/dev/null || true
    sleep 1
    kill -9 $pids 2>/dev/null || true
  fi
}

if command -v lsof >/dev/null 2>&1; then
  kill_port "$BACKEND_PORT"
  kill_port "$MODEL_PORT"
  kill_port "$FRONTEND_PORT"
fi
pkill -f "backend/model/server.py" 2>/dev/null || true

# --- dependency setup ---------------------------------------------------------
if [[ "$SKIP_INSTALL" == false ]]; then
  if [[ ! -d "$BACKEND_DIR/node_modules" ]]; then
    info "Installing backend Node dependencies..."
    (cd "$BACKEND_DIR" && npm install)
  fi

  if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    info "Installing frontend Node dependencies..."
    (cd "$FRONTEND_DIR" && npm install)
  fi

  if [[ ! -x "$MODEL_DIR/venv/bin/python" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      info "Creating Python virtualenv for the Bayesian Network model service..."
      python3 -m venv "$MODEL_DIR/venv"
      "$MODEL_DIR/venv/bin/pip" install --quiet --upgrade pip
      info "Installing Python model dependencies (this can take a minute)..."
      "$MODEL_DIR/venv/bin/pip" install --quiet -r "$MODEL_DIR/requirements.txt"
    else
      warn "python3 not found - the prediction microservice will fail to start."
      warn "Install Python 3 and re-run, or run with --no-install and set it up manually."
    fi
  fi
fi

if [[ ! -f "$BACKEND_DIR/data.db" ]]; then
  warn "backend/data.db not found. Patient search/dashboard pages need seeding:"
  warn "  (cd backend && npm run seed)"
fi

if [[ ! -f "$MODEL_DIR/cleaned_hill_climb_bn_model.pkl" ]]; then
  warn "Model file backend/model/cleaned_hill_climb_bn_model.pkl is missing - live predictions will fail."
fi

# --- launch services -----------------------------------------------------------
PIDS=()
CLEANING_UP=false

cleanup() {
  if [[ "$CLEANING_UP" == true ]]; then return; fi
  CLEANING_UP=true
  trap - EXIT INT TERM
  echo ""
  info "Shutting down..."
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null
    fi
  done
  # The model service is spawned by the backend (not a direct child of this
  # script), so it needs to be stopped explicitly.
  pkill -f "backend/model/server.py" 2>/dev/null || true
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null
    fi
  done
  pkill -9 -f "backend/model/server.py" 2>/dev/null || true
  ok "Stopped."
}
trap cleanup EXIT INT TERM

info "Starting backend (API :$BACKEND_PORT, model service :$MODEL_PORT)..."
(cd "$BACKEND_DIR" && exec node server.js) > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
PIDS+=("$BACKEND_PID")

info "Starting frontend (Vite :$FRONTEND_PORT)..."
# Invoke the vite binary directly (not via `npm run dev`) so $! is the real
# vite process and not npm's wrapper, which doesn't reliably forward signals.
(cd "$FRONTEND_DIR" && exec node_modules/.bin/vite --port "$FRONTEND_PORT" --strictPort) > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
PIDS+=("$FRONTEND_PID")

# Stream both logs with colored prefixes while services are up.
tail -n +1 -F "$LOG_DIR/backend.log" 2>/dev/null | sed -u "s/^/$(color '1;36' '[backend] ')/" &
PIDS+=("$!")
tail -n +1 -F "$LOG_DIR/frontend.log" 2>/dev/null | sed -u "s/^/$(color '1;35' '[frontend]')/" &
PIDS+=("$!")

sleep 2
ok "Backend:    http://localhost:$BACKEND_PORT"
ok "Model API:  http://localhost:$MODEL_PORT (auto-started by backend)"
ok "Frontend:   http://localhost:$FRONTEND_PORT"
info "Logs: $LOG_DIR/backend.log, $LOG_DIR/frontend.log"
info "Press Ctrl+C to stop everything."

wait "$BACKEND_PID" "$FRONTEND_PID"
