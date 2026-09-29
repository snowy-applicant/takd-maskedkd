#!/usr/bin/env bash
# Progress of every run, the runner process, the GPU and the last log lines.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
require_venv
"$PY" -m takd.cli status "$@"
PIDFILE=outputs/logs/runner.pid
if runner_alive; then
  echo "runner: active (pid $(cat "$PIDFILE"))"
else
  echo "runner: not running"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
fi
if [[ -f outputs/logs/latest.log ]]; then
  echo "--- last log lines (outputs/logs/latest.log)"
  tail -n 5 outputs/logs/latest.log
fi
