#!/usr/bin/env bash
# Stop the background runner and its current training worker. Rerun run_all.sh to resume.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
PIDFILE=outputs/logs/runner.pid
if ! runner_alive; then
  echo "No active runner (removing any stale pid file)."
  rm -f "$PIDFILE"
  exit 0
fi
PID="$(cat "$PIDFILE")"
# run_all.sh starts the runner with setsid, so it leads its own process group (runner + worker).
if ! kill -TERM -- "-$PID" 2>/dev/null; then
  pkill -TERM -P "$PID" || true
  kill -TERM "$PID" || true
fi
group_alive() { kill -0 -- "-$PID" 2>/dev/null || kill -0 "$PID" 2>/dev/null; }
for _ in $(seq 1 60); do
  group_alive || break
  sleep 1
done
if group_alive; then
  echo "Still running after 60 s; sending SIGKILL to the runner's process group."
  kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
  sleep 2
fi
rm -f "$PIDFILE"
echo "Stopped runner $PID. Resume with:  bash scripts/run_all.sh"
