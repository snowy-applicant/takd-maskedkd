#!/usr/bin/env bash
# Train every run in the background (safe to close the SSH session).
#   bash scripts/run_all.sh                      # hkd suite, then logit suite; seeds 0 1 2
#   bash scripts/run_all.sh --suite hkd          # one suite (any `takd.cli run` option works)
#   bash scripts/run_all.sh --foreground ...     # stay attached (e.g. inside tmux)
# Rerunning the same command resumes: finished runs are verified and skipped, and an
# interrupted run continues from its last resume.pt (saved every 5 epochs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
require_venv
if [[ ! -f data/cache/coco_single_224.pt ]]; then
  echo "Data cache missing: run  bash scripts/prepare_data.sh" >&2
  exit 1
fi
FOREGROUND=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--foreground" ]]; then FOREGROUND=1; else ARGS+=("$arg"); fi
done
if [[ ${#ARGS[@]} -eq 0 ]]; then ARGS=(--suite all); fi
mkdir -p outputs/logs
PIDFILE=outputs/logs/runner.pid
if runner_alive; then
  echo "A runner is already active (pid $(cat "$PIDFILE")). See: bash scripts/status.sh" >&2
  exit 1
fi
LOG="outputs/logs/run_$(date +%Y%m%d_%H%M%S).log"
ln -sfn "$(basename "$LOG")" outputs/logs/latest.log
if [[ "$FOREGROUND" == 1 ]]; then
  echo $$ > "$PIDFILE"
  trap 'rm -f "$PIDFILE"' EXIT
  set +e
  "$PY" -m takd.cli run "${ARGS[@]}" 2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi
if command -v setsid >/dev/null 2>&1; then
  setsid nohup "$PY" -m takd.cli run "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
else
  nohup "$PY" -m takd.cli run "${ARGS[@]}" >"$LOG" 2>&1 < /dev/null &
fi
echo $! > "$PIDFILE"
echo "Started runner pid $(cat "$PIDFILE") -> $LOG"
echo "  follow:  tail -f outputs/logs/latest.log"
echo "  status:  bash scripts/status.sh"
echo "  stop:    bash scripts/stop.sh      (rerun this script later to resume)"
