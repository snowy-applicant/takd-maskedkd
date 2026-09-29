#!/usr/bin/env bash
# Print the run order and the analytic training FLOPs of every group (no GPU needed).
#   bash scripts/plan.sh               # both suites
#   bash scripts/plan.sh --suite hkd
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
require_venv
"$PY" -m takd.cli plan "$@"
