#!/usr/bin/env bash
# Rebuild results/<suite>/ (CSVs, SUMMARY.md, figures) from outputs/. Safe while training runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
require_venv
"$PY" -m takd.cli report "$@"
