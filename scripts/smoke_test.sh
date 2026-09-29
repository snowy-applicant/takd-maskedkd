#!/usr/bin/env bash
# A few minutes, no download: unit tests, then the whole pipeline (teacher, both assistants,
# groups A-F, two seeds, both suites, report) with tiny models on synthetic data.
#   bash scripts/smoke_test.sh               # on the GPU for a CUDA install, else CPU
#   bash scripts/smoke_test.sh --device cpu
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
require_venv
"$PY" -m pytest -q
rm -rf outputs/smoke
DEVICE_ARGS=()
if [[ " $* " != *" --device "* ]] && [[ "$(cat .tools/torch-extra 2>/dev/null || echo cpu)" != "cpu" ]]; then
  DEVICE_ARGS=(--device cuda)   # a CUDA install must pass on the GPU, not silently fall back to CPU
fi
"$PY" -m takd.cli smoke "${DEVICE_ARGS[@]+"${DEVICE_ARGS[@]}"}" "$@"
