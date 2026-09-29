#!/usr/bin/env bash
# Download and validate COCO single (same selection and split as kshs-aimlab-benchmarks),
# build the 224x224 GPU cache and fetch the official DeiT-S / DeiT-B ImageNet weights.
# Resumable: rerun after an interruption. Extra arguments go to `takd.cli prepare`, e.g.
#   bash scripts/prepare_data.sh --data-root /abs/path/to/existing/coco_single --workers 16
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
require_venv
"$PY" -m takd.cli prepare "$@"
"$PY" -m takd.cli plan --suite all
