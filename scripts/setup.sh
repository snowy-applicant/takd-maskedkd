#!/usr/bin/env bash
# One-time setup of a repo-local environment:
#   .tools/uv/bin/uv      uv binary (no receipt, no PATH/profile edits)
#   .tools/python/        uv-managed CPython 3.12 (the system Python is never used)
#   .venv/                torch 2.14.0 (+cu126 or +cu130, chosen from the NVIDIA driver)
# Nothing is written outside this folder; no sudo is needed.
# Override the build with:  TAKD_TORCH_EXTRA=cu126|cu130|cpu bash scripts/setup.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO_ROOT"
UV_VERSION="0.12.20"

if [[ ! -x "$UV_BIN_DIR/uv" ]]; then
  echo "== Installing uv $UV_VERSION into $UV_BIN_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh
  else
    # Standard library only (system python3 is only used to download): verify the SHA-256.
    python3 "$REPO_ROOT/scripts/fetch_uv.py" "$UV_VERSION" "$UV_BIN_DIR"
  fi
fi
uv --version

PYTHON_VERSION="$(cat .python-version)"
echo "== Installing CPython $PYTHON_VERSION into $UV_PYTHON_INSTALL_DIR"
uv python install "$PYTHON_VERSION" --no-bin

EXTRA="${TAKD_TORCH_EXTRA:-}"
if [[ -z "$EXTRA" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1 | tr -dc '0-9')"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
    if [[ -z "$DRIVER_MAJOR" ]]; then
      echo "Could not read the NVIDIA driver version from nvidia-smi; set TAKD_TORCH_EXTRA=cu126 or cu130." >&2
      exit 1
    fi
    if (( DRIVER_MAJOR >= 580 )); then
      EXTRA=cu130
    elif (( DRIVER_MAJOR >= 525 )); then
      EXTRA=cu126
    else
      echo "NVIDIA driver $DRIVER_MAJOR is older than 525; torch 2.14 CUDA wheels cannot run." >&2
      exit 1
    fi
  else
    echo "WARNING: nvidia-smi not found; installing the CPU build (smoke tests only)." >&2
    EXTRA=cpu
  fi
fi
echo "$EXTRA" > .tools/torch-extra
echo "== Syncing .venv (torch extra: $EXTRA, plus pytest)"
if ! uv sync --locked --extra "$EXTRA"; then
  echo "uv.lock does not match pyproject.toml; resolving again without --locked" >&2
  uv sync --extra "$EXTRA"
fi

echo "== Checking the installation"
ldd --version 2>/dev/null | head -n1 || true
"$PY" "$REPO_ROOT/scripts/check_gpu.py"
echo "Setup complete. Next:  bash scripts/smoke_test.sh   then   bash scripts/prepare_data.sh"
