# shellcheck shell=bash
# Source this file (the scripts do it for you); never add it to a shell profile.
# Keeps uv, the managed CPython, every cache and temp file inside the repository,
# so the server's system Python (3.14.7) and $HOME stay untouched.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export REPO_ROOT
export UV_BIN_DIR="$REPO_ROOT/.tools/uv/bin"
export PATH="$UV_BIN_DIR:$PATH"                               # this process only
export UV_CACHE_DIR="$REPO_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$REPO_ROOT/.tools/python"
export UV_PYTHON_BIN_DIR="$REPO_ROOT/.tools/python-bin"
export UV_PYTHON_INSTALL_BIN=0
export UV_PYTHON_CACHE_DIR="$REPO_ROOT/.cache/uv-python-archives"
export UV_PYTHON_PREFERENCE=only-managed                      # never use the system interpreter
export UV_PYTHON_DOWNLOADS=automatic
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv"
export UV_LINK_MODE=hardlink
export UV_TOOL_DIR="$REPO_ROOT/.tools/uv-tools"
export UV_TOOL_BIN_DIR="$REPO_ROOT/.tools/uv-tools/bin"
export UV_HTTP_TIMEOUT=300
export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"
export TORCH_HOME="$REPO_ROOT/.cache/torch"                   # DeiT ImageNet weights
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache/inductor"
export TRITON_HOME="$REPO_ROOT/.cache"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$REPO_ROOT/.cache/nv"
export MPLCONFIGDIR="$REPO_ROOT/.cache/matplotlib"
export TMPDIR="$REPO_ROOT/.tmp"
export PYTHONUNBUFFERED=1
export PY="$REPO_ROOT/.venv/bin/python"
mkdir -p "$UV_BIN_DIR" "$UV_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" \
         "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"

# True only if the pid file names a live process that is really our runner (guards
# against a stale pid file whose number the kernel has since given to another process).
runner_alive() {
  local pidfile="$REPO_ROOT/outputs/logs/runner.pid" pid
  [[ -f "$pidfile" ]] || return 1
  pid="$(cat "$pidfile")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r "/proc/$pid/cmdline" ]]; then
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -qE "takd\.cli run|scripts/run_all\.sh" || return 1
  fi
  return 0
}

require_venv() {
  if [[ ! -x "$PY" ]]; then
    echo "No environment yet: run  bash scripts/setup.sh" >&2
    exit 1
  fi
}
