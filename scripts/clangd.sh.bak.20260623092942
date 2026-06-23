#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [ -n "${LLVM_ROOT:-}" ]; then
  exec "$LLVM_ROOT/bin/clangd" "$@"
fi

exec clangd "$@"
