#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [ -n "${LLVM_ROOT:-}" ]; then
  exec "$LLVM_ROOT/bin/clangd" "$@"
fi

export LD_LIBRARY_PATH=/work/7/uw07387/opt/libstdcxx-gcc15/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec /work/7/uw07387/opt/LLVM-22.1.7-Linux-X64/bin/clangd "$@"
