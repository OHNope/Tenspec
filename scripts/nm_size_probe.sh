#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"

mode="${1:-debug}"
max_lines="${MAX_LINES:-80}"
native_target="binary_size_libtorch_probe"
typed_target="binary_size_tensor_probe"

xmake f -m "$mode" --yes --python="$PYTHON"
xmake build "$native_target"
xmake build "$typed_target"

native_probe="build/linux/x86_64/$mode/$native_target"
typed_probe="build/linux/x86_64/$mode/$typed_target"
if [ ! -x "$native_probe" ] || [ ! -x "$typed_probe" ]; then
  echo "failed to locate size probe binaries under build/" >&2
  exit 1
fi

echo "-- binaries --"
ls -lh "$native_probe" "$typed_probe"

echo
echo "-- size --"
size "$native_probe" "$typed_probe" || true

echo
echo "-- native project symbols --"
nm -S --size-sort --demangle "$native_probe" 2>/dev/null |
  sed -n '/tenspec_binary_size_libtorch_probe/p' |
  tail -n "$max_lines"

echo
echo "-- tenspec project symbols --"
nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
  sed -n '/tenspec/p' |
  tail -n "$max_lines"

echo
echo "-- symbol summary --"
native_symbol_count="$(
  nm -S --size-sort --demangle "$native_probe" 2>/dev/null |
    awk '/tenspec_binary_size_libtorch_probe/ { count += 1 } END { print count + 0 }'
)"
native_symbol_bytes="$(
  nm -S --size-sort --demangle "$native_probe" 2>/dev/null |
    awk '/tenspec_binary_size_libtorch_probe/ { bytes += strtonum("0x" $2) } END { print bytes + 0 }'
)"
typed_probe_symbol_count="$(
  nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
    awk '/tenspec_binary_size_tensor_probe/ { count += 1 } END { print count + 0 }'
)"
typed_probe_symbol_bytes="$(
  nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
    awk '/tenspec_binary_size_tensor_probe/ { bytes += strtonum("0x" $2) } END { print bytes + 0 }'
)"
tensor_symbol_count="$(
  nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
    awk '! /tenspec_binary_size_tensor_probe/ && /tenspecW7tenspec6Tensor|tenspec10TensorBase/ { count += 1 } END { print count + 0 }'
)"
tensor_symbol_bytes="$(
  nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
    awk '! /tenspec_binary_size_tensor_probe/ && /tenspecW7tenspec6Tensor|tenspec10TensorBase/ { bytes += strtonum("0x" $2) } END { print bytes + 0 }'
)"
tensor_unique_symbol_count="$(
  nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
    awk '! /tenspec_binary_size_tensor_probe/ && /tenspecW7tenspec6Tensor|tenspec10TensorBase/ && !seen[$1]++ { count += 1 } END { print count + 0 }'
)"
tensor_unique_symbol_bytes="$(
  nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
    awk '! /tenspec_binary_size_tensor_probe/ && /tenspecW7tenspec6Tensor|tenspec10TensorBase/ && !seen[$1]++ { bytes += strtonum("0x" $2) } END { print bytes + 0 }'
)"
runtime_check_count="$(
  nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
    awk '/check_shape_compatible_runtime|check_tensor_compatible_runtime|dtype_matches|device_matches|layout_matches/ { count += 1 } END { print count + 0 }'
)"
runtime_check_bytes="$(
  nm -S --size-sort --demangle "$typed_probe" 2>/dev/null |
    awk '/check_shape_compatible_runtime|check_tensor_compatible_runtime|dtype_matches|device_matches|layout_matches/ { bytes += strtonum("0x" $2) } END { print bytes + 0 }'
)"

printf "native_probe_functions bytes=%s symbols=%s\n" "$native_symbol_bytes" "$native_symbol_count"
printf "typed_probe_functions bytes=%s symbols=%s\n" "$typed_probe_symbol_bytes" "$typed_probe_symbol_count"
printf "typed_tensor_wrapper bytes=%s symbols=%s\n" "$tensor_symbol_bytes" "$tensor_symbol_count"
printf "typed_tensor_wrapper_unique bytes=%s symbols=%s\n" "$tensor_unique_symbol_bytes" "$tensor_unique_symbol_count"
printf "typed_runtime_checks bytes=%s symbols=%s\n" "$runtime_check_bytes" "$runtime_check_count"
