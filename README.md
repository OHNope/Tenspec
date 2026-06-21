# Typetorch

Typetorch is an experimental C++26 project for type-safe LibTorch tensor wrappers
and lightweight Python C API bindings.

The core wrapper carries tensor shape, dtype, device, and layout information in
C++ types while forwarding runtime work to LibTorch. LibTorch still owns tensor
storage, kernels, dispatch, autograd, and device behavior; Typetorch adds a typed
contract layer around `torch::Tensor`.

The public LibTorch-facing surface now uses the C++ frontend namespace
(`torch::Tensor`, `torch::Scalar`, `torch::TensorOptions`, and `torch::*`
factory functions).

## Status

This repository is a research/prototype codebase. It intentionally depends on
very recent C++ compiler features, especially C++26 modules and static
reflection. Expect toolchain sensitivity.

## Requirements

Minimum practical requirements:

| Component | Requirement | Notes |
| --- | --- | --- |
| C++ compiler | GCC 16.1 or newer with modules and static reflection support | Older GCC releases are not expected to build this project. |
| Build system | xmake | Used for C++26 module builds and local package resolution. |
| LibTorch | 2.8.0 + CUDA 12.8 archive | The repository includes an xmake package recipe in `packages/l/libtorch-bin/`. |
| Python headers | CPython development headers | Required for `typetorch_capi_ext`. |
| Python runtime | Python with `torch` installed | Required only to import and exercise the extension. |
| OS/tooling | Linux, `nm`, `size`, common binutils | Measurement scripts assume Unix-like tooling. |

The repository does not hard-code a machine layout. Put machine-specific paths
in local config files that are ignored by git:

```bash
cp scripts/env.local.example.sh scripts/env.local.sh
cp typetorch.local.example.lua typetorch.local.lua
```

`scripts/env.sh` automatically sources `scripts/env.local.sh` when it exists.
`xmake.lua` automatically includes `typetorch.local.lua` when it exists.

Use `scripts/env.local.sh` for shell environment setup:

```bash
export GCC_ROOT=/path/to/gcc-16.1-or-newer
export GCC_TOOLCHAIN_ROOT=/path/to/x-tools/x86_64-focal-linux-gnu  # optional
export GCC_TRIPLET=x86_64-focal-linux-gnu                         # optional
export GCC_VERSION=16.1.0                                         # optional
export GCC_SYSROOT=/path/to/sysroot                               # optional
export GCC_RUNTIME_LIB=/path/to/gcc/runtime/lib64                 # optional
export XMAKE_ROOT=/path/to/xmake-prefix             # optional
export PYTHON=/path/to/python                       # optional
```

Use `typetorch.local.lua` for xmake-only configuration:

```lua
typetorch_config({
    gcc_root = "/path/to/gcc-prefix",
    gcc_toolchain_root = "/path/to/x-tools/x86_64-focal-linux-gnu",
    gcc_triplet = "x86_64-focal-linux-gnu",
    gcc_version = "16.1.0",
    gcc_sysroot = "/path/to/x-tools/x86_64-focal-linux-gnu/x86_64-focal-linux-gnu/sysroot",
    python_include_dir = "/path/to/python/include",
})
```

If your GCC driver cannot find startup objects such as `crti.o` or
`crtbeginS.o`, set `gcc_toolchain_root`/`gcc_sysroot` or the equivalent
environment variables above. As an escape hatch, `gcc_relocate_ldflags` or
`TYPETORCH_GCC_RELOCATE_LDFLAGS` may contain the exact `-B... --sysroot=...`
flags for your installation.

## Build From Scratch

Clone and initialize submodules:

```bash
git clone <repo-url> typetorch
cd typetorch
git submodule update --init --recursive
```

Prepare the environment:

```bash
source scripts/env.sh
bash scripts/check_env.sh
```

Configure and run the tensor API test:

```bash
xmake f -m debug --yes --python="${PYTHON:-python3}"
xmake build typetorch_tensor_arithmetic_test
xmake run typetorch_tensor_arithmetic_test
```

Build and run the C++ debug example:

```bash
xmake build typetorch_cpp_debug
xmake run typetorch_cpp_debug
```

Build the Python extension:

```bash
xmake build typetorch_capi_ext
```

The extension is emitted under `build/<platform>/<arch>/<mode>/` with basename
`typetorch_capi`.

For release measurements:

```bash
xmake f -m release --yes --python="${PYTHON:-python3}"
xmake build typetorch_forwarding_benchmark
xmake build binary_size_libtorch_probe
xmake build binary_size_tensor_checked_probe
xmake build binary_size_tensor_unsafe_probe
```

## API Sketch

```cpp
import typetorch;

using Matrix = typetorch::Tensor<
    typetorch::Shape<2, 3>,
    typetorch::DType::F32,
    typetorch::Device::CPU,
    typetorch::Layout::Contiguous>;

auto options = torch::TensorOptions{}.dtype(torch::kFloat).device(torch::kCPU);
torch::Tensor raw = torch::arange(6, options).view({2, 3});
auto typed = Matrix::retain(raw);        // runtime contract check
auto flat = typed.flatten<>();           // result type is Shape<6>
torch::Tensor back = std::move(flat).unwrap();
```

Important differences from ordinary LibTorch code:

| Topic | LibTorch | Typetorch |
| --- | --- | --- |
| Tensor type | Mostly dynamic `torch::Tensor` metadata | Shape, dtype, device, and layout are part of the C++ type. |
| Shape errors | Usually runtime errors | Statically known shape mismatches fail during compilation. |
| Dynamic dimensions | Tensor sizes are runtime values | `dyn` marks dimensions that remain runtime-checked. |
| Ownership style | `torch::Tensor` is copyable handle type | `Tensor<...>` is move-only; use `unwrap()` to recover raw LibTorch. |
| Trust boundary | Any `torch::Tensor` can flow anywhere | `retain()` validates raw tensors before they enter typed code. |
| Layout | Runtime property | `Layout::Contiguous`, `Layout::NonContiguous`, or `Layout::Any` can be tracked. |

See [docs/api.md](docs/api.md) for the supported operation style and the places
where Typetorch intentionally differs from LibTorch.

## Current Measurement Snapshot

These numbers are a snapshot from the `Tensor`-only wrapper layout after
removing the former `TensorBase` layer. Reproduce them with the commands in
[docs/performance-and-size.md](docs/performance-and-size.md). Treat them as
indicative, not as a stable benchmark promise.

### Forwarding Performance

Release build, CPU tensors, 1 thread, 1000 iterations. Ratio is
`Typetorch / raw torch::Tensor`; values near 1.0 mean no measurable forwarding loss.

| Operation | Raw ns/op | Typetorch ns/op | Ratio | Overhead |
| --- | ---: | ---: | ---: | ---: |
| `add.dynamic` | 1984.773 | 1977.399 | 0.9963 | -0.37% |
| `add.static` | 2008.366 | 1990.744 | 0.9912 | -0.88% |
| `matmul.static` | 12711.788 | 12760.018 | 1.0038 | +0.38% |
| `transpose.static` | 994.180 | 998.066 | 1.0039 | +0.39% |
| `view.static` | 748.612 | 745.286 | 0.9956 | -0.44% |
| `permute.static` | 1125.074 | 1112.860 | 0.9891 | -1.09% |

### Symbol Growth Probe

Binary-size probes now separate the two Typetorch usage modes that matter for
interpretation:

| Target | Role |
| --- | --- |
| `binary_size_libtorch_probe` | Native LibTorch baseline using only `torch::Tensor`. |
| `binary_size_tensor_checked_probe` | Fully checked Typetorch boundary: raw tensors enter through `retain()`. |
| `binary_size_tensor_unsafe_probe` | Fully trusted Typetorch boundary: raw tensors enter through `unsafe_retain()`. |
| `binary_size_tensor_probe` | Legacy mixed probe; keep only for comparing older measurements. |

Run the symbol report locally on an interactive machine:

```bash
bash scripts/nm_size_probe.sh debug
```

All three current probes print `16` and run the same high-level
retain/add/unwrap/numel flow. Compare checked vs native for the full contract
cost, unsafe vs native for trusted forwarding overhead, and checked vs unsafe
for the isolated runtime validation cost.

### Binary Size Snapshot

Build release measurement targets manually; they are intentionally not part of
CI:

```bash
source scripts/env.sh
xmake f -m release --yes --python="${PYTHON:-python3}"
xmake build typetorch_forwarding_benchmark
xmake build binary_size_libtorch_probe
xmake build binary_size_tensor_checked_probe
xmake build binary_size_tensor_unsafe_probe
size \
  build/linux/x86_64/release/binary_size_libtorch_probe \
  build/linux/x86_64/release/binary_size_tensor_checked_probe \
  build/linux/x86_64/release/binary_size_tensor_unsafe_probe
```

Current release snapshot:

| Target | text | data | bss | dec | hex |
| --- | ---: | ---: | ---: | ---: | ---: |
| `binary_size_libtorch_probe` | 61973 | 1712 | 200 | 63885 | f98d |
| `binary_size_tensor_checked_probe` | 65873 | 1744 | 200 | 67817 | 108e9 |
| `binary_size_tensor_unsafe_probe` | 62205 | 1712 | 200 | 64117 | fa75 |

The older `binary_size_tensor_probe` mixed checked and unsafe retain paths, so
it should not be used as the headline Typetorch-vs-LibTorch comparison.

## Documentation

- [API Notes](docs/api.md)
- [Design Notes](docs/design.md)
- [Performance and Size Notes](docs/performance-and-size.md)
- [Implementation Report](docs/implementation-report-2026-06-06.md)
