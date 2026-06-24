[![CI](https://github.com/OHNope/Typetorch/actions/workflows/ci.yml/badge.svg)](https://github.com/OHNope/Typetorch/actions/workflows/ci.yml)
# Typetorch

Typetorch is an experimental C++26 library that makes tensor shape, dtype, device, and layout part of
the **C++ type system**. It wraps LibTorch's `torch::Tensor` in a move-only typed contract layer:
shape mismatches, dtype conflicts, and layout violations are caught at **compile time** rather than
at runtime. LibTorch still owns all storage, kernels, autograd, and device execution.

## At a Glance

```cpp
import typetorch;

// Define typed tensor aliases — these ARE the types
using Matrix = typetorch::Tensor<
    typetorch::Shape<2, 3>,          // 2x3 matrix — both dims known at compile time
    typetorch::DType::F32,           // float32
    typetorch::Device::CPU,          // CPU tensor
    typetorch::Layout::Contiguous>;  // dense row-major storage

using Vector  = typetorch::Tensor<typetorch::Shape<typetorch::dyn>, ...>;
//                                     ^^^ runtime-only dimension

// Enter typed code: validate raw tensor at the boundary
auto raw  = torch::arange(6, opts).view({2, 3});
auto x    = Matrix::retain(raw);   // runtime contract check
auto y    = Matrix::ones();        // factory, no raw tensor needed
auto z    = x.add(y);             // shapes match -> compiles, result type computed statically
auto flat = x.flatten<>();        // Shape<2,3> -> Shape<6>, all at compile time

// Leave typed code: recover the raw handle
torch::Tensor out = std::move(flat).unwrap();
```

## Why Typetorch

In ordinary LibTorch code, tensor metadata is dynamic:
- `x.sizes()` returns a runtime vector.
- `x.add(y)` succeeds or fails at runtime based on broadcast compatibility.
- `x.view({...})` throws if the view is invalid.

Typetorch moves these checks **earlier**:
- Shape, dtype, device, and layout are template parameters of `Tensor`.
- Operations compute result types at compile time via `consteval` reflection.
- Mismatches become `static_assert` failures or concept violations — the code
  never reaches the linker.

The runtime cost is **zero**: all wrapper methods are unconditionally inlined,
forwarding directly to the underlying LibTorch call.

## The Type System

```cpp
typetorch::Tensor</*Shape*/ Shape, /*DType*/ DType, /*Device*/ Device, /*Layout*/ Layout>
```

| Parameter  | Values | Meaning |
| --- | --- | --- |
| `Shape` | `Shape<2, 3, 4>`, `Shape<dyn, 3>`, ... | Static dimensions; use `typetorch::dyn` (-2) for runtime-only dims, `typetorch::infer` (-1) for view inference |
| `DType` | `Any`, `F32`, `F16`, `BF16`, `I64`, `Bool` | Element type |
| `Device` | `Any`, `CPU`, `CUDA` | Device placement |
| `Layout` | `Any`, `Contiguous`, `NonContiguous` | Memory layout; many ops preserve or degrade this statically |

Type aliases keep code readable:

```cpp
using Matrix = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
                                 typetorch::Device::CPU, typetorch::Layout::Contiguous>;
```

## What Compiles

These are the key compile-time guarantees Typetorch provides. See
[docs/examples.md](docs/examples.md) for the full catalog.

### Broadcasting

```cpp
using Matrix23 = Tensor<Shape<2, 3>, DType::F32, Device::CPU, Layout::Contiguous>;
using Vector3  = Tensor<Shape<3>,    DType::F32, Device::CPU, Layout::Contiguous>;

auto y = Matrix23::ones().add(Vector3::ones());
// OK: Shape<3> broadcasts with Shape<2,3> -> result is Shape<2,3>
```

### matmul

```cpp
using In  = Tensor<Shape<dyn, 3>, DType::F32, Device::CPU, Layout::Any>;
using Wt  = Tensor<Shape<3, 2>,   DType::F32, Device::CPU, Layout::Any>;

auto y = In::retain(raw_input).matmul(Wt::retain(raw_weight));
// OK: inner dims (3,3) match -> result Shape<dyn, 2>
```

### transpose/permute tracks layout

```cpp
using Cube = Tensor<Shape<2, 3, 4>, DType::F32, Device::CPU, Layout::Contiguous>;

auto t  = std::move(Cube::ones()).transpose<0, 1>();
// OK: Shape<3,2,4>, Layout::Any (transpose breaks contiguity)

auto id = std::move(Cube::ones()).permute<0, 1, 2>();
// OK: identity permute — Layout::Contiguous preserved
```

### view checks element count at compile time

```cpp
auto v = std::move(Matrix23::ones()).view<6>();
// OK: 2 x 3 = 6 elements -> Shape<6>

auto v2 = std::move(Matrix23::ones()).view<typetorch::infer>();
// OK: -1 inference -> Shape<6>
```

## What Does NOT Compile

These are the errors Typetorch exists to catch. See
[docs/examples.md](docs/examples.md) for more cases.

### Shape mismatch

```cpp
auto z = Matrix23::ones().add(Matrix32::ones());
// ERROR: static assertion failed: tensor shapes are not broadcast-compatible
//   Shape<2,3> vs Shape<3,2>: dim 0 is 2 vs 3, neither is 1
```

### matmul inner dimension

```cpp
auto z = Tensor<Shape<2,3>>::ones().matmul(Tensor<Shape<2,4>>::ones());
// ERROR: consteval throw — A's last dim (3) != B's second-to-last dim (2)
```

### view with wrong element count

```cpp
std::move(Matrix23::ones()).view<3, 3>();
// ERROR: consteval throw — 2x3=6 != 3x3=9
```

### Factory on dynamic shape

```cpp
Tensor<Shape<dyn>>::zeros();
// ERROR: static assertion failed — requires a fully static shape
```

### Integer dtype on float-only ops

```cpp
Tensor<Shape<2,3>, DType::I64>::ones().gelu();
// ERROR: static assertion — gelu() requires floating tensor dtype
```

### View on non-contiguous tensor

```cpp
auto t = std::move(cube).transpose<0, 2>();
std::move(t).view<2, 4, 3>();
// ERROR: consteval throw — Tensor should be contiguous!
// Fix: call .contiguous() first
```

### Conservative widening (not an error, but important)

```cpp
auto z = F32Mat::ones().add(I64Mat::ones());   // result dtype -> DType::Any
auto z = CPUMat::ones().add(CUDAMat::ones());  // result device -> Device::Any
// Different dtypes or devices widen to Any rather than erroring.
// This is intentional but worth knowing: Any propagates through subsequent ops.
```

### Runtime retain with wrong shape

```cpp
auto raw = torch::arange(9, opts).view({3, 3});
auto x = Matrix23::retain(raw);
// RUNTIME ERROR: expected shape <2,3>, got <3,3>
```

## Trust Boundary: retain vs unsafe

Typetorch provides two paths for raw tensor entry:

| Method | Behavior | When to use |
| --- | --- | --- |
| `Tensor::retain(raw)` | Runtime checks: shape, dtype, device, layout | At API boundaries, untrusted inputs |
| `Tensor::unsafe_retain(raw)` | No checks, zero overhead | Inside trusted code where contract is already established |
| `Tensor::unsafe_move(raw&&)` | No checks, moves the raw handle | Same, but avoids the retain bump |

```cpp
// Trusted internal code — zero-overhead path
auto typed = Matrix::unsafe_retain(internal_tensor);
auto result = typed.add(Matrix::ones());

// Untrusted boundary — checked path
auto typed2 = Matrix::retain(user_provided_tensor);
```

`retain()` internally calls `check_tensor_compatible_runtime`, which validates
all four contract parameters against the raw tensor's actual metadata. After
validation, it delegates to `unsafe_retain`.

## Move-Only Semantics

`Tensor<...>` is move-only. This is intentional:

```cpp
auto a = Matrix::ones();
auto b = a;              // ERROR: deleted copy constructor
auto c = std::move(a);   // OK: move is allowed
```

Use `unsafe_raw()` for const reference access without giving up ownership:

```cpp
void inspect(Matrix const &x) {
    torch::Tensor const &raw = x.unsafe_raw();  // OK: read-only access
}
```

Use `unwrap()` to permanently leave the typed world:

```cpp
torch::Tensor raw = std::move(typed).unwrap();  // OK: typed handle consumed
```

## Operation Quick Reference

See [docs/api.md](docs/api.md) for the full operation catalog and contract details.

## Requirements

| Component | Requirement |
| --- | --- |
| C++ compiler | GCC 16.1+ with `-fmodules -freflection` |
| Build system | xmake |
| LibTorch | 2.8.0 + CUDA 12.8 |
| Python headers | For `typetorch_capi_ext` target only |

## Quick Start

### 1. Clone and prepare

```bash
git clone <repo-url> typetorch
cd typetorch
git submodule update --init --recursive
```

### 2. Configure local paths

Typetorch needs to know where your GCC 16.1+ toolchain lives. Create two config
files (both are git-ignored):

```bash
cp scripts/env.local.example.sh scripts/env.local.sh
cp typetorch.local.example.lua typetorch.local.lua
```

**`scripts/env.local.sh`** — shell environment, sourced by `scripts/env.sh`:

```bash
# REQUIRED: path to your GCC 16.1+ installation prefix.
# $GCC_ROOT/bin/gcc and $GCC_ROOT/bin/g++ must exist.
export GCC_ROOT="/home/7/uw07387/gcc-16.1.0"

# OPTIONAL: if your GCC was built as a relocated toolchain (common in HPC
# environments), set these so the linker finds crti.o, crtbeginS.o, etc.
# Leave unset if `gcc -v` shows the right sysroot automatically.
export GCC_TOOLCHAIN_ROOT="/home/7/uw07387/x-tools/x86_64-focal-linux-gnu"
export GCC_TRIPLET="x86_64-focal-linux-gnu"
export GCC_VERSION="16.1.0"
export GCC_SYSROOT="$GCC_TOOLCHAIN_ROOT/$GCC_TRIPLET/sysroot"

# OPTIONAL: extra GCC runtime library path if libstdc++.so is not on the
# default linker search path.
export GCC_RUNTIME_LIB="$GCC_TOOLCHAIN_ROOT/$GCC_TRIPLET/lib64"

# OPTIONAL: custom xmake installation prefix (adds $XMAKE_ROOT/bin to PATH).
export XMAKE_ROOT="/home/7/uw07387/xmake"

# OPTIONAL: Python interpreter. scripts/env.sh auto-detects the include dir.
export PYTHON="python3"
```

Only `GCC_ROOT` is mandatory. If you installed GCC from a tarball into a single
prefix and `gcc -v` prints reasonable `--with-sysroot` and `COLLECT_GCC_OPTIONS`,
you likely only need the first line.

**`typetorch.local.lua`** — xmake configuration, included by `xmake.lua`:

```lua
typetorch_config({
    -- REQUIRED: same value as GCC_ROOT above.
    gcc_root = "/home/7/uw07387/gcc-16.1.0",

    -- OPTIONAL: same values as the shell env vars above.
    -- Only needed when the GCC driver is relocated.
    gcc_toolchain_root = "/home/7/uw07387/x-tools/x86_64-focal-linux-gnu",
    gcc_triplet = "x86_64-focal-linux-gnu",
    gcc_version = "16.1.0",
    gcc_sysroot = "/home/7/uw07387/x-tools/x86_64-focal-linux-gnu/x86_64-focal-linux-gnu/sysroot",

    -- OPTIONAL: escape hatch for manual -B/--sysroot linker flags.
    -- Only set this if the toolchain-root settings above are not enough.
    gcc_relocate_ldflags = "",

    -- OPTIONAL: Python include dir; auto-detected in most cases.
    python_include_dir = "/home/7/uw07387/.venv/include/python3.12",
})
```

### 3. Build and run

```bash
source scripts/env.sh
bash scripts/check_env.sh

xmake f -m debug --yes
xmake build typetorch_cpp_debug
xmake run typetorch_cpp_debug

# Run tensor arithmetic tests
xmake build typetorch_tensor_arithmetic_test
xmake run typetorch_tensor_arithmetic_test
```

## Performance

Release-build forwarding overhead is within measurement noise for all core
operations — see [docs/performance-and-size.md](docs/performance-and-size.md) for
the full benchmark tables and methodology.

### Forwarding Benchmark (release, 2026-06-24, 10000 iterations)

| Operation | Raw ns/op | Typetorch ns/op | Ratio |
| --- | ---: | ---: | ---: |
| `add.dynamic` | 2201.94 | 2030.83 | 0.92 |
| `add.static` | 2244.88 | 2222.38 | 0.99 |
| `matmul.static` | 10176.55 | 10195.68 | 1.00 |
| `transpose.static` | 759.13 | 765.81 | 1.01 |
| `view.static` | 598.36 | 589.59 | 0.99 |
| `permute.static` | 882.68 | 879.24 | 1.00 |

All ratios are within 2.3% of 1.0 — **zero-cost abstraction confirmed.**

### Binary Size (release, 2026-06-24, with attribute/extensions)

Compiler attributes and `--gc-sections` reduce binary sizes by 32–64%
compared to the previous snapshot.

| Target | text | data | bss | dec | hex |
| --- | ---: | ---: | ---: | ---: | ---: |
| `typetorch_forwarding_benchmark` | 49430 | 1808 | 200 | 51438 | c8ee |
| `binary_size_libtorch_probe` | 22356 | 1696 | 200 | 24252 | 5ebc |
| `binary_size_tensor_checked_probe` | 30117 | 1728 | 200 | 32045 | 7d2d |
| `binary_size_tensor_unsafe_probe` | 22236 | 1696 | 200 | 24132 | 5e44 |
| `typetorch_tensor_arithmetic_test` | 84238 | 2432 | 200 | 86870 | 15356 |

Key comparisons:
- **unsafe vs native** (22236 vs 22356): **-120 bytes (-0.5%)** — Typetorch forwarding is a true zero-cost abstraction.
- **checked vs native** (30117 vs 22356): +7761 bytes (+34.7%) — cost of Typetorch wrapper + runtime boundary validation.
- **checked vs unsafe** (30117 vs 22236): +7881 bytes (+35.4%) — isolated cost of runtime contract checks.

Arithmetic test (84238 text) exercises all op categories: arithmetic, view, NN, aggregate.
All arithmetic tests pass in both debug and release builds.

### Size Reduction vs Previous Snapshot

| Target | text before | text after | reduction |
| --- | ---: | ---: | ---: |
| `typetorch_forwarding_benchmark` | 73363 | 49430 | **-32.6%** |
| `binary_size_libtorch_probe` | 61968 | 22356 | **-63.9%** |
| `binary_size_tensor_checked_probe` | 66156 | 30117 | **-54.5%** |
| `binary_size_tensor_unsafe_probe` | 62488 | 22236 | **-64.4%** |
| `typetorch_tensor_arithmetic_test` | 124074 | 84238 | **-32.1%** |

The reduction is driven by `-ffunction-sections -fdata-sections -Wl,--gc-sections`
combined with `TYPETORCH_COLD` on error-reporting paths — the linker can discard
cold functions that are never called in a given binary.



## Using Typetorch in Your Project

Typetorch is currently distributed as a source dependency with a local xmake
package recipe. It is not yet published in the official xmake package
repository, so your project must clone or vendor this repository first.

### Requirements

- Linux x86-64
- GCC 16.1 or newer with C++26 modules and reflection support
- xmake
- Git

The current package recipe uses LibTorch 2.8.0 with CUDA 12.8. A GPU is not
required for CPU-only programs, but CUDA execution requires a compatible
NVIDIA driver.

### 1. Add Typetorch to your project

Using a Git submodule is recommended:

```bash
git submodule add https://github.com/OHNope/Typetorch.git third_party/typetorch
git submodule update --init --recursive
```

The resulting layout should be:

```text
your-project/
├── xmake.lua
├── src/
│   └── main.cpp
└── third_party/
    └── typetorch/
```

When cloning a project that already contains the submodule, use:

```bash
git clone --recurse-submodules <your-project-url>
```

### 2. Configure xmake

Add the local repository and Typetorch module rule to your project's
`xmake.lua`:

```lua
set_project("my-project")
set_languages("c++26")
set_policy("build.c++.modules.gcc.cxx11abi", true)

add_repositories("typetorch-repo third_party/typetorch")
includes("third_party/typetorch/xmake_package.lua")
add_requires("typetorch")

target("my_app")
    set_kind("binary")
    add_packages("typetorch")
    add_rules("typetorch.modules")
    add_files("src/main.cpp")
```

`add_packages("typetorch")` propagates the LibTorch include paths, libraries,
runtime paths, and required compiler options. `typetorch.modules` adds the
installed Typetorch module sources in dependency order. Do not list the module
files manually.

### 3. Import Typetorch

```cpp
// src/main.cpp
import std;
import typetorch;

using Matrix = typetorch::Tensor<typetorch::Shape<64, 128>,
                                 typetorch::DType::F32,
                                 typetorch::Device::CPU,
                                 typetorch::Layout::Contiguous>;

int main() {
    auto x = Matrix::randn();
    auto w = Matrix::randn();
    auto y = x.add(w);
    return std::move(y).unwrap().numel() == 64 * 128 ? 0 : 1;
}
```

Do not mix `import typetorch;` with textual inclusion of Typetorch or LibTorch
headers in the same translation unit. That can cause duplicate declarations.

### 4. Build and run

```bash
xmake f -m release --yes
xmake build my_app
xmake run my_app
```

The first configuration downloads and installs the pinned LibTorch package.
It is large. Later configurations reuse xmake's package cache unless the cache
is deleted or installation is explicitly forced.

If GCC is installed in a nonstandard or relocated location, configure the
compiler before running xmake:

```bash
export GCC_ROOT=/path/to/gcc-16.1.0
project_root="$PWD"
source third_party/typetorch/scripts/env.sh
cd "$project_root"

xmake f -m release --yes
xmake build my_app
xmake run my_app
```

`scripts/env.sh` changes into the Typetorch repository, which is why the
example saves and restores the consumer project's directory.

### Common issues

**"fatal error: module 'typetorch:types' not found"**

Use both `add_packages("typetorch")` and `add_rules("typetorch.modules")` on
the consuming target. Do not add the `.mpp` files manually.

**xmake downloads LibTorch again**

A normal `xmake f` reuses the package cache. Commands such as
`xmake require --force`, or deleting xmake's package cache, force LibTorch to
be downloaded and installed again.

**"crti.o: cannot open" or "crtbeginS.o: no such file"**

Your GCC is a relocated toolchain. Source typetorch's `scripts/env.sh` before
building, or set `TYPETORCH_GCC_RELOCATE_LDFLAGS` in your environment.

**"undefined reference to torch::..."**

Ensure the consuming target has `add_packages("typetorch")`. This propagates
the LibTorch link libraries.


## Documentation

- [Compile-Time Examples](docs/examples.md) — exhaustive what-compiles / what-does-not examples
- [API Notes](docs/api.md) — contract details, operation style, differences from LibTorch
- [Design Notes](docs/design.md) — architecture decisions
- [Performance and Size Notes](docs/performance-and-size.md) — benchmarks and binary analysis
