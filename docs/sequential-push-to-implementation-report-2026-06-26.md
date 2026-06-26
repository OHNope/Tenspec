# TypedSequential push_back/to Implementation Comparison

Date: 2026-06-26

## Summary

This report compares three implementation strategies for `TypedSequentialImpl::push_back` and `TypedSequentialImpl::to`:

1. `std::apply`: the original implementation style.
2. `template_for_assign`: the proposed `template for` implementation that default-constructs a result tuple and assigns each element.
3. `index_sequence`: an alternative implementation that avoids `std::apply` while still directly constructing the result module from expanded tuple elements.

The main conclusion is that the proposed `template_for_assign` form is not a good replacement. Its binary size and runtime cost increase because it default-constructs the whole result tuple and then assigns into it. The `index_sequence` implementation avoids that issue and performs similarly to, or slightly better than, the original `std::apply` implementation in this test.

## Tested Implementations

### Original std::apply

The original form uses `std::apply` to expand `modules_` into constructor arguments:

```cpp
template <TypedModuleHolder NewModule>
[[nodiscard]] inline auto push_back(NewModule module) const
{
    using ResultImpl = TypedSequentialImpl<
        Modules..., typename[: ::std::meta::remove_cvref(^^NewModule):]>;

    return ::std::apply(
        [&module](auto const &...modules) {
            return ::std::make_shared<ResultImpl>(
                modules..., ::std::move(module));
        },
        modules_);
}
```

The corresponding `to` implementation also uses `std::apply` and expands `modules.template to<...>()...` directly into the result constructor.

### Proposed template_for_assign

The proposed form uses `template for`, but it first default-constructs the result tuple and then assigns each element:

```cpp
using ResultModules = typename ResultImpl::Modules;
ResultModules result_modules{};

template for (constexpr ::std::size_t I :
    ::std::views::iota(::std::size_t{}, sizeof...(Modules)))
{
    ::std::get<I>(result_modules) = ::std::get<I>(modules_);
}

::std::get<sizeof...(Modules)>(result_modules) = ::std::move(module);

return ::std::make_shared<ResultImpl>(::std::move(result_modules));
```

This form changes the object-lifetime pattern. It requires default construction of every destination module holder before assigning the desired values. For module holders such as `Linear`, this is not a trivial operation. It pulls in default construction, assignment, destruction, and shared ownership/module registration related code paths that are not needed by direct construction.

### index_sequence Direct Construction

The improved non-`std::apply` implementation uses `std::index_sequence` to expand `std::get<I>(modules_)...` directly into the result constructor:

```cpp
template <class ResultImpl, TypedModuleHolder NewModule, ::std::size_t... I>
[[nodiscard]] inline auto push_back_index_sequence(
    NewModule module, ::std::index_sequence<I...>) const
{
    return ::std::make_shared<ResultImpl>(
        ::std::get<I>(modules_)..., ::std::move(module));
}

template <TypedModuleHolder NewModule>
[[nodiscard]] inline auto push_back(NewModule module) const
{
    using StoredNewModule =
        typename[: ::std::meta::remove_cvref(^^NewModule):];
    using ResultImpl = TypedSequentialImpl<Modules..., StoredNewModule>;

    return push_back_index_sequence<ResultImpl>(
        ::std::move(module),
        ::std::make_index_sequence<sizeof...(Modules)>{});
}
```

For `to`:

```cpp
template <class ResultImpl, Device ndevice, DType ndtype,
          ::std::size_t... I>
[[nodiscard]] inline auto to_index_sequence(::std::index_sequence<I...>) const
{
    return ::std::make_shared<ResultImpl>(
        ::std::get<I>(modules_).template to<ndevice, ndtype>()...);
}
```

This preserves the important property of the original `std::apply` implementation: the result object is constructed directly from the final arguments.

## Test Setup

A temporary benchmark source was placed under `tests/nnmodules/` and compiled through three temporary xmake targets:

- `typetorch_nnmodules_sequential_push_to_apply_compare`
- `typetorch_nnmodules_sequential_push_to_template_for_compare`
- `typetorch_nnmodules_sequential_push_to_index_sequence_compare`

The benchmark used a four-layer base sequence:

```cpp
using L0 = typetorch::Linear<16, 32>;
using L1 = typetorch::Linear<32, 32>;
using L2 = typetorch::Linear<32, 16>;
using L3 = typetorch::Linear<16, 8>;
using L4 = typetorch::Linear<8, 4>;
```

It measured:

- compile time by rebuilding each target with `xmake build -j 1`
- runtime for repeated `push_back` and `to` calls
- executable file size
- `.text` section size
- test object file size
- filtered object symbol bytes for `CompareSequentialImpl`, `push_back`, `to`, and related shared pointer control block symbols

The final command used for the three-way comparison was:

```bash
cd /work/7/uw07387/libtorch-reflect-wrapper
MODE=release BUILD_JOBS=1 ROUNDS=3 RUNTIME_ROUNDS=5 BENCH_ITERS=100 \
  tests/nnmodules/run_sequential_push_to_compare.sh
```

The test was run on the login node, so compile-time values should be treated as approximate. The size and object-code trends are more stable.

## Results

### Compile Time

| implementation | mean compile time |
| --- | ---: |
| `std::apply` | 18.479575 s |
| `template_for_assign` | 21.208624 s |
| `index_sequence` | 18.446941 s |

`std::apply` and `index_sequence` were effectively tied. `template_for_assign` was slower in this run.

### Runtime

Each runtime row is the mean of 5 runs, with 100 iterations per run.

| implementation | `push_back` mean | `to` mean |
| --- | ---: | ---: |
| `std::apply` | 548685 ns | 8031514 ns |
| `template_for_assign` | 4311781 ns | 9493499 ns |
| `index_sequence` | 566500 ns | 5880328 ns |

`template_for_assign` was much slower for `push_back`. `index_sequence` was similar to `std::apply` for `push_back` and faster for `to` in this test.

### Size

| implementation | executable file | object file | object symbols | `.text` |
| --- | ---: | ---: | ---: | ---: |
| `std::apply` | 109296 | 932888 | 293 | 93944 |
| `template_for_assign` | 113392 | 969144 | 300 | 97484 |
| `index_sequence` | 109296 | 931536 | 293 | 93881 |

`template_for_assign` increased executable size, object size, symbol count, and `.text` size. `index_sequence` was slightly smaller than `std::apply` in object and `.text` size.

### Filtered Symbol Bytes

| implementation | filtered bytes | filtered symbols |
| --- | ---: | ---: |
| `std::apply` | 17597 | 56 |
| `template_for_assign` | 21192 | 55 |
| `index_sequence` | 17612 | 56 |

The proposed `template_for_assign` version generated roughly 3.6 KB more filtered symbol bytes than the direct-construction forms.

## Why template_for_assign Increased Binary Size

The size increase was caused by the construction strategy, not by `template for` itself.

The `template_for_assign` version performs these extra steps:

1. Default-construct a full `ResultModules` tuple.
2. Default-construct every module holder inside that tuple.
3. Assign each existing module holder into the default-constructed destination holder.
4. Assign the new module into the last tuple slot.
5. Move the tuple into `ResultImpl`.

For `torch::nn::ModuleHolder`-style objects, default construction and assignment are not zero-cost operations. They involve shared ownership state and can pull in destructor and assignment paths. In the symbol output, the larger implementation included tuple-constructor paths such as constructors from `std::tuple<...>` and larger `push_back`/`to` bodies. It also retained more code related to constructing, assigning, and cleaning up intermediate module holder state.

By contrast, both `std::apply` and `index_sequence` directly construct `ResultImpl` from the final module arguments. They do not create a default result tuple and then mutate it.

## Recommendation

Do not replace the current `std::apply` implementation with the proposed `template_for_assign` form.

If the goal is minimal code churn and clear standard-library expression, keep `std::apply`.

If the goal is to avoid `std::apply` while keeping direct construction, use the `index_sequence` helper form. Based on this benchmark, `index_sequence` has the best overall result:

- compile time effectively tied with `std::apply`
- `.text` slightly smaller than `std::apply`
- object file slightly smaller than `std::apply`
- `push_back` runtime effectively tied with `std::apply`
- `to` runtime better than `std::apply` in this test

`template for` is still appropriate for statement-level repeated work such as `register_module` loops or `get<I>`-based operations where no parameter-pack expression is needed. For this particular case, the operation is naturally a constructor argument pack, and `index_sequence` is the better tool.
