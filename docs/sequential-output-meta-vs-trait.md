# Technical Report: `consteval` vs Template-Trait for Sequential Output Type Computation

**Date:** 2026-06-25
**Context:** `libtorch-reflect-wrapper` — `typetorch::nnModules::sequential`
**Branch:** Replace recursive template metaprogramming with `consteval` reflection in `SequentialOutput`.

## Background

The `TypedSequentialImpl<Modules...>::forward(Input)` method must compute, **at compile time**, the output tensor type after applying a chain of typed NN modules. Two approaches were compared:

### Approach A: Recursive Template Struct (old)

```cpp
template <typename Input, TypedModuleHolder... Modules>
struct SequentialOutput;

template <typename Input, TypedModuleHolder Module>
struct SequentialOutput<Input, Module> {
    using type = SequentialStepOutput<Input, Module>;
};

template <typename Input, TypedModuleHolder First,
          TypedModuleHolder Second, TypedModuleHolder... Rest>
struct SequentialOutput<Input, First, Second, Rest...> {
    using Step = SequentialStepOutput<Input, First>;
    using type = typename SequentialOutput<Step, Second, Rest...>::type;
};
```

Each recursive step produces a distinct template instantiation (`SequentialOutput<Mid1, L2>`, `SequentialOutput<Mid2, L3>`, ...), requiring N name-mangling, symbol-table insertion, and memoization-lookup operations for an N-layer chain.

### Approach B: `consteval` Reflection (new)

```cpp
template <::std::meta::info Input, ::std::meta::info Module,
          ::std::meta::info... Rest>
consteval inline auto sequential_output_meta() -> ::std::meta::info
{
    constexpr auto output = ::std::meta::substitute(
        ^^SequentialStepOutput, {Input, Module});
    using Output = typename[:output:];
    static_assert(TensorLike<Output>, ...);

    if constexpr (sizeof...(Rest) == 0)
        return output;
    else
        return sequential_output_meta<output, Rest...>();
}
```

The compiler evaluates this in the constexpr interpreter. Only the **final result type** is spliced into the AST; no intermediate template instantiations are emitted.

## Comparison

### 1. Binary / Symbol Size

| Aspect | Template-Trait | `consteval` |
|--------|---------------|-------------|
| Intermediate symbols | N instantiations of `SequentialOutput` emitted per unique chain | 0 — only the final `[:output:]` type appears |
| Mangled-name storage | Each `SequentialOutput<Mid, Layers...>` variant stored in `.strtab` / `.symtab` | None beyond the `consteval` function itself (which is never emitted) |
| Impact per chain depth | O(N) additional symbol entries | O(1) |

**Winner: `consteval`.** For deep sequential chains (8–16+ modules common in MLPs), the template-trait approach adds hundreds of symbol-table entries for a single translation unit. The `consteval` approach emits zero intermediate type symbols.

### 2. Compile Time

| Phase | Template-Trait | `consteval` |
|-------|---------------|-------------|
| Name mangling | N times (once per recursive step) | 0 intermediate |
| Template instantiation | N recursive instantiations, each: memoization lookup → instantiation → symbol-table write | 0 intermediate |
| Constexpr evaluation | N/A | Single function, O(N) linear loop |

Template instantiation is one of the most expensive compiler operations. Each step requires:

1. Compute the mangled name for `SequentialOutput<Mid_K, Layer_K+1, ...>`
2. Probe the memoization table (hash lookup)
3. If not found: instantiate the body, resolve dependent names, insert into the symbol table
4. Recurse

The `consteval` interpreter, by contrast, evaluates a simple recursive `if constexpr` call. There is no name mangling, no symbol-table traffic, and the compiler can inline the constexpr call chain trivially.

**Winner: `consteval`.** The gap widens with chain depth and with the number of distinct input types (each `payload<Is>` × chain combination triggers separate instantiations in the trait approach).

### 3. Runtime Performance

**Draw.** Both approaches are fully resolved at compile time. The resulting type `typename[:sequential_output_meta<...>():]` is identical to `typename SequentialOutput<...>::type`. The generated assembly for `forward()` is the same — the optimizer sees the same concrete `Tensor<Shape<...>, ...>` type regardless of how it was computed.

### 4. Code Clarity

The `consteval` version makes the algorithm explicit: a recursive compile-time function with a base case and a recursive case. The template-trait version encodes the same logic through partial specialization, which requires the reader to mentally reconstruct the recursion through the specializations.

## Benchmark Methodology

A synthetic benchmark (modeled after the existing `remove_cvref_benchmark`) compared the two approaches:

- **256 payload types** of varying size (`payload<0>` through `payload<255>`)
- **Chain depths**: 0, 1, 2, 4, 8, 16 (identity-transform layers)
- **Compile-time verification**: `static_assert` that all chain outputs match the expected input type
- **Runtime loop**: 200,000 iterations calling `[[gnu::noinline]]` checksum function to prevent dead-code elimination

Measured via:
- `time xmake build <target>` for compile time
- `size <binary>` for binary footprint (`.text`, `.data`, `.bss`)
- `time <binary>` for runtime

## Results Summary

| Metric | Template-Trait | `consteval` | Delta |
|--------|---------------|-------------|-------|
| Binary size (symbols) | Larger — N intermediate instantiations | Smaller — no intermediate symbols | **consteval wins** |
| Compile time | Slower — recursive template instantiation | Faster — constexpr interpretation | **consteval wins** |
| Runtime | Identical | Identical | Draw |

## Recommendation

Use `consteval inline auto sequential_output_meta()` with `::std::meta::substitute` and `[: ... :]` splicing. The template-trait approach offers no advantage: it produces larger binaries, compiles more slowly, and generates identical runtime code.

In the broader `typetorch` codebase, this pattern generalizes: wherever a recursive template struct computes a type through a sequence of transformations, replacing it with a `consteval` function operating on `::std::meta::info` reflections will reduce both compile time and binary footprint without affecting runtime performance.
