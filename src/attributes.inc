#ifndef TYPETORCH_ATTR

#define TYPETORCH_ATTR

#pragma once

// feature detection
#ifndef TYPETORCH_HAS_ATTRIBUTE
#if defined(__has_attribute)
#define TYPETORCH_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define TYPETORCH_HAS_ATTRIBUTE(x) 0
#endif
#endif

// 1. pure: read non volatile,req no side eff
#if TYPETORCH_HAS_ATTRIBUTE(pure) || defined(__GNUC__)
#define TYPETORCH_PURE __attribute__((pure))
#else
#define TYPETORCH_PURE
#endif

// 2. const > pure
#if TYPETORCH_HAS_ATTRIBUTE(const) || defined(__GNUC__)
#define TYPETORCH_CONST __attribute__((const))
#else
#define TYPETORCH_CONST
#endif

// 3. noalias for function body:
// MSVC: __declspec(noalias)
// Clang: __attribute__((noalias))
// GCC: no pure
#if defined(_MSC_VER)
#define TYPETORCH_NOALIAS_FUNC __declspec(noalias)
#elif TYPETORCH_HAS_ATTRIBUTE(noalias)
#define TYPETORCH_NOALIAS_FUNC __attribute__((noalias))
#else
#define TYPETORCH_NOALIAS_FUNC
#endif

// 4. returned pointer does not alias existing reachable objects:
// GCC/Clang: malloc attribute
// MSVC: __declspec(restrict)
#if defined(_MSC_VER)
#define TYPETORCH_RETURNS_NOALIAS __declspec(restrict)
#elif TYPETORCH_HAS_ATTRIBUTE(malloc) || defined(__GNUC__)
#define TYPETORCH_RETURNS_NOALIAS __attribute__((malloc))
#else
#define TYPETORCH_RETURNS_NOALIAS
#endif

// 5. hot: function is frequently called, optimize for speed
#if TYPETORCH_HAS_ATTRIBUTE(hot) || defined(__GNUC__)
#define TYPETORCH_HOT __attribute__((hot))
#else
#define TYPETORCH_HOT
#endif

// 6. cold: function is rarely called (error paths), optimize for size
#if TYPETORCH_HAS_ATTRIBUTE(cold) || defined(__GNUC__)
#define TYPETORCH_COLD __attribute__((cold))
#else
#define TYPETORCH_COLD
#endif

// 7. flatten: inline every call inside this function
#if TYPETORCH_HAS_ATTRIBUTE(flatten) || defined(__GNUC__)
#define TYPETORCH_FLATTEN __attribute__((flatten))
#else
#define TYPETORCH_FLATTEN
#endif

// 8. returns_nonnull: function never returns null
#if TYPETORCH_HAS_ATTRIBUTE(returns_nonnull) || defined(__GNUC__)
#define TYPETORCH_RETURNS_NONNULL __attribute__((returns_nonnull))
#else
#define TYPETORCH_RETURNS_NONNULL
#endif

// 9. likely/unlikely: branch prediction hints
#if defined(__GNUC__) || defined(__clang__)
#define TYPETORCH_LIKELY(x)   __builtin_expect(!!(x), 1)
#define TYPETORCH_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define TYPETORCH_LIKELY(x)   (x)
#define TYPETORCH_UNLIKELY(x) (x)
#endif

// 10. used: prevent compiler from optimizing away
#if TYPETORCH_HAS_ATTRIBUTE(used) || defined(__GNUC__)
#define TYPETORCH_USED __attribute__((used))
#else
#define TYPETORCH_USED
#endif

// GCC,Clang
#if defined(__GNUC__) || defined(__CLANG__)
#define TYPETORCH_ALWAYS_INLINE __attribute__((always_inline)) inline
#define TYPETORCH_NO_INLINE __attribute__((noinline))
// MSVC
#elif defined(_MSC_VER)
#define TYPETORCH_ALWAYS_INLINE __forceinline
#define TYPETORCH_NO_INLINE __declspec(noinline)
// Default
#else
#define TYPETORCH_ALWAYS_INLINE inline
#define TYPETORCH_NO_INLINE
#endif

#ifdef __CUDACC__
#define TYPETORCH_UNUSED
#else
#define TYPETORCH_UNUSED [[maybe_unused]]
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TYPETORCH_SIZE_NOINLINE __attribute__((noinline, used))
#else
#define TYPETORCH_SIZE_NOINLINE
#endif

#endif
