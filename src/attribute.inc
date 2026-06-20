// GCC,Clang
#if defined(__GNUC__) || defined(__CLANG__)
#define TENSPEC_ALWAYS_INLINE __attribute__((always_inline)) inline
#define TENSPEC_NO_INLINE __attribute__((noinline))
// MSVC
#elif defined(_MSC_VER)
#define TENSPEC_ALWAYS_INLINE __forceinline
#define TENSPEC_NO_INLINE __declspec(noinline)
// Default
#else
#define TENSPEC_ALWAYS_INLINE inline
#define TENSPEC_NO_INLINE
#endif