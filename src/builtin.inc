#ifndef TYPETORCH_BUILTIN
#define TYPETORCH_BUILTIN
#pragma once

#ifdef __has_builtin
#define TYPETORCH_HAS_BUILTIN(x) __has_builtin(x)
#else
#define TYPETORCH_HAS_BUILTIN(x) 0
#endif


#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define TYPETORCH_HAS_STD_UNREACHABLE 1
#else
#define TYPETORCH_HAS_STD_UNREACHABLE 0
#endif


#if TYPETORCH_HAS_BUILTIN(__builtin_unreachable)
#define TYPETORCH_UNREACHABLE __builtin_unreachable();

#elif defined(__GNUC__) || defined(__clang__)
#define TYPETORCH_UNREACHABLE __builtin_unreachable();

#elif defined(_MSC_VER)
#define TYPETORCH_UNREACHABLE __assume(0);

#elif TYPETORCH_HAS_STD_UNREACHABLE
#define TYPETORCH_UNREACHABLE std::unreachable();

#else
#define TYPETORCH_UNREACHABLE ::std::abort();
#endif


#endif