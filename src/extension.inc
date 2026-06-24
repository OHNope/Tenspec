#ifndef TYPETORCH_EXTENSION
#define TYPETORCH_EXTENSION

#pragma once

#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

#endif