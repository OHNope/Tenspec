#include <cstddef>
#include "../../third_party/fast_io/include/fast_io.h"
#include <meta>
#include <type_traits>
#include <utility>

#define NORMALIZE_TYPE(Type) [: ::std::meta::remove_cvref(^^Type) :]
#include "remove_cvref_benchmark.inc"
