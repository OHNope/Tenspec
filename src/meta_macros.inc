#ifndef TYPETORCH_META_MACROS

#define TYPETORCH_META_MACROS

#pragma once

#define REMOVE_CVREF(Arg) \
	typename[: ::std::meta::remove_cvref(^^Arg):]


#endif