import std;
import libtorch;
import fastio;

#if defined(__GNUC__) || defined(__clang__)
#define TENSPEC_SIZE_NOINLINE __attribute__((noinline, used))
#else
#define TENSPEC_SIZE_NOINLINE
#endif

namespace tenspec_binary_size_libtorch_probe
{

TENSPEC_SIZE_NOINLINE auto raw_identity(::at::Tensor const &x) -> ::at::Tensor
{
	return x;
}

TENSPEC_SIZE_NOINLINE auto raw_add(::at::Tensor const &a, ::at::Tensor const &b)
	-> ::at::Tensor
{
	return a.add(b);
}

TENSPEC_SIZE_NOINLINE auto raw_numel(::at::Tensor const &x) -> ::std::int64_t
{
	return x.numel();
}

TENSPEC_SIZE_NOINLINE auto native_raw_ref(::at::Tensor const &x)
	-> ::at::Tensor const &
{
	return x;
}

TENSPEC_SIZE_NOINLINE auto native_retain_checked(::at::Tensor const &x)
	-> ::at::Tensor
{
	return x;
}

TENSPEC_SIZE_NOINLINE auto native_retain_unsafe(::at::Tensor const &x)
	-> ::at::Tensor
{
	return x;
}

TENSPEC_SIZE_NOINLINE auto native_unwrap(::at::Tensor x) -> ::at::Tensor
{
	return x;
}

TENSPEC_SIZE_NOINLINE auto native_add(::at::Tensor const &a,
									  ::at::Tensor const &b) -> ::at::Tensor
{
	return a.add(b);
}

TENSPEC_SIZE_NOINLINE auto native_numel(::at::Tensor const &x) -> ::std::int64_t
{
	return x.numel();
}

} // namespace tenspec_binary_size_libtorch_probe

int main()
{
	namespace probe = tenspec_binary_size_libtorch_probe;

	auto options{::at::TensorOptions().dtype(::at::kFloat).device(::at::kCPU)};
	auto a{::at::ones({4}, options)};
	auto b{::at::ones({4}, options)};

	auto raw_sum{probe::raw_add(a, b)};
	auto native_a{probe::native_retain_unsafe(a)};
	auto native_b{probe::native_retain_checked(b)};
	auto native_sum{probe::native_add(native_a, native_b)};
	auto unwrapped{probe::native_unwrap(::std::move(native_sum))};

	auto const total{probe::raw_numel(raw_sum) + probe::native_numel(native_a) +
					 probe::native_raw_ref(native_a).numel() +
					 probe::raw_identity(unwrapped).numel()};
	::fast_io::io::println(total);
}
