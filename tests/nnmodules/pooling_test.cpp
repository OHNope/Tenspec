import std;
import libtorch;
import typetorch;
import fast_io;

namespace
{

using Input =
	typetorch::Tensor<typetorch::Shape<2, 3, 9, 10>,
					  typetorch::DType::F32, typetorch::Device::CPU,
					  typetorch::Layout::Contiguous>;
using MaxOutput =
	typetorch::Tensor<typetorch::Shape<2, 3, 5, 5>,
					  typetorch::DType::F32, typetorch::Device::CPU,
					  typetorch::Layout::Any>;
using AvgOutput =
	typetorch::Tensor<typetorch::Shape<2, 3, 4, 4>,
					  typetorch::DType::F32, typetorch::Device::CPU,
					  typetorch::Layout::Any>;
using AdaptiveOutput =
	typetorch::Tensor<typetorch::Shape<2, 3, 2, 3>,
					  typetorch::DType::F32, typetorch::Device::CPU,
					  typetorch::Layout::Any>;
using AdaptiveKeepHeightOutput =
	typetorch::Tensor<typetorch::Shape<2, 3, 9, 3>,
					  typetorch::DType::F32, typetorch::Device::CPU,
					  typetorch::Layout::Any>;

auto options() -> ::torch::TensorOptions
{
	return ::torch::TensorOptions{}
		.dtype(::torch::kFloat)
		.device(::torch::kCPU);
}

void expect_equal(char const *name, ::torch::Tensor const &actual,
				  ::torch::Tensor const &expected)
{
	if (!actual.equal(expected))
	{
		::fast_io::io::perrln(name, " mismatch");
        ::std::exit(1);
	}
}

} // namespace

int main()
{
	using Max = typetorch::MaxPool2d<
		typetorch::Shape<3, 2>, typetorch::Shape<2, 2>,
		typetorch::Shape<1, 0>, typetorch::Shape<1, 1>, true>;
	using Avg = typetorch::AvgPool2d<
		typetorch::Shape<3, 3>, typetorch::Shape<2, 2>,
		typetorch::Shape<0, 0>, false, false>;
	using Adaptive =
		typetorch::AdaptiveAvgPool2d<typetorch::Shape<2, 3>>;
	using AdaptiveKeepHeight =
		typetorch::AdaptiveAvgPool2d<
			typetorch::Shape<typetorch::dyn, 3>>;

	static_assert(::std::same_as<
				  decltype(::std::declval<Max::Impl &>().forward(
					  ::std::declval<Input const &>())),
				  MaxOutput>);
	static_assert(::std::same_as<
				  decltype(::std::declval<Avg::Impl &>().forward(
					  ::std::declval<Input const &>())),
				  AvgOutput>);
	static_assert(::std::same_as<
				  decltype(::std::declval<Adaptive::Impl &>().forward(
					  ::std::declval<Input const &>())),
				  AdaptiveOutput>);
	static_assert(::std::same_as<
				  decltype(::std::declval<AdaptiveKeepHeight::Impl &>()
							   .forward(::std::declval<Input const &>())),
				  AdaptiveKeepHeightOutput>);

	auto input{::torch::randn({2, 3, 9, 10}, options())};

	Max max_pool;
	auto raw_max{::torch::nn::MaxPool2d{
		::torch::nn::MaxPool2dOptions{{3, 2}}
			.stride({2, 2})
			.padding({1, 0})
			.dilation({1, 1})
			.ceil_mode(true)}};
	expect_equal(
		"MaxPool2d",
		max_pool->forward(Input::unsafe_retain(input)).unsafe_raw(),
		raw_max->forward(input));

	Avg avg_pool;
	auto raw_avg{::torch::nn::AvgPool2d{
		::torch::nn::AvgPool2dOptions{{3, 3}}
			.stride({2, 2})
			.padding({0, 0})
			.ceil_mode(false)
			.count_include_pad(false)}};
	expect_equal(
		"AvgPool2d",
		avg_pool->forward(Input::unsafe_retain(input)).unsafe_raw(),
		raw_avg->forward(input));

	Adaptive adaptive_pool;
	auto raw_adaptive{::torch::adaptive_avg_pool2d(
		input, ::std::array<::std::int64_t, 2>{2, 3})};
	expect_equal(
		"AdaptiveAvgPool2d",
		adaptive_pool->forward(Input::unsafe_retain(input)).unsafe_raw(),
		raw_adaptive);
	AdaptiveKeepHeight adaptive_keep_height;
	auto raw_adaptive_keep_height{::torch::adaptive_avg_pool2d(
		input, ::std::array<::std::int64_t, 2>{9, 3})};
	expect_equal(
		"AdaptiveAvgPool2d keep height",
		adaptive_keep_height->forward(Input::unsafe_retain(input)).unsafe_raw(),
		raw_adaptive_keep_height);


	auto sequence{typetorch::Sequential{
		typetorch::MaxPool2d<typetorch::Shape<2, 2>>{},
		typetorch::AdaptiveAvgPool2d<typetorch::Shape<1, 1>>{},
		typetorch::Flatten<1, -1>{}}};
	using SequenceOutput =
		typetorch::Tensor<typetorch::Shape<2, 3>,
						  typetorch::DType::F32,
						  typetorch::Device::CPU,
						  typetorch::Layout::Any>;
	static_assert(::std::same_as<
				  decltype(sequence->forward(
					  ::std::declval<Input const &>())),
				  SequenceOutput>);

	::fast_io::io::println("typetorch nnModules pooling tests passed");
}
