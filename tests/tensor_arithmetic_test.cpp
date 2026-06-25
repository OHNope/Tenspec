import std;
import libtorch;
import typetorch;
import fast_io;

namespace
{

using Matrix = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
								 typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using MatrixAny = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
									typetorch::Device::CPU, typetorch::Layout::Any>;
using Bias = typetorch::Tensor<typetorch::Shape<3>, typetorch::DType::F32,
							   typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using Column = typetorch::Tensor<typetorch::Shape<2, 1>, typetorch::DType::F32,
								 typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using MatrixFlat = typetorch::Tensor<typetorch::Shape<6>, typetorch::DType::F32,
									 typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using MatrixUnsqueezed = typetorch::Tensor<typetorch::Shape<1, 2, 3>, typetorch::DType::F32,
										   typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using Squeezable = typetorch::Tensor<typetorch::Shape<1, 2, 1, 3>, typetorch::DType::F32,
									 typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using Squeezed = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
								   typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using BoolMatrix = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::Bool,
									 typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using CatRows = typetorch::Tensor<typetorch::Shape<4, 3>, typetorch::DType::F32,
								  typetorch::Device::CPU, typetorch::Layout::Any>;
using CatCols = typetorch::Tensor<typetorch::Shape<2, 6>, typetorch::DType::F32,
								  typetorch::Device::CPU, typetorch::Layout::Any>;
using StackOuter = typetorch::Tensor<typetorch::Shape<2, 2, 3>, typetorch::DType::F32,
									 typetorch::Device::CPU, typetorch::Layout::Any>;
using StackLast = typetorch::Tensor<typetorch::Shape<2, 3, 2>, typetorch::DType::F32,
									typetorch::Device::CPU, typetorch::Layout::Any>;

static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().sub(
				  ::std::declval<Bias const &>())),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().mul(
				  ::std::declval<Column const &>())),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().div(
				  ::std::declval<Bias const &>())),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().sub(::torch::Scalar{1.0F})),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().mul(::torch::Scalar{2.0F})),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().div(::torch::Scalar{2.0F})),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::typetorch::softmax(::std::declval<Matrix const &>(), 1)),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::typetorch::relu(::std::declval<Matrix const &>())),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::typetorch::gelu(::std::declval<Matrix const &>())),
			  Matrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().flatten<>()),
			  MatrixFlat>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().unsqueeze<0>()),
			  MatrixUnsqueezed>);
static_assert(::std::same_as<
			  decltype(::std::declval<Squeezable const &>().squeeze()),
			  Squeezed>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>() > ::torch::Scalar{2.0F}),
			  BoolMatrix>);
static_assert(::std::same_as<
			  decltype(::std::declval<Matrix const &>().masked_fill(
				  ::std::declval<BoolMatrix const &>(), ::torch::Scalar{0.0F})),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(Matrix::where(::std::declval<BoolMatrix const &>(),
									 ::std::declval<Matrix const &>(),
									 ::std::declval<Matrix const &>())),
			  MatrixAny>);
static_assert(::std::same_as<
			  decltype(Matrix::cat<0>(::std::declval<Matrix const &>(),
									  ::std::declval<Matrix const &>())),
			  CatRows>);
static_assert(::std::same_as<
			  decltype(typetorch::cat<1>(::std::declval<Matrix const &>(),
										 ::std::declval<Matrix const &>())),
			  CatCols>);
static_assert(::std::same_as<
			  decltype(Matrix::stack<0>(::std::declval<Matrix const &>(),
										::std::declval<Matrix const &>())),
			  StackOuter>);
static_assert(::std::same_as<
			  decltype(typetorch::stack<2>(::std::declval<Matrix const &>(),
										   ::std::declval<Matrix const &>())),
			  StackLast>);

auto options() -> ::torch::TensorOptions
{
	return ::torch::TensorOptions{}.dtype(::torch::kFloat).device(::torch::kCPU);
}

auto matrix_raw() -> ::torch::Tensor
{
	return ::torch::arange(6, options()).view({2, 3});
}

void expect_allclose(char const *name, ::torch::Tensor const &actual,
					 ::torch::Tensor const &expected)
{
	if (!actual.equal(expected))
	{
		::fast_io::io::perrln(name, " mismatch; actual=", actual.toString(),
                               ", expected=", expected.toString());
        ::std::exit(1);
	}
}

} // namespace

int main()
{
	auto bias_raw{::torch::tensor({10.0F, 20.0F, 30.0F}, options())};
	auto column_raw{::torch::tensor({1.0F, 2.0F}, options()).view({2, 1})};
	auto twos_raw{::torch::ones({2, 3}, options()).mul(2.0F)};

	expect_allclose(
		"sub_tensor_same_shape",
		Matrix::retain(matrix_raw()).sub(Matrix::retain(twos_raw)).unsafe_raw(),
		matrix_raw().sub(twos_raw));
	expect_allclose(
		"sub_tensor_broadcast_alpha",
		Matrix::retain(matrix_raw()).sub(Bias::retain(bias_raw), 0.5F).unsafe_raw(),
		matrix_raw().sub(bias_raw, 0.5F));
	expect_allclose("sub_scalar",
					Matrix::retain(matrix_raw()).sub(::torch::Scalar{1.5F}).unsafe_raw(),
					matrix_raw().sub(1.5F));

	expect_allclose(
		"mul_tensor_same_shape",
		Matrix::retain(matrix_raw()).mul(Matrix::retain(twos_raw)).unsafe_raw(),
		matrix_raw().mul(twos_raw));
	expect_allclose(
		"mul_tensor_broadcast",
		Matrix::retain(matrix_raw()).mul(Column::retain(column_raw)).unsafe_raw(),
		matrix_raw().mul(column_raw));
	expect_allclose("mul_scalar",
					Matrix::retain(matrix_raw()).mul(::torch::Scalar{3.0F}).unsafe_raw(),
					matrix_raw().mul(3.0F));

	expect_allclose(
		"div_tensor_same_shape",
		Matrix::retain(matrix_raw()).div(Matrix::retain(twos_raw)).unsafe_raw(),
		matrix_raw().div(twos_raw));
	expect_allclose(
		"div_tensor_broadcast",
		Matrix::retain(matrix_raw()).div(Bias::retain(bias_raw)).unsafe_raw(),
		matrix_raw().div(bias_raw));
	expect_allclose("div_scalar",
					Matrix::retain(matrix_raw()).div(::torch::Scalar{2.0F}).unsafe_raw(),
					matrix_raw().div(2.0F));

	expect_allclose("softmax_runtime_dim",
					::typetorch::softmax(Matrix::retain(matrix_raw()), 1).unsafe_raw(),
					matrix_raw().softmax(1));
	expect_allclose("relu",
					::typetorch::relu(Matrix::retain(matrix_raw())).unsafe_raw(),
					matrix_raw().relu());
	expect_allclose("gelu_default",
					::typetorch::gelu(Matrix::retain(matrix_raw())).unsafe_raw(),
					::torch::gelu(matrix_raw()));
	expect_allclose("gelu_tanh",
					::typetorch::gelu(Matrix::retain(matrix_raw()), "tanh").unsafe_raw(),
					::torch::gelu(matrix_raw(), "tanh"));

	expect_allclose("flatten_static",
					Matrix::retain(matrix_raw()).flatten<>().unsafe_raw(),
					matrix_raw().flatten());
	expect_allclose("unsqueeze_static",
					Matrix::retain(matrix_raw()).unsqueeze<0>().unsafe_raw(),
					matrix_raw().unsqueeze(0));
	expect_allclose(
		"squeeze_static",
		Squeezable::retain(matrix_raw().view({1, 2, 1, 3})).squeeze().unsafe_raw(),
		matrix_raw().view({1, 2, 1, 3}).squeeze());
	auto mask_raw{::torch::gt(matrix_raw(), ::torch::Scalar{2.0F})};
	expect_allclose("comparison_bool",
					(Matrix::retain(matrix_raw()) > ::torch::Scalar{2.0F}).unsafe_raw(),
					mask_raw);
	expect_allclose(
		"masked_fill_static",
		Matrix::retain(matrix_raw())
			.masked_fill(BoolMatrix::retain(mask_raw), ::torch::Scalar{-1.0F})
			.unsafe_raw(),
		matrix_raw().masked_fill(mask_raw, -1.0F));
	expect_allclose(
		"where_static",
		Matrix::where(BoolMatrix::retain(mask_raw), Matrix::retain(matrix_raw()),
					  Matrix::retain(twos_raw))
			.unsafe_raw(),
		::torch::where(mask_raw, matrix_raw(), twos_raw));
	expect_allclose(
		"cat_dim0_static",
		Matrix::cat<0>(Matrix::retain(matrix_raw()), Matrix::retain(twos_raw))
			.unsafe_raw(),
		::torch::cat({matrix_raw(), twos_raw}, 0));
	expect_allclose(
		"cat_dim1_free",
		typetorch::cat<1>(Matrix::retain(matrix_raw()), Matrix::retain(twos_raw))
			.unsafe_raw(),
		::torch::cat({matrix_raw(), twos_raw}, 1));
	expect_allclose(
		"stack_dim0_static",
		Matrix::stack<0>(Matrix::retain(matrix_raw()), Matrix::retain(twos_raw))
			.unsafe_raw(),
		::torch::stack({matrix_raw(), twos_raw}, 0));
	expect_allclose(
		"stack_dim2_free",
		typetorch::stack<2>(Matrix::retain(matrix_raw()), Matrix::retain(twos_raw))
			.unsafe_raw(),
		::torch::stack({matrix_raw(), twos_raw}, 2));

	::fast_io::io::println("typetorch tensor arithmetic tests passed");
}
