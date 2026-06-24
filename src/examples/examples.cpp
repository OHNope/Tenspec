module typetorch.examples;

import std;
import libtorch;
import typetorch;

namespace typetorch_examples
{
namespace detail
{

using Weight3To2 = typetorch::Tensor<typetorch::Shape<3, 2>, typetorch::DType::F32,
								   typetorch::Device::CPU,
								   typetorch::Layout::Any>;
using StaticVector = typetorch::Tensor<typetorch::Shape<6>, typetorch::DType::F32,
									 typetorch::Device::CPU,
									 typetorch::Layout::Contiguous>;
using StaticMatrix = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
									 typetorch::Device::CPU,
									 typetorch::Layout::Contiguous>;
using StaticMatrixT = typetorch::Tensor<typetorch::Shape<3, 2>, typetorch::DType::F32,
									  typetorch::Device::CPU,
									  typetorch::Layout::Any>;
using StaticCube = typetorch::Tensor<typetorch::Shape<2, 3, 4>, typetorch::DType::F32,
								   typetorch::Device::CPU,
								   typetorch::Layout::Contiguous>;
using StaticCubePermuted =
	typetorch::Tensor<typetorch::Shape<2, 4, 3>, typetorch::DType::F32,
					typetorch::Device::CPU, typetorch::Layout::Any>;
using StaticCubePermutedContiguous =
	typetorch::Tensor<typetorch::Shape<2, 4, 3>, typetorch::DType::F32,
					typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using StaticCubeIdentity =
	typetorch::Tensor<typetorch::Shape<2, 3, 4>, typetorch::DType::F32,
					typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using StaticImage =
	typetorch::Tensor<typetorch::Shape<3, 32, 48>, typetorch::DType::F32,
					typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using StaticImageConv =
	typetorch::Tensor<typetorch::Shape<8, 16, 48>, typetorch::DType::F32,
					typetorch::Device::CPU, typetorch::Layout::Any>;
using DynamicImageBatch =
	typetorch::Tensor<
		typetorch::Shape<typetorch::dyn, 3, typetorch::dyn, 48>,
		typetorch::DType::F32, typetorch::Device::CPU,
		typetorch::Layout::Any>;
using DynamicImageBatchConv =
	typetorch::Tensor<
		typetorch::Shape<typetorch::dyn, 8, typetorch::dyn, 48>,
		typetorch::DType::F32, typetorch::Device::CPU,
		typetorch::Layout::Any>;
using TestConv2d =
	typetorch::Conv2d<3, 8, typetorch::Shape<3, 5>,
					  typetorch::Shape<2, 1>, typetorch::Shape<1, 2>>;

static_assert(::std::is_same_v<decltype(Vector::retain(
								   ::std::declval<::torch::Tensor const &>())),
							   Vector>);
static_assert(::std::is_same_v<decltype(Vector::unsafe_retain(
								   ::std::declval<::torch::Tensor const &>())),
							   Vector>);
static_assert(::std::is_same_v<decltype(Vector::unsafe_move(
								   ::std::declval<::torch::Tensor>())),
							   Vector>);
static_assert(::std::is_same_v<decltype(Vector::retain(
								   ::std::declval<::torch::Tensor const &>())),
							   Vector>);
static_assert(::std::is_same_v<decltype(::std::declval<Vector const &>().unsafe_raw()),
							   ::torch::Tensor const &>);
static_assert(::std::is_same_v<decltype(::std::declval<Vector &&>().unwrap()),
							   ::torch::Tensor>);

static_assert(::std::is_same_v<decltype(::std::declval<Matrix const &>().add(
								   ::std::declval<Vector const &>())),
							   Matrix>);
static_assert(::std::is_same_v<decltype(::std::declval<Vector const &>().add(
								   ::std::declval<::torch::Scalar const &>())),
							   Vector>);
static_assert(
	::std::is_same_v<decltype(::std::declval<Matrix3In const &>().matmul(
						 ::std::declval<Matrix3Out const &>())),
					 Matrix>);
static_assert(
	::std::is_same_v<decltype(::std::declval<Matrix3In const &>().matmul(
						 ::std::declval<Weight3To2 const &>())),
					 Matrix2Out>);
static_assert(::std::is_same_v<decltype(::std::declval<StaticMatrix &&>()
											.transpose<0, 1>()),
							   StaticMatrixT>);
static_assert(::std::is_same_v<decltype(::std::declval<StaticCube &&>()
											.permute<0, 2, 1>()),
							   StaticCubePermuted>);
static_assert(::std::is_same_v<decltype(::std::declval<StaticCube &&>()
											.permute<0, -1, 1>()),
							   StaticCubePermuted>);
static_assert(::std::is_same_v<decltype(::std::declval<StaticCube &&>()
											.permute<0, 1, 2>()),
							   StaticCubeIdentity>);
static_assert(::std::is_same_v<decltype(::std::declval<StaticMatrix &&>()
											.view<typetorch::infer>()),
							   StaticVector>);
static_assert(::std::is_same_v<decltype(::std::declval<StaticCubePermuted &&>()
											.contiguous()),
							   StaticCubePermutedContiguous>);
static_assert(::std::is_same_v<decltype(StaticMatrix::ones()), StaticMatrix>);
static_assert(::std::is_same_v<decltype(StaticMatrix::empty()), StaticMatrix>);
static_assert(::std::is_same_v<decltype(StaticMatrix::zeros()), StaticMatrix>);
static_assert(::std::is_same_v<decltype(StaticMatrix::rand()), StaticMatrix>);
static_assert(::std::is_same_v<decltype(StaticMatrix::randn()), StaticMatrix>);
static_assert(::std::is_same_v<decltype(StaticMatrix::arange<6>()), StaticVector>);
static_assert(::std::is_same_v<decltype(StaticMatrix::arange<0, 6>()),
							   StaticVector>);
static_assert(::std::is_same_v<
			  decltype(StaticMatrix::arange<0, 6, 2, typetorch::DType::I64>()),
			  typetorch::Tensor<typetorch::Shape<3>, typetorch::DType::I64,
							  typetorch::Device::CPU,
							  typetorch::Layout::Contiguous>>);
static_assert(::std::is_same_v<
			  decltype(::std::declval<TestConv2d::Impl &>().forward(
				  ::std::declval<StaticImage const &>())),
			  StaticImageConv>);
static_assert(::std::is_same_v<
			  decltype(::std::declval<TestConv2d::Impl &>().forward(
				  ::std::declval<DynamicImageBatch const &>())),
			  DynamicImageBatchConv>);
static_assert(::std::is_same_v<
			  decltype(::std::declval<TestConv2d::Impl const &>().typed_weight()),
			  typetorch::Tensor<typetorch::Shape<8, 3, 3, 5>,
							  typetorch::DType::F32,
							  typetorch::Device::CPU>>);
static_assert(::std::is_default_constructible_v<TestConv2d>);

::torch::Tensor fixed_linear_weight();

TestConv2d make_test_conv2d()
{
	return TestConv2d{};
}

auto convert_test_conv2d(TestConv2d const &conv)
{
	return conv.to<typetorch::Device::CPU, typetorch::DType::F16>();
}

} // namespace detail

SizeVector checked_vector_sizes(Vector const &x)
{
	auto sizes{x.unsafe_raw().sizes().vec()};
	return SizeVector::unsafe_move(
		::torch::tensor(sizes, x.unsafe_raw().options().dtype(::torch::kLong).device(::torch::kCPU)));
}

Vector add_vectors(Vector const &a, Vector const &b)
{
	return a.add(b);
}

Matrix add_bias(Matrix const &x, Vector const &bias)
{
	return x.add(bias);
}

Matrix matmul_3_inner(Matrix3In const &x, Matrix3Out const &weight)
{
	return x.matmul(weight);
}

Matrix2Out linear_3_to_2(Matrix3In const &x)
{
	auto weight{detail::Weight3To2::retain(
		detail::fixed_linear_weight().transpose(0, 1))};
	return x.matmul(weight);
}

Matrix transpose_matrix(Matrix const &x)
{
	return Matrix::unsafe_move(x.unsafe_raw().transpose(0, 1));
}

::std::int64_t numel_plus(Matrix const &x, ::std::int64_t extra)
{
	return x.unsafe_raw().numel() + extra;
}

namespace detail
{

::torch::Tensor fixed_linear_weight()
{
	auto options{::torch::TensorOptions().dtype(::torch::kFloat).device(::torch::kCPU)};
	return ::torch::tensor({0.5F, 0.5F, 0.0F, 0.0F, 1.5F, 0.5F}, options).view({2, 3});
}

} // namespace detail

} // namespace typetorch_examples
