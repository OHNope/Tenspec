import std;
import libtorch;
import typetorch;
import fast_io;

namespace {

using BatchInput = typetorch::Tensor<typetorch::Shape<2, 3, 8, 8>, typetorch::DType::F32,
                                      typetorch::Device::CPU, typetorch::Layout::Contiguous>;

auto options() -> ::torch::TensorOptions {
    return ::torch::TensorOptions{}.dtype(::torch::kFloat).device(::torch::kCPU);
}

void expect_allclose(char const* name, ::torch::Tensor const& actual,
                     ::torch::Tensor const& expected) {
    if (!actual.equal(expected)) {
        ::fast_io::io::perrln(name, " mismatch; actual=", actual.toString(),
                               ", expected=", expected.toString());
        ::std::exit(1);
    }
}

} // namespace

int main() {
    // --- Test 1: forward with bias, default stride/pad ---
    {
        ::torch::nn::Conv2d raw(
            ::torch::nn::Conv2dOptions(3, 6, 3).bias(true));
        using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>>;
        Conv typed;

        typed->weight.set_data(raw->weight);
        typed->bias.set_data(raw->bias);

        auto input = ::torch::randn({2, 3, 8, 8}, options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(BatchInput::unsafe_retain(input));

        expect_allclose("conv2d_default", actual.unsafe_raw(), expected);
    }

    // --- Test 2: forward with stride=2, pad=1, no bias ---
    {
        ::torch::nn::Conv2d raw(
            ::torch::nn::Conv2dOptions(3, 6, 3)
                .stride(2).padding(1).bias(false));
        using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>,
                                        typetorch::Shape<2, 2>, typetorch::Shape<1, 1>>;
        Conv typed;

        typed->weight.set_data(raw->weight);

        auto input = ::torch::randn({2, 3, 8, 8}, options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(BatchInput::unsafe_retain(input));

        expect_allclose("conv2d_stride2_pad1", actual.unsafe_raw(), expected);
    }

    // --- Test 3: to(dtype) F16 ---
    {
        ::torch::nn::Conv2d raw(
            ::torch::nn::Conv2dOptions(3, 6, 3).bias(true));
        using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>>;
        Conv typed;
        typed->weight.set_data(raw->weight);
        typed->bias.set_data(raw->bias);

        auto f16 = typed->template to<typetorch::DType::F16>();
        if (!f16->weight.equal(raw->weight.to(::torch::kHalf))) {
            ::fast_io::io::perrln("conv2d_to_f16: weight mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 4: typed_weight / typed_bias shapes ---
    {
        using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>>;
        Conv typed;
        auto w = typed->typed_weight();
        auto b = typed->typed_bias();
        static_assert(::std::same_as<decltype(w)::shape, typetorch::Shape<6, 3, 3, 3>>);
        static_assert(::std::same_as<decltype(b)::shape, typetorch::Shape<6>>);
    }

    // --- Test 5: static constexpr members ---
    {
        using Conv = typetorch::Conv2d<3, 6, typetorch::Shape<3, 3>>;
        static_assert(Conv::in_channels == 3);
        static_assert(Conv::out_channels == 6);
        static_assert(Conv::groups == 1);
    }

    ::fast_io::io::println("typetorch nnModules conv2d tests passed");
}
