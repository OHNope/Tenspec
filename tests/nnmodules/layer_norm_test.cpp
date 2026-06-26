import std;
import libtorch;
import typetorch;
import fast_io;

namespace {

using Input23 = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
                                   typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using Input224 = typetorch::Tensor<typetorch::Shape<2, 2, 4>, typetorch::DType::F32,
                                    typetorch::Device::CPU, typetorch::Layout::Contiguous>;

auto options() -> ::torch::TensorOptions {
    return ::torch::TensorOptions{}.dtype(::torch::kFloat).device(::torch::kCPU);
}

void expect_allclose(char const* name, ::torch::Tensor const& actual,
                     ::torch::Tensor const& expected) {
    if (!::torch::allclose(actual, expected)) {
        ::fast_io::io::perrln(name, " mismatch; actual=", actual.toString(),
                               ", expected=", expected.toString());
        ::std::exit(1);
    }
}

} // namespace

int main() {
    // --- Test 1: forward affine=true ---
    {
        ::torch::nn::LayerNorm raw(
            ::torch::nn::LayerNormOptions({2, 3}));
        using LN = typetorch::LayerNorm<typetorch::Shape<2, 3>>;
        LN typed;

        typed->weight.set_data(raw->weight);
        typed->bias.set_data(raw->bias);

        auto input = ::torch::randn({2, 3}, options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(Input23::unsafe_retain(input));

        if (!::torch::allclose(actual.unsafe_raw(), expected)) {
            ::fast_io::io::perrln("layernorm_affine: mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 2: forward affine=false ---
    {
        ::torch::nn::LayerNorm raw(
            ::torch::nn::LayerNormOptions({4}).elementwise_affine(false));
        using LN = typetorch::LayerNorm<typetorch::Shape<4>>;
        LN typed(/*eps=*/1e-5, /*elementwise_affine=*/false);

        auto input = ::torch::randn({2, 2, 4}, options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(Input224::unsafe_retain(input));

        if (!::torch::allclose(actual.unsafe_raw(), expected)) {
            ::fast_io::io::perrln("layernorm_no_affine: mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 3: to(dtype) F16 ---
    {
        ::torch::nn::LayerNorm raw(
            ::torch::nn::LayerNormOptions({2, 3}));
        using LN = typetorch::LayerNorm<typetorch::Shape<2, 3>>;
        LN typed;
        typed->weight.set_data(raw->weight);
        typed->bias.set_data(raw->bias);

        auto f16 = typed->template to<typetorch::DType::F16>();
        if (!f16->weight.equal(raw->weight.to(::torch::kHalf))) {
            ::fast_io::io::perrln("layernorm_to_f16: weight mismatch");
            ::std::exit(1);
        }
        if (!f16->bias.equal(raw->bias.to(::torch::kHalf))) {
            ::fast_io::io::perrln("layernorm_to_f16: bias mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 4: typed_weight / typed_bias shapes ---
    {
        using LN = typetorch::LayerNorm<typetorch::Shape<2, 3>>;
        LN typed;
        auto w = typed->typed_weight();
        auto b = typed->typed_bias();
        static_assert(::std::same_as<decltype(w)::shape, typetorch::Shape<2, 3>>);
        static_assert(::std::same_as<decltype(b)::shape, typetorch::Shape<2, 3>>);
    }

    ::fast_io::io::println("typetorch nnModules layer_norm tests passed");
}
