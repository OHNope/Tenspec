import std;
import libtorch;
import typetorch;
import fast_io;

namespace {

using FloatInput = typetorch::Tensor<typetorch::Shape<4, 3>, typetorch::DType::F32,
                                     typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using SingleInput = typetorch::Tensor<typetorch::Shape<2, 4>, typetorch::DType::F32,
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
    // --- Test 1: forward with bias ---
    {
        auto raw{::torch::nn::Linear{::torch::nn::LinearOptions(3, 2).bias(true)}};
        using Linear32 = typetorch::Linear<3, 2>;
        Linear32 typed;

        typed->weight.set_data(raw->weight);
        typed->bias.set_data(raw->bias);

        auto input = ::torch::randn({4, 3}, options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(FloatInput::unsafe_retain(input));

        expect_allclose("linear_3_to_2_with_bias", actual.unsafe_raw(), expected);
    }

    // --- Test 2: forward without bias ---
    {
        auto raw{::torch::nn::Linear{::torch::nn::LinearOptions(4, 1).bias(false)}};
        using LinearNoBias = typetorch::Linear<4, 1>;
        LinearNoBias typed;

        typed->weight.set_data(raw->weight);

        auto input = ::torch::randn({2, 4}, options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(SingleInput::unsafe_retain(input));

        expect_allclose("linear_4_to_1_no_bias", actual.unsafe_raw(), expected);
    }

    // --- Test 3: to(dtype) F16 ---
    {
        auto raw{::torch::nn::Linear{::torch::nn::LinearOptions(3, 2).bias(true)}};
        using Linear32 = typetorch::Linear<3, 2>;
        Linear32 typed;

        typed->weight.set_data(raw->weight);
        typed->bias.set_data(raw->bias);

        auto f16 = typed->template to<typetorch::DType::F16>();
        auto raw_f16_weight = raw->weight.to(::torch::kHalf);
        auto raw_f16_bias = raw->bias.to(::torch::kHalf);

        if (!f16->weight.equal(raw_f16_weight)) {
            ::fast_io::io::perrln("linear_to_f16: weight mismatch");
            ::std::exit(1);
        }
        if (!f16->bias.equal(raw_f16_bias)) {
            ::fast_io::io::perrln("linear_to_f16: bias mismatch");
            ::std::exit(1);
        }

        auto input = ::torch::randn({4, 3}, options());
        auto f16_input = input.to(::torch::kHalf);
        using F16Input = typetorch::Tensor<typetorch::Shape<4, 3>, typetorch::DType::F16,
                                            typetorch::Device::CPU, typetorch::Layout::Contiguous>;
        auto actual_f16 = f16->forward(F16Input::unsafe_retain(f16_input));
        auto raw_f16_result = f16_input.mm(raw_f16_weight.t()).add(raw_f16_bias);
        if (!actual_f16.unsafe_raw().to(::torch::kFloat).equal(raw_f16_result.to(::torch::kFloat))) {
            ::fast_io::io::perrln("linear_to_f16: forward mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 4: typed_weight / typed_bias shapes ---
    {
        using Linear32 = typetorch::Linear<3, 2>;
        Linear32 typed;
        auto w = typed->typed_weight();
        auto b = typed->typed_bias();
        static_assert(::std::same_as<decltype(w)::shape, typetorch::Shape<2, 3>>);
        static_assert(::std::same_as<decltype(b)::shape, typetorch::Shape<2>>);
    }

    // --- Test 5: static constexpr members ---
    {
        static_assert(typetorch::Linear<3, 2>::in_features == 3);
        static_assert(typetorch::Linear<3, 2>::out_features == 2);
    }

    ::fast_io::io::println("typetorch nnModules linear tests passed");
}
