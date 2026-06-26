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

void expect_allclose(char const* name, int exit_code, ::torch::Tensor const& actual,
                     ::torch::Tensor const& expected, double rtol = 1e-4,
                     double atol = 1e-5) {
    if (!::torch::allclose(actual, expected, rtol, atol)) {
        auto actual_f32 = actual.to(::torch::kFloat).reshape({-1});
        auto expected_f32 = expected.to(::torch::kFloat).reshape({-1});
        auto diff = actual_f32.sub(expected_f32).abs();
        auto max_index = diff.argmax().item<::std::int64_t>();
        ::fast_io::io::perrln(::std::string_view{name}, " mismatch; actual=", actual.toString(),
                               ", expected=", expected.toString(),
                               ", max_abs_diff=", diff.max().item<double>(),
                               ", max_index=", max_index,
                               ", actual_value=", actual_f32[max_index].item<double>(),
                               ", expected_value=", expected_f32[max_index].item<double>());
        ::std::exit(exit_code);
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

        expect_allclose("linear_3_to_2_with_bias", 11, actual.unsafe_raw(), expected);
    }

    // --- Test 2: forward without bias ---
    {
        auto raw{::torch::nn::Linear{::torch::nn::LinearOptions(4, 1).bias(false)}};
        using LinearNoBias = typetorch::Linear<4, 1>;
        LinearNoBias typed{/*with_bias=*/false};

        typed->weight.set_data(raw->weight);

        auto input = ::torch::randn({2, 4}, options());
        auto expected = raw->forward(input);
        auto actual = typed->forward(SingleInput::unsafe_retain(input));

        expect_allclose("linear_4_to_1_no_bias", 12, actual.unsafe_raw(), expected);
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

        expect_allclose("linear_to_f16_weight", 13, f16->weight, raw_f16_weight, 0.0, 0.0);
        expect_allclose("linear_to_f16_bias", 14, f16->bias, raw_f16_bias, 0.0, 0.0);

        auto input = ::torch::randn({4, 3}, options());
        auto f16_input = input.to(::torch::kHalf);
        auto raw_f16{::torch::nn::Linear{::torch::nn::LinearOptions(3, 2).bias(true)}};
        raw_f16->weight.set_data(raw_f16_weight);
        raw_f16->bias.set_data(raw_f16_bias);
        using F16Input = typetorch::Tensor<typetorch::Shape<4, 3>, typetorch::DType::F16,
                                            typetorch::Device::CPU, typetorch::Layout::Contiguous>;
        auto actual_f16 = f16->forward(F16Input::unsafe_retain(f16_input));
        auto actual_f16_f32 = actual_f16.unsafe_raw().to(::torch::kFloat);
        auto expected_f16_f32 = raw_f16->forward(f16_input).to(::torch::kFloat);
        expect_allclose(
            "linear_to_f16_forward", 15, actual_f16_f32, expected_f16_f32, 1e-3, 1e-3);
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
