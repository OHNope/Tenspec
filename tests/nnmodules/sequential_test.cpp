import std;
import libtorch;
import typetorch;
import fast_io;

namespace {

using Input = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::F32,
                                 typetorch::Device::CPU, typetorch::Layout::Contiguous>;

auto options() -> ::torch::TensorOptions {
    return ::torch::TensorOptions{}.dtype(::torch::kFloat).device(::torch::kCPU);
}

} // namespace

int main() {
    // --- Test 1: Two Linear layers chained ---
    {
        // Raw equivalent
        auto raw = ::torch::nn::Sequential(
            ::torch::nn::Linear(::torch::nn::LinearOptions(3, 5).bias(true)),
            ::torch::nn::Linear(::torch::nn::LinearOptions(5, 2).bias(false)));
        raw[0]->as<::torch::nn::Linear>()->weight.set_data(
            ::torch::randn({5, 3}, options()));
        raw[0]->as<::torch::nn::Linear>()->bias.set_data(
            ::torch::randn({5}, options()));
        raw[1]->as<::torch::nn::Linear>()->weight.set_data(
            ::torch::randn({2, 5}, options()));

        // Typed equivalent
        using LinearA = typetorch::Linear<3, 5>;
        using LinearB = typetorch::Linear<5, 2>;
        LinearA a;
        LinearB b(/*with_bias=*/false);
        a->weight.set_data(raw[0]->as<::torch::nn::Linear>()->weight);
        a->bias.set_data(raw[0]->as<::torch::nn::Linear>()->bias);
        b->weight.set_data(raw[1]->as<::torch::nn::Linear>()->weight);

        auto seq = typetorch::Sequential(a, b);

        auto input = ::torch::randn({2, 3}, options());
        auto expected = raw->forward(input);
        auto actual = seq->forward(Input::unsafe_retain(input));

        if (!::torch::allclose(actual.unsafe_raw(), expected)) {
            ::fast_io::io::perrln("sequential_two_linear: mismatch; actual=",
                                   actual.unsafe_raw().toString(),
                                   ", expected=", expected.toString());
            ::std::exit(1);
        }
    }

    // --- Test 2: Sequential size ---
    {
        using LinearA = typetorch::Linear<3, 5>;
        using LinearB = typetorch::Linear<5, 2>;
        LinearA a;
        LinearB b;
        auto seq = typetorch::Sequential(a, b);
        static_assert(decltype(seq)::size == 2);
    }

    // --- Test 3: get<N>() returns correct types ---
    {
        using LinearA = typetorch::Linear<3, 5>;
        using LinearB = typetorch::Linear<5, 2>;
        LinearA a;
        LinearB b;
        auto seq = typetorch::Sequential(a, b);
        auto& first = seq.template get<0>();
        auto& second = seq.template get<1>();
        static_assert(::std::same_as<decltype(first), LinearA&>);
        static_assert(::std::same_as<decltype(second), LinearB&>);
    }

    // --- Test 4: Linear + LayerNorm chain ---
    {
        auto raw = ::torch::nn::Sequential(
            ::torch::nn::Linear(::torch::nn::LinearOptions(3, 3).bias(false)),
            ::torch::nn::LayerNorm(::torch::nn::LayerNormOptions({3})));
        raw[0]->as<::torch::nn::Linear>()->weight.set_data(
            ::torch::randn({3, 3}, options()));
        raw[1]->as<::torch::nn::LayerNorm>()->weight.set_data(
            ::torch::ones({3}, options()));
        raw[1]->as<::torch::nn::LayerNorm>()->bias.set_data(
            ::torch::zeros({3}, options()));

        using L = typetorch::Linear<3, 3>;
        using LN = typetorch::LayerNorm<typetorch::Shape<3>>;
        L lin(/*with_bias=*/false);
        LN ln;
        lin->weight.set_data(raw[0]->as<::torch::nn::Linear>()->weight);
        ln->weight.set_data(raw[1]->as<::torch::nn::LayerNorm>()->weight);
        ln->bias.set_data(raw[1]->as<::torch::nn::LayerNorm>()->bias);

        auto seq = typetorch::Sequential(lin, ln);

        auto input = ::torch::randn({2, 3}, options());
        auto expected = raw->forward(input);
        auto actual = seq->forward(Input::unsafe_retain(input));

        if (!::torch::allclose(actual.unsafe_raw(), expected)) {
            ::fast_io::io::perrln("sequential_linear_layernorm: mismatch");
            ::std::exit(1);
        }
    }

    ::fast_io::io::println("typetorch nnModules sequential tests passed");
}
