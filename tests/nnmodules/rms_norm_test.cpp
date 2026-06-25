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
        using RN = typetorch::RMSNorm<typetorch::Shape<2, 3>>;
        RN typed;

        // Set weight to known values
        typed->weight.set_data(::torch::ones({2, 3}, options()).mul(0.5));

        auto input = ::torch::randn({2, 3}, options());
        auto typed_result = typed->forward(Input23::unsafe_retain(input));

        // Compare against torch::rms_norm
        auto raw_result = ::torch::rms_norm(input, {2, 3}, typed->weight);

        expect_allclose("rmsnorm_affine", typed_result.unsafe_raw(), raw_result);
    }

    // --- Test 2: forward affine=false ---
    {
        using RN = typetorch::RMSNorm<typetorch::Shape<4>>;
        RN typed(/*eps=*/1e-5, /*elementwise_affine=*/false);

        auto input = ::torch::randn({2, 2, 4}, options());
        auto typed_result = typed->forward(Input224::unsafe_retain(input));

        auto raw_result = ::torch::rms_norm(input, {4}, {}, 1e-5);

        expect_allclose("rmsnorm_no_affine", typed_result.unsafe_raw(), raw_result);
    }

    // --- Test 3: to(dtype) F16 ---
    {
        using RN = typetorch::RMSNorm<typetorch::Shape<2, 3>>;
        RN typed;
        typed->weight.set_data(::torch::ones({2, 3}, options()));

        auto f16 = typed->template to<typetorch::DType::F16>();
        if (!f16->weight.equal(typed->weight.to(::torch::kHalf))) {
            ::fast_io::io::perrln("rmsnorm_to_f16: weight mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 4: typed_weight shape ---
    {
        using RN = typetorch::RMSNorm<typetorch::Shape<2, 3>>;
        RN typed;
        auto w = typed->typed_weight();
        static_assert(::std::same_as<decltype(w)::shape, typetorch::Shape<2, 3>>);
    }

    ::fast_io::io::println("typetorch nnModules rms_norm tests passed");
}
