import std;
import libtorch;
import typetorch;
import fast_io;

namespace {

using Indices1D = typetorch::Tensor<typetorch::Shape<4>, typetorch::DType::I64,
                                     typetorch::Device::CPU, typetorch::Layout::Contiguous>;
using Indices2D = typetorch::Tensor<typetorch::Shape<2, 3>, typetorch::DType::I64,
                                     typetorch::Device::CPU, typetorch::Layout::Contiguous>;

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
    // --- Test 1: forward with 1D indices ---
    {
        ::torch::nn::EmbeddingImpl raw(
            ::torch::nn::EmbeddingOptions(10, 4));
        using Emb = typetorch::Embedding<10, 4>;
        Emb typed;

        typed->weight.set_data(raw.weight);

        auto indices = ::torch::tensor({1L, 3L, 5L, 7L},
            ::torch::TensorOptions{}.dtype(::torch::kLong));
        auto expected = raw.forward(indices);
        auto actual = typed->forward(Indices1D::unsafe_retain(indices));

        expect_allclose("embedding_1d", actual.unsafe_raw(), expected);
    }

    // --- Test 2: forward with 2D indices ---
    {
        ::torch::nn::EmbeddingImpl raw(
            ::torch::nn::EmbeddingOptions(10, 4));
        using Emb = typetorch::Embedding<10, 4>;
        Emb typed;
        typed->weight.set_data(raw.weight);

        auto indices = ::torch::tensor({{0L, 2L, 4L}, {6L, 8L, 1L}},
            ::torch::TensorOptions{}.dtype(::torch::kLong));
        auto expected = raw.forward(indices);
        auto actual = typed->forward(Indices2D::unsafe_retain(indices));

        expect_allclose("embedding_2d", actual.unsafe_raw(), expected);
    }

    // --- Test 3: to(dtype) F16 ---
    {
        ::torch::nn::EmbeddingImpl raw(
            ::torch::nn::EmbeddingOptions(10, 4));
        using Emb = typetorch::Embedding<10, 4>;
        Emb typed;
        typed->weight.set_data(raw.weight);

        auto f16 = typed->template to<typetorch::DType::F16>();
        if (!f16->weight.equal(raw.weight.to(::torch::kHalf))) {
            ::fast_io::io::perrln("embedding_to_f16: weight mismatch");
            ::std::exit(1);
        }

        auto indices = ::torch::tensor({1L, 3L},
            ::torch::TensorOptions{}.dtype(::torch::kLong));
        using F16Indices = typetorch::Tensor<typetorch::Shape<2>, typetorch::DType::I64,
                                              typetorch::Device::CPU, typetorch::Layout::Contiguous>;
        auto actual_f16 = f16->forward(F16Indices::unsafe_retain(indices));
        auto expected_f16 = raw.weight.to(::torch::kHalf).index_select(0, indices);
        if (!actual_f16.unsafe_raw().to(::torch::kFloat).equal(
                expected_f16.to(::torch::kFloat))) {
            ::fast_io::io::perrln("embedding_to_f16: forward mismatch");
            ::std::exit(1);
        }
    }

    // --- Test 4: typed_weight shape ---
    {
        using Emb = typetorch::Embedding<10, 4>;
        Emb typed;
        auto w = typed->typed_weight();
        static_assert(::std::same_as<decltype(w)::shape, typetorch::Shape<10, 4>>);
    }

    ::fast_io::io::println("typetorch nnModules embedding tests passed");
}
