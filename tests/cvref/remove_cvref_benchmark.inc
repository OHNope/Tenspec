namespace remove_cvref_benchmark
{

template <::std::size_t I>
struct payload
{
    ::std::byte bytes[(I % 251) + 1];
};

template <::std::size_t I>
using qualified_payload = ::std::conditional_t<
    (I % 3) == 0, payload<I> const &,
    ::std::conditional_t<(I % 3) == 1, payload<I> volatile &&,
                        payload<I> const volatile>>;

template <::std::size_t I>
using normalized_payload = NORMALIZE_TYPE(qualified_payload<I>);

template <::std::size_t... Is>
consteval auto compile_time_checksum(::std::index_sequence<Is...>) -> ::std::size_t
{
    static_assert((::std::same_as<normalized_payload<Is>, payload<Is>> && ...));
    return (sizeof(normalized_payload<Is>) + ... + 0);
}

inline constexpr auto instance_count = 2048UZ;
inline constexpr auto expected_checksum =
    compile_time_checksum(::std::make_index_sequence<instance_count>{});

template <::std::size_t... Is>
[[gnu::noinline]] auto runtime_checksum(::std::size_t seed,
                                        ::std::index_sequence<Is...>) noexcept
    -> ::std::size_t
{
    ((seed = (seed * 0x9e3779b185ebca87ULL) ^
             (sizeof(normalized_payload<Is>) + alignof(normalized_payload<Is>))),
      ...);
    return seed;
}

} // namespace remove_cvref_benchmark

int main(int argc, char **)
{
    using namespace remove_cvref_benchmark;
    ::std::size_t value = static_cast<::std::size_t>(argc);
    constexpr auto indices = ::std::make_index_sequence<instance_count>{};
    for (auto iteration = 0; iteration != 200000; ++iteration)
        value = runtime_checksum(value, indices);

    ::fast_io::io::println(expected_checksum, " ", value);
    return value == 0;
}
