module;

#include <fast_io.h>

module fast_io;

namespace fastio_detail
{

template <typename T>
void write_one(T const &value)
{
	::fast_io::io::print(value);
}

template <typename T>
void err_write_one(T const &value)
{
	::fast_io::io::perr(value);
}

} // namespace fastio_detail

namespace fast_io
{
inline namespace io
{

void write(char const *value, size_type size)
{
	fastio_detail::write_one(::fast_io::mnp::strvw(value, value + size));
}

void write(char value)
{
	fastio_detail::write_one(value);
}
void write(bool value)
{
	fastio_detail::write_one(value);
}
void write(short value)
{
	fastio_detail::write_one(value);
}
void write(unsigned short value)
{
	fastio_detail::write_one(value);
}
void write(int value)
{
	fastio_detail::write_one(value);
}
void write(unsigned int value)
{
	fastio_detail::write_one(value);
}
void write(long value)
{
	fastio_detail::write_one(value);
}
void write(unsigned long value)
{
	fastio_detail::write_one(value);
}
void write(long long value)
{
	fastio_detail::write_one(value);
}
void write(unsigned long long value)
{
	fastio_detail::write_one(value);
}
void write(float value)
{
	fastio_detail::write_one(value);
}
void write(double value)
{
	fastio_detail::write_one(value);
}
void write(long double value)
{
	fastio_detail::write_one(static_cast<double>(value));
}

void errwrite(char const *value, size_type size)
{
	fastio_detail::err_write_one(::fast_io::mnp::strvw(value, value + size));
}

void errwrite(char value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(bool value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(short value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(unsigned short value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(int value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(unsigned int value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(long value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(unsigned long value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(long long value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(unsigned long long value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(float value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(double value)
{
	fastio_detail::err_write_one(value);
}
void errwrite(long double value)
{
	fastio_detail::err_write_one(static_cast<double>(value));
}

} // namespace io
} // namespace fast_io