#ifndef included_AMP_arrayHelpers
#define included_AMP_arrayHelpers

#include <array>
#include <cmath>
#include <ostream>


namespace AMP {


/****************************************************************
 * Output array                                                  *
 ****************************************************************/
template<std::size_t N>
inline std::ostream &operator<<( std::ostream &out, const std::array<double, N> &x )
{
    out << "(" << x[0];
    for ( size_t i = 1; i < N; i++ )
        out << "," << x[i];
    out << ")";
    return out;
}


/****************************************************************
 * Scalar-array arithmetic operations                            *
 ****************************************************************/
template<std::size_t N>
constexpr std::array<double, N> operator*( double a, const std::array<double, N> &b )
{
    if constexpr ( N == 1 ) {
        return { a * b[0] };
    } else if constexpr ( N == 2 ) {
        return { a * b[0], a * b[1] };
    } else if constexpr ( N == 3 ) {
        return { a * b[0], a * b[1], a * b[2] };
    } else {
        auto c = b;
        for ( size_t i = 0; i < N; i++ )
            c[i] *= a;
        return c;
    }
}
template<std::size_t N>
constexpr std::array<double, N> operator*( const std::array<double, N> &b, double a )
{
    if constexpr ( N == 1 ) {
        return { a * b[0] };
    } else if constexpr ( N == 2 ) {
        return { a * b[0], a * b[1] };
    } else if constexpr ( N == 3 ) {
        return { a * b[0], a * b[1], a * b[2] };
    } else {
        auto c = b;
        for ( size_t i = 0; i < N; i++ )
            c[i] *= a;
        return c;
    }
}
template<std::size_t N>
constexpr std::array<double, N> operator+( const std::array<double, N> &a, double b )
{
    if constexpr ( N == 1 ) {
        return { a[0] + b };
    } else if constexpr ( N == 2 ) {
        return { a[0] + b, a[1] + b };
    } else if constexpr ( N == 3 ) {
        return { a[0] + b, a[1] + b, a[2] + b };
    } else {
        auto c = a;
        for ( size_t i = 0; i < N; i++ )
            c[i] += b;
        return c;
    }
}
template<std::size_t N>
constexpr std::array<double, N> operator-( const std::array<double, N> &a, double b )
{
    if constexpr ( N == 1 ) {
        return { a[0] - b };
    } else if constexpr ( N == 2 ) {
        return { a[0] - b, a[1] - b };
    } else if constexpr ( N == 3 ) {
        return { a[0] - b, a[1] - b, a[2] - b };
    } else {
        auto c = a;
        for ( size_t i = 0; i < N; i++ )
            c[i] -= b;
        return c;
    }
}
template<std::size_t N>
constexpr std::array<double, N> operator-( const std::array<double, N> &a )
{
    if constexpr ( N == 1 ) {
        return { -a[0] };
    } else if constexpr ( N == 2 ) {
        return { -a[0], -a[1] };
    } else if constexpr ( N == 3 ) {
        return { -a[0], -a[1], -a[2] };
    } else {
        std::array<double, N> c = { 0 };
        for ( size_t i = 0; i < N; i++ )
            c[i] = -a[i];
        return c;
    }
}


/****************************************************************
 * array-array arithmetic operations                             *
 ****************************************************************/
template<std::size_t N>
constexpr std::array<double, N> operator+( const std::array<double, N> &x,
                                           const std::array<double, N> &y )
{
    if constexpr ( N == 1 ) {
        return { x[0] + y[0] };
    } else if constexpr ( N == 2 ) {
        return { x[0] + y[0], x[1] + y[1] };
    } else if constexpr ( N == 3 ) {
        return { x[0] + y[0], x[1] + y[1], x[2] + y[2] };
    } else {
        auto z = x;
        for ( size_t i = 0; i < N; i++ )
            z[i] += y[i];
        return z;
    }
}
template<std::size_t N>
constexpr std::array<double, N> operator-( const std::array<double, N> &x,
                                           const std::array<double, N> &y )
{
    if constexpr ( N == 1 ) {
        return { x[0] - y[0] };
    } else if constexpr ( N == 2 ) {
        return { x[0] - y[0], x[1] - y[1] };
    } else if constexpr ( N == 3 ) {
        return { x[0] - y[0], x[1] - y[1], x[2] - y[2] };
    } else {
        auto z = x;
        for ( size_t i = 0; i < N; i++ )
            z[i] -= y[i];
        return z;
    }
}


/****************************************************************
 * dot/cross products                                            *
 ****************************************************************/
template<std::size_t N>
constexpr double dot( const std::array<double, N> &x, const std::array<double, N> &y )
{
    if constexpr ( N == 1 ) {
        return x[0] * y[0];
    } else if constexpr ( N == 2 ) {
        return x[0] * y[0] + x[1] * y[1];
    } else if constexpr ( N == 3 ) {
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    } else {
        double d = 0;
        for ( size_t i = 0; i < N; i++ )
            d += x[i] * y[i];
        return d;
    }
}
constexpr double cross( const std::array<double, 2> &x, const std::array<double, 2> &y )
{
    return x[0] * y[1] - x[1] * y[0];
}
constexpr std::array<double, 3> cross( const std::array<double, 3> &x,
                                       const std::array<double, 3> &y )
{
    return { x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0] };
}
template<std::size_t N>
constexpr double norm( const std::array<double, N> &x )
{
    if constexpr ( N == 1 ) {
        return x[0] * x[0];
    } else if constexpr ( N == 2 ) {
        return x[0] * x[0] + x[1] * x[1];
    } else if constexpr ( N == 3 ) {
        return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
    } else {
        double n = 0;
        for ( size_t i = 0; i < N; i++ )
            n += x[i] * x[i];
        return n;
    }
}
template<std::size_t N>
constexpr std::array<double, N> normalize( const std::array<double, N> &x )
{
    double tmp = 1.0 / std::sqrt( dot( x, x ) );
    auto y     = x;
    for ( size_t i = 0; i < N; i++ )
        y[i] *= tmp;
    return y;
}

} // namespace AMP


#endif
