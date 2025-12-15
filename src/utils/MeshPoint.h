#ifndef included_AMP_MeshPoint
#define included_AMP_MeshPoint


#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <math.h>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>


namespace AMP::Mesh {


/**
 * \class Point
 * \brief A class used to store information for a point
 */
template<class TYPE>
class MeshPoint final
{
public:
    //! Empty constructor
    constexpr MeshPoint() noexcept : d_ndim( 0 ), d_data{ 0, 0, 0 } {}

    // Constructors
    constexpr MeshPoint( std::initializer_list<TYPE> x ) : d_ndim( x.size() ), d_data{ 0, 0, 0 }
    {
        if ( d_ndim > 3 || d_ndim == 0 )
            throw std::logic_error( "Invalid Dimension" );
        auto it   = x.begin();
        d_data[0] = *it;
        for ( size_t d = 1; d < d_ndim; d++ )
            d_data[d] = *( ++it );
    }
    constexpr explicit MeshPoint( size_t ndim ) noexcept : d_ndim( ndim ), d_data{ 0, 0, 0 } {}
    constexpr explicit MeshPoint( const TYPE &x ) noexcept : d_ndim( 1 ), d_data{ x, 0, 0 } {}
    constexpr explicit MeshPoint( const TYPE &x, const TYPE &y ) noexcept
        : d_ndim( 2 ), d_data{ x, y, 0 }
    {
    }
    constexpr explicit MeshPoint( const TYPE &x, const TYPE &y, const TYPE &z ) noexcept
        : d_ndim( 3 ), d_data{ x, y, z }
    {
    }
    constexpr explicit MeshPoint( const size_t ndim, const TYPE *x ) noexcept
        : d_ndim( ndim ), d_data{ 0, 0, 0 }
    {
        for ( size_t d = 0; d < d_ndim; d++ )
            d_data[d] = x[d];
    }
    constexpr MeshPoint( const size_t ndim, std::initializer_list<TYPE> x )
        : d_ndim( ndim ), d_data{ 0, 0, 0 }
    {
        if ( d_ndim > 3 )
            throw std::logic_error( "Invalid Dimension" );
        auto it   = x.begin();
        d_data[0] = *it;
        for ( size_t d = 1; d < std::min<size_t>( d_ndim, x.size() ); d++ )
            d_data[d] = *( ++it );
    }
    template<std::size_t NDIM>
    constexpr MeshPoint( const std::array<TYPE, NDIM> &x ) : d_ndim( NDIM ), d_data{ 0, 0, 0 }
    {
        for ( size_t d = 0; d < NDIM; d++ )
            d_data[d] = x[d];
    }
    MeshPoint( const std::vector<TYPE> &x ) : d_ndim( x.size() ), d_data{ 0, 0, 0 }
    {
        for ( size_t d = 0; d < x.size(); d++ )
            d_data[d] = x[d];
    }

    // Typecast operators
    constexpr operator std::array<TYPE, 1>() const { return { d_data[0] }; }
    constexpr operator std::array<TYPE, 2>() const { return { d_data[0], d_data[1] }; }
    constexpr operator std::array<TYPE, 3>() const { return { d_data[0], d_data[1], d_data[2] }; }


    // Copy/assigment operators
    constexpr MeshPoint( MeshPoint && ) noexcept      = default;
    constexpr MeshPoint( const MeshPoint & ) noexcept = default;
    constexpr MeshPoint &operator=( MeshPoint && ) noexcept = default;
    constexpr MeshPoint &operator=( const MeshPoint & ) noexcept = default;

    // Copy point of a different type
    template<class TYPE2>
    constexpr MeshPoint( const MeshPoint<TYPE2> &rhs ) noexcept : d_ndim( rhs.d_ndim )
    {
        d_data[0] = rhs.d_data[0];
        d_data[1] = rhs.d_data[1];
        d_data[2] = rhs.d_data[2];
    }

    // Number of dimensions
    constexpr size_t size() const { return d_ndim; }
    constexpr uint8_t ndim() const { return d_ndim; }
    constexpr void setNdim( uint8_t N ) { d_ndim = N; }

    // Accessors
    constexpr TYPE *data() noexcept { return d_data; }
    constexpr const TYPE *data() const noexcept { return d_data; }
    constexpr TYPE &x() noexcept { return d_data[0]; }
    constexpr TYPE &y() noexcept { return d_data[1]; }
    constexpr TYPE &z() noexcept { return d_data[2]; }
    constexpr const TYPE &x() const { return d_data[0]; }
    constexpr const TYPE &y() const { return d_data[1]; }
    constexpr const TYPE &z() const { return d_data[2]; }
    constexpr TYPE &operator[]( std::size_t i )
    {
        if ( i >= d_ndim )
            throw std::logic_error( "Invalid index" );
        return d_data[i];
    }
    constexpr const TYPE &operator[]( std::size_t i ) const
    {
        if ( i >= d_ndim )
            throw std::logic_error( "Invalid index" );
        return d_data[i];
    }

    // Iterators
    constexpr TYPE *begin() noexcept { return d_data; }
    constexpr TYPE *end() noexcept { return d_data + d_ndim; }
    constexpr const TYPE *begin() const noexcept { return d_data; }
    constexpr const TYPE *end() const noexcept { return d_data + d_ndim; }

    // Arithmetic operators
    constexpr MeshPoint &operator+=( const TYPE rhs ) noexcept
    {
        d_data[0] += rhs;
        d_data[1] += rhs;
        d_data[2] += rhs;
        return *this;
    }
    template<class TYPE2>
    constexpr MeshPoint &operator+=( const MeshPoint<TYPE2> &rhs ) noexcept
    {
        d_data[0] += rhs.d_data[0];
        d_data[1] += rhs.d_data[1];
        d_data[2] += rhs.d_data[2];
        return *this;
    }
    constexpr MeshPoint &operator-=( const TYPE rhs ) noexcept
    {
        d_data[0] -= rhs;
        d_data[1] -= rhs;
        d_data[2] -= rhs;
        return *this;
    }
    template<class TYPE2>
    constexpr MeshPoint &operator-=( const MeshPoint<TYPE2> &rhs ) noexcept
    {
        d_data[0] -= rhs.d_data[0];
        d_data[1] -= rhs.d_data[1];
        d_data[2] -= rhs.d_data[2];
        return *this;
    }
    constexpr MeshPoint &operator*=( const TYPE &rhs ) noexcept
    {
        d_data[0] *= rhs;
        d_data[1] *= rhs;
        d_data[2] *= rhs;
        return *this;
    }

    // Comparison operators
    constexpr bool operator==( const MeshPoint &rhs ) const
    {
        return d_ndim == rhs.d_ndim && d_data[0] == rhs.d_data[0] && d_data[1] == rhs.d_data[1] &&
               d_data[2] == rhs.d_data[2];
    }
    constexpr bool operator!=( const MeshPoint &rhs ) const
    {
        return d_ndim != rhs.d_ndim || d_data[0] != rhs.d_data[0] || d_data[1] != rhs.d_data[1] ||
               d_data[2] != rhs.d_data[2];
    }
    constexpr bool operator>( const MeshPoint &rhs ) const
    {
        if ( d_ndim != rhs.d_ndim )
            return d_ndim > rhs.d_ndim;
        for ( int d = 0; d < 3; d++ ) {
            if ( d_data[d] != rhs.d_data[d] )
                return d_data[d] > rhs.d_data[d];
        }
        return false;
    }
    constexpr bool operator>=( const MeshPoint &rhs ) const
    {
        if ( d_ndim != rhs.d_ndim )
            return d_ndim > rhs.d_ndim;
        for ( int d = 0; d < 3; d++ ) {
            if ( d_data[d] != rhs.d_data[d] )
                return d_data[d] > rhs.d_data[d];
        }
        return true;
    }
    constexpr bool operator<( const MeshPoint &rhs ) const
    {
        if ( d_ndim != rhs.d_ndim )
            return d_ndim > rhs.d_ndim;
        for ( int d = 0; d < 3; d++ ) {
            if ( d_data[d] != rhs.d_data[d] )
                return d_data[d] < rhs.d_data[d];
        }
        return false;
    }
    constexpr bool operator<=( const MeshPoint &rhs ) const
    {
        if ( d_ndim != rhs.d_ndim )
            return d_ndim < rhs.d_ndim;
        for ( int d = 0; d < 3; d++ ) {
            if ( d_data[d] != rhs.d_data[d] )
                return d_data[d] < rhs.d_data[d];
        }
        return true;
    }

    //! Return the squared magnitude
    constexpr TYPE norm() const
    {
        return d_data[0] * d_data[0] + d_data[1] * d_data[1] + d_data[2] * d_data[2];
    }

    //! Return the magnitude
    inline TYPE abs() const { return std::sqrt( norm() ); }

    //! Print the point
    inline void print( std::ostream &os ) const
    {
        if ( d_ndim == 0 ) {
            os << "()";
        } else {
            os << "(" << d_data[0];
            for ( int d = 1; d < d_ndim; d++ )
                os << "," << d_data[d];
            os << ")";
        }
    }

    //! Print the point
    inline std::string print() const
    {
        std::ostringstream stream;
        print( stream );
        return stream.str();
    }

private:
    uint8_t d_ndim;
    TYPE d_data[3];
};


using Point = MeshPoint<double>;


/****************************************************************
 * Operator overloading                                          *
 ****************************************************************/
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator+( const AMP::Mesh::MeshPoint<TYPE> &a,
                                                const AMP::Mesh::MeshPoint<TYPE> &b )
{
    TYPE c[3] = { a.x() + b.x(), a.y() + b.y(), a.z() + b.z() };
    return AMP::Mesh::MeshPoint<TYPE>( a.size(), c );
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator+( const AMP::Mesh::MeshPoint<TYPE> &a, const TYPE &b )
{
    TYPE c[3] = { a.x() + b, a.y() + b, a.z() + b };
    return AMP::Mesh::MeshPoint<TYPE>( a.size(), c );
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator+( const TYPE &a, const AMP::Mesh::MeshPoint<TYPE> &b )
{
    TYPE c[3] = { a + b.x(), a + b.y(), a + b.z() };
    return AMP::Mesh::MeshPoint<TYPE>( b.size(), c );
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator-( const AMP::Mesh::MeshPoint<TYPE> &a,
                                                const AMP::Mesh::MeshPoint<TYPE> &b )
{
    TYPE c[3] = { a.x() - b.x(), a.y() - b.y(), a.z() - b.z() };
    return AMP::Mesh::MeshPoint<TYPE>( a.size(), c );
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator-( const AMP::Mesh::MeshPoint<TYPE> &a, const TYPE &b )
{
    TYPE c[3] = { a.x() - b, a.y() - b, a.z() - b };
    return AMP::Mesh::MeshPoint<TYPE>( a.size(), c );
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator-( const TYPE &a, const AMP::Mesh::MeshPoint<TYPE> &b )
{
    TYPE c[3] = { a - b.x(), a - b.y(), a - b.z() };
    return AMP::Mesh::MeshPoint<TYPE>( b.size(), c );
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator-( const AMP::Mesh::MeshPoint<TYPE> &a )
{
    TYPE c[3] = { -a.x(), -a.y(), -a.z() };
    return AMP::Mesh::MeshPoint<TYPE>( a.size(), c );
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator*( const AMP::Mesh::MeshPoint<TYPE> &a, const TYPE &b )
{
    auto c = a;
    c.x() *= b;
    c.y() *= b;
    c.z() *= b;
    return c;
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> operator*( const TYPE &a, const AMP::Mesh::MeshPoint<TYPE> &b )
{
    auto c = b;
    c.x() *= a;
    c.y() *= a;
    c.z() *= a;
    return c;
}


/****************************************************************
 * Helper functions                                              *
 ****************************************************************/
template<class TYPE>
constexpr TYPE abs( const AMP::Mesh::MeshPoint<TYPE> &x )
{
    return x.abs();
}
template<class TYPE>
constexpr TYPE dot( const AMP::Mesh::MeshPoint<TYPE> &a, const AMP::Mesh::MeshPoint<TYPE> &b )
{
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> cross( const AMP::Mesh::MeshPoint<TYPE> &a,
                                            const AMP::Mesh::MeshPoint<TYPE> &b )
{
    return AMP::Mesh::MeshPoint<TYPE>( a.y() * b.z() - a.z() * b.y(),
                                       a.z() * b.x() - a.x() * b.z(),
                                       a.x() * b.y() - a.y() * b.x() );
}
template<class TYPE>
constexpr AMP::Mesh::MeshPoint<TYPE> normalize( const AMP::Mesh::MeshPoint<TYPE> &x )
{
    auto y   = x;
    double t = 1.0 / x.abs();
    y.x() *= t;
    y.y() *= t;
    y.z() *= t;
    return y;
}
template<class TYPE>
std::ostream &operator<<( std::ostream &out, const AMP::Mesh::MeshPoint<TYPE> &x )
{
    x.print( out );
    return out;
}

} // namespace AMP::Mesh


#endif
