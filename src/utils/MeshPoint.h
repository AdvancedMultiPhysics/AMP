#ifndef included_AMP_MeshPoint
#define included_AMP_MeshPoint


#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <math.h>
#include <ostream>
#include <stdexcept>
#include <vector>


namespace AMP::Mesh {


/**
 * \class Point
 * \brief A class used to store information for a point
 */
class MeshPoint final
{
public:
    //! Empty constructor
    constexpr MeshPoint() noexcept : d_ndim( 0 ), d_data{ 0, 0, 0 } {}

    // Constructors
    constexpr MeshPoint( std::initializer_list<double> x ) : d_ndim( x.size() ), d_data{ 0, 0, 0 }
    {
        if ( d_ndim > 3 || d_ndim == 0 )
            throw std::logic_error( "Invalid Dimension" );
        auto it   = x.begin();
        d_data[0] = *it;
        for ( size_t d = 1; d < d_ndim; d++ )
            d_data[d] = *( ++it );
    }
    constexpr explicit MeshPoint( size_t ndim ) noexcept : d_ndim( ndim ), d_data{ 0, 0, 0 } {}
    constexpr explicit MeshPoint( const double &x ) noexcept : d_ndim( 1 ), d_data{ x, 0, 0 } {}
    constexpr explicit MeshPoint( const double &x, const double &y ) noexcept
        : d_ndim( 2 ), d_data{ x, y, 0 }
    {
    }
    constexpr explicit MeshPoint( const double &x, const double &y, const double &z ) noexcept
        : d_ndim( 3 ), d_data{ x, y, z }
    {
    }
    constexpr explicit MeshPoint( const size_t ndim, const double *x ) noexcept
        : d_ndim( ndim ), d_data{ 0, 0, 0 }
    {
        for ( size_t d = 0; d < d_ndim; d++ )
            d_data[d] = x[d];
    }
    constexpr MeshPoint( const size_t ndim, std::initializer_list<double> x )
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
    constexpr MeshPoint( const std::array<double, NDIM> &x ) : d_ndim( NDIM ), d_data{ 0, 0, 0 }
    {
        for ( size_t d = 0; d < NDIM; d++ )
            d_data[d] = x[d];
    }
    MeshPoint( const std::vector<double> &x ) : d_ndim( x.size() ), d_data{ 0, 0, 0 }
    {
        for ( size_t d = 0; d < x.size(); d++ )
            d_data[d] = x[d];
    }

    // typecast operators
    constexpr operator std::array<double, 1>() const { return { d_data[0] }; }
    constexpr operator std::array<double, 2>() const { return { d_data[0], d_data[1] }; }
    constexpr operator std::array<double, 3>() const { return { d_data[0], d_data[1], d_data[2] }; }


    // Copy/assignment operators
    constexpr MeshPoint( MeshPoint && ) noexcept                 = default;
    constexpr MeshPoint( const MeshPoint & ) noexcept            = default;
    constexpr MeshPoint &operator=( MeshPoint && ) noexcept      = default;
    constexpr MeshPoint &operator=( const MeshPoint & ) noexcept = default;

    // Number of dimensions
    constexpr size_t size() const { return d_ndim; }
    constexpr uint8_t ndim() const { return d_ndim; }
    constexpr void setNdim( uint8_t N ) { d_ndim = N; }

    // Accessors
    constexpr double *data() noexcept { return d_data; }
    constexpr const double *data() const noexcept { return d_data; }
    constexpr double &x() noexcept { return d_data[0]; }
    constexpr double &y() noexcept { return d_data[1]; }
    constexpr double &z() noexcept { return d_data[2]; }
    constexpr const double &x() const { return d_data[0]; }
    constexpr const double &y() const { return d_data[1]; }
    constexpr const double &z() const { return d_data[2]; }
    constexpr double &operator[]( std::size_t i )
    {
        if ( i >= d_ndim )
            throw std::logic_error( "Invalid index" );
        return d_data[i];
    }
    constexpr const double &operator[]( std::size_t i ) const
    {
        if ( i >= d_ndim )
            throw std::logic_error( "Invalid index" );
        return d_data[i];
    }

    // Iterators
    constexpr double *begin() noexcept { return d_data; }
    constexpr double *end() noexcept { return d_data + d_ndim; }
    constexpr const double *begin() const noexcept { return d_data; }
    constexpr const double *end() const noexcept { return d_data + d_ndim; }

    // Arithmetic operators
    constexpr MeshPoint &operator+=( const double rhs ) noexcept
    {
        d_data[0] += rhs;
        d_data[1] += rhs;
        d_data[2] += rhs;
        return *this;
    }
    constexpr MeshPoint &operator+=( const MeshPoint &rhs ) noexcept
    {
        d_data[0] += rhs.d_data[0];
        d_data[1] += rhs.d_data[1];
        d_data[2] += rhs.d_data[2];
        return *this;
    }
    constexpr MeshPoint &operator-=( const double rhs ) noexcept
    {
        d_data[0] -= rhs;
        d_data[1] -= rhs;
        d_data[2] -= rhs;
        return *this;
    }
    constexpr MeshPoint &operator-=( const MeshPoint &rhs ) noexcept
    {
        d_data[0] -= rhs.d_data[0];
        d_data[1] -= rhs.d_data[1];
        d_data[2] -= rhs.d_data[2];
        return *this;
    }
    constexpr MeshPoint &operator*=( const double &rhs ) noexcept
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
    constexpr double norm() const
    {
        return d_data[0] * d_data[0] + d_data[1] * d_data[1] + d_data[2] * d_data[2];
    }

    //! Return the magnitude
    inline double abs() const { return std::sqrt( norm() ); }

    //! Print the point
    void print( std::ostream &os ) const;

    //! Print the point
    std::string print() const;

private:
    uint8_t d_ndim;
    double d_data[3];
};


using Point = MeshPoint;


/****************************************************************
 * Operator overloading                                          *
 ****************************************************************/
constexpr AMP::Mesh::MeshPoint operator+( const AMP::Mesh::MeshPoint &a,
                                          const AMP::Mesh::MeshPoint &b )
{
    double c[3] = { a.x() + b.x(), a.y() + b.y(), a.z() + b.z() };
    return AMP::Mesh::MeshPoint( a.size(), c );
}
constexpr AMP::Mesh::MeshPoint operator+( const AMP::Mesh::MeshPoint &a, const double &b )
{
    double c[3] = { a.x() + b, a.y() + b, a.z() + b };
    return AMP::Mesh::MeshPoint( a.size(), c );
}
constexpr AMP::Mesh::MeshPoint operator+( const double &a, const AMP::Mesh::MeshPoint &b )
{
    double c[3] = { a + b.x(), a + b.y(), a + b.z() };
    return AMP::Mesh::MeshPoint( b.size(), c );
}
constexpr AMP::Mesh::MeshPoint operator-( const AMP::Mesh::MeshPoint &a,
                                          const AMP::Mesh::MeshPoint &b )
{
    double c[3] = { a.x() - b.x(), a.y() - b.y(), a.z() - b.z() };
    return AMP::Mesh::MeshPoint( a.size(), c );
}
constexpr AMP::Mesh::MeshPoint operator-( const AMP::Mesh::MeshPoint &a, const double &b )
{
    double c[3] = { a.x() - b, a.y() - b, a.z() - b };
    return AMP::Mesh::MeshPoint( a.size(), c );
}
constexpr AMP::Mesh::MeshPoint operator-( const double &a, const AMP::Mesh::MeshPoint &b )
{
    double c[3] = { a - b.x(), a - b.y(), a - b.z() };
    return AMP::Mesh::MeshPoint( b.size(), c );
}
constexpr AMP::Mesh::MeshPoint operator-( const AMP::Mesh::MeshPoint &a )
{
    double c[3] = { -a.x(), -a.y(), -a.z() };
    return AMP::Mesh::MeshPoint( a.size(), c );
}
constexpr AMP::Mesh::MeshPoint operator*( const AMP::Mesh::MeshPoint &a, const double &b )
{
    double c[3] = { b * a.x(), b * a.y(), b * a.z() };
    return AMP::Mesh::MeshPoint( a.size(), c );
}
constexpr AMP::Mesh::MeshPoint operator*( const double &a, const AMP::Mesh::MeshPoint &b )
{
    double c[3] = { a * b.x(), a * b.y(), a * b.z() };
    return AMP::Mesh::MeshPoint( b.size(), c );
}


/****************************************************************
 * Helper functions                                              *
 ****************************************************************/
inline double abs( const AMP::Mesh::MeshPoint &x ) { return x.abs(); }
constexpr double dot( const AMP::Mesh::MeshPoint &a, const AMP::Mesh::MeshPoint &b )
{
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
constexpr AMP::Mesh::MeshPoint cross( const AMP::Mesh::MeshPoint &a, const AMP::Mesh::MeshPoint &b )
{
    return AMP::Mesh::MeshPoint( a.y() * b.z() - a.z() * b.y(),
                                 a.z() * b.x() - a.x() * b.z(),
                                 a.x() * b.y() - a.y() * b.x() );
}
inline AMP::Mesh::MeshPoint normalize( const AMP::Mesh::MeshPoint &x )
{
    auto y   = x;
    double t = 1.0 / x.abs();
    y.x() *= t;
    y.y() *= t;
    y.z() *= t;
    return y;
}
std::ostream &operator<<( std::ostream &, const AMP::Mesh::MeshPoint & );


} // namespace AMP::Mesh


#endif
