#include <algorithm>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>

#include "ProfilerApp.h"


// #define R_INT 0x20000000
#define R_INT 1000000


// Compute the L2 norm of a vector ||x||
inline double L2norm( const AMP::Array<double> &x, const bool *mask = nullptr )
{
    size_t N    = x.length();
    double norm = 0;
    if ( mask == nullptr ) {
        for ( size_t i = 0; i < N; i++ )
            norm += x( i ) * x( i );
    } else {
        for ( size_t i = 0; i < N; i++ )
            norm += mask[i] ? x( i ) * x( i ) : 0;
    }
    return std::sqrt( norm );
}


// Compute the L2 error norm of a vector ||x1-x2||
inline double
L2errNorm( const AMP::Array<double> &x1, const AMP::Array<double> &x2, const bool *mask = nullptr )
{
    AMP_ASSERT( x1.size() == x2.size() );
    size_t N    = x1.length();
    double norm = 0;
    if ( mask == nullptr ) {
        for ( size_t i = 0; i < N; i++ )
            norm += ( x1( i ) - x2( i ) ) * ( x1( i ) - x2( i ) );
    } else {
        for ( size_t i = 0; i < N; i++ )
            norm += mask[i] ? ( x1( i ) - x2( i ) ) * ( x1( i ) - x2( i ) ) : 0;
    }
    return std::sqrt( norm );
}


// Structure to store a point
template<int NDIM>
struct PointInt {
    int x[NDIM];
    PointInt() { memset( x, 0, sizeof( x ) ); }
    inline uint64_t R2() const
    {
        uint64_t R2 = 0;
        for ( auto &elem : x ) {
            uint64_t x2 = abs( elem );
            R2 += x2 * x2;
        }
        return R2;
    }
    inline bool operator==( const PointInt &rhs ) const
    {
        bool equal = true;
        for ( int d = 0; d < NDIM; d++ )
            equal = equal && x[d] == rhs.x[d];
        return equal;
    }
    inline bool operator!=( const PointInt &rhs ) const { return !operator==( rhs ); }
    inline bool operator>=( const PointInt &rhs ) const
    {
        for ( int d = 0; d < NDIM; d++ ) {
            if ( x[d] > rhs.x[d] )
                return true;
            if ( rhs.x[d] > x[d] )
                return false;
        }
        return true;
    }
    inline bool operator<=( const PointInt &rhs ) const
    {
        for ( int d = 0; d < NDIM; d++ ) {
            if ( x[d] < rhs.x[d] )
                return true;
            if ( rhs.x[d] < x[d] )
                return false;
        }
        return true;
    }
    inline bool operator>( const PointInt &rhs ) const
    {
        return operator>=( rhs ) && !operator==( rhs );
    }
    inline bool operator<( const PointInt &rhs ) const
    {
        return operator<=( rhs ) && !operator==( rhs );
    }
};


// Create a set of random points in a Nd-hypersphere
// Note: we use a combination of a fixed grid and random points to control the error
template<class TYPE>
AMP::Array<TYPE> createRandomPoints( int ndim, int N );
template<int NDIM>
std::vector<PointInt<NDIM>> createRandomPointsInt( int N )
{
    std::vector<PointInt<NDIM>> points;
    if ( N == NDIM + 1 ) {
        points.resize( N );
        for ( int d = 0; d < NDIM; d++ )
            points[d + 1].x[d] = R_INT;
        return points;
    }
    PROFILE( "createRandomPoints", 1 );
    const uint64_t R_INT2 = static_cast<uint64_t>( R_INT ) * static_cast<uint64_t>( R_INT );
    points.reserve( N + 10 );
    // Create a logical Nd-hypercube on [-1,1] and keep only the points within R<=R_INT
    if ( N > 10 ) {
        int Nd          = static_cast<int>( floor( std::pow( N, 1.0 / NDIM ) ) );
        Nd              = std::min( N / 2, Nd );
        Nd              = 2 * ( Nd / 2 ) + 1;
        const double dx = 2.0 / static_cast<double>( Nd - 1 );
        for ( size_t k = 0; k < std::pow( Nd, NDIM ); k++ ) {
            PointInt<NDIM> p;
            size_t j = k;
            for ( int d = 0; d < NDIM; d++ ) {
                double pos = ( j % Nd ) * dx - 1.0;
                p.x[d]     = static_cast<int>( round( static_cast<double>( R_INT ) * pos ) );
                j /= Nd;
            }
            if ( p.R2() <= R_INT2 )
                points.push_back( p );
        }
    }
    // Add random points
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_int_distribution<int> dist( -R_INT, R_INT );
    while ( static_cast<int>( points.size() ) < N + 5 ) {
        PointInt<NDIM> p;
        for ( int d = 0; d < NDIM; d++ )
            p.x[d] = dist( gen );
        if ( p.R2() <= R_INT2 )
            points.push_back( p );
    }
    // Check for duplicates and remove them
    std::sort( points.begin(), points.end() );
    for ( size_t i = points.size() - 1; i > 0; i-- ) {
        if ( points[i] == points[i - 1] ) {
            points[i] = points.back();
            points.resize( points.size() - 1 );
        }
    }
    points.resize( std::min<size_t>( points.size(), N ) );
    // Resort the points in random order
    std::shuffle( points.begin(), points.end(), std::mt19937( std::random_device()() ) );
    return points;
}
AMP::Array<int> getPointListInt( int ndim, int N )
{
    AMP::Array<int> points;
    if ( ndim == 1 ) {
        auto points2 = createRandomPointsInt<1>( N );
        points.resize( ndim, points2.size() );
        for ( size_t i = 0; i < points2.size(); i++ )
            points( 0, i ) = points2[i].x[0];
    } else if ( ndim == 2 ) {
        auto points2 = createRandomPointsInt<2>( N );
        points.resize( ndim, points2.size() );
        for ( size_t i = 0; i < points2.size(); i++ ) {
            points( 0, i ) = points2[i].x[0];
            points( 1, i ) = points2[i].x[1];
        }
    } else if ( ndim == 3 ) {
        auto points2 = createRandomPointsInt<3>( N );
        points.resize( ndim, points2.size() );
        for ( size_t i = 0; i < points2.size(); i++ ) {
            points( 0, i ) = points2[i].x[0];
            points( 1, i ) = points2[i].x[1];
            points( 2, i ) = points2[i].x[2];
        }
    } else if ( ndim == 4 ) {
        auto points2 = createRandomPointsInt<4>( N );
        points.resize( ndim, points2.size() );
        for ( size_t i = 0; i < points2.size(); i++ ) {
            for ( int d = 0; d < ndim; d++ )
                points( d, i ) = points2[i].x[d];
        }
    } else if ( ndim == 5 ) {
        auto points2 = createRandomPointsInt<5>( N );
        points.resize( ndim, points2.size() );
        for ( size_t i = 0; i < points2.size(); i++ ) {
            for ( int d = 0; d < ndim; d++ )
                points( d, i ) = points2[i].x[d];
        }
    }
    return points;
}
template<>
AMP::Array<int> createRandomPoints<int>( int ndim, int N )
{
    return getPointListInt( ndim, N );
}
template<>
AMP::Array<double> createRandomPoints<double>( int ndim, int N )
{
    PROFILE( "createRandomPointsDouble", 1 );
    AMP::Array<double> points( ndim, N );
    int i = 0;
    // Create a Nd-hypercube on [-1,1] and keep only the points within R<=1
    int Nd = static_cast<int>( floor( std::pow( N, 1.0 / ndim ) ) );
    Nd     = std::min( N / 2, Nd );
    for ( size_t k = 0; k < std::pow( Nd, ndim ); k++ ) {
        double x[10] = { 0 };
        double R     = 0.0;
        size_t j     = k;
        for ( int d = 0; d < ndim; d++ ) {
            x[d] = 2.0 * ( ( j % Nd ) + 0.5 ) / static_cast<double>( Nd ) - 1.0;
            R += x[d] * x[d];
            j /= Nd;
        }
        if ( R <= 1.0 ) {
            for ( int d = 0; d < ndim; d++ )
                points( d, i ) = x[d];
            i++;
        }
    }
    // Add random points
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dist( -1, 1 );
    while ( i < N ) {
        double x[10] = { 0 };
        double R     = 0.0;
        for ( int d = 0; d < ndim; d++ ) {
            x[d] = dist( gen );
            R += x[d] * x[d];
        }
        if ( R <= 1.0 ) {
            for ( int d = 0; d < ndim; d++ )
                points( d, i ) = x[d];
            i++;
        }
    }
    return points;
}
