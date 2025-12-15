#ifndef included_AMP_NearestPairSearch_hpp
#define included_AMP_NearestPairSearch_hpp

#include "AMP/utils/NearestPairSearch.h"
#include "AMP/utils/Utilities.hpp"

#include "ProfilerApp.h"

#include <math.h>


namespace AMP {


// subroutine to find the minimum distance between any two points
template<int NDIM, class TYPE>
inline std::pair<int, int> find_min_dist2( const size_t N, const TYPE *const *x, double &dist )

{
    PROFILE( "find_min_dist2", 3 );
    // Number of points below which we will switch to a brute force search
    constexpr size_t N_brute = 10;
    if ( N < N_brute ) {
        // Use a brute force method
        double dist1 = 1e100;
        std::pair<int, int> index;
        for ( size_t i = 0; i < N; i++ ) {
            for ( size_t j = i + 1; j < N; j++ ) {
                double dist2 = 0.0;
                for ( int d = 0; d < NDIM; d++ ) {
                    double tmp = static_cast<double>( x[d][i] - x[d][j] );
                    dist2 += tmp * tmp;
                }
                if ( dist2 < dist1 ) {
                    dist1        = dist2;
                    index.first  = static_cast<int>( i );
                    index.second = static_cast<int>( j );
                }
            }
        }
        dist = std::sqrt( dist1 );
        return index;
    }
    // Choose the split direction as the largest dimension
    TYPE range[2 * NDIM];
    for ( int d = 0; d < NDIM; d++ ) {
        TYPE x_min = x[d][0];
        TYPE x_max = x[d][0];
        for ( size_t i = 0; i < N; i++ ) {
            TYPE x0 = x[d][i];
            x_min   = std::min( x0, x_min );
            x_max   = std::max( x0, x_max );
        }
        range[2 * d + 0] = x_min;
        range[2 * d + 1] = x_max;
    }
    int split       = 0;
    TYPE dist_range = range[1] - range[0];
    TYPE median     = ( range[1] + range[0] ) / 2;
    for ( int d = 1; d < NDIM; d++ ) {
        if ( range[2 * d + 1] - range[2 * d + 0] > dist_range ) {
            split      = d;
            dist_range = range[2 * d + 1] - range[2 * d + 0];
            median     = ( range[2 * d + 1] + range[2 * d + 0] ) / 2;
        }
    }
    if ( dist_range == 0 ) {
        // All points are degenerate
        dist = 0;
        return std::pair<int, int>( 0, 1 );
    }
    // First get the points sorted along the desired direction
    TYPE *x2[NDIM] = { nullptr };
    for ( int d = 0; d < NDIM; d++ )
        x2[d] = new TYPE[N];
    for ( size_t i = 0; i < N; i++ )
        x2[split][i] = x[split][i];
    int *I = new int[N];
    for ( size_t i = 0; i < N; i++ )
        I[i] = static_cast<int>( i );
    AMP::Utilities::quicksort( N, x2[split], I );
    for ( int d = 0; d < NDIM; d++ ) {
        if ( d != split ) {
            for ( size_t i = 0; i < N; i++ )
                x2[d][i] = x[d][I[i]];
        }
    }
    // Divide the points in half and recursively find the closest pair in each half
    size_t k = AMP::Utilities::findfirst<TYPE>( N, x2[split], median );
    AMP_ASSERT( k > 0 && k < N );
    TYPE *x3[NDIM] = { nullptr };
    for ( int d = 0; d < NDIM; d++ )
        x3[d] = &x2[d][k];
    double dist1 = 0, dist2 = 0;
    std::pair<int, int> pair1 = find_min_dist2<NDIM, TYPE>( k, x2, dist1 );
    std::pair<int, int> pair2 = find_min_dist2<NDIM, TYPE>( N - k, x3, dist2 );
    pair2.first += static_cast<int>( k );
    pair2.second += static_cast<int>( k );
    std::pair<int, int> index;
    if ( dist1 <= dist2 ) {
        dist  = dist1;
        index = pair1;
    } else {
        dist  = dist2;
        index = pair2;
    }
    // Check for pairs of points that are closer than dist and span the median
    size_t i1 = AMP::Utilities::findfirst<TYPE>( N, x2[split], x2[split][k] - dist );
    size_t i2 = AMP::Utilities::findfirst<TYPE>( N, x2[split], x2[split][k - 1] + dist );
    i2        = std::min<size_t>( i2 + 1, N );
    if ( k - i1 < 5 || i2 - k < 5 || i2 - i1 > 0.5 * N ) {
        // Use a brute force method
        double dist3 = dist * dist;
        for ( size_t i = i1; i < k; i++ ) {
            for ( size_t j = k; j < i2; j++ ) {
                double dist4 = 0.0;
                for ( int d = 0; d < NDIM; d++ ) {
                    double tmp = static_cast<double>( x2[d][i] - x2[d][j] );
                    dist4 += tmp * tmp;
                }
                if ( dist4 < dist3 ) {
                    dist3        = dist4;
                    index.first  = static_cast<int>( i );
                    index.second = static_cast<int>( j );
                }
            }
        }
        dist = std::sqrt( dist3 );
    } else {
        // Search recursively
        for ( int d = 0; d < NDIM; d++ )
            x3[d] = &x2[d][i1];
        double dist3               = 1e100;
        std::pair<int, int> index3 = find_min_dist2<NDIM, TYPE>( i2 - i1, x3, dist3 );
        if ( dist3 < dist ) {
            dist         = dist3;
            index.first  = index3.first + static_cast<int>( i1 );
            index.second = index3.second + static_cast<int>( i1 );
        }
    }
    // Convert the index to the original x positions
    index.first  = I[index.first];
    index.second = I[index.second];
    // Free memory and return
    for ( int d = 0; d < NDIM; d++ )
        delete[] x2[d];
    delete[] I;
    return index;
}


// Find the minimum distance between any 2 points in 1D
template<class TYPE>
inline std::pair<int, int> find_min_dist_1d( const int N, const TYPE *x )
{
    int *I   = new int[N];
    TYPE *x2 = new TYPE[N];
    for ( int i = 0; i < N; i++ )
        I[i] = i;
    for ( int i = 0; i < N; i++ )
        x2[i] = x[i];
    AMP::Utilities::quicksort( N, x2, I );
    int index = 1;
    TYPE dist = std::max( x2[1], x2[0] ) - std::min( x2[1], x2[0] );
    for ( int i = 2; i < N; i++ ) {
        if ( ( x2[i] - x2[i - 1] ) < dist ) {
            index = i;
            dist  = std::max( x2[i], x2[i - 1] ) - std::min( x2[i], x2[i - 1] );
        }
    }
    std::pair<int, int> ans( I[index - 1], I[index] );
    delete[] I;
    delete[] x2;
    return ans;
}


template<int NDIM, class TYPE>
inline std::pair<int, int> find_min_dist( const int N, const TYPE *x )
{
    if ( N < 2 )
        return std::pair<int, int>( 0, 0 );
    if ( NDIM == 1 )
        return find_min_dist_1d( N, x );
    TYPE *x2[NDIM] = { nullptr };
    for ( int d = 0; d < NDIM; d++ ) {
        x2[d] = new TYPE[N];
        for ( int i = 0; i < N; i++ )
            x2[d][i] = x[d + i * NDIM];
    }
    double dist               = 0;
    std::pair<int, int> index = find_min_dist2<NDIM, TYPE>( N, x2, dist );
    for ( int d = 0; d < NDIM; d++ )
        delete[] x2[d];
    return index;
}


} // namespace AMP

#endif
