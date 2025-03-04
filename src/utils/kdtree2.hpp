#ifndef included_AMP_kdtree2_hpp
#define included_AMP_kdtree2_hpp

#include "AMP/utils/Utilities.h"
#include "AMP/utils/arrayHelpers.h"
#include "AMP/utils/kdtree2.h"

#include "ProfilerApp.h"

#include <cstring>
#include <memory>


namespace AMP {


/****************************************************************
 * data_struct                                                   *
 ****************************************************************/
template<uint8_t NDIM, class TYPE>
kdtree2<NDIM, TYPE>::data_struct::data_struct( size_t N0 )
{
    N    = N0;
    x    = (Point *) malloc( N * sizeof( Point ) );
    data = (TYPE *) malloc( N * sizeof( TYPE ) );
}
template<uint8_t NDIM, class TYPE>
kdtree2<NDIM, TYPE>::data_struct::~data_struct()
{
    free( x );
    free( data );
    x    = nullptr;
    data = nullptr;
}
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::data_struct::add( const Point &x2, const TYPE &d2 )
{
    x       = (Point *) realloc( x, ( N + 1 ) * sizeof( Point ) );
    data    = (TYPE *) realloc( data, ( N + 1 ) * sizeof( TYPE ) );
    x[N]    = x2;
    data[N] = d2;
    N++;
}


/****************************************************************
 * Compute the distance to a box                                 *
 ****************************************************************/
template<uint8_t NDIM, class TYPE>
double kdtree2<NDIM, TYPE>::distanceToBox( const std::array<double, NDIM> &pos,
                                           const std::array<double, NDIM> &ang,
                                           const std::array<double, NDIM> &lb,
                                           const std::array<double, NDIM> &ub )
{
    double d = std::numeric_limits<double>::infinity();
    // Check if the intersection of each surface is within the bounds of the box
    auto inside = [&lb, &ub]( const std::array<double, NDIM> &p ) {
        bool in = true;
        for ( size_t d = 0; d < NDIM; d++ )
            in = in && ( p[d] >= lb[d] - 1e-12 ) && ( p[d] <= ub[d] + 1e-12 );
        return in;
    };
    // Compute the distance to each surface and check if it is closer
    for ( size_t i = 0; i < NDIM; i++ ) {
        double d1 = ( lb[i] - pos[i] ) / ang[i];
        double d2 = ( ub[i] - pos[i] ) / ang[i];
        if ( d1 >= 0 ) {
            auto p = pos + d1 * ang;
            if ( inside( p ) )
                d = std::min( d, d1 );
        }
        if ( d2 >= 0 ) {
            auto p = pos + d2 * ang;
            if ( inside( p ) )
                d = std::min( d, d2 );
        }
    }
    // Return the distance
    if ( inside( pos ) && d < 1e100 )
        d = -d;
    return d;
}


/********************************************************
 * Constructor                                           *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
kdtree2<NDIM, TYPE>::kdtree2( size_t N, std::array<double, NDIM> *x, TYPE *data )
{
    initialize( N, x, data );
}
template<uint8_t NDIM, class TYPE>
kdtree2<NDIM, TYPE>::kdtree2( const std::vector<std::array<double, NDIM>> &x,
                              const std::vector<TYPE> &data )
{
    AMP_ASSERT( x.size() == data.size() );
    size_t N = x.size();
    auto x2  = new std::array<double, NDIM>[N];
    auto d2  = new TYPE[N];
    for ( size_t i = 0; i < N; i++ ) {
        x2[i] = x[i];
        d2[i] = data[i];
    }
    initialize( N, x2, d2 );
    delete[] x2;
    delete[] d2;
}
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::initialize( size_t N, Point *x, TYPE *data )
{
    PROFILE( "initialize" );
    d_N = N;
    // Update the box
    d_lb.fill( 1e100 );
    d_ub.fill( -1e100 );
    for ( size_t i = 0; i < d_N; i++ ) {
        for ( int d = 0; d < NDIM; d++ ) {
            d_lb[d] = std::min( d_lb[d], x[i][d] );
            d_ub[d] = std::max( d_ub[d], x[i][d] );
        }
    }
    // If we have more than the threshold split the tree
    constexpr uint64_t threshold = 40; // Optimize for performance
    if ( d_N > threshold ) {
        // Split the tree and recurse
        splitData( N, x, data );
    } else {
        // Store the data
        d_data = std::make_unique<data_struct>( N );
        for ( size_t i = 0; i < N; i++ ) {
            d_data->x[i]    = x[i];
            d_data->data[i] = data[i];
        }
    }
}
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::splitData( size_t N, Point *x, TYPE *data )
{
    // Check that points are unique
    bool allMatch = true;
    for ( size_t i = 0; i < N; i++ )
        allMatch = allMatch && x[i] == x[0];
    AMP_ASSERT( !allMatch );
    // Choose the splitting direction (and tolerance)
    int dir    = 0;
    double tmp = d_ub[0] - d_lb[0];
    for ( int d = 0; d < NDIM; d++ ) {
        if ( ( d_ub[d] - d_lb[d] ) > tmp ) {
            dir = d;
            tmp = d_ub[d] - d_lb[d];
        }
    }
    // Resort the data along the splitting direction
    auto x2 = new double[N];
    for ( size_t i = 0; i < N; i++ )
        x2[i] = x[i][dir];
    AMP::Utilities::quicksort( N, x2, x, data );
    // Find the ideal point to split
    size_t k = find_split( N, x2 );
    if ( k == 0 ) {
        printf( "%i, %f\n", dir, tmp );
        for ( int d = 0; d < NDIM; d++ )
            printf( "%i: %f %f\n", d, d_lb[d], d_ub[d] );
        auto msg = AMP::Utilities::stringf( "Error in split (k==0), %i", N );
        AMP_ERROR( msg );
    }
    // Recursively split
    d_split_dim = dir;
    d_split     = 0.5 * ( x2[k - 1] + x2[k] );
    delete[] x2;
    d_left  = std::unique_ptr<kdtree2>( new kdtree2( k, x, data ) );
    d_right = std::unique_ptr<kdtree2>( new kdtree2( N - k, &x[k], &data[k] ) );
}


/********************************************************
 * Return the domain box                                 *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::add( const Point &p, const TYPE &data )
{
    d_N++;
    // Update the bounding box
    for ( int d = 0; d < NDIM; d++ ) {
        d_lb[d] = std::min( d_lb[d], p[d] );
        d_ub[d] = std::max( d_ub[d], p[d] );
    }
    if ( d_left ) {
        // Figure out which half we belong to and split
        if ( p[d_split_dim] <= d_split )
            d_left->add( p, data );
        else
            d_right->add( p, data );
    } else {
        // Add the point to the current leaf node
        d_data->add( p, data );
        // Split the leaf node if needed
        constexpr uint64_t threshold = 40; // Optimize for performance
        if ( d_N > threshold ) {
            splitData( d_data->N, d_data->x, d_data->data );
            d_data.reset();
        }
    }
}


/********************************************************
 * Return the domain box                                 *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
std::array<double, 2 * NDIM> kdtree2<NDIM, TYPE>::box() const
{
    std::array<double, 2 * NDIM> b;
    for ( int d = 0; d < NDIM; d++ ) {
        b[2 * d + 0] = d_lb[d];
        b[2 * d + 1] = d_ub[d];
    }
    return b;
}


/********************************************************
 * Return the points in the domain                       *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
std::vector<std::array<double, NDIM>> kdtree2<NDIM, TYPE>::getPoints() const
{
    std::vector<Point> x;
    getPoints( x );
    return x;
}
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::getPoints( std::vector<Point> &x ) const
{
    if ( d_left ) {
        d_left->getPoints( x );
        d_right->getPoints( x );
    } else {
        x.insert( x.end(), d_data->x, d_data->x + d_data->N );
    }
}


/********************************************************
 * Find the ideal point to split such that we divide     *
 *   both the space and points as much as possible       *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
size_t kdtree2<NDIM, TYPE>::find_split( size_t N, const double *x )
{
    // Find the largest gap such that we also divide the points and space
    double lb = x[0];
    double ub = x[N - 1];
    int k     = 0;
    double q  = 0;
    for ( size_t i = 1; i < N; i++ ) {
        // Compute the quality of the split at the current location
        double q2 = ( x[i] - x[i - 1] ) * ( x[i] - lb ) * ( ub - x[i - 1] ) * i * ( N - i - 0 );
        if ( q2 > q ) {
            q = q2;
            k = i;
        }
    }
    return k;
}


/********************************************************
 * Check if the point (and its radius intersects the box *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
bool kdtree2<NDIM, TYPE>::intersect( const Point &x, double dist ) const
{
    // Check if the point (and its radius) intersects with the current box
    double dist2 = 0.0;
    for ( int k = 0; k < NDIM; k++ ) {
        double d = std::max( d_lb[k] - x[k], x[k] - d_ub[k] );
        if ( d > 0.0 )
            dist2 += d * d;
    }
    return dist2 <= dist;
}


/********************************************************
 * Nearest neighbor search                               *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
std::tuple<std::array<double, NDIM>, TYPE>
kdtree2<NDIM, TYPE>::findNearest( const kdtree2::Point &x ) const
{
    double d;
    std::tuple<std::array<double, NDIM>, TYPE> nearest;
    findNearest( x, 1, &nearest, &d );
    return nearest;
}
template<uint8_t NDIM, class TYPE>
std::vector<std::tuple<std::array<double, NDIM>, TYPE>>
kdtree2<NDIM, TYPE>::findNearest( const kdtree2::Point &x, int N ) const
{
    std::vector<double> dist( N );
    std::vector<std::tuple<std::array<double, NDIM>, TYPE>> nearest( N );
    findNearest( x, N, nearest.data(), dist.data() );
    return nearest;
}
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::findNearest( const Point &x,
                                       size_t N,
                                       std::tuple<Point, TYPE> *nearest,
                                       double *dist ) const
{
    // First, find dive into the structure to find where the position would be stored
    if ( d_left != nullptr ) {
        // Drill down the tree to find the node that should contain the point
        // As we travel back check the neighboring trees for any points that might be closer
        if ( x[d_split_dim] <= d_split ) {
            d_left->findNearest( x, N, nearest, dist );
            d_right->checkNearest( x, N, nearest, dist );
        } else {
            d_right->findNearest( x, N, nearest, dist );
            d_left->checkNearest( x, N, nearest, dist );
        }
    } else {
        // We are at the final node, find the closest values using the naive approach
        for ( size_t i = 0; i < N; i++ )
            dist[i] = 1e200;
        for ( size_t i = 0; i < d_N; i++ ) {
            double d = norm( x - d_data->x[i] );
            if ( d < dist[N - 1] ) {
                dist[N - 1]    = d;
                nearest[N - 1] = std::tuple<Point, TYPE>( d_data->x[i], d_data->data[i] );
                for ( size_t j = N - 1; j > 0; j-- ) {
                    if ( dist[j] < dist[j - 1] ) {
                        std::swap( dist[j], dist[j - 1] );
                        std::swap( nearest[j], nearest[j - 1] );
                    }
                }
            }
        }
    }
}
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::checkNearest( const Point &x,
                                        size_t N,
                                        std::tuple<Point, TYPE> *nearest,
                                        double *dist ) const
{
    // Check if the point (and its radius) intersects with the current box
    if ( !intersect( x, dist[N - 1] ) )
        return;
    // Recursively search the subtrees
    if ( d_left != nullptr ) {
        d_left->checkNearest( x, N, nearest, dist );
        d_right->checkNearest( x, N, nearest, dist );
        return;
    }
    // We are at a base node, check the points for any that might be closer
    for ( size_t i = 0; i < d_N; i++ ) {
        double dist2 = norm( x - d_data->x[i] );
        if ( dist2 < dist[N - 1] ) {
            dist[N - 1]    = dist2;
            nearest[N - 1] = std::tuple<Point, TYPE>( d_data->x[i], d_data->data[i] );
            for ( size_t j = N - 1; j > 0; j-- ) {
                if ( dist[j] < dist[j - 1] ) {
                    std::swap( dist[j], dist[j - 1] );
                    std::swap( nearest[j], nearest[j - 1] );
                }
            }
        }
    }
}


/********************************************************
 * Find the point within a predefined distance           *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
std::vector<std::tuple<std::array<double, NDIM>, TYPE>>
kdtree2<NDIM, TYPE>::findNearest( const Point &x, double dist ) const
{
    std::vector<std::tuple<Point, TYPE>> ans;
    findNearest( x, dist, ans );
    return ans;
}
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::findNearest( const Point &x,
                                       double dist,
                                       std::vector<std::tuple<Point, TYPE>> &nearest ) const
{
    // Check if the box is within the distance
    if ( !intersect( x, dist ) )
        return;
    if ( d_left != nullptr ) {
        // Check the left and right boxes
        d_left->findNearest( x, dist, nearest );
        d_right->findNearest( x, dist, nearest );
    } else {
        // We are at the final node, find the closest values using the naive approach
        double dist2 = dist * dist;
        for ( size_t i = 0; i < d_N; i++ ) {
            double d = norm( x - d_data->x[i] );
            if ( d <= dist2 )
                nearest.emplace_back( d_data->x[i], d_data->data[i] );
        }
    }
}


/********************************************************
 * Ray-neighborhood intersection                         *
 ********************************************************/
template<uint8_t NDIM, class TYPE>
std::vector<std::tuple<std::array<double, NDIM>, TYPE>>
kdtree2<NDIM, TYPE>::findNearestRay( const Point &x, const Point &dir, double dist ) const
{
    auto v = AMP::normalize( dir );
    Point inv_v;
    for ( int d = 0; d < NDIM; d++ )
        inv_v[d] = 1.0 / v[d];
    std::vector<std::tuple<Point, TYPE>> ans;
    findNearestRay( x, v, inv_v, dist, ans );
    return ans;
}
template<uint8_t NDIM, class TYPE>
void kdtree2<NDIM, TYPE>::findNearestRay( const Point &x,
                                          const Point &dir,
                                          const Point &inv_dir,
                                          double dist,
                                          std::vector<std::tuple<Point, TYPE>> &nearest ) const
{
    // Compute the nearest point to a ray
    auto intersect = []( const Point &p0, const Point &v, const Point &x ) {
        auto t = dot( v, x - p0 );
        t      = std::max( t, 0.0 );
        auto p = p0 + v * t;
        return p;
    };
    // Check the distance between the ray and the expanded box
    auto lb     = d_lb - dist;
    auto ub     = d_ub + dist;
    double tmin = -std::numeric_limits<double>::infinity();
    double tmax = std::numeric_limits<double>::infinity();
    for ( int d = 0; d < NDIM; d++ ) {
        double t1 = inv_dir[d] * ( lb[d] - x[d] );
        double t2 = inv_dir[d] * ( ub[d] - x[d] );
        tmin      = std::max( tmin, std::min( t1, t2 ) );
        tmax      = std::min( tmax, std::max( t1, t2 ) );
    }
    bool intersectsBox = tmax >= tmin;
    if ( !intersectsBox ) {
        // The ray does not intersect the expanded box
    } else if ( d_left ) {
        // Check the left and right boxes
        d_left->findNearestRay( x, dir, inv_dir, dist, nearest );
        d_right->findNearestRay( x, dir, inv_dir, dist, nearest );
    } else {
        // We are at the final node, for each point check if it is closer to the ray
        const double dist2 = dist * dist;
        for ( size_t i = 0; i < d_N; i++ ) {
            auto p  = d_data->x[i];           // Current point to check
            auto pi = intersect( x, dir, p ); // Find the intersection with the ray
            auto d2 = norm( pi - p );         // Distance: ray-nearest
            if ( d2 <= dist2 )
                nearest.emplace_back( p, d_data->data[i] );
        }
    }
}


} // namespace AMP

#endif
