#include "AMP/utils/DelaunayTessellation.h"
#include "AMP/utils/DelaunayFaceList.h"
#include "AMP/utils/DelaunayFaceList.hpp"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/NearestPairSearch.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/typeid.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>


using AMP::DelaunayHelpers::getETYPE;


namespace AMP::DelaunayTessellation {


// Structure to hold surfaces that need to be tested
struct check_surface_struct {
    uint8_t test;
    uint8_t f1;
    uint8_t f2;
    int t1;
    int t2;
};


/********************************************************************
 * Convert coordinates                                               *
 * We want to use int with a range of (-2^30:2^30)                   *
 ********************************************************************/
template<class TYPE>
static Array<int> convert( const Array<TYPE> &x )
{
    TYPE max = 0;
    for ( auto y : x )
        max = std::max<TYPE>( std::abs( y ), max );
    Array<int> y( x.size() );
    if constexpr ( std::is_floating_point_v<TYPE> ) {
        auto scale = pow( 2.0, 30 - ceil( log2( max ) ) );
        for ( size_t i = 0; i < x.length(); i++ ) {
            auto z = scale * x( i );
            AMP_ASSERT( z < static_cast<TYPE>( std::numeric_limits<int>::max() ) );
            y( i ) = z;
        }
    } else {
        constexpr int64_t int_max = pow( 2.0, 30 );
        int shift                 = 0;
        while ( static_cast<int64_t>( max ) > int_max ) {
            shift++;
            max = max >> 1;
        }
        if ( shift == 0 ) {
            for ( size_t i = 0; i < x.length(); i++ )
                y( i ) = x( i );
        } else {
            for ( size_t i = 0; i < x.length(); i++ )
                y( i ) = x( i ) >> shift;
        }
    }
    return y;
}


/********************************************************************
 * Forward declares                                                  *
 ********************************************************************/
template<int NDIM>
static void clean_triangles( const int N,
                             const std::array<int, NDIM> *x,
                             size_t &N_tri,
                             std::array<int, NDIM + 1> *tri,
                             std::array<int, NDIM + 1> *tri_nab );
template<int NDIM>
static void swap_triangles( int i1,
                            int i2,
                            std::array<int, NDIM + 1> *tri,
                            std::array<int, NDIM + 1> *tri_nab );
template<int NDIM>
static bool find_flip( const std::array<int, NDIM> *x,
                       const std::array<int, NDIM + 1> *tri,
                       const std::array<int, NDIM + 1> *tri_nab,
                       const double TOL_VOL,
                       std::vector<check_surface_struct> &check_surface,
                       int &N_tri_old,
                       int &N_tri_new,
                       int *index_old,
                       int *new_tri,
                       int *new_tri_nab );


/********************************************************************
 * Compute the dot product of two vectors                            *
 * Note: We are already using an increased precision, and want to    *
 * maintain the maximum degree of accuracy possible.                 *
 ********************************************************************/
static inline double dot( int N, const long double *x, const long double *y )
{
    // Approximate dot product for long double precision
    bool sign[8];
    long double z[8];
    for ( int i = 0; i < N; i++ ) {
        z[i]    = x[i] * y[i];
        sign[i] = z[i] >= 0;
    }
    long double ans = z[0];
    z[0]            = 0;
    while ( true ) {
        int index     = -1;
        bool sign_ans = ans >= 0;
        for ( int i = 1; i < N; i++ ) {
            if ( ( sign_ans != sign[i] ) && z[i] != 0 )
                index = i;
        }
        if ( index == -1 )
            break;
        ans += z[index];
        z[index] = 0;
    }
    for ( int i = 1; i < N; i++ ) {
        if ( z[i] != 0 )
            ans += z[i];
    }
    return static_cast<double>( ans );
}
// Integer based dot products
static inline double dot( int N, const int *x, const int *y )
{
    int64_t ans( 0 );
    for ( int i = 0; i < N; i++ )
        ans += int64_t( x[i] ) * int64_t( y[i] );
    return static_cast<double>( ans );
}
static inline double dot( int N, const int64_t *x, const int64_t *y )
{
    int128_t ans( 0 );
    for ( int i = 0; i < N; i++ )
        ans += int128_t( x[i] ) * int128_t( y[i] );
    return static_cast<double>( ans );
}
static inline double dot( int N, const int128_t *x, const int128_t *y )
{
    int256_t ans( 0 );
    for ( int i = 0; i < N; i++ )
        ans += int256_t( x[i] ) * int256_t( y[i] );
    return static_cast<double>( ans );
}
static inline double dot( int N, const int256_t *x, const int256_t *y )
{
    int512_t ans( 0 );
    for ( int i = 0; i < N; i++ )
        ans += int512_t( x[i] ) * int512_t( y[i] );
    return static_cast<double>( ans );
}


/********************************************************************
 * Check if the closest pointer are ~ the same                       *
 ********************************************************************/
template<int NDIM>
static std::array<std::array<int, NDIM>, 2> getDomain( const std::vector<std::array<int, NDIM>> &x )
{
    auto xmin = x[0];
    auto xmax = x[0];
    for ( size_t i = 1; i < x.size(); i++ ) {
        for ( int d = 0; d < NDIM; d++ ) {
            xmin[d] = std::min( xmin[d], x[i][d] );
            xmax[d] = std::max( xmax[d], x[i][d] );
        }
    }
    return { xmin, xmax };
}
template<int NDIM>
static inline void checkClosest( const std::vector<std::array<int, NDIM>> &x,
                                 const std::pair<int, int> &index_pair )
{
    // Get the bounding box for the domain
    auto [xmin, xmax]  = getDomain<NDIM>( x );
    double domain_size = 0;
    for ( int d = 0; d < NDIM; d++ ) {
        int64_t dist = xmax[d] - xmin[d];
        domain_size += dist * dist;
    }
    domain_size = std::sqrt( domain_size );

    // Check that the two closest two points are not the same
    int i1       = index_pair.first;
    int i2       = index_pair.second;
    double r_min = 0;
    for ( int d = 0; d < NDIM; d++ ) {
        auto tmp = static_cast<double>( x[i1][d] - x[i2][d] );
        r_min += tmp * tmp;
    }
    r_min = std::sqrt( r_min );
    if ( r_min < 1e-8 * domain_size )
        throw std::logic_error( "Duplicate or nearly duplicate points detected: " +
                                std::to_string( r_min ) );
}


/********************************************************************
 * Test if 3 points are co-linear                                    *
 ********************************************************************/
template<int NDIM, class TYPE>
static inline bool collinear( const std::array<TYPE, NDIM> *x, double tol )
{
    double r1[NDIM];
    double r2[NDIM];
    double tmp1 = 0, tmp2 = 0;
    for ( int j = 0; j < NDIM; j++ ) {
        r1[j] = static_cast<double>( x[1][j] - x[0][j] );
        r2[j] = static_cast<double>( x[2][j] - x[0][j] );
        tmp1 += r1[j] * r1[j];
        tmp2 += r2[j] * r2[j];
    }
    tmp1 = 1.0 / std::sqrt( tmp1 );
    tmp2 = 1.0 / std::sqrt( tmp2 );
    for ( int j = 0; j < NDIM; j++ ) {
        r1[j] *= tmp1;
        r2[j] *= tmp2;
    }
    double dot = 0.0;
    for ( int j = 0; j < NDIM; j++ ) {
        dot += r1[j] * r2[j];
    }
    bool is_collinear = fabs( 1.0 - fabs( dot ) ) <= tol;
    return is_collinear;
}


/********************************************************************
 * Test if all points are co-linear                                  *
 ********************************************************************/
template<int NDIM>
static inline bool allCollinear( const Array<int> &x )
{
    using Point = std::array<int, NDIM>;
    auto y      = DelaunayHelpers::convert<int, NDIM>( x );
    auto [i, j] = find_min_dist<NDIM, int>( y.size(), y[0].data() );
    checkClosest<NDIM>( y, { i, j } );
    Point x2[3];
    x2[0] = y[i];
    x2[1] = y[j];
    for ( int k = 0; k < (int) y.size(); k++ ) {
        if ( k == i || k == j )
            continue;
        x2[2] = y[k];
        if ( !collinear<NDIM, int>( x2, 1e-12 ) )
            return false;
    }
    return true;
}
template<class TYPE>
bool collinear( const Array<TYPE> &x )
{
    auto y   = convert( x );
    int NDIM = y.size( 0 );
    if ( NDIM == 1 ) {
        return allCollinear<1>( y );
    } else if ( NDIM == 2 ) {
        return allCollinear<2>( y );
    } else if ( NDIM == 3 ) {
        return allCollinear<3>( y );
    } else {
        throw std::logic_error( "collinear is not supported for dimension > 5" );
    }
}
template bool collinear<short>( const Array<short> & );
template bool collinear<int>( const Array<int> & );
template bool collinear<float>( const Array<float> & );
template bool collinear<double>( const Array<double> & );
template bool collinear<long double>( const Array<long double> & );


/********************************************************************
 * Test if 4 points are co-planar                                    *
 *    |  x1  y1  z1  1 |                                             *
 *    |  x2  y2  z2  1 | = 0                                         *
 *    |  x3  y3  z3  1 |                                             *
 *    |  x4  y4  z4  1 |                                             *
 * Note: for exact math this requires N^D precision                  *
 ********************************************************************/
template<int NDIM, class TYPE>
static inline bool coplanar( const std::array<TYPE, NDIM> *x, TYPE tol )
{
    using ETYPE      = typename getETYPE<NDIM, TYPE>::ETYPE;
    bool is_coplanar = true;
    if constexpr ( NDIM == 3 ) {
        ETYPE det( 0 ), one( 1 ), neg( -1 );
        for ( int d = 0; d < NDIM + 1; d++ ) {
            ETYPE M[9];
            for ( int i1 = 0; i1 < d; i1++ ) {
                for ( int i2 = 0; i2 < NDIM; i2++ ) {
                    M[i1 + i2 * NDIM] = ETYPE( x[i1][i2] );
                }
            }
            for ( int i1 = d + 1; i1 < NDIM + 1; i1++ ) {
                for ( int i2 = 0; i2 < NDIM; i2++ ) {
                    M[i1 - 1 + i2 * NDIM] = ETYPE( x[i1][i2] );
                }
            }
            const ETYPE &sign = ( ( NDIM + d ) % 2 == 0 ) ? one : neg;
            det += sign * DelaunayHelpers::det<ETYPE, NDIM>( M );
        }
        is_coplanar = fabs( static_cast<double>( det ) ) <= tol;
    } else {
        AMP_ERROR( "Not programmed for dimensions != 3" );
    }
    return is_coplanar;
}


/********************************************************************
 * Increase the storage for tri and tri_nab to hold N_tri triangles  *
 ********************************************************************/
template<int NDIM>
static inline void check_tri_size( size_t size_new,
                                   std::vector<std::array<int, NDIM + 1>> &tri,
                                   std::vector<std::array<int, NDIM + 1>> &tri_nab )
{
    size_t size_old = tri.size();
    tri.resize( size_new );
    tri_nab.resize( size_new );
    for ( size_t s = size_old; s < size_new; s++ ) {
        for ( int d = 0; d <= NDIM; d++ ) {
            tri[s][d]     = -1;
            tri_nab[s][d] = -1;
        }
    }
}


/********************************************************************
 * Choose the order to add points                                    *
 * We will do this by sorting the points by order of their distance  *
 * from the closest pair                                             *
 ********************************************************************/
template<int NDIM, class TYPE>
static std::vector<int> getInsertionOrder( const std::array<TYPE, NDIM> &x0,
                                           const std::vector<std::array<TYPE, NDIM>> &x )
{
    PROFILE( "getInsertionOrder", 3 );
    // Compute the square of the radius
    int N = x.size();
    std::vector<double> R2( N, 0.0 );
    for ( int j = 0; j < N; j++ ) {
        for ( int d = 0; d < NDIM; d++ ) {
            double tmp = x[j][d] - x0[d];
            R2[j] += tmp * tmp;
        }
    }
    // Sort the points, keeping track of the index
    std::vector<int> I( N );
    for ( int j = 0; j < N; j++ ) {
        I[j] = j;
    }
    AMP::Utilities::quicksort( R2, I );
    return I;
}


/********************************************************************
 * Create the initial triangle                                       *
 ********************************************************************/
template<int NDIM>
static std::tuple<std::array<int, NDIM + 1>, std::vector<int>>
createInitialTriangle( const std::array<int, NDIM> &x0,
                       const std::vector<std::array<int, NDIM>> &x,
                       int iteration = 0 )
{
    using Point    = std::array<int, NDIM>;
    using Triangle = std::array<int, NDIM + 1>;

    int N = x.size();
    AMP_INSIST( N > NDIM, "Insufficient number of points" );

    // Next we need to create a list of the order in which we want to insert the values
    auto I = getInsertionOrder<NDIM, int>( x0, x );

    // Resort the first few points so that the first ndim+1 points are not collinear or coplanar
    int ik = 2;
    for ( int i = 2; i <= NDIM; i++ ) {
        switch ( i ) {
        case 2: {
            // Find the first point that is not collinear with the first 2 points in I
            Point x2[3];
            x2[0] = x[I[0]];
            x2[1] = x[I[1]];
            while ( true ) {
                x2[2]             = x[I[ik]];
                bool is_collinear = collinear<NDIM, int>( x2, 1e-12 );
                if ( !is_collinear )
                    break;
                ik++;
                if ( ik >= N ) {
                    // No 3 non-collinear points were found
                    throw std::logic_error( "Error: All points are collinear" );
                }
            }
            break;
        }
        case 3:
            if constexpr ( NDIM == 3 ) {
                // Find the first point that is not coplanar with the first 3 points in I
                Point x2[4];
                for ( int i1 = 0; i1 < 3; i1++ )
                    x2[i1] = x[I[i1]];
                while ( true ) {
                    x2[3]            = x[I[ik]];
                    bool is_coplanar = coplanar<NDIM, int>( x2, 0 );
                    if ( !is_coplanar )
                        break;
                    ik++;
                    if ( ik >= N ) {
                        // No 4 non-coplanar points were found
                        throw std::logic_error( "Error: All points are coplanar" );
                    }
                }
            } else {
                throw std::logic_error(
                    "Error: Co-planar check is not programmed for dimensions other than 3" );
            }
            break;
        default:
            throw std::logic_error( "Error: Not programmed for this number of dimensions" );
        }
        // Move the points if necessary
        if ( i != ik ) {
            int tmp = I[ik];
            for ( int j = ik; j > i; j-- )
                I[j] = I[j - 1];
            I[i] = tmp;
        }
    }

    // Initial amount of memory to allocate for tri
    Triangle tri;
    Point x2[NDIM + 1];
    for ( int d = 0; d <= NDIM; d++ ) {
        tri[d] = I[d];
        x2[d]  = x[tri[d]];
    }
    double volume = DelaunayHelpers::calcVolume<NDIM, int>( x2 );
    if ( fabs( volume ) <= 0 ) {
        if ( iteration < 10 ) {
            // Choose a random point and try again
            auto [xmin, xmax] = getDomain<NDIM>( x );
            std::random_device rd;
            std::mt19937 gen( rd() );
            std::array<int, NDIM> x2;
            for ( int d = 0; d < NDIM; d++ ) {
                std::uniform_int_distribution<int> dis( xmin[d], xmax[d] );
                x2[d] = dis( gen );
            }
            std::tie( tri, I ) = createInitialTriangle<NDIM>( x2, x, iteration++ );
        } else {
            AMP_ERROR( "Error creating initial triangle" );
        }
    } else if ( volume < 0 ) {
        // The volume is negitive, swap the last two indicies
        std::swap( tri[NDIM - 1], tri[NDIM] );
    }
    return std::tie( tri, I );
}


/********************************************************************
 * This is the main function that creates the tessellation           *
 ********************************************************************/
template<int NDIM>
static std::tuple<std::vector<std::array<int, NDIM + 1>>, std::vector<std::array<int, NDIM + 1>>>
create_tessellation( const std::vector<std::array<int, NDIM>> &x )
{
    int N          = x.size();
    using Point    = std::array<int, NDIM>;
    using Triangle = std::array<int, NDIM + 1>;

    PROFILE2( AMP::Utilities::stringf( "create_tessellation<%i>", NDIM ), 1 );

    // First, get the two closest points
    auto index_pair = find_min_dist<NDIM, int>( N, x[0].data() );
    checkClosest<NDIM>( x, index_pair );

    // Initial amount of memory to allocate for tri
    std::vector<Triangle> tri, tri_nab;
    check_tri_size<NDIM>( 2 * N, tri, tri_nab );

    // Create an initial triangle
    std::vector<int> I;
    std::tie( tri[0], I ) = createInitialTriangle<NDIM>( x[index_pair.first], x, 0 );
    size_t N_tri          = 1;

    // Maintain a list of the triangle faces on the convex hull
    FaceList<NDIM> face_list( N, x.data(), 0, tri[0] );

    // Maintain a list of the unused triangles (those that are all -1, but less than N_tri)
    std::vector<size_t> unused;
    unused.reserve( 512 );

    // Maintain a list of surfaces to check
    std::vector<check_surface_struct> check_surface;
    check_surface.reserve( 256 );

    // Subsequently add each point to the convex hull
    std::vector<Triangle> new_tri;
    std::vector<Triangle> new_tri_nab;
    std::vector<int> neighbor;
    std::vector<int> face;
    std::vector<uint32_t> new_tri_id;
    for ( int i = NDIM + 1; i < N; i++ ) {
        PROFILE( "create-add_points", 3 );
        // Add a point to the convex hull and create the new triangles
        face_list.add_node( I[i], unused, N_tri, new_tri_id, new_tri, new_tri_nab, neighbor, face );
        int N_tri_new = new_tri_id.size();
        // Increase the storage for tri and tri_nab if necessary
        check_tri_size<NDIM>( N_tri, tri, tri_nab );
        // Add each triangle and update the structures
        for ( int j = 0; j < N_tri_new; j++ ) {
            int index_new = new_tri_id[j];
            for ( int j1 = 0; j1 <= NDIM; j1++ )
                tri[index_new][j1] = new_tri[j][j1];
            for ( int j1 = 0; j1 <= NDIM; j1++ )
                tri_nab[index_new][j1] = new_tri_nab[j][j1];
            std::swap( tri_nab[neighbor[j]][face[j]], index_new );
            AMP_ASSERT( index_new == -1 );
        }
#if DEBUG_CHECK == 2
        bool all_valid =
            check_current_triangles<NDIM>( N, x.data(), N_tri, tri, tri_nab, unused, TOL_VOL );
        if ( !all_valid )
            throw std::logic_error( "Failed internal triangle check" );
#endif
        // Get a list of the surfaces we need to check for a valid tesselation
        for ( int j = 0; j < N_tri_new; j++ ) {
            int index_new = new_tri_id[j];
            for ( int j1 = 0; j1 <= NDIM; j1++ ) {
                if ( tri_nab[index_new][j1] != -1 ) {
                    bool found = false;
                    for ( auto &elem : check_surface ) {
                        if ( index_new == elem.t1 && j1 == elem.f1 ) {
                            // The surface is already in the list to check
                            found = true;
                            break;
                        }
                        if ( index_new == elem.t2 && j1 == elem.f2 ) {
                            // The surface is already in the list to check
                            found = true;
                            break;
                        }
                    }
                    if ( !found ) {
                        check_surface_struct tmp;
                        tmp.test = 0x00;
                        tmp.t1   = index_new;
                        tmp.f1   = j1;
                        tmp.t2   = tri_nab[index_new][j1];
                        int m    = tri_nab[index_new][j1];
                        for ( int j2 = 0; j2 <= NDIM; j2++ ) {
                            if ( tri_nab[m][j2] == index_new )
                                tmp.f2 = j2;
                        }
                        check_surface.push_back( tmp );
                    }
                }
            }
        }

        // Now that we have created a new triangle, perform any edge flips that
        // are necessary to create a valid tesselation
        size_t it = 1;
        AMP_ASSERT( !check_surface.empty() );
        while ( !check_surface.empty() ) {
            PROFILE( "create-edge_flips", 4 );
            if ( it > std::max<size_t>( 500, N_tri ) )
                throw std::logic_error( "Error: infinite loop detected" );
            // First, lets eliminate all the surfaces that are fine
            for ( auto &elem : check_surface ) {
                /* Check the surface to see if the triangle pairs need to undergo a flip
                 * The surface is invalid and the triangles need to undergo a flip
                 * if the vertex of either triangle lies within the circumsphere of the other.
                 * Note: if the vertex of one triangle lies within the circumsphere of the
                 * other, then the vertex of the other triangle will lie within the circumsphere
                 * of the current triangle.  Consequently we only need to check one
                 * vertex/triangle pair
                 */
                if ( ( elem.test & 0x01 ) != 0 ) {
                    // We already checked this surface
                    continue;
                }
                Point x2[NDIM + 1];
                for ( int j1 = 0; j1 < NDIM + 1; j1++ ) {
                    int m  = tri[elem.t1][j1];
                    x2[j1] = x[m];
                }
                int m     = tri[elem.t2][elem.f2];
                int test  = test_in_circumsphere<NDIM, int>( x2, x[m], 0 );
                elem.test = ( test != 1 ) ? 0xFF : 0x01;
            }
            // Remove all surfaces that are good
            size_t n = 0;
            for ( size_t k = 0; k < check_surface.size(); k++ ) {
                if ( check_surface[k].test != 0xFF ) {
                    std::swap( check_surface[n], check_surface[k] );
                    n++;
                }
            }
            check_surface.resize( n );
            if ( check_surface.size() == 0 ) {
                // All surfaces are good, we are finished
                break;
            }
            // Find a valid flip
            // The maximum flip currently supported is a 4-4 flip
            int index_old[5], new_tri[( NDIM + 1 ) * 4], new_tri_nab[( NDIM + 1 ) * 4];
            int N_tri_old     = 0;
            int N_tri_new     = 0;
            bool flipped_edge = find_flip<NDIM>( x.data(),
                                                 tri.data(),
                                                 tri_nab.data(),
                                                 0,
                                                 check_surface,
                                                 N_tri_old,
                                                 N_tri_new,
                                                 index_old,
                                                 new_tri,
                                                 new_tri_nab );
            if ( !flipped_edge ) {
                // We did not find any valid flips, this is an error
                for ( auto &elem : check_surface )
                    elem.test = 0;
                bool test = find_flip<NDIM>( x.data(),
                                             tri.data(),
                                             tri_nab.data(),
                                             0,
                                             check_surface,
                                             N_tri_old,
                                             N_tri_new,
                                             index_old,
                                             new_tri,
                                             new_tri_nab );
                if ( test )
                    printf( "   Valid flips were detected if we reset check_surface\n" );
                throw std::logic_error( "Error: no valid flips detected" );
            }
            // Check that we conserved the boundary triangles
            int old_tri_nab[( NDIM + 1 ) * 4]; // The maximum flip currently supported is a 4-4 flip
            for ( int j1 = 0; j1 < N_tri_old; j1++ ) {
                for ( int j2 = 0; j2 <= NDIM; j2++ ) {
                    int tmp = tri_nab[index_old[j1]][j2];
                    for ( int j3 = 0; j3 < N_tri_old; j3++ ) {
                        if ( tmp == index_old[j3] )
                            tmp = -2 - tmp;
                    }
                    old_tri_nab[j2 + j1 * ( NDIM + 1 )] = tmp;
                }
            }
            bool pass = conserved_neighbors(
                N_tri_old * ( NDIM + 1 ), old_tri_nab, N_tri_new * ( NDIM + 1 ), new_tri_nab );
            if ( !pass )
                throw std::logic_error( "Error: triangle neighbors not conserved" );
            // Delete the old triangles, add the new ones, and update the structures
            // First get the indicies where we will store the new triangles
            int index_new[4];
            for ( int j = 0; j < N_tri_new; j++ ) {
                if ( j < N_tri_old ) {
                    // We can just reuse the storage of the old indicies
                    index_new[j] = index_old[j];
                } else if ( !unused.empty() ) {
                    // Use a previously freed place
                    index_new[j] = (int) unused.back();
                    unused.pop_back();
                } else {
                    // We will need to expand N_tri
                    check_tri_size<NDIM>( N_tri + 1, tri, tri_nab );
                    index_new[j] = (int) N_tri;
                    N_tri++;
                }
            }
            // Delete the old triangles
            int N_face_update = 0;
            int old_tri_id[4 * ( NDIM + 1 )], old_face_id[4 * ( NDIM + 1 )],
                new_tri_id[4 * ( NDIM + 1 )], new_face_id[4 * ( NDIM + 1 )];
            for ( int j = N_tri_new; j < N_tri_old; j++ )
                unused.push_back( index_old[j] );
            for ( int j1 = 0; j1 < N_tri_old; j1++ ) {
                for ( int j2 = 0; j2 <= NDIM; j2++ ) {
                    tri[index_old[j1]][j2] = -1;
                    if ( tri_nab[index_old[j1]][j2] == -1 ) {
                        // The given face is on the convex hull
                        old_tri_id[N_face_update]  = index_old[j1];
                        old_face_id[N_face_update] = j2;
                        N_face_update++;
                    } else {
                        // The given face is not on the convex hull
                        /*bool is_old = false;
                        for (int j3=0; j3<N_tri_old; j3++) {
                            if ( tri_nab[index_old[j1]][j2]==index_old[j3] )
                                is_old = true;
                        }*/
                    }
                    tri_nab[index_old[j1]][j2] = -2;
                }
            }
            for ( int j1 = 0; j1 < N_tri_old; j1++ ) {
                size_t j2 = 0;
                while ( j2 < check_surface.size() ) {
                    if ( check_surface[j2].t1 == index_old[j1] ||
                         check_surface[j2].t2 == index_old[j1] ) {
                        std::swap( check_surface[j2], check_surface.back() );
                        check_surface.pop_back();
                    } else {
                        j2++;
                    }
                }
            }
            // Create the new triangles
            for ( int j1 = 0; j1 < N_tri_new; j1++ ) {
                for ( int j2 = 0; j2 <= NDIM; j2++ )
                    tri[index_new[j1]][j2] = new_tri[j2 + j1 * ( NDIM + 1 )];
            }
            // Update the triangle neighbors and determine the new lists of surfaces to check
            int N_face_update2 = 0;
            for ( int j1 = 0; j1 < N_tri_new; j1++ ) {
                for ( int j2 = 0; j2 <= NDIM; j2++ ) {
                    int k1 = index_new[j1];
                    int k2 = new_tri_nab[j2 + j1 * ( NDIM + 1 )];
                    if ( k2 < -1 ) {
                        // Neighbor triangle is one of the new triangles, we only need to update
                        // tr_nab
                        tri_nab[k1][j2] = index_new[-k2 - 2];
                    } else if ( k2 == -1 ) {
                        // Face is on the convex hull, we need to update tri_nab and store the
                        // face for updating face_list
                        tri_nab[k1][j2]             = -1;
                        new_tri_id[N_face_update2]  = index_new[j1];
                        new_face_id[N_face_update2] = j2;
                        N_face_update2++;
                    } else {
                        // Neighbor triangle is an existing triangle, we need to update tr_nab
                        // for the new triangle, the exisiting triangle, and add the face to
                        // check_surface
                        tri_nab[k1][j2] = k2;
                        int tmp[NDIM];
                        for ( int m = 0; m < j2; m++ )
                            tmp[m] = tri[k1][m];
                        for ( int m = j2 + 1; m <= NDIM; m++ )
                            tmp[m - 1] = tri[k1][m];
                        int face = -1;
                        for ( int m1 = 0; m1 <= NDIM; m1++ ) {
                            bool found = false;
                            for ( auto &elem : tmp ) {
                                if ( tri[k2][m1] == elem )
                                    found = true;
                            }
                            if ( !found ) {
                                face = m1;
                                break;
                            }
                        }
                        tri_nab[k2][face] = index_new[j1];
                        check_surface_struct tmp2;
                        tmp2.test = 0x00;
                        tmp2.t1   = k1;
                        tmp2.f1   = j2;
                        tmp2.t2   = k2;
                        tmp2.f2   = face;
                        check_surface.push_back( tmp2 );
                    }
                }
            }
            // Update the faces on the convex hull
            if ( N_face_update != N_face_update2 ) {
                printf( "N_face_update = %i, k = %i\n", N_face_update, N_face_update2 );
                throw std::logic_error( "internal error" );
            }
            face_list.update_face(
                N_face_update, old_tri_id, old_face_id, new_tri_id, new_face_id, tri.data() );
// Check the current triangles for errors (Only when debug is set, very expensive)
#if DEBUG_CHECK == 2
            all_valid =
                check_current_triangles<NDIM>( N, x.data(), N_tri, tri, tri_nab, unused, TOL_VOL );
            if ( !all_valid )
                throw std::logic_error( "Failed internal triangle check" );
#endif
            it++;
        }
// Check the current triangles for errors (Only when debug is set, very expensive)
#if DEBUG_CHECK == 2
        all_valid = check_current_triangles<NDIM>(
            N, x.data(), N_tri, tri.data(), tri_nab.data(), unused, TOL_VOL );
        if ( !all_valid )
            throw std::logic_error( "Failed internal triangle check" );
#endif
    }
    check_surface.clear();
    I = std::vector<int>();

    // Delete any unused triangles
    PROFILE( "clean up", 3 );
    while ( !unused.empty() ) {
        // First, see if any unused slots are at the end of the list
        size_t index = unused.size();
        for ( size_t i = 0; i < unused.size(); i++ ) {
            if ( unused[i] == N_tri - 1 ) {
                index = i;
                break;
            }
        }
        if ( index < unused.size() ) {
            unused[index] = unused.back();
            unused.pop_back();
            N_tri--;
            continue;
        }
        // Move the last triangle to fill the last gap
        auto new_tri_id = static_cast<int>( unused.back() ); // The new triangle number
        auto old_tri_id = static_cast<int>( N_tri - 1 );     // The old triangle number
        for ( int j = 0; j <= NDIM; j++ ) {
            tri[new_tri_id][j]     = tri[old_tri_id][j];
            tri_nab[new_tri_id][j] = tri_nab[old_tri_id][j];
            // Update the neighbors
            if ( tri_nab[new_tri_id][j] != -1 ) {
                bool found = false;
                for ( int j2 = 0; j2 <= NDIM; j2++ ) {
                    if ( tri_nab[tri_nab[new_tri_id][j]][j2] == old_tri_id ) {
                        tri_nab[tri_nab[new_tri_id][j]][j2] = new_tri_id;
                        found                               = true;
                    }
                }
                if ( !found )
                    throw std::logic_error(
                        "Error with internal structures (delete any unused triangles)" );
            }
        }
        // Update the face list
        for ( int j = 0; j <= NDIM; j++ ) {
            if ( tri_nab[new_tri_id][j] == -1 )
                face_list.update_face( 1, &old_tri_id, &j, &new_tri_id, &j, tri.data() );
        }
        unused.pop_back();
        N_tri--;
    }
    bool test = false;
    for ( size_t i = 0; i < N_tri; i++ ) {
        for ( int d = 0; d <= NDIM; d++ ) {
            if ( tri[i][d] == -1 )
                test = true;
        }
    }
    if ( test ) {
        // We should have removed all the NULL triangles in the previous step
        throw std::logic_error( "Error with internal structures (NULL tri)\n" );
    }

    // Remove triangles that are only likely to create problems
    // clean_triangles<NDIM>( N, x.data(), N_tri, tri, tri_nab );

    // Resort the triangle indicies so the smallest index is first (should help with caching)
    // Note: this causes tests to fail (not sure why)
    /*if constexpr ( NDIM==2 ) {
        for (size_t i=0; i<N_tri; i++) {
            int *t1 = tri[i].data();
            int *n1 = tri_nab[i].data();
            int t2[NDIM+1], n2[NDIM+1];
            memcpy(t2,t1,(NDIM+1)*sizeof(int));
            memcpy(n2,n1,(NDIM+1)*sizeof(int));
            int j = 0;
            for (int d=1; d<=NDIM; d++) {
                if ( t1[d] < t1[j] )
                    j = d;
            }
            for (int d=0; d<=NDIM; d++) {
                t1[d] = t2[(j+d)%(NDIM+1)];
                n1[d] = n2[(j+d)%(NDIM+1)];
            }
        }
    }*/

    // Resort the triangles so the are sorted by the first index (improves caching)

    // Check the final set of triangles to make sure they are all valid
    bool all_valid =
        check_current_triangles<NDIM>( N, x.data(), N_tri, tri.data(), tri_nab.data(), unused, 0 );
    if ( !all_valid )
        throw std::logic_error( "Final check of triangles failed" );

    // Resize the output vectors
    tri.resize( N_tri );
    tri_nab.resize( N_tri );
    return std::tie( tri, tri_nab );
}


/********************************************************************
 * Function to remove sliver triangles on the surface                *
 ********************************************************************/
constexpr double vol_sphere( int NDIM, double r )
{
    if ( NDIM == 1 )
        return 2.0 * r;
    else if ( NDIM == 2 )
        return 3.141592653589793 * r * r;
    else if ( NDIM == 3 )
        return 4.188790204786391 * r * r * r;
    else
        throw std::logic_error( "vol_sphere is undefined for the NDIM" );
}
template<int NDIM>
static void clean_triangles( const int,
                             const std::array<int, NDIM> *x,
                             size_t &N_tri,
                             std::array<int, NDIM + 1> *tri,
                             std::array<int, NDIM + 1> *tri_nab )
{
    // Get a list of all triangles on the boundary and a figure of merit
    // We will use the ratio of the volume of the circumsphere to the volume of the simplex
    std::vector<std::pair<double, size_t>> index;
    index.reserve( 1000 );
    for ( size_t i = 0; i < N_tri; i++ ) {
        bool on_boundary = false;
        for ( int j = 0; j < NDIM + 1; j++ ) {
            if ( tri_nab[i][j] == -1 )
                on_boundary = true;
        }
        if ( on_boundary ) {
            std::array<int, NDIM> x2[NDIM + 1];
            for ( int j = 0; j < NDIM + 1; j++ ) {
                int k = tri[i][j];
                x2[j] = x[k];
            }
            double vol = DelaunayHelpers::calcVolume<NDIM, int>( x2 );
            double R, center[NDIM];
            get_circumsphere<NDIM, int>( x2, R, center );
            double quality = vol_sphere( NDIM, R ) / vol;
            index.emplace_back( quality, i );
        }
    }
    // Remove any triangles that fill more than tol of the sphere
    const double tol = 1e-4;
    for ( size_t i = 0; i < index.size(); i++ ) {
        if ( index[i].first > 1.0 / tol ) {
            std::swap( index[i], index.back() );
            index.pop_back();
        }
    }
    if ( index.empty() )
        return;
    // Sort the triangles based on the quality (worst last)
    std::sort( index.begin(), index.end() );
    // Remove the triangles
    while ( !index.empty() ) {
        size_t i = index.back().second;
        index.pop_back();
        // Check that all nodes are shared by at least 1 other triangle
        // we cannot remove any triangles that wil orphan a node
        bool remove = true;
        for ( int j1 = 0; j1 < NDIM + 1; j1++ ) {
            int n        = tri[i][j1];
            bool found_n = false;
            for ( int j2 = 0; j2 < NDIM + 1; j2++ ) {
                int k = tri_nab[i][j2];
                if ( k == -1 )
                    continue;
                for ( int j3 = 0; j3 < NDIM + 1; j3++ ) {
                    if ( tri[k][j3] == n )
                        found_n = true;
                }
            }
            if ( !found_n )
                remove = false;
        }
        // Remove the triangle
        if ( remove ) {
            size_t i2 = N_tri - 1;
            swap_triangles( i, i2, tri, tri_nab );
            for ( int j = 0; j < NDIM + 1; j++ ) {
                int k          = tri_nab[i2][j];
                tri[i2][j]     = -1;
                tri_nab[i2][j] = -1;
                for ( int j2 = 0; j2 < NDIM + 1; j2++ ) {
                    if ( tri_nab[k][j2] == static_cast<int>( i2 ) )
                        tri_nab[k][j2] = -1;
                }
            }
            N_tri--;
        }
    }
}


/********************************************************************
 * Function to swap two triangle indicies                            *
 ********************************************************************/
template<int NDIM>
static void
swap_triangles( int i1, int i2, std::array<int, NDIM + 1> *tri, std::array<int, NDIM + 1> *tri_nab )
{
    // First swap the triangle data
    for ( int j = 0; j < NDIM + 1; j++ ) {
        std::swap( tri[i1][j], tri[i2][j] );
        std::swap( tri_nab[i1][j], tri_nab[i2][j] );
    }
    // Get a unique list of all neighbors of either triangle (and the triangles themselves)
    int neighbors[2 * NDIM + 4];
    for ( int j = 0; j < NDIM + 1; j++ ) {
        neighbors[2 * j + 0] = tri_nab[i1][j];
        neighbors[2 * j + 1] = tri_nab[i2][j];
    }
    neighbors[2 * NDIM + 2] = i1;
    neighbors[2 * NDIM + 3] = i2;
    int N                   = 2 * NDIM + 4;
    int i                   = 0;
    while ( i < N ) {
        bool keep = neighbors[i] != -1;
        for ( int j = 0; j < i; j++ )
            keep = keep && neighbors[i] != neighbors[j];
        if ( !keep ) {
            std::swap( neighbors[i], neighbors[N - 1] );
            N--;
        } else {
            i++;
        }
    }
    // For each triangle swap any values for i1 and i2
    for ( i = 0; i < N; i++ ) {
        int k = neighbors[i];
        for ( int j = 0; j < NDIM + 1; j++ ) {
            if ( tri_nab[k][j] == i1 ) {
                tri_nab[k][j] = i2;
            } else if ( tri_nab[k][j] == i2 ) {
                tri_nab[k][j] = i1;
            }
        }
    }
}


/************************************************************************
 * This function tests if a point is inside the circumsphere of an       *
 *    nd-simplex.                                                        *
 * For performance, I assume the points are ordered properly such that   *
 * the volume of the simplex (as calculated by calc_volume) is positive. *
 *                                                                       *
 * The point is inside the circumsphere if the determinant is positive   *
 * for points stored in a clockwise manner.  If the order is not known,  *
 * we can compare to a point we know is inside the cicumsphere.          *
 *    |  x1-xi   y1-yi   z1-zi   (x1-xi)^2+(y1-yi)^2+(z1-yi)^2  |        *
 *    |  x2-xi   y2-yi   z2-zi   (x2-xi)^2+(y2-yi)^2+(z2-yi)^2  |        *
 *    |  x3-xi   y3-yi   z3-zi   (x3-xi)^2+(y3-yi)^2+(z3-yi)^2  |        *
 *    |  x4-xi   y4-yi   z4-zi   (x4-xi)^2+(y4-yi)^2+(z4-yi)^2  |        *
 * det(A) == 0:  We are on the circumsphere                              *
 * det(A) > 0:   We are inside the circumsphere                          *
 * det(A) < 0:   We are outside the circumsphere                         *
 *                                                                       *
 * Note: this implementation requires N^D precision                      *
 ************************************************************************/
template<int NDIM, class TYPE>
int test_in_circumsphere( const std::array<TYPE, NDIM> x[],
                          const std::array<TYPE, NDIM> &xi,
                          double TOL_VOL )
{
    using ETYPE = typename getETYPE<NDIM, TYPE>::ETYPE;
    if constexpr ( NDIM == 1 ) {
        TYPE x1 = std::min( x[0][0], x[1][0] );
        TYPE x2 = std::max( x[0][0], x[1][0] );
        if ( fabs( xi[0] - x1 ) <= TOL_VOL || fabs( xi[0] - x1 ) <= TOL_VOL )
            return 0; // We are on the circumsphere
        if ( xi[0] > x1 && xi[0] < x2 )
            return 1; // We inside the circumsphere
        return -1;    // We outside the circumsphere
    }
    // Solve the sub-determinants (requires N^NDIM precision)
    double R2 = 0.0;
    const ETYPE one( 1 ), neg( -1 );
    ETYPE det2[NDIM + 1], R[NDIM + 1];
    for ( int d = 0; d <= NDIM; d++ ) {
        ETYPE A2[NDIM * NDIM];
        ETYPE sum( 0 );
        for ( int j = 0; j < NDIM; j++ ) {
            ETYPE tmp( x[d][j] - xi[j] );
            sum += tmp * tmp;
            for ( int i = 0; i < d; i++ )
                A2[i + j * NDIM] = ETYPE( x[i][j] - xi[j] );
            for ( int i = d + 1; i <= NDIM; i++ )
                A2[i - 1 + j * NDIM] = ETYPE( x[i][j] - xi[j] );
        }
        R[d] = sum;
        R2 += static_cast<double>( R[d] );
        const ETYPE &sign = ( ( NDIM + d ) % 2 == 0 ) ? one : neg;
        det2[d]           = sign * DelaunayHelpers::det<ETYPE, NDIM>( A2 );
    }
    // Compute the determinate (requires N^(NDIM+2) precision, used internally in dot)
    double det_A = dot( NDIM + 1, det2, R );
    if ( fabs( det_A ) <= 0.1 * R2 * TOL_VOL ) {
        // We are on the circumsphere
        return 0;
    } else if ( det_A > 0 ) {
        // We inside the circumsphere
        return 1;
    } else {
        // We outside the circumsphere
        return -1;
    }
}
// clang-format off
template int test_in_circumsphere<1,int>( const std::array<int,1>[], const std::array<int,1> &, double );
template int test_in_circumsphere<2,int>( const std::array<int,2>[], const std::array<int,2> &, double );
template int test_in_circumsphere<3,int>( const std::array<int,3>[], const std::array<int,3> &, double );
template int test_in_circumsphere<1,double>( const std::array<double,1>[], const std::array<double,1> &, double );
template int test_in_circumsphere<2,double>( const std::array<double,2>[], const std::array<double,2> &, double );
template int test_in_circumsphere<3,double>( const std::array<double,3>[], const std::array<double,3> &, double );
// clang-format on


/********************************************************************
 * This function tests if the edge flip is valid.                    *
 * To be valid, the line between the two points that are not on the  *
 * surface and the intersection of the face must lie within the face *
 * The point of intersection lies within the face if the Barycentric *
 * coordinates for all but the given face are positive               *
 ********************************************************************/
template<int NDIM>
static bool
test_flip_valid( const std::array<int, NDIM> x[], const int i, const std::array<int, NDIM> &xi )
{
    auto L        = DelaunayHelpers::computeBarycentric<NDIM, int>( x, xi );
    bool is_valid = true;
    for ( int j = 0; j <= NDIM; j++ )
        is_valid = is_valid && ( j == i || L[j] >= 0 );
    return is_valid;
}


/************************************************************************
 * This function performs a flip in 2D                                   *
 * Note: all indicies are hard-coded for ndim=2 and some loops have been *
 * unrolled to simplify the code and improve performance                 *
 ************************************************************************/
static bool flip_2D( const std::array<int, 2> x[],
                     const int tri[],
                     const int tri_nab[],
                     const int t1,
                     const int s1,
                     const int t2,
                     const int s2,
                     int *index_old,
                     int *new_tri,
                     int *new_tri_nab,
                     const double TOL_VOL )
{
    // Check if the flip is valid (it should always be valid in 2D if it is necessary)
    std::array<int, 2> x2[3], xi;
    for ( int i = 0; i < 3; i++ ) {
        int k = tri[i + t1 * 3];
        x2[i] = x[k];
    }
    int k        = tri[s2 + t2 * 3];
    xi           = x[k];
    bool isvalid = test_flip_valid<2>( x2, s1, xi );
    if ( !isvalid ) {
        // We need to do an edge flip, and the edge flip is not valid
        // This should not actually occur
        return false;
    }
    // Create the new triangles that are formed by the edge flips and the neighbor list
    index_old[0] = t1;
    index_old[1] = t2;
    int is[2], tn[4];
    if ( s1 == 0 ) {
        is[0] = tri[1 + 3 * t1];
        is[1] = tri[2 + 3 * t1];
        tn[0] = tri_nab[1 + 3 * t1];
        tn[1] = tri_nab[2 + 3 * t1];
    } else if ( s1 == 1 ) {
        is[0] = tri[0 + 3 * t1];
        is[1] = tri[2 + 3 * t1];
        tn[0] = tri_nab[0 + 3 * t1];
        tn[1] = tri_nab[2 + 3 * t1];
    } else {
        is[0] = tri[0 + 3 * t1];
        is[1] = tri[1 + 3 * t1];
        tn[0] = tri_nab[0 + 3 * t1];
        tn[1] = tri_nab[1 + 3 * t1];
    }
    if ( is[0] == tri[0 + 3 * t2] )
        tn[2] = tri_nab[0 + 3 * t2];
    else if ( is[0] == tri[1 + 3 * t2] )
        tn[2] = tri_nab[1 + 3 * t2];
    else
        tn[2] = tri_nab[2 + 3 * t2];
    if ( is[1] == tri[0 + 3 * t2] )
        tn[3] = tri_nab[0 + 3 * t2];
    else if ( is[1] == tri[1 + 3 * t2] )
        tn[3] = tri_nab[1 + 3 * t2];
    else
        tn[3] = tri_nab[2 + 3 * t2];
    int in1        = tri[s1 + 3 * t1];
    int in2        = tri[s2 + 3 * t2];
    new_tri[0]     = in1;
    new_tri[1]     = in2;
    new_tri[2]     = is[0];
    new_tri[3]     = in1;
    new_tri[4]     = in2;
    new_tri[5]     = is[1];
    new_tri_nab[0] = tn[3];
    new_tri_nab[1] = tn[1];
    new_tri_nab[2] = -3;
    new_tri_nab[3] = tn[2];
    new_tri_nab[4] = tn[0];
    new_tri_nab[5] = -2;
    // Check that the new triagles are valid (and have the proper ordering)
    isvalid = true;
    for ( int it = 0; it < 2; it++ ) {
        for ( int i = 0; i < 3; i++ ) {
            int k = new_tri[i + it * 3];
            x2[i] = x[k];
        }
        double volume = DelaunayHelpers::calcVolume<2, int>( x2 );
        if ( fabs( volume ) <= TOL_VOL ) {
            // The triangle is invalid (collinear)
            isvalid = false;
        } else if ( volume < 0 ) {
            // The ordering of the points is invalid, swap the last two points (we want the volume
            // to be positive)
            int tmp                 = new_tri[1 + it * 3];
            new_tri[1 + it * 3]     = new_tri[2 + it * 3];
            new_tri[2 + it * 3]     = tmp;
            tmp                     = new_tri_nab[1 + it * 3];
            new_tri_nab[1 + it * 3] = new_tri_nab[2 + it * 3];
            new_tri_nab[2 + it * 3] = tmp;
        }
    }
    return isvalid;
}


/************************************************************************
 * This function performs a 2-2 flip in 3D                               *
 * Note: all indicies are hard-coded for ndim=3 and some loops may be    *
 * unrolled to simplify the code or improve performance                  *
 ************************************************************************/
static bool flip_3D_22( const std::array<int, 3> x[],
                        const int tri[],
                        const int tri_nab[],
                        const int t1,
                        const int s1,
                        const int t2,
                        const int s2,
                        int *index_old,
                        int *new_tri,
                        int *new_tri_nab,
                        const double TOL_VOL )
{
    // The 2-2 fip is only valid when 4 of the vertices lie on a plane on the convex hull
    // This is likely for structured grids
    for ( int i1 = 0; i1 < 4; i1++ ) {
        if ( tri_nab[i1 + 4 * t1] != -1 ) {
            // The face does not lie on the convex hull
            continue;
        }
        for ( int i2 = 0; i2 < 4; i2++ ) {
            if ( tri_nab[i2 + 4 * t2] != -1 ) {
                // The face does not lie on the convex hull
                continue;
            }
            if ( tri[i1 + 4 * t1] != tri[i2 + 4 * t2] ) {
                // The 5th vertex does not match for the two triangles
                continue;
            }
            // Get the list of the 4 vertices on the convex hull
            int v1, v2, v3, v4;
            v1 = tri[s1 + 4 * t1];
            v2 = tri[s2 + 4 * t2];
            v3 = -1;
            v4 = -1;
            for ( int j = 0; j < 4; j++ ) {
                if ( j == i1 || j == s1 )
                    continue;
                if ( v3 == -1 )
                    v3 = tri[j + 4 * t1];
                else
                    v4 = tri[j + 4 * t1];
            }
            // Check if the 4 vertices are coplanar (the resulting simplex will have a volume of 0)
            std::array<int, 3> x2[4] = { x[v1], x[v2], x[v3], x[v4] };
            double vol               = fabs( DelaunayHelpers::calcVolume<3, int>( x2 ) );
            if ( vol > TOL_VOL ) {
                // The points are not coplanar
                continue;
            }
            // Create the new triangles that are formed by the 2-2 edge flip
            int v5       = tri[i1 + 4 * t1];
            index_old[0] = t1;
            index_old[1] = t2;
            new_tri[0]   = v1;
            new_tri[1]   = v2;
            new_tri[2]   = v3;
            new_tri[3]   = v5; // The first triangle
            new_tri[4]   = v1;
            new_tri[5]   = v2;
            new_tri[6]   = v4;
            new_tri[7]   = v5; // The second triangle
            // Check that the new triagles are valid (this can occur if they are planar)
            bool isvalid = true;
            for ( int it = 0; it < 2; it++ ) {
                for ( int i = 0; i < 4; i++ ) {
                    int k = new_tri[i + it * 4];
                    x2[i] = x[k];
                }
                double volume = DelaunayHelpers::calcVolume<3, int>( x2 );
                if ( fabs( volume ) <= TOL_VOL ) {
                    // The triangle is invalid (collinear)
                    isvalid = false;
                } else if ( volume < 0 ) {
                    // The ordering of the points is invalid, swap the last two points (we want the
                    // volume to be positive)
                    int tmp             = new_tri[2 + it * 4];
                    new_tri[2 + it * 4] = new_tri[3 + it * 4];
                    new_tri[3 + it * 4] = tmp;
                }
            }
            if ( !isvalid )
                continue;
            // Check that the new triangles are Delanuay (we have already checked that they are
            // valid)
            for ( int j = 0; j < 4; j++ ) {
                int k = new_tri[j];
                x2[j] = x[k];
            }
            int test = test_in_circumsphere<3, int>( x2, x[v4], TOL_VOL );
            if ( test == 1 ) {
                // The flip did not fix the Delaunay condition
                continue;
            }
            // Compute the triangle neighbors (this is a less efficient, but general method)
            const int N_old = 2;
            const int N_new = 2;
            int f1, f2;
            for ( int j = 0; j < N_new * 4; j++ )
                new_tri_nab[j] = -1;
            for ( int j1 = 0; j1 < N_new; j1++ ) {
                for ( int j2 = j1 + 1; j2 < N_new;
                      j2++ ) { // We only need to loop through unchecked triangles
                    bool are_neighbors =
                        are_tri_neighbors( 3, &new_tri[j1 * 4], &new_tri[j2 * 4], &f1, &f2 );
                    if ( are_neighbors ) {
                        new_tri_nab[j1 * 4 + f1] = -j2 - 2;
                        new_tri_nab[j2 * 4 + f2] = -j1 - 2;
                    } else {
                        printf( "Internal error (flip_3D_22)\n" );
                        return false;
                    }
                }
            }
            int neighbors[8];
            for ( int j1 = 0; j1 < N_old; j1++ ) {
                for ( int j2 = 0; j2 < 4; j2++ )
                    neighbors[j2 + j1 * 4] = tri_nab[j2 + index_old[j1] * 4];
            }
            for ( auto &neighbor : neighbors ) {
                for ( int j2 = 0; j2 < N_old; j2++ ) {
                    if ( neighbor == index_old[j2] )
                        neighbor = -1;
                }
            }
            for ( int j1 = 0; j1 < N_new; j1++ ) {
                for ( auto &neighbor : neighbors ) {
                    if ( neighbor == -1 )
                        continue;
                    bool are_neighbors =
                        are_tri_neighbors( 3, &new_tri[j1 * 4], &tri[neighbor * 4], &f1, &f2 );
                    if ( are_neighbors )
                        new_tri_nab[j1 * 4 + f1] = neighbor;
                }
            }
            return true;
        }
    }
    return false;
}


/************************************************************************
 * This function performs a 3-2 flip in 3D                               *
 * Note: all indicies are hard-coded for ndim=3 and some loops may be    *
 * unrolled to simplify the code or improve performance                  *
 ************************************************************************/
static bool flip_3D_32( const std::array<int, 3> x[],
                        const int tri[],
                        const int tri_nab[],
                        const int t1,
                        const int,
                        const int t2,
                        const int,
                        int *index_old,
                        int *new_tri,
                        int *new_tri_nab,
                        const double TOL_VOL )
{
    // Search for triangles that are neighbors to both t1 and t2
    int nab_list[4];
    int N_nab = 0;
    for ( int i1 = 0; i1 < 4; i1++ ) {
        for ( int i2 = 0; i2 < 4; i2++ ) {
            if ( tri_nab[i1 + 4 * t1] == tri_nab[i2 + 4 * t2] && tri_nab[i1 + 4 * t1] != -1 ) {
                nab_list[N_nab] = tri_nab[i1 + 4 * t1];
                N_nab++;
            }
        }
    }
    if ( N_nab == 0 ) {
        // No triangle neighbors both t1 and t2
        return false;
    }
    for ( int i = 0; i < N_nab; i++ ) {
        // The nodes that are common to all 3 triangles are the unique vertices on the 3 new
        // triangles
        int nodes[4];
        for ( int j = 0; j < 4; j++ )
            nodes[j] = tri[j + 4 * nab_list[i]];
        for ( auto &node : nodes ) {
            bool found1 = false;
            bool found2 = false;
            for ( int j2 = 0; j2 < 4; j2++ ) {
                if ( node == tri[j2 + 4 * t1] )
                    found1 = true;
                if ( node == tri[j2 + 4 * t2] )
                    found2 = true;
            }
            if ( !found1 || !found2 )
                node = -1;
        }
        int N_nodes = 0;
        for ( auto &node : nodes ) {
            if ( node != -1 ) {
                nodes[N_nodes] = node;
                N_nodes++;
            }
        }
        if ( N_nodes != 2 ) {
            // This should not occur
            printf( "Unexpected error\n" );
            return false;
        }
        int t[3];
        t[0] = t1;
        t[1] = t2;
        t[2] = nab_list[i];
        int surf[4];
        int k = 0;
        for ( auto &elem : t ) {
            for ( int j2 = 0; j2 < 4; j2++ ) {
                int tmp = tri[j2 + 4 * elem];
                if ( tmp != nodes[0] && tmp != nodes[1] ) {
                    bool in_surf = false;
                    for ( int j3 = 0; j3 < k; j3++ ) {
                        if ( tmp == surf[j3] )
                            in_surf = true;
                    }
                    if ( !in_surf ) {
                        surf[k] = tmp;
                        k++;
                    }
                }
            }
        }
        if ( k != 3 ) {
            // This should not occur
            printf( "Unexpected error\n" );
            return false;
        }
        // Create the new triangles
        index_old[0] = t1;
        index_old[1] = t2;
        index_old[2] = nab_list[i];
        new_tri[0]   = nodes[0];
        new_tri[1]   = surf[0];
        new_tri[2]   = surf[1];
        new_tri[3]   = surf[2];
        new_tri[4]   = nodes[1];
        new_tri[5]   = surf[0];
        new_tri[6]   = surf[1];
        new_tri[7]   = surf[2];
        // Check that the new triangles are valid (this can occur if they are planar)
        std::array<int, 3> x2[4];
        bool isvalid = true;
        for ( int it = 0; it < 2; it++ ) {
            for ( int i = 0; i < 4; i++ ) {
                int k = new_tri[i + it * 4];
                x2[i] = x[k];
            }
            double volume = DelaunayHelpers::calcVolume<3, int>( x2 );
            if ( fabs( volume ) <= TOL_VOL ) {
                // The triangle is invalid (collinear)
                isvalid = false;
            } else if ( volume < 0 ) {
                // The ordering of the points is invalid, swap the last two points (we want the
                // volume to be positive)
                int tmp             = new_tri[2 + it * 4];
                new_tri[2 + it * 4] = new_tri[3 + it * 4];
                new_tri[3 + it * 4] = tmp;
            }
        }
        if ( !isvalid )
            continue;
        // Check that the new triangles are Delanuay (we have already checked that they are valid)
        for ( int j = 0; j < 4; j++ ) {
            int k = new_tri[j];
            x2[j] = x[k];
        }
        int test = test_in_circumsphere<3, int>( x2, x[nodes[1]], TOL_VOL );
        if ( test == 1 ) {
            // The new triangles did not fixed the surface
            continue;
        }
        // Compute the triangle neighbors (this is a less efficient, but general method)
        const int N_old = 3;
        const int N_new = 2;
        int f1, f2;
        for ( int j = 0; j < N_new * 4; j++ )
            new_tri_nab[j] = -1;
        for ( int j1 = 0; j1 < N_new; j1++ ) {
            for ( int j2 = j1 + 1; j2 < N_new;
                  j2++ ) { // We only need to loop through unchecked triangles
                bool are_neighbors =
                    are_tri_neighbors( 3, &new_tri[j1 * 4], &new_tri[j2 * 4], &f1, &f2 );
                if ( are_neighbors ) {
                    new_tri_nab[j1 * 4 + f1] = -j2 - 2;
                    new_tri_nab[j2 * 4 + f2] = -j1 - 2;
                } else {
                    printf( "Internal error (flip_3D_32)\n" );
                    return false;
                }
            }
        }
        int neighbors[12];
        for ( int j1 = 0; j1 < N_old; j1++ ) {
            for ( int j2 = 0; j2 < 4; j2++ )
                neighbors[j2 + j1 * 4] = tri_nab[j2 + index_old[j1] * 4];
        }
        for ( auto &neighbor : neighbors ) {
            for ( int j2 = 0; j2 < N_old; j2++ ) {
                if ( neighbor == index_old[j2] )
                    neighbor = -1;
            }
        }
        for ( int j1 = 0; j1 < N_new; j1++ ) {
            for ( auto &neighbor : neighbors ) {
                if ( neighbor == -1 )
                    continue;
                bool are_neighbors =
                    are_tri_neighbors( 3, &new_tri[j1 * 4], &tri[neighbor * 4], &f1, &f2 );
                if ( are_neighbors )
                    new_tri_nab[j1 * 4 + f1] = neighbor;
            }
        }
        // Finished
        return true;
    }
    return false;
}


/************************************************************************
 * This function performs a 2-3 flip in 3D                               *
 * Note: all indicies are hard-coded for ndim=3 and some loops may be    *
 * unrolled to simplify the code or improve performance                  *
 ************************************************************************/
static bool flip_3D_23( const std::array<int, 3> x[],
                        const int tri[],
                        const int tri_nab[],
                        const int t1,
                        const int s1,
                        const int t2,
                        const int s2,
                        int *index_old,
                        int *new_tri,
                        int *new_tri_nab,
                        const double TOL_VOL )
{
    // First lets check if the flip is valid
    std::array<int, 3> x2[4];
    for ( int i = 0; i < 4; i++ ) {
        int k = tri[i + 4 * t1];
        x2[i] = x[k];
    }
    int k        = tri[s2 + 4 * t2];
    bool isvalid = test_flip_valid<3>( x2, s1, x[k] );
    if ( !isvalid ) {
        // We need to do an edge flip, and the edge flip is not valid
        return false;
    }
    // Form the new triangles
    int is[3], in[2] = { 0, 0 };
    k = 0;
    for ( int i = 0; i < 4; i++ ) {
        if ( i == s1 ) {
            in[0] = tri[i + 4 * t1];
        } else {
            is[k] = tri[i + 4 * t1];
            k++;
        }
    }
    in[1]        = tri[s2 + 4 * t2];
    index_old[0] = t1;
    index_old[1] = t2;
    new_tri[0]   = in[0];
    new_tri[1]   = in[1];
    new_tri[2]   = is[0];
    new_tri[3]   = is[1];
    new_tri[4]   = in[0];
    new_tri[5]   = in[1];
    new_tri[6]   = is[0];
    new_tri[7]   = is[2];
    new_tri[8]   = in[0];
    new_tri[9]   = in[1];
    new_tri[10]  = is[1];
    new_tri[11]  = is[2];
    // Check that the new triagles are valid (this can occur if they are planar)
    isvalid = true;
    for ( int it = 0; it < 3; it++ ) {
        for ( int i = 0; i < 4; i++ ) {
            int k = new_tri[i + it * 4];
            x2[i] = x[k];
        }
        double volume = DelaunayHelpers::calcVolume<3, int>( x2 );
        if ( fabs( volume ) <= TOL_VOL ) {
            // The triangle is invalid (collinear)
            isvalid = false;
        } else if ( volume < 0 ) {
            // The ordering of the points is invalid, swap the last two points (we want the volume
            // to be positive)
            int tmp             = new_tri[2 + it * 4];
            new_tri[2 + it * 4] = new_tri[3 + it * 4];
            new_tri[3 + it * 4] = tmp;
        }
    }
    if ( !isvalid )
        return false;
    // Check that the new triangles are Delanuay (we have already checked that they are valid)
    for ( int it = 0; it < 3; it++ ) {
        for ( int i = 0; i < 4; i++ ) {
            int k = new_tri[i + it * 4];
            x2[i] = x[k];
        }
        int k;
        if ( it == 0 ) {
            k = is[2];
        } else if ( it == 1 ) {
            k = is[1];
        } else {
            k = is[0];
        }
        int test = test_in_circumsphere<3, int>( x2, x[k], TOL_VOL );
        if ( test == 1 ) {
            // The new triangles did not fix the surface
            isvalid = false;
        }
    }
    if ( !isvalid )
        return false;
    // Compute the triangle neighbors (this is a less efficient, but general method)
    const int N_old = 2;
    const int N_new = 3;
    int f1, f2;
    for ( int j = 0; j < N_new * 4; j++ )
        new_tri_nab[j] = -1;
    for ( int j1 = 0; j1 < N_new; j1++ ) {
        for ( int j2 = j1 + 1; j2 < N_new;
              j2++ ) { // We only need to loop through unchecked triangles
            bool are_neighbors =
                are_tri_neighbors( 3, &new_tri[j1 * 4], &new_tri[j2 * 4], &f1, &f2 );
            if ( are_neighbors ) {
                new_tri_nab[j1 * 4 + f1] = -j2 - 2;
                new_tri_nab[j2 * 4 + f2] = -j1 - 2;
            } else {
                printf( "Internal error (flip_3D_23)\n" );
                return false;
            }
        }
    }
    int neighbors[8];
    for ( int j1 = 0; j1 < N_old; j1++ ) {
        for ( int j2 = 0; j2 < 4; j2++ )
            neighbors[j2 + j1 * 4] = tri_nab[j2 + index_old[j1] * 4];
    }
    for ( auto &neighbor : neighbors ) {
        for ( int j2 = 0; j2 < N_old; j2++ ) {
            if ( neighbor == index_old[j2] )
                neighbor = -1;
        }
    }
    for ( int j1 = 0; j1 < N_new; j1++ ) {
        for ( auto &neighbor : neighbors ) {
            if ( neighbor == -1 )
                continue;
            bool are_neighbors =
                are_tri_neighbors( 3, &new_tri[j1 * 4], &tri[neighbor * 4], &f1, &f2 );
            if ( are_neighbors )
                new_tri_nab[j1 * 4 + f1] = neighbor;
        }
    }
    return true;
}


/************************************************************************
 * This function performs a 4-4 flip in 3D                               *
 * Note: all indicies are hard-coded for ndim=3 and some loops may be    *
 * unrolled to simplify the code or improve performance                  *
 ************************************************************************/
static bool flip_3D_44( const std::array<int, 3> x[],
                        const int tri[],
                        const int tri_nab[],
                        const int t1,
                        const int s1,
                        const int t2,
                        const int s2,
                        int *index_old,
                        int *new_tri,
                        int *new_tri_nab,
                        const double TOL_VOL )
{
    // Loop through the triangle neighbors for both t1 and t2, to find
    // a pair of triangles that form and octahedron
    for ( int i1 = 0; i1 < 4; i1++ ) {
        int t3 = tri_nab[i1 + 4 * t1];
        for ( int i2 = 0; i2 < 4; i2++ ) {
            int t4 = tri_nab[i2 + 4 * t2];
            if ( t3 == -1 || t3 == t2 || t4 == -1 || t4 == t1 || t3 == t4 ) {
                // We do not have 4 unique triangles
                continue;
            }
            if ( tri_nab[0 + 4 * t3] != t4 && tri_nab[1 + 4 * t3] != t4 &&
                 tri_nab[2 + 4 * t3] != t4 && tri_nab[3 + 4 * t3] != t4 ) {
                // The 4 triangles do not form an octahedron
                continue;
            }
            // Get a list of the surface ids
            int s12, s13, s21, s24, s31 = 0, s34 = 0, s42 = 0, s43 = 0;
            s12 = s1;
            s13 = i1;
            s21 = s2;
            s24 = i2;
            for ( int i = 0; i < 4; i++ ) {
                if ( tri_nab[i + 4 * t3] == t1 )
                    s31 = i;
                if ( tri_nab[i + 4 * t3] == t4 )
                    s34 = i;
                if ( tri_nab[i + 4 * t4] == t2 )
                    s42 = i;
                if ( tri_nab[i + 4 * t4] == t3 )
                    s43 = i;
            }
            // Get a list of the unique nodes
            int nodes[16];
            for ( auto &node : nodes )
                node = -1;
            for ( int i = 0; i < 4; i++ )
                nodes[i] = tri[i + 4 * t1];
            int N_nodes = 4;
            for ( int j1 = 0; j1 < 4; j1++ ) {
                bool found = false;
                for ( int j2 = 0; j2 < N_nodes; j2++ ) {
                    if ( tri[j1 + 4 * t2] == nodes[j2] )
                        found = true;
                }
                if ( !found ) {
                    nodes[N_nodes] = tri[j1 + 4 * t2];
                    N_nodes++;
                }
            }
            for ( int j1 = 0; j1 < 4; j1++ ) {
                bool found = false;
                for ( int j2 = 0; j2 < N_nodes; j2++ ) {
                    if ( tri[j1 + 4 * t3] == nodes[j2] )
                        found = true;
                }
                if ( !found ) {
                    nodes[N_nodes] = tri[j1 + 4 * t3];
                    N_nodes++;
                }
            }
            for ( int j1 = 0; j1 < 4; j1++ ) {
                bool found = false;
                for ( int j2 = 0; j2 < N_nodes; j2++ ) {
                    if ( tri[j1 + 4 * t4] == nodes[j2] )
                        found = true;
                }
                if ( !found ) {
                    nodes[N_nodes] = tri[j1 + 4 * t4];
                    N_nodes++;
                }
            }
            if ( N_nodes != 6 ) {
                // We still do not have a valid octahedron
                printf( "Unexpected number of nodes\n" );
                continue;
            }
            // Check if any 4 of the nodes are coplanar (the resulting simplex will have a volume of
            // 0)
            // Note: while there are 15 combinations of 4 points, in reality
            // because the triangle faces already specify groups of 3 that
            // are coplanar, we only need to check 2 groups of points
            int set1[4], set2[4];
            for ( int i = 0; i < 4; i++ ) {
                set1[i] = tri[i + 4 * t1];
                set2[i] = tri[i + 4 * t1];
            }
            set1[s13] = tri[s21 + t2 * 4];
            set2[s12] = tri[s31 + t3 * 4];
            std::array<int, 3> x2[4];
            double vol1, vol2;
            for ( int j1 = 0; j1 < 4; j1++ ) {
                x2[j1] = x[set1[j1]];
            }
            vol1 = fabs( DelaunayHelpers::calcVolume<3, int>( x2 ) );
            for ( int j1 = 0; j1 < 4; j1++ ) {
                x2[j1] = x[set2[j1]];
            }
            vol2 = fabs( DelaunayHelpers::calcVolume<3, int>( x2 ) );
            for ( int it = 0; it < 2; it++ ) { // Loop through the sets (2)
                if ( it == 0 ) {
                    // Check set 1
                    if ( vol1 > TOL_VOL ) {
                        // Set 1 is not coplanar
                        continue;
                    }
                    // Set 1 is coplanar, check if the line between the two unused points passes
                    // through the 4 points in the set. If it does, then test_flip_valid will be
                    // true
                    // using either the current triangle and neighbor, or using the other 2
                    // triangles
                    for ( int j1 = 0; j1 < 4; j1++ ) {
                        int k  = tri[j1 + 4 * t1];
                        x2[j1] = x[k];
                    }
                    auto xi        = x[tri[s21 + 4 * t2]];
                    bool is_valid1 = test_flip_valid<3>( x2, s12, xi );
                    for ( int j1 = 0; j1 < 4; j1++ ) {
                        int k  = tri[j1 + 4 * t3];
                        x2[j1] = x[k];
                    }
                    bool is_valid2 = test_flip_valid<3>( x2, s34, xi );
                    if ( !is_valid1 && !is_valid2 ) {
                        // The flip is not valid
                        continue;
                    }
                } else if ( it == 1 ) {
                    // Check set 2
                    if ( vol2 > TOL_VOL ) {
                        // Set 1 is not coplanar
                        continue;
                    }
                    // Set 2 is coplanar, check if the line between the two unused points passes
                    // through the 4 points in the set. If it does, then test_flip_valid will be
                    // true
                    // using either the current triangle and neighbor, or using the other 2
                    // triangles
                    for ( int j1 = 0; j1 < 4; j1++ ) {
                        int k  = tri[j1 + 4 * t1];
                        x2[j1] = x[k];
                    }
                    auto xi        = x[tri[s31 + 4 * t3]];
                    bool is_valid1 = test_flip_valid<3>( x2, s13, xi );
                    for ( int j1 = 0; j1 < 4; j1++ ) {
                        int k  = tri[j1 + 4 * t2];
                        x2[j1] = x[k];
                    }
                    bool is_valid2 = test_flip_valid<3>( x2, s24, xi );
                    if ( !is_valid1 && !is_valid2 ) {
                        // The flip is not valid
                        continue;
                    }
                } else {
                    printf( "Unexpected error\n" );
                    return false;
                }
                // Get a list of the nodes common to all triangles
                int node_all[4];
                int k = 0;
                for ( int i = 0; i < 4; i++ ) {
                    int tmp     = tri[i + 4 * t1];
                    bool found2 = false;
                    for ( int j = 0; j < 4; j++ ) {
                        if ( tri[j + 4 * t2] == tmp )
                            found2 = true;
                    }
                    bool found3 = false;
                    for ( int j = 0; j < 4; j++ ) {
                        if ( tri[j + 4 * t3] == tmp )
                            found3 = true;
                    }
                    bool found4 = false;
                    for ( int j = 0; j < 4; j++ ) {
                        if ( tri[j + 4 * t3] == tmp )
                            found4 = true;
                    }
                    if ( found2 && found3 && found4 ) {
                        node_all[k] = tmp;
                        k++;
                    }
                }
                if ( k != 2 ) {
                    printf( "Unexpected error\n" );
                    return false;
                }
                // We have a valid flip, create the triangles
                index_old[0] = t1;
                index_old[1] = t2;
                index_old[2] = t3;
                index_old[3] = t4;
                if ( it == 0 ) {
                    // The 1-3 and 2-4 faces form the plane
                    new_tri[0]  = tri[s12 + 4 * t1];
                    new_tri[1]  = tri[s21 + 4 * t2];
                    new_tri[2]  = tri[s13 + 4 * t1];
                    new_tri[3]  = node_all[0];
                    new_tri[4]  = tri[s12 + 4 * t1];
                    new_tri[5]  = tri[s21 + 4 * t2];
                    new_tri[6]  = tri[s13 + 4 * t1];
                    new_tri[7]  = node_all[1];
                    new_tri[8]  = tri[s34 + 4 * t3];
                    new_tri[9]  = tri[s43 + 4 * t4];
                    new_tri[10] = tri[s31 + 4 * t3];
                    new_tri[11] = node_all[0];
                    new_tri[12] = tri[s34 + 4 * t3];
                    new_tri[13] = tri[s43 + 4 * t4];
                    new_tri[14] = tri[s31 + 4 * t3];
                    new_tri[15] = node_all[1];
                } else {
                    // The 1-2 and 3-4 faces form the plane
                    new_tri[0]  = tri[s13 + 4 * t1];
                    new_tri[1]  = tri[s31 + 4 * t3];
                    new_tri[2]  = tri[s12 + 4 * t1];
                    new_tri[3]  = node_all[0];
                    new_tri[4]  = tri[s13 + 4 * t1];
                    new_tri[5]  = tri[s31 + 4 * t3];
                    new_tri[6]  = tri[s12 + 4 * t1];
                    new_tri[7]  = node_all[1];
                    new_tri[8]  = tri[s24 + 4 * t2];
                    new_tri[9]  = tri[s42 + 4 * t4];
                    new_tri[10] = tri[s21 + 4 * t2];
                    new_tri[11] = node_all[0];
                    new_tri[12] = tri[s24 + 4 * t2];
                    new_tri[13] = tri[s42 + 4 * t4];
                    new_tri[14] = tri[s21 + 4 * t2];
                    new_tri[15] = node_all[1];
                }
                // Check that the new triagles are valid (this can occur if they are planar)
                bool isvalid = true;
                for ( int it2 = 0; it2 < 4; it2++ ) {
                    for ( int i = 0; i < 4; i++ ) {
                        int k = new_tri[i + it2 * 4];
                        x2[i] = x[k];
                    }
                    double volume = DelaunayHelpers::calcVolume<3, int>( x2 );
                    if ( fabs( volume ) <= TOL_VOL ) {
                        // The triangle is invalid (coplanar)
                        isvalid = false;
                    } else if ( volume < 0 ) {
                        // The ordering of the points is invalid, swap the last two points (we want
                        // the volume to be positive)
                        int tmp              = new_tri[2 + it2 * 4];
                        new_tri[2 + it2 * 4] = new_tri[3 + it2 * 4];
                        new_tri[3 + it2 * 4] = tmp;
                    }
                }
                if ( !isvalid )
                    continue;
                // Check that the new triangles are Delanuay (we have already checked that they are
                // valid)
                for ( int it2 = 0; it2 < 4; it2++ ) { // Loop through the test cases
                    for ( int i = 0; i < 4; i++ ) {
                        int k = new_tri[i + it2 * 4];
                        x2[i] = x[k];
                    }
                    int k = -1;
                    if ( it2 == 0 || it2 == 2 )
                        k = new_tri[7];
                    else if ( it2 == 1 || it2 == 3 )
                        k = new_tri[3];
                    int test = test_in_circumsphere<3, int>( x2, x[k], TOL_VOL );
                    if ( test == 1 )
                        isvalid = false;
                }
                if ( !isvalid )
                    continue;
                // Compute the triangle neighbors (this is a less efficient, but general method)
                const int N_old = 4;
                const int N_new = 4;
                int f1, f2;
                for ( int j = 0; j < N_new * 4; j++ )
                    new_tri_nab[j] = -1;
                for ( int j1 = 0; j1 < N_new; j1++ ) {
                    for ( int j2 = j1 + 1; j2 < N_new;
                          j2++ ) { // We only need to loop through unchecked triangles
                        bool are_neighbors =
                            are_tri_neighbors( 3, &new_tri[j1 * 4], &new_tri[j2 * 4], &f1, &f2 );
                        if ( are_neighbors ) {
                            new_tri_nab[j1 * 4 + f1] = -j2 - 2;
                            new_tri_nab[j2 * 4 + f2] = -j1 - 2;
                        }
                    }
                }
                int neighbors[16];
                for ( int j1 = 0; j1 < N_old; j1++ ) {
                    for ( int j2 = 0; j2 < 4; j2++ )
                        neighbors[j2 + j1 * 4] = tri_nab[j2 + index_old[j1] * 4];
                }
                for ( auto &neighbor : neighbors ) {
                    for ( int j2 = 0; j2 < N_old; j2++ ) {
                        if ( neighbor == index_old[j2] )
                            neighbor = -1;
                    }
                }
                for ( int j1 = 0; j1 < N_new; j1++ ) {
                    for ( auto &neighbor : neighbors ) {
                        if ( neighbor == -1 )
                            continue;
                        bool are_neighbors =
                            are_tri_neighbors( 3, &new_tri[j1 * 4], &tri[neighbor * 4], &f1, &f2 );
                        if ( are_neighbors )
                            new_tri_nab[j1 * 4 + f1] = neighbor;
                    }
                }
                // Finished
                return true;
            }
        }
    }
    return false;
}


/************************************************************************
 * Function to find a valid flip                                         *
 * Note: the flip must also calculate the triangle neighbors.            *
 *   While this requires more effort when programming flips, the         *
 *   of the flips allow them to potentially calculate the                *
 *   neighbors much faster than a more generic method.                   *
 *      new_tri_nab >= 0   - Triangle neighbor is an existing triangle   *
 *                           (with the given index)                      *
 *      new_tri_nab -(i+1) - Triangle neighbor is the ith new triangle   *
 *      new_tri_nab ==-1   - Triangle face is on the convex hull         *
 ************************************************************************/
static bool find_flip_2D( const std::array<int, 2> *x,
                          const int *tri,
                          const int *tri_nab,
                          const double TOL_VOL,
                          std::vector<check_surface_struct> &check_surface,
                          int &N_tri_old,
                          int &N_tri_new,
                          int *index_old,
                          int *new_tri,
                          int *new_tri_nab )
{
    PROFILE( "find_flip<2>", 4 );
    // In 2D we only have one type of flip (2-2 flip)
    bool found = false;
    for ( auto &elem : check_surface ) {
        if ( ( elem.test & 0x02 ) != 0 ) {
            // We already tried this flip
            continue;
        }
        int s1 = elem.f1;
        int s2 = elem.f2;
        int t1 = elem.t1;
        int t2 = elem.t2;
#if DEBUG_CHECK >= 1
        TYPE x1[6] = { 0 }, x2[6] = { 0 }, xi1[2] = { 0 }, xi2[2] = { 0 };
        for ( int j = 0; j < 3; j++ ) {
            int m1 = tri[j + t1 * 3];
            int m2 = tri[j + t2 * 3];
            x1[j]  = x[m1];
            x2[j]  = x[m2];
        }
        xi1[0]    = x2[0 + 2 * s2];
        xi1[1]    = x2[1 + 2 * s2];
        xi2[0]    = x1[0 + 2 * s1];
        xi2[1]    = x1[1 + 2 * s1];
        int test1 = test_in_circumsphere<2, int>( x1, xi1, TOL_VOL );
        int test2 = test_in_circumsphere<2, int>( x2, xi2, TOL_VOL );
        if ( test1 != 1 || test2 != 1 )
            throw std::logic_error( "Internal error" );
#endif
        bool flipped_edge =
            flip_2D( x, tri, tri_nab, t1, s1, t2, s2, index_old, new_tri, new_tri_nab, TOL_VOL );
        if ( flipped_edge ) {
            N_tri_old = 2;
            N_tri_new = 2;
            found     = true;
            break;
        } else {
            // The 2-2 flip is not valid, set the second bit to indicate we checked this
            elem.test |= 0x02;
            printf( "Warning: necessary flips in 2D should always be valid\n" );
        }
    }
    return found;
}
static bool find_flip_3D( const std::array<int, 3> *x,
                          const int *tri,
                          const int *tri_nab,
                          const double TOL_VOL,
                          std::vector<check_surface_struct> &check_surface,
                          int &N_tri_old,
                          int &N_tri_new,
                          int *index_old,
                          int *new_tri,
                          int *new_tri_nab )
{
    PROFILE( "find_flip<3>", 4 );
    bool found = false;
    // In 3D there are several possible types of flips
    // First let's see if there are any valid 2-2 flips
    if ( !found ) {
        for ( auto &elem : check_surface ) {
            if ( ( elem.test & 0x02 ) != 0 ) {
                // We already tried this flip
                continue;
            }
            int s1            = elem.f1;
            int s2            = elem.f2;
            int t1            = elem.t1;
            int t2            = elem.t2;
            bool flipped_edge = flip_3D_22(
                x, tri, tri_nab, t1, s1, t2, s2, index_old, new_tri, new_tri_nab, TOL_VOL );
            if ( flipped_edge ) {
                N_tri_old = 2;
                N_tri_new = 2;
                found     = true;
                break;
            } else {
                // The 2-2 flip is not valid, set the second bit to indicate we checked this
                elem.test |= 0x02;
            }
        }
    }
    // If we did not find a valid flip, check for 2-3 flips
    if ( !found ) {
        for ( auto &elem : check_surface ) {
            if ( ( elem.test & 0x08 ) != 0 ) {
                // We already tried this flip
                continue;
            }
            int s1            = elem.f1;
            int s2            = elem.f2;
            int t1            = elem.t1;
            int t2            = elem.t2;
            bool flipped_edge = flip_3D_23(
                x, tri, tri_nab, t1, s1, t2, s2, index_old, new_tri, new_tri_nab, TOL_VOL );
            if ( flipped_edge ) {
                N_tri_old = 2;
                N_tri_new = 3;
                found     = true;
                break;
            } else {
                // The 2-3 flip is not valid, set the fourth bit to indicate we checked this
                elem.test |= 0x08;
            }
        }
    }
    // If we did not find a valid flip, check for 3-2 flips
    // Note: we must recheck this if we modified a neighboring triangle
    if ( !found ) {
        for ( auto &elem : check_surface ) {
            int s1            = elem.f1;
            int s2            = elem.f2;
            int t1            = elem.t1;
            int t2            = elem.t2;
            bool flipped_edge = flip_3D_32(
                x, tri, tri_nab, t1, s1, t2, s2, index_old, new_tri, new_tri_nab, TOL_VOL );
            if ( flipped_edge ) {
                N_tri_old = 3;
                N_tri_new = 2;
                found     = true;
                break;
            } else {
                // The 3-2 flip is not valid, set the third bit to indicate we checked this
                elem.test |= 0x04;
            }
        }
    }
    // If we did not find a valid flip, check for 4-4 flips
    // Note: we must recheck this if we modified a neighboring triangle
    if ( !found ) {
        for ( auto &elem : check_surface ) {
            int s1            = elem.f1;
            int s2            = elem.f2;
            int t1            = elem.t1;
            int t2            = elem.t2;
            bool flipped_edge = flip_3D_44(
                x, tri, tri_nab, t1, s1, t2, s2, index_old, new_tri, new_tri_nab, TOL_VOL );
            if ( flipped_edge ) {
                N_tri_old = 4;
                N_tri_new = 4;
                found     = true;
                break;
            } else {
                // The 4-4 flip is not valid, set the fifth bit to indicate we checked this
                elem.test |= 0x10;
            }
        }
    }
    return found;
}
template<int NDIM>
static bool find_flip( const std::array<int, NDIM> *x,
                       const std::array<int, NDIM + 1> *tri,
                       const std::array<int, NDIM + 1> *tri_nab,
                       const double TOL_VOL,
                       std::vector<check_surface_struct> &check_surface,
                       int &N_tri_old,
                       int &N_tri_new,
                       int *index_old,
                       int *new_tri,
                       int *new_tri_nab )
{
    bool valid = false;
    if constexpr ( NDIM == 2 ) {
        valid = find_flip_2D( x,
                              tri[0].data(),
                              tri_nab[0].data(),
                              TOL_VOL,
                              check_surface,
                              N_tri_old,
                              N_tri_new,
                              index_old,
                              new_tri,
                              new_tri_nab );
    } else if constexpr ( NDIM == 3 ) {
        valid = find_flip_3D( x,
                              tri[0].data(),
                              tri_nab[0].data(),
                              TOL_VOL,
                              check_surface,
                              N_tri_old,
                              N_tri_new,
                              index_old,
                              new_tri,
                              new_tri_nab );
    }
    return valid;
}


/********************************************************************
 * Primary interfaces                                                *
 ********************************************************************/
template<class TYPE>
std::tuple<AMP::Array<int>, AMP::Array<int>> create_tessellation( const Array<TYPE> &x )
{
    int NDIM = x.size( 0 );
    // Convert to integer with max 30 bits
    auto y = convert( x );
    // Run the problem
    AMP::Array<int> tri, nab;
    if ( NDIM == 2 ) {
        auto x2           = DelaunayHelpers::convert<int, 2>( y );
        auto [tri2, nab2] = create_tessellation<2>( x2 );
        tri               = DelaunayHelpers::convert<int, 3>( tri2 );
        nab               = DelaunayHelpers::convert<int, 3>( nab2 );
    } else if ( NDIM == 3 ) {
        auto x2           = DelaunayHelpers::convert<int, 3>( y );
        auto [tri2, nab2] = create_tessellation<3>( x2 );
        tri               = DelaunayHelpers::convert<int, 4>( tri2 );
        nab               = DelaunayHelpers::convert<int, 4>( nab2 );
    } else {
        throw std::logic_error( "Unsupported dimension" );
    }
    return std::tie( tri, nab );
}
// clang-format off
template std::tuple<AMP::Array<int>, AMP::Array<int>> create_tessellation<short>( const Array<short> & );
template std::tuple<AMP::Array<int>, AMP::Array<int>> create_tessellation<int>( const Array<int> & );
template std::tuple<AMP::Array<int>, AMP::Array<int>> create_tessellation<int64_t>( const Array<int64_t> & );
template std::tuple<AMP::Array<int>, AMP::Array<int>> create_tessellation<float>( const Array<float> & );
template std::tuple<AMP::Array<int>, AMP::Array<int>> create_tessellation<double>( const Array<double> & );
// clang-format on


/********************************************************************
 * Primary interfaces                                                *
 ********************************************************************/
template<std::size_t NDIM, class TYPE>
static inline void copy_x2( const TYPE *x, std::array<TYPE, NDIM> x2[] )
{
    for ( size_t i = 0; i <= NDIM; i++ )
        for ( size_t d = 0; d < NDIM; d++ )
            x2[i][d] = x[d + i * NDIM];
}
template<size_t NDIM>
static inline double calc_volume2( const double x[] )
{
    std::array<double, NDIM> x2[NDIM + 1];
    copy_x2( x, x2 );
    return DelaunayHelpers::calcVolume<NDIM, double>( x2 );
}
double calc_volume( int ndim, const double x[] )
{
    double vol = 0.0;
    if ( ndim == 1 ) {
        vol = x[1] - x[0];
    } else if ( ndim == 2 ) {
        vol = calc_volume2<2>( x );
    } else if ( ndim == 3 ) {
        vol = calc_volume2<3>( x );
    } else if ( ndim == 4 ) {
        vol = calc_volume2<4>( x );
    } else {
        throw std::logic_error( "Unsupported dimension" );
    }
    return vol;
}


/************************************************************************
 * This function computes the circumsphere containing the simplex        *
 *     http://mathworld.wolfram.com/Circumsphere.html                    *
 * We have 4 points that define the circumsphere (x1, x2, x3, x4).       *
 * We will work in a reduced coordinate system with x1 at the center     *
 *    so that we can reduce the size of the determinant and the number   *
 *    of significant digits.                                             *
 *                                                                       *
 *     | x2-x1  y2-y1  z2-z1 |                                           *
 * a = | x3-x1  y3-y1  z3-z1 |                                           *
 *     | x4-x1  y4-y1  z4-z1 |                                           *
 *                                                                       *
 *      | (x2-x1)^2+(y2-y1)^2+(z2-z1)^2  y2  z2 |                        *
 * Dx = | (x3-x1)^2+(y3-y1)^2+(z3-z1)^2  y3  z3 |                        *
 *      | (x4-x1)^2+(y4-y1)^2+(z4-z1)^2  y4  z4 |                        *
 *                                                                       *
 *        | (x2-x1)^2+(y2-y1)^2+(z2-z1)^2  x2  z2 |                      *
 * Dy = - | (x3-x1)^2+(y3-y1)^2+(z3-z1)^2  x3  z3 |                      *
 *        | (x4-x1)^2+(y4-y1)^2+(z4-z1)^2  x4  z4 |                      *
 *                                                                       *
 *      | (x2-x1)^2+(y2-y1)^2+(z2-z1)^2  x2  y2 |                        *
 * Dz = | (x3-x1)^2+(y3-y1)^2+(z3-z1)^2  x3  y3 |                        *
 *      | (x4-x1)^2+(y4-y1)^2+(z4-z1)^2  x4  y4 |                        *
 *                                                                       *
 * c = 0                                                                 *
 *                                                                       *
 * Note: This version does not use exact math                            *
 ************************************************************************/
template<int NDIM, class TYPE>
void get_circumsphere( const std::array<TYPE, NDIM> x0[], double &R, double *center )
{
    if constexpr ( NDIM == 1 ) {
        center[0] = 0.5 * ( x0[0][0] + x0[1][0] );
        R         = 0.5 * std::abs( x0[0][0] - x0[1][0] );
        return;
    }
    long double x[NDIM * NDIM];
    for ( int i = 0; i < NDIM; i++ ) {
        for ( int j = 0; j < NDIM; j++ )
            x[j + i * NDIM] = static_cast<double>( x0[i + 1][j] - x0[0][j] );
    }
    long double A[NDIM * NDIM], D[NDIM][NDIM * NDIM];
    for ( int i = 0; i < NDIM; i++ ) {
        long double tmp( 0 );
        for ( int j = 0; j < NDIM; j++ ) {
            long double x2 = static_cast<double>( x[j + i * NDIM] );
            tmp += x2 * x2;
            A[i + j * NDIM] = x2;
            for ( int k = j + 1; k < NDIM; k++ )
                D[k][i + ( j + 1 ) * NDIM] = x2;
            for ( int k = 0; k < j; k++ )
                D[k][i + j * NDIM] = x2;
        }
        for ( auto &elem : D )
            elem[i] = tmp;
    }
    long double a = static_cast<double>( DelaunayHelpers::det<long double, NDIM>( A ) );
    R             = 0.0;
    for ( int i = 0; i < NDIM; i++ ) {
        long double d = ( ( i % 2 == 0 ) ? 1 : -1 ) *
                        static_cast<double>( DelaunayHelpers::det<long double, NDIM>( D[i] ) );
        center[i] = static_cast<double>( d / ( 2 * a ) + static_cast<long double>( x0[0][i] ) );
        R += d * d;
    }
    R = std::sqrt( R ) / fabs( static_cast<double>( 2 * a ) );
}
// clang-format off
template void get_circumsphere<1,int>( const std::array<int,1> x0[], double &R, double *center );
template void get_circumsphere<2,int>( const std::array<int,2> x0[], double &R, double *center );
template void get_circumsphere<3,int>( const std::array<int,3> x0[], double &R, double *center );
template void get_circumsphere<1,double>( const std::array<double,1> x0[], double &R, double *center );
template void get_circumsphere<2,double>( const std::array<double,2> x0[], double &R, double *center );
template void get_circumsphere<3,double>( const std::array<double,3> x0[], double &R, double *center );
// clang-format on


} // namespace AMP::DelaunayTessellation
