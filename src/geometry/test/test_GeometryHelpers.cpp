#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>


using namespace AMP::Geometry::GeometryHelpers;
using AMP::Mesh::Point;


/****************************************************************
 * Vector operations                                             *
 ****************************************************************/
static inline Point3D operator+( const Point3D &x, const Point3D &y )
{
    return { x[0] + y[0], x[1] + y[1], x[2] + y[2] };
}
static inline Point3D operator-( const Point3D &x, const Point3D &y )
{
    return { x[0] - y[0], x[1] - y[1], x[2] - y[2] };
}
static inline Point3D operator*( double x, const Point3D &y )
{
    return { x * y[0], x * y[1], x * y[2] };
}
static inline double dot( const Point3D &x, const Point3D &y )
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}
static inline Point3D normalize( const Point3D &x )
{
    double tmp = 1.0 / std::sqrt( dot( x, x ) );
    return { tmp * x[0], tmp * x[1], tmp * x[2] };
}
static inline Point2D convert2( const Point &x ) { return { x.x(), x.y() }; }

// Test distance to line
void test_dist_line( int N, AMP::UnitTest &ut )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( -50, 50 );
    // Test ray - line-segment intersection
    bool pass = true;
    auto t1   = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < N; i++ ) {
        // Generate an intersection point, then the rays and line segments
        Point pi   = { dis( gen ), dis( gen ) };
        Point p0   = { dis( gen ), dis( gen ) };
        Point p1   = p0 + 1.34 * ( pi - p0 );
        Point p2   = p0 + 0.98 * ( pi - p0 );
        Point p    = { dis( gen ), dis( gen ) };
        Point dir  = normalize( pi - p );
        double d   = ( pi.x() - p.x() ) / dir.x();
        double d1  = distanceToLine( convert2( p ), dir, p0, p1 );
        double d2  = distanceToLine( convert2( p ), dir, p0, p2 );
        double d3  = distanceToLine( convert2( p ), -dir, p0, p2 );
        double d4  = distanceToLine( convert2( p ), dir, p1, p0 );
        bool pass1 = fabs( d - d1 ) < 1e-8;
        bool pass2 = fabs( d2 ) > 1e200;
        bool pass3 = fabs( d3 ) > 1e200;
        bool pass4 = fabs( d - d4 ) < 1e-8;
        pass       = pass && pass1 && pass2 && pass3 && pass4;
        if ( !pass1 )
            printf( "distanceToLine (1): %f %f %e\n", d, d1, d - d1 );
        if ( !pass4 )
            printf( "distanceToLine (2): %f %f %e\n", d, d4, d - d4 );
        if ( !pass2 || !pass3 )
            printf( "distanceToLine (3): %f %f %e\n", d, d2, d3 );
    }
    auto t2    = std::chrono::high_resolution_clock::now();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
    printf( "distanceToLine: %i ns\n", static_cast<int>( ns / ( 4 * N ) ) );
    ut.pass_fail( pass, "distanceToLine (2D)" );
}


// Test the mapping to/from a logical circle
void test_map_logical_circle( int N, AMP::UnitTest &ut )
{
    auto distance = []( double x, double y, std::array<double, 2> xy ) {
        return std::sqrt( ( x - xy[0] ) * ( x - xy[0] ) + ( y - xy[1] ) * ( y - xy[1] ) );
    };
    std::random_device rd;
    std::mt19937 gen( rd() );
    const double tol = 1e-10;
    for ( int method = 1; method <= 3; method++ ) {
        const double r = 2.0;
        std::uniform_real_distribution<> dis( 0, 1 );
        bool pass = true;
        auto t1   = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < N; i++ ) {
            double x  = dis( gen );
            double y  = dis( gen );
            auto p    = map_logical_circle( r, method, x, y );
            auto p2   = map_circle_logical( r, method, p[0], p[1] );
            double r2 = std::sqrt( p[0] * p[0] + p[1] * p[1] );
            pass      = pass && r2 < r + 1e-15;
            pass      = pass && distance( x, y, p2 ) < tol;
            if ( !( distance( x, y, p2 ) < tol ) )
                printf( "%e %e %e %e %e %e\n", x, y, p2[0], p2[1], p2[0] - x, p2[1] - y );
        }
        auto t2    = std::chrono::high_resolution_clock::now();
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
        printf( "map_logical_circle - %i: %i ns\n", method, static_cast<int>( ns / ( 4 * N ) ) );
        ut.pass_fail( pass, "circle logical-physical-logical - " + std::to_string( method ) );
    }
    for ( int method = 1; method <= 3; method++ ) {
        std::uniform_real_distribution<> dis( -3.0, 3.0 );
        bool pass = true;
        for ( int i = 0; i < N; i++ ) {
            double x = dis( gen );
            double y = dis( gen );
            auto p   = map_circle_logical( 1.0, method, x, y );
            auto p2  = map_logical_circle( 1.0, method, p[0], p[1] );
            pass     = pass && distance( x, y, p2 ) < tol;
            if ( !( distance( x, y, p2 ) < tol ) )
                printf( "%e %e %e %e %e %e\n", x, y, p2[0], p2[1], p2[0] - x, p2[1] - y );
        }
        ut.pass_fail( pass, "circle physical-logical-physical - " + std::to_string( method ) );
    }
}


// Test the mapping to/from a regular polygon
void test_map_logical_poly( int N, AMP::UnitTest &ut )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( 0, 1 );
    const double tol = 1e-10;
    const double r   = 2.3;
    for ( int Np = 3; Np <= 10; Np++ ) {
        bool pass = true;
        auto t1   = std::chrono::high_resolution_clock::now();
        for ( int i = 0; i < N; i++ ) {
            double x  = dis( gen );
            double y  = dis( gen );
            auto p    = map_logical_poly( Np, r, x, y );
            auto p2   = map_poly_logical( Np, r, p[0], p[1] );
            double r2 = std::sqrt( p[0] * p[0] + p[1] * p[1] );
            pass      = pass && r2 < r + 1e-15;
            pass      = pass && fabs( p2[0] - x ) < tol && fabs( p2[1] - y ) < tol;
            if ( fabs( p2[0] - x ) > tol || fabs( p2[1] - y ) > tol )
                printf( "%e %e %e %e %e %e\n", x, y, p2[0], p2[1], p2[0] - x, p2[1] - y );
        }
        auto t2    = std::chrono::high_resolution_clock::now();
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
        printf( "map_logical_poly - %i: %i ns\n", Np, static_cast<int>( ns / N ) );
        ut.pass_fail( pass, "map_logical_poly - " + std::to_string( Np ) );
    }
}


// Test the mapping to/from the surface of a sphere
void test_map_logical_sphere_surface( int N, AMP::UnitTest &ut )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( 0, 1 );
    const double tol = 1e-10;
    const double r   = 2.0;
    bool pass        = true;
    for ( int method = 1; method <= 3; method++ ) {
        auto t1      = std::chrono::high_resolution_clock::now();
        int N_failed = 0;
        for ( int i = 0; i < N; i++ ) {
            double x   = dis( gen );
            double y   = dis( gen );
            auto p     = map_logical_sphere_surface( method, r, x, y );
            auto p2    = map_sphere_surface_logical( method, r, p[0], p[1], p[2] );
            double r2  = std::sqrt( p[0] * p[0] + p[1] * p[1] + p[2] * p[2] );
            double err = std::max( fabs( p2[0] - x ), fabs( p2[1] - y ) );
            bool pass2 = r2 < r + tol && err < tol;
            N_failed += pass2 ? 0 : 1;
            if ( !pass2 && N_failed < 20 )
                printf(
                    "%i: %e %e %e %e %e %e\n", method, x, y, p2[0], p2[1], p2[0] - x, p2[1] - y );
        }
        if ( N_failed > 3 ) {
            pass = false;
            ut.failure( "map_logical_sphere_surface: " + std::to_string( method ) );
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        int ns  = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / N;
        printf( "map_logical_sphere_surface (%i): %i ns\n", method, ns );
    }
    int N2 = 100;
    for ( int i = 0; i < N2; i++ ) {
        double x = i / static_cast<double>( N - 1 );
        auto p1  = map_logical_sphere_surface( 1, r, 0, x );
        auto p2  = map_logical_sphere_surface( 1, r, 1, x );
        if ( p1 != p2 ) {
            printf(
                "Failed boundary 1: (%e,%e,%e)\n", p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2] );
            pass = false;
        }
    }
    auto testMap = []( const AMP::Mesh::Point &p0 ) {
        auto p2 = map_sphere_surface_logical( 1, 1.0, p0[0], p0[1], p0[2] );
        auto p  = map_logical_sphere_surface( 1, 1.0, p2[0], p2[1] );
        auto d  = p0 - AMP::Mesh::Point( p );
        return d.abs() < 1e-12;
    };
    pass = pass && testMap( { 0, 0, -1 } );
    pass = pass && testMap( { 0, 0, 1 } );
    pass = pass && testMap( { 0, -1, 0 } );
    pass = pass && testMap( { 0, 1, 0 } );
    pass = pass && testMap( { -1, 0, 0 } );
    pass = pass && testMap( { 1, 0, 0 } );
    ut.pass_fail( pass, "map_logical_sphere_surface" );
}


// Create random triangle/ray/distance sets in the same plane
static std::tuple<std::array<Point3D, 3>, Point3D, Point3D, double> createTriRayPlane()
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( -1.0, 1.0 );
    // Create the center of a triangle and point of intersection
    Point3D c  = { dis( gen ), dis( gen ), dis( gen ) };
    Point3D pi = { dis( gen ), dis( gen ), dis( gen ) };
    // Create the ray
    auto dir = normalize( c - pi );
    double d = 1.0 + dis( gen );
    auto pos = pi - d * dir;
    // Create the triangle points
    Point3D v1 = normalize( { dis( gen ), dis( gen ), dis( gen ) } );
    std::array<Point3D, 3> tri;
    tri[0] = pi + fabs( dis( gen ) ) * v1;
    tri[1] = pi - fabs( dis( gen ) ) * v1;
    tri[2] = pi + 2 * dir;
    return std::tie( tri, pos, dir, d );
}


// Create random triangle/ray/distance sets in the same plane
static std::tuple<std::array<Point3D, 3>, Point3D, Point3D, double> createTriRayVol()
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( -1.0, 1.0 );
    // Create the triangle points
    std::array<Point3D, 3> tri;
    tri[0] = { dis( gen ), dis( gen ), dis( gen ) };
    tri[1] = { dis( gen ), dis( gen ), dis( gen ) };
    tri[2] = { dis( gen ), dis( gen ), dis( gen ) };
    // Create a random point within the triangle and direction
    Point3D L = { fabs( dis( gen ) ), fabs( dis( gen ) ), fabs( dis( gen ) ) };
    double t  = 1.0 / ( L[0] + L[1] + L[2] );
    L         = { t * L[0], t * L[1], t * L[2] };
    auto pi   = L[0] * tri[0] + L[1] * tri[1] + L[2] * tri[2];
    auto dir  = normalize( { dis( gen ), dis( gen ), dis( gen ) } );
    double d  = 1.0 + fabs( dis( gen ) );
    auto pos  = pi - d * dir;
    return std::tie( tri, pos, dir, d );
}


// Test ray-triangle intersection
void test_ray_triangle_intersection( int N, AMP::UnitTest &ut )
{
    bool pass = true;
    // Create some sample triangles
    std::array<Point2D, 3> t1 = { { { 0, 0 }, { 1, 0 }, { 0, 1 } } };
    std::array<Point3D, 3> t2 = { { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } } };
    // Test 2D triangle
    double d11 = distanceToTriangle( t1, { -1, 0 }, { 1, 0 } );
    double d12 = distanceToTriangle( t1, { -1, 0.5 }, { 1, 0 } );
    double d13 = distanceToTriangle( t1, { -1, -0.5 }, { 1, 0 } );
    double d14 = distanceToTriangle( t1, { 0.1, 0.2 }, { -1, 0 } );
    pass       = pass && fabs( d11 - 1.0 ) < 1e-12;
    pass       = pass && fabs( d12 - 1.0 ) < 1e-12;
    pass       = pass && fabs( d13 ) > 1e200;
    pass       = pass && fabs( d14 + 0.1 ) < 1e-12;
    // Test 2D triangles in 3D
    double d21 = distanceToTriangle( t2, { -1, 0, 0 }, { 1, 0, 0 } );
    double d22 = distanceToTriangle( t2, { -1, 0.5, 0 }, { 1, 0, 0 } );
    double d23 = distanceToTriangle( t2, { -1, -0.5, 0 }, { 1, 0, 0 } );
    pass       = pass && fabs( d21 - 1.0 ) < 1e-12;
    pass       = pass && fabs( d22 - 1.0 ) < 1e-12;
    pass       = pass && fabs( d23 ) > 1e200;
    auto start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < N; i++ ) {
        auto [tri, pos, dir, d] = createTriRayPlane();
        double d2               = distanceToTriangle( tri, pos, dir );
        if ( fabs( d - d2 ) > 1e-8 ) {
            pass = false;
        }
    }
    for ( int i = 0; i < N; i++ ) {
        auto [tri, pos, dir, d] = createTriRayVol();
        double d2               = distanceToTriangle( tri, pos, dir );
        if ( fabs( d - d2 ) > 1e-8 ) {
            pass = false;
        }
    }
    auto end   = std::chrono::high_resolution_clock::now();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count();
    printf( "ray-triangle intersection: %i ns\n", static_cast<int>( ns / ( 2 * N ) ) );
    ut.pass_fail( pass, "ray-triangle intersection" );
}


// Test point cloud
template<uint8_t NDIM>
void testPointCloud( int N, AMP::UnitTest &ut )
{
    using TYPE  = std::array<double, NDIM>;
    int N_ranks = 120;
    std::vector<TYPE> x( N );
    AMP::Utilities::fillRandom( x );
    auto ranks = assignRanks( x, N_ranks );
    AMP_ASSERT( (int) ranks.size() == N );
    std::array<double, 2 * NDIM> range0;
    for ( uint8_t d = 0; d < NDIM; d++ ) {
        range0[2 * d + 0] = 1e100;
        range0[2 * d + 1] = -1e100;
    }
    std::vector<int> count( N_ranks, 0 );
    std::vector<std::array<double, 2 * NDIM>> range( N_ranks, range0 );
    for ( int i = 0; i < N; i++ ) {
        AMP_ASSERT( ranks[i] >= 0 && ranks[i] < N_ranks );
        int r = ranks[i];
        count[r]++;
        for ( uint8_t d = 0; d < NDIM; d++ ) {
            range[r][2 * d + 0] = std::min( x[i][d], range[r][2 * d + 0] );
            range[r][2 * d + 1] = std::max( x[i][d], range[r][2 * d + 1] );
        }
    }
    int countAvg = N / N_ranks;
    double total = 0;
    std::vector<double> volume( N_ranks, 0 );
    bool pass = true;
    for ( int i = 0; i < N_ranks; i++ ) {
        pass      = pass && std::abs( count[i] - countAvg ) < 10;
        volume[i] = 1;
        for ( uint8_t d = 0; d < NDIM; d++ )
            volume[i] *= range[i][2 * d + 1] - range[i][2 * d + 0];
        total += volume[i];
    }
    pass = pass && total >= 0 && total < 1.0;
    if constexpr ( NDIM == 1 ) {
        for ( int i = 1; i < N_ranks; i++ )
            pass = pass && range[i][0] >= range[i - 1][1];
    }
    ut.pass_fail( pass, "assignRanks<" + std::to_string( NDIM ) + ">" );
}


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    // Run the tests
    test_dist_line( 10000, ut );
    test_map_logical_poly( 10000, ut );
    test_map_logical_circle( 10000, ut );
    test_map_logical_sphere_surface( 10000, ut );
    test_ray_triangle_intersection( 1000, ut );
    testPointCloud<1>( 10000, ut );
    testPointCloud<2>( 10000, ut );
    testPointCloud<3>( 10000, ut );

    // Print the results and return
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
