#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>


/****************************************************************
 * Map the logical coordinates to a circle                       *
 ****************************************************************/
using Point = const std::array<double, 2>;
static inline Point map_c2p( double xc, double yc )
{
    if ( fabs( xc ) < 1e-12 && fabs( yc ) < 1e-12 )
        return { 0.0, 0.0 };
    if ( fabs( yc ) > fabs( xc ) ) {
        auto [yp, xp] = map_c2p( yc, xc );
        return { xp, yp };
    }
    double scale = std::max( { 1.0, xc, yc } );
    if ( scale > 1.0 ) {
        xc /= scale;
        yc /= scale;
    }
    double xp             = 0;
    double yp             = 0;
    const double invsqrt2 = 0.7071067811865475244;
    double D              = invsqrt2 * xc * ( 2 - xc );
    double center         = D - sqrt( 1.0 - D * D );
    yp                    = invsqrt2 * ( 2 - xc ) * yc;
    xp                    = center + sqrt( 1.0 - yp * yp );
    return { scale * xp, scale * yp };
}
static inline Point map_p2c( double xp, double yp )
{
    // Perform the inverse mapping as map_c2p
    if ( fabs( xp ) < 1e-12 && fabs( yp ) < 1e-12 )
        return { 0.0, 0.0 };
    if ( fabs( yp ) > fabs( xp ) ) {
        auto [yc, xc] = map_p2c( yp, xp );
        return { xc, yc };
    }
    double scale = std::max( sqrt( xp * xp + yp * yp ), 1.0 );
    if ( scale > 1.0 ) {
        xp /= scale;
        yp /= scale;
    }
    double xc          = 0;
    double yc          = 0;
    const double sqrt2 = 1.4142135623730950488;
    auto z             = xp - sqrt( 1 - yp * yp );
    auto D             = 0.5 * ( z + sqrt( 2 - z * z ) );
    xc                 = 1.0 - sqrt( std::max( 1 - D * sqrt2, 0.0 ) );
    yc                 = yp * sqrt2 / ( 2 - xc );
    return { scale * xc, scale * yc };
}
Point map_logical_circle( double r, double x, double y )
{
    const double xc = 2 * x - 1; // Change domain to [-1,1]
    const double yc = 2 * y - 1; // Change domain to [-1,1]
    auto [xp, yp]   = map_c2p( fabs( xc ), fabs( yc ) );
    if ( xc < 0.0 )
        xp = -xp;
    if ( yc < 0.0 )
        yp = -yp;
    return { r * xp, r * yp };
}
Point map_circle_logical( double r, double x, double y )
{
    // Get the points in the unit circle
    double xp = x / r;
    double yp = y / r;
    // Perform the inverse mapping to [-1,1]
    auto [xc, yc] = map_p2c( fabs( xp ), fabs( yp ) );
    if ( xp < 0.0 )
        xc = -xc;
    if ( yp < 0.0 )
        yc = -yc;
    // Change domain to [0,1]
    return { 0.5 * ( xc + 1 ), 0.5 * ( yc + 1 ) };
}
double distance( const Point &x, Point &y )
{
    return sqrt( ( x[0] - y[0] ) * ( x[0] - y[0] ) + ( x[1] - y[1] ) * ( x[1] - y[1] ) );
};


// Test the mapping to/from a logical circle
using PointDist = std::array<double, 3>;
std::tuple<std::vector<PointDist>, std::vector<PointDist>> test_map_logical_circle( int N )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    const double r = 2.0;
    std::uniform_real_distribution<> dis( 0, 1 );
    // Test logical->physical->logical
    std::vector<PointDist> points1( N );
    for ( int i = 0; i < N; i++ ) {
        double x    = dis( gen );
        double y    = dis( gen );
        auto p      = map_logical_circle( r, x, y );
        auto p2     = map_circle_logical( r, p[0], p[1] );
        double dist = distance( { x, y }, p2 );
        points1.push_back( { x, y, dist } );
    }
    std::sort( points1.begin(), points1.end(), []( auto a, auto b ) { return a[2] > b[2]; } );
    // Test physical->logical->physical
    dis = std::uniform_real_distribution<>( -3.0, 3.0 );
    std::vector<PointDist> points2( N );
    for ( int i = 0; i < N; i++ ) {
        double x    = dis( gen );
        double y    = dis( gen );
        auto p      = map_circle_logical( 1.0, x, y );
        auto p2     = map_logical_circle( 1.0, p[0], p[1] );
        double dist = distance( { x, y }, p2 );
        points2.push_back( { x, y, dist } );
    }
    std::sort( points2.begin(), points2.end(), []( auto a, auto b ) { return a[2] > b[2]; } );
    return std::tie( points1, points2 );
}


// Write/Read a single point
void writePoint( std::FILE *fid, const PointDist x, const Point &logical, const Point &physical )
{
    double data[7] = { x[0], x[1], x[2], logical[0], logical[1], physical[0], physical[1] };
    std::ignore    = std::fwrite( data, sizeof( double ), 7, fid );
}
std::tuple<Point, double, Point, Point> readPoint( std::FILE *fid )
{
    double data[7] = { 0 };
    std::ignore    = std::fread( data, sizeof( double ), 7, fid );
    Point x        = { data[0], data[1] };
    double d       = data[2];
    Point logical  = { data[3], data[4] };
    Point physical = { data[5], data[6] };
    return std::tie( x, d, logical, physical );
}


// Run and compare the failed points
void testFailedPoints( const char *filename, double tol )
{
    // Open the file
    auto fid = std::fopen( filename, "rb" );
    if ( !fid ) {
        std::cerr << "Unable to open file for writing\n";
        return;
    }
    auto print = []( const char *name, const Point &p1, const Point &p2 ) {
        printf(
            "   %s: (%f,%f) (%f,%f) %e\n", name, p1[0], p1[1], p2[0], p2[1], distance( p1, p2 ) );
    };
    // Load the points and compare the steps
    int Np[2]      = { 0 };
    std::ignore    = std::fread( Np, sizeof( int ), 2, fid );
    const double r = 2.0;
    printf( "Checking logical->physical->logical\n" );
    for ( int i = 0; i < Np[0]; i++ ) {
        auto [p, d1, l1, p1] = readPoint( fid );
        auto p2              = map_logical_circle( r, p[0], p[1] );
        auto l2              = map_circle_logical( r, p2[0], p2[1] );
        auto l3              = map_circle_logical( r, p1[0], p1[1] );
        bool t1              = distance( p1, p2 ) < tol;
        bool t2              = distance( l1, l3 ) < tol;
        bool t3              = distance( p, l2 ) < tol;
        if ( d1 < tol && t1 && t2 && t3 ) {
            // Original point passed
        } else if ( t1 && t2 && t3 ) {
            printf( "(%f %f) - matches, %e\n", p[0], p[1], d1 );
        } else {
            printf( "(%f,%f):\n", p[0], p[1] );
            if ( !t1 )
                print( "map_logical_circle", p1, p2 );
            if ( !t2 )
                print( "map_circle_logical", l1, l3 );
            if ( !t3 )
                print( "final", p, l2 );
        }
    }
    printf( "\nChecking physical->logical->physical\n" );
    for ( int i = 0; i < Np[1]; i++ ) {
        auto [p, d1, l1, p1] = readPoint( fid );
        auto l2              = map_circle_logical( 1.0, p[0], p[1] );
        auto p2              = map_logical_circle( 1.0, l2[0], l2[1] );
        auto p3              = map_logical_circle( 1.0, l1[0], l1[1] );
        bool t1              = distance( l1, l2 ) < tol;
        bool t2              = distance( p1, p3 ) < tol;
        bool t3              = distance( p, p2 ) < tol;
        if ( d1 < tol && t1 && t2 && t3 ) {
            // Original point passed
        } else if ( t1 && t2 && t3 ) {
            printf( "(%f %f) - matches, %e\n", p[0], p[1], d1 );
        } else {
            printf( "(%f %f):\n", p[0], p[1] );
            if ( !t1 )
                print( "map_circle_logical", l1, l2 );
            if ( !t2 )
                print( "map_logical_circle", p1, p3 );
            if ( !t3 )
                print( "final", p, p2 );
        }
    }
    std::fclose( fid );
    printf( "\n" );
}


// Write the failed geometry points
int writeFailedPoints( std::vector<PointDist> &p1,
                       std::vector<PointDist> &p2,
                       double tol,
                       const char *filename )
{
    // Verify the points are sorted
    std::is_sorted( p1.begin(), p1.end(), []( auto a, auto b ) { return a[2] > b[2]; } );
    std::is_sorted( p2.begin(), p2.end(), []( auto a, auto b ) { return a[2] > b[2]; } );
    // Check how many failures we have
    int N_failed = 0;
    for ( size_t i = 0; i < p1.size(); i++ ) {
        if ( p1[i][2] > tol )
            N_failed++;
    }
    for ( size_t i = 0; i < p2.size(); i++ ) {
        if ( p2[i][2] > tol )
            N_failed++;
    }
    if ( N_failed == 0 )
        return N_failed;
    // Open the file
    auto fid = std::fopen( filename, "wb" );
    if ( !fid ) {
        std::cerr << "Unable to open file for writing\n";
        return N_failed;
    }
    // Keep the points with the largest error
    int N = 1000;
    p1.resize( std::min<int>( p1.size(), N ) );
    p2.resize( std::min<int>( p2.size(), N ) );
    // Write the points (and intermediate steps)
    int Np[2] = { (int) p1.size(), (int) p2.size() };
    std::fwrite( Np, sizeof( int ), 2, fid );
    const double r = 2.0;
    for ( size_t i = 0; i < p1.size(); i++ ) {
        auto physical = map_logical_circle( r, p1[i][0], p1[i][1] );
        auto logical  = map_circle_logical( r, physical[0], physical[1] );
        writePoint( fid, p1[i], logical, physical );
    }
    for ( size_t i = 0; i < p2.size(); i++ ) {
        auto logical  = map_circle_logical( 1.0, p2[i][0], p2[i][1] );
        auto physical = map_logical_circle( 1.0, logical[0], logical[1] );
        writePoint( fid, p2[i], logical, physical );
    }
    std::fclose( fid );
    // Open the file and retest the points for consistency
    testFailedPoints( filename, tol );
    return N_failed;
}


// Main function
int main( int argc, char **argv )
{
    // Check inputs
    if ( argc > 2 ) {
        std::cerr << "Error calling " << argv[0] << std::endl;
        std::cerr << "   " << argv[0] << " <filename>" << std::endl;
        return -1;
    }

    // Standalone test
    const double tol = 1e-10;
    if ( argc == 1 ) {
        // Run the test
        auto [p1, p2] = test_map_logical_circle( 10000 );
        // Write failed points to a file
        int N_failed = writeFailedPoints( p1, p2, tol, "failedGeometryHelpersPoints.data" );
        printf( "N_failed = %i\n", N_failed );
        printf( "max error = %e\n\n", std::max( p1[0][2], p2[0][2] ) );
        // Finished
        if ( N_failed == 0 ) {
            printf( "Tests passed\n" );
            return 0;
        } else {
            printf( "Tests failed\n" );
            return 1;
        }
    }

    // Load existing failed points and compare the answers step by step
    if ( argc == 2 ) {
        testFailedPoints( argv[1], tol );
        return 0;
    }

    // Invalid call
    std::cerr << "Error calling " << argv[0] << std::endl;
    std::cerr << "   " << argv[0] << " <filename>" << std::endl;
    return -1;
}
