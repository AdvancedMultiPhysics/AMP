#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/arrayHelpers.h"
#include "AMP/utils/kdtree.h"
#include "AMP/utils/kdtree2.h"

#include "ProfilerApp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <set>
#include <vector>


using namespace AMP;


// Add the result to the tests
void checkResult( AMP::UnitTest &ut, bool pass, const std::string &msg )
{
    if ( pass )
        ut.passes( msg );
    else
        ut.failure( msg );
}


// Compute the average distance between N points in a NDIM hypercube
// This is done by dividing the volume of the hypercube by the volume of the hyperspheres
double compute_avg_dist( int DIM, int N )
{
    double C        = 1.0;
    const double pi = 3.14159265359;
    if ( DIM == 1 ) {
        C = 1.0;
    } else if ( DIM == 2 ) {
        C = 4.0 / pi;
    } else if ( DIM == 3 ) {
        C = 6.0 / pi;
    } else if ( DIM == 4 ) {
        C = 32 / ( pi * pi );
    } else if ( DIM == 5 ) {
        C = 60.0 / pi * pi;
    }
    return std::pow( C / ( (double) N ), 1.0 / ( (double) DIM ) );
}


// Run the kdtree tests
void run_kdtree_test( AMP::UnitTest &ut, int DIM, size_t Nx, size_t Ns )
{
    auto prefix = AMP::Utilities::stringf( "kdtree<%i,%i,%i>", DIM, Nx, Ns );
    PROFILE2( prefix );
    prefix += "::";

    // Initialize the random number
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dis( 0, 1 );

    // Create the coordinates
    std::vector<std::vector<double>> points( DIM, std::vector<double>( Nx, 0.0 ) );
    for ( int d = 0; d < DIM; d++ ) {
        for ( size_t i = 0; i < Nx; i++ )
            points[d][i] = dis( gen );
    }

    // Create the search points
    std::vector<std::vector<double>> search( DIM, std::vector<double>( Ns, 0.0 ) );
    for ( int d = 0; d < DIM; d++ ) {
        for ( size_t i = 0; i < Ns; i++ )
            search[d][i] = dis( gen );
    }

    // Create the kdtree
    const auto x = new const double *[DIM];
    for ( int d = 0; d < DIM; d++ )
        x[d] = &points[d][0];
    AMP::kdtree tree( DIM, Nx, x );
    delete[] x;

    // Check the bounding box
    auto box  = tree.box();
    bool pass = (int) box.size() == 2 * DIM;
    for ( int d = 0; d < DIM; d++ ) {
        pass = pass && box[2 * d + 0] >= 0.0;
        pass = pass && box[2 * d + 1] <= 1.0;
    }
    checkResult( ut, pass, prefix + "box" );

    // Search for the local points
    {
        PROFILE( "search_local" );
        pass = true;
        double dist, xs[100], pos[100];
        for ( size_t i = 0; i < Nx; i++ ) {
            for ( int d = 0; d < DIM; d++ )
                xs[d] = points[d][i];
            size_t j = tree.find_nearest( xs, &dist, pos );
            if ( j != i || dist != 0.0 )
                pass = false;
            for ( int d = 0; d < DIM; d++ )
                if ( pos[d] != xs[d] )
                    pass = false;
        }
    }
    checkResult( ut, pass, prefix + "find_nearest (local)" );

    // Search for the unknown points
    {
        PROFILE( "search_unknown" );
        pass            = true;
        double dist_max = std::sqrt( (double) DIM ); // Maximum possible distance between two points
        double dist_avg = compute_avg_dist( DIM, (int) Nx ); // Average distance between two points
        double dist, xs[100], pos[100];
        for ( size_t i = 0; i < Ns; i++ ) {
            for ( int d = 0; d < DIM; d++ )
                xs[d] = search[d][i];
            size_t j = tree.find_nearest( xs, &dist, pos );
            if ( j >= Nx || dist <= 0.0 || dist > dist_max || dist > 100.0 * dist_avg )
                pass = false;
        }
    }
    checkResult( ut, pass, prefix + "find_nearest (unknown)" );
}


// Run the kdtree2 specific tests (note: kdtree uses kdtree2 under the hood)
template<int DIM>
void run_kdtree2_test( [[maybe_unused]] AMP::UnitTest &ut, size_t N )
{
    // Initialize the random number
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dis( 0, 1 );

    // Create the coordinates
    std::vector<int> index( N, -1 );
    std::vector<std::array<double, DIM>> x( N, { 0.0 } );
    for ( size_t i = 0; i < N; i++ ) {
        index[i] = i;
        for ( int d = 0; d < DIM; d++ )
            x[i][d] = dis( gen );
    }

    // Create the tree
    auto tree = AMP::kdtree2<DIM, int>( x, index );

    // Check for ray intersections
    std::array<double, 3> p0 = { 10, 20, 20 };
    std::array<double, 3> v  = { -1, -2, -2 };
    auto result              = tree.findNearestRay( p0, v, 0.01 );
    std::cout << "Ray-point intersection:\n";
    auto intersect = []( const std::array<double, 3> &p0,
                         const std::array<double, 3> &v,
                         const std::array<double, 3> &x ) {
        auto t = dot( v, x - p0 );
        t      = std::max( t, 0.0 );
        auto p = p0 + v * t;
        return p;
    };
    for ( auto tmp : result ) {
        auto p  = std::get<0>( tmp );
        auto pi = intersect( p0, v, p );
        auto d  = std::sqrt( norm( pi - p ) );
        printf( "   (%0.2f,%0.2f,%0.2f)  (%0.2f,%0.2f,%0.2f)  %0.5f\n",
                p[0],
                p[1],
                p[2],
                pi[0],
                pi[1],
                pi[2],
                d );
    }
}


// Main
int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    PROFILE_ENABLE( 3 );
    PROFILE_ENABLE_TRACE();

    // Run a 1D test
    run_kdtree_test( ut, 1, 10, 10 );
    run_kdtree_test( ut, 1, 10000, 1000 );

    // Run a 2D test
    run_kdtree_test( ut, 2, 10, 10 );
    run_kdtree_test( ut, 2, 100000, 10000 );

    // Run a 3D test
    run_kdtree_test( ut, 3, 10, 10 );
    run_kdtree_test( ut, 3, 100000, 100000 );

    // Run a 5D test
    run_kdtree_test( ut, 5, 10, 10 );
    run_kdtree_test( ut, 5, 10000, 2000 );

    // Run a 10D test
    // run_kdtree_test( ut, 10, 10, 10 );
    // run_kdtree_test( ut, 10, 10000, 2000 );

    // Run specific kdtree2 tests
    run_kdtree2_test<3>( ut, 1000 );

    // Save the results
    PROFILE_SAVE( "test_kdtree" );
    ut.report();
    int N_errors = ut.NumFailGlobal();
    ut.reset();
    AMP::AMPManager::shutdown();
    return N_errors;
}
