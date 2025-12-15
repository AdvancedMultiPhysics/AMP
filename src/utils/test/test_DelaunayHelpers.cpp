#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/extended_int.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>


using int128_t  = AMP::extended::int128_t;
using int256_t  = AMP::extended::int256_t;
using int512_t  = AMP::extended::int512_t;
using int1024_t = AMP::extended::int1024_t;
using int2048_t = AMP::extended::int2048_t;


template<class TYPE>
static TYPE createShift( int i0, int shift )
{
    TYPE tmp( i0 );
    if ( shift >= 0 )
        tmp <<= shift;
    else
        tmp >>= -shift;
    return tmp;
}


// Helper function to check if two numbers are approximately equal
inline bool approx_equal( double x, double y, double tol = 1e-8 )
{
    return 2.0 * fabs( x - y ) <= tol * fabs( x + y );
}


// Return the class name
template<class TYPE>
std::string className();
template<>
std::string className<float>()
{
    return "float";
}
template<>
std::string className<double>()
{
    return "double";
}
template<>
std::string className<long double>()
{
    return "long double";
}
template<>
std::string className<int128_t>()
{
    return "int128_t";
}
template<>
std::string className<int256_t>()
{
    return "int256_t";
}
template<>
std::string className<int512_t>()
{
    return "int512_t";
}


// Run the test for a matrix
template<class TYPE, std::size_t NDIM>
void runMatTest( const int64_t *M0,
                 const int64_t *b0,
                 const int128_t &d0,
                 const int128_t *x0,
                 AMP::UnitTest &ut )
{
    std::string name = "<" + className<TYPE>() + "," + std::to_string( NDIM ) + ">";
    auto tol         = static_cast<double>( std::numeric_limits<TYPE>::epsilon() );
    if ( std::is_same_v<TYPE, long double> && AMP::Utilities::running_valgrind() )
        tol = std::numeric_limits<double>::epsilon();
    TYPE M[NDIM * NDIM], b[NDIM];
    for ( size_t i = 0; i < NDIM * NDIM; i++ )
        M[i] = TYPE( M0[i] );
    for ( size_t i = 0; i < NDIM; i++ )
        b[i] = TYPE( b0[i] );
    // Check the determinant
    TYPE d   = static_cast<TYPE>( d0 );
    TYPE det = AMP::DelaunayHelpers::det<TYPE, NDIM>( M );
    auto err = fabs( static_cast<double>( det - d ) / static_cast<double>( d0 ) );
    if ( err <= 10 * tol )
        ut.passes( "determinant" + name );
    else
        ut.failure( AMP::Utilities::stringf( "determinant%s: %e", name.data(), err ) );

    // Check the solution
    TYPE det_M, x[NDIM];
    AMP::DelaunayHelpers::solve<TYPE, NDIM>( M, b, x, det_M );
    err = fabs( static_cast<double>( det_M - d ) / static_cast<double>( d0 ) );
    for ( size_t i = 0; i < NDIM; i++ ) {
        auto x2 = static_cast<TYPE>( x0[i] );
        auto e2 = fabs( static_cast<double>( x[i] - x2 ) / static_cast<double>( x0[i] ) );
        err     = std::max( err, e2 );
    }
    if ( fabs( err ) <= 10 * tol )
        ut.passes( "solve" + name );
    else
        ut.failure( AMP::Utilities::stringf( "solve%s: %e", name.data(), err ) );
}


// Helper function to check if two numbers are approximately equal
inline void testMatOps( AMP::UnitTest &ut )
{
    // clang-format off
    // Create some matricies
    int64_t M1[] = { 23048816020 };
    int64_t M2[] = { 5870447045, 2077422927, 3012463303, 4709233485 };
    int64_t M3[] = { 508508655, 510771564, 817627708, 794831417, 644318130,
                     378609383, 811580458, 532825589, 350727104 };
    int64_t M4[] = { 547008895, 296320811, 744692816, 188955029,
                     686775437, 183511160, 368484607, 625618565,
                     780227443,  81125774, 929385979, 775712686,
                     486791639, 435858593, 446783751, 306349473 };
    // rhs
    int64_t b1[] = { 8443087927 };
    int64_t b2[] = { 1947642896, 225921781 };
    int64_t b3[] = { 435698684, 311102287, 923379642 };
    int64_t b4[] = { 430207391, 184816320, 904880969, 979748378 };
    
    // solution * det
    int128_t x1[] = { int128_t( "8443087927" ) };
    int128_t x2[] = { int128_t( "8491324068054669917" ), int128_t( "-2719816154086489447" ) };
    int128_t x3[] = { int128_t( "-72360915059620974405447760" ),
                      int128_t( "93391984307512611733644090" ),
                      int128_t( "-75325940808692452445123071" ) };
    int128_t x4[] = { int128_t( "75421797283001082069974646004256134" ),
                      int128_t( "64981030633990940166503573744994630" ),
                      int128_t( "-78922842163937317826618251692079107" ),
                      int128_t( "-76871213525117559611967916800395752" ) };
    
    // determinant
    int128_t d1( "23048816020" );
    int128_t d2( "21387145463834953944" );
    int128_t d3( "-54391557236731530985383257" );
    int128_t d4( "-30483580554324799216520791201229879" );

    // clang-format on

    // Run the tests
    runMatTest<int512_t, 1>( M1, b1, d1, x1, ut );
    runMatTest<int512_t, 2>( M2, b2, d2, x2, ut );
    runMatTest<int512_t, 3>( M3, b3, d3, x3, ut );
    runMatTest<int512_t, 4>( M4, b4, d4, x4, ut );

    runMatTest<float, 1>( M1, b1, d1, x1, ut );
    runMatTest<float, 2>( M2, b2, d2, x2, ut );
    runMatTest<float, 3>( M3, b3, d3, x3, ut );
    runMatTest<float, 4>( M4, b4, d4, x4, ut );

    runMatTest<double, 1>( M1, b1, d1, x1, ut );
    runMatTest<double, 2>( M2, b2, d2, x2, ut );
    runMatTest<double, 3>( M3, b3, d3, x3, ut );
    runMatTest<double, 4>( M4, b4, d4, x4, ut );

    runMatTest<long double, 1>( M1, b1, d1, x1, ut );
    runMatTest<long double, 2>( M2, b2, d2, x2, ut );
    runMatTest<long double, 3>( M3, b3, d3, x3, ut );
    runMatTest<long double, 4>( M4, b4, d4, x4, ut );
}


// Main
int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    PROFILE_ENABLE( 3 ); // 0: code, 1: tests, 3: basic timers, 5: all timers

    // Check that we can create "random" points
    testMatOps( ut );

    PROFILE_SAVE( "test_DelaunayInterpolation" );
    ut.report();
    auto N_errors = static_cast<int>( ut.NumFailGlobal() );
    AMP::AMPManager::shutdown();
    return N_errors;
}
