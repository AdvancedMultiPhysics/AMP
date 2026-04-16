// This file tests the polynomial class
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/polynomial.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>


#if defined( AMP_USE_LAPACK ) && defined( AMP_USE_LAPACK_WRAPPERS )
    #define USE_LAPACK
    #include "LapackWrappers.h"
#endif


#define to_ms( x ) std::chrono::duration_cast<std::chrono::milliseconds>( x ).count()
#define to_us( x ) std::chrono::duration_cast<std::chrono::microseconds>( x ).count()
#define to_ns( x ) std::chrono::duration_cast<std::chrono::nanoseconds>( x ).count()


using AMP::Polynomial;


static inline bool approx_equal( double a, double b, double tol = 1e-8 )
{
    return 2.0 * fabs( a - b ) <= tol * fabs( a + b );
}
static inline bool
approx_equal( const std::complex<double> &a, const std::complex<double> &b, double tol = 1e-8 )
{
    if ( a == b )
        return true;
    return std::abs( 2.0 * ( a - b ) / ( a + b ) ) < tol;
}


// Test creating polynomials
void testCreate( AMP::UnitTest &ut )
{
    double p1_coeff[5] = { 1, 2, 3, 0, 0 }; // Extra zeros will be removed by constructor
    double p2_coeff[5] = { 5, 4, 3, 2, 1 };
    double p3_coeff[7] = { 5, 14, 26, 20, 14, 8, 3 }; // p3=p1*p2

    // Create some polynomials
    Polynomial one( 1.0 );
    Polynomial two( 2.0 );
    Polynomial p1( 4, p1_coeff );
    Polynomial p2( 4, p2_coeff );
    Polynomial p3( 6, p3_coeff );
    p2.print();
    AMP::pout << std::endl;

    // Check the created polynomials
    bool pass = true;
    pass      = pass && one.order() == 0;
    pass      = pass && two.order() == 0;
    pass      = pass && p1.order() == 2;
    pass      = pass && p2.order() == 4;
    pass      = pass && p3.order() == 6;
    pass      = pass && one.getCoeff( 0 ) == 1.0;
    pass      = pass && two.getCoeff( 0 ) == 2.0;
    for ( int i = 0; i <= p1.order(); i++ )
        pass = pass && p1.getCoeff( i ) == p1_coeff[i];
    for ( int i = 0; i <= p2.order(); i++ )
        pass = pass && p2.getCoeff( i ) == p2_coeff[i];
    for ( int i = 0; i <= p3.order(); i++ )
        pass = pass && p3.getCoeff( i ) == p3_coeff[i];
    if ( pass )
        ut.passes( "Basic tests" );
    else
        ut.failure( "Basic tests" );
}


// Test polynomial functions
void testPolynomial( AMP::UnitTest &ut )
{
    // Create the polynomials
    double p1_coeff[5]  = { 1, 2, 3, 0, 0 }; // Extra zeros will be removed by constructor
    double p2_coeff[5]  = { 5, 4, 3, 2, 1 };
    double p3_coeff[7]  = { 5, 14, 26, 20, 14, 8, 3 }; // p3=p1*p2
    double dp2_coeff[4] = { 4, 6, 6, 4 };              // Derivative of p2
    Polynomial one( 1.0 );
    Polynomial two( 2.0 );
    Polynomial p1( 4, p1_coeff );
    Polynomial p2( 4, p2_coeff );
    Polynomial p3( 6, p3_coeff );

    // Check setCoeff
    Polynomial p0 = p1;
    p0.setCoeff( 1, 0.0 );
    p0.setCoeff( 2, 0.0 );
    if ( p0 != one )
        ut.failure( "setCoeff" );

    // Check the comparison operators
    if ( !( one == one ) || one != one || !( one != two ) || one == two )
        ut.failure( "comparison-1" );
    if ( !( p1 == p1 ) || p1 != p1 || !( p1 != p2 ) || p1 == p2 || Polynomial( p1 ) != p1 )
        ut.failure( "comparison-2" );

    // Check arithmetic operators
    if ( one + two != 3.0 || 2.0 * one != two || two - one != one )
        ut.failure( "arithmetic-1" );
    if ( p1 + p1 != 2 * p1 || p1 - p1 != 0 || p1 * p2 != p3 )
        ut.failure( "arithmetic-2" );
    Polynomial Q, R;
    std::tie( Q, R ) = p3 / p2;
    if ( Q != p1 || R != 0 )
        ut.failure( "arithmetic-3" );
    std::tie( Q, R ) = p2 / p1;
    if ( !approx_equal( Q[0], 16.0 / 27.0 ) || !approx_equal( Q[1], 4.0 / 9.0 ) ||
         !approx_equal( Q[2], 1.0 / 3.0 ) || !approx_equal( R[0], 119.0 / 27.0 ) ||
         !approx_equal( R[1], 64.0 / 27.0 ) )
        ut.failure( "arithmetic-4" );

    // Check the derivative
    if ( p2.derivative() != Polynomial( 3, dp2_coeff ) || two.derivative() != 0 )
        ut.failure( "derivative" );

    // Check eval and evalDerivative
    if ( p1.eval( 1.5 ) != 10.75 || p2.eval( 2.5 ) != 104.0625 )
        ut.failure( "eval" );
    if ( p1.evalDerivative( 1.5 ) != 11 || p2.evalDerivative( 2.5 ) != 119 )
        ut.failure( "evalDerivative" );
}


// Test finding roots
void testBasicRoots( AMP::UnitTest &ut )
{
    double p1_coeff[5] = { 0, 0, 0, 1, -2 };  // Polynomial with roots 0,0,0,0.5
    double p2_coeff[4] = { -21, 31, -11, 1 }; // Polynomial with roots 1,3,7
    double p3_coeff[5] = { 5, 4, 3, 2, 1 };   // Polynomial with complex roots
    Polynomial p1( 4, p1_coeff );
    Polynomial p2( 3, p2_coeff );
    Polynomial p3( 4, p3_coeff );
    auto roots1 = p1.roots();
    auto roots2 = p2.roots();
    auto roots3 = p3.roots();
    if ( roots1[0] != 0.0 || roots1[1] != 0.0 || roots1[2] != 0.0 ||
         !approx_equal( roots1[3].real(), 0.5 ) )
        ut.failure( "roots-1" );
    if ( fabs( p2.rootInterval( 0, 2 ) - 1 ) > 1e-8 || fabs( p2.rootInterval( 2, 6 ) - 3 ) > 1e-8 ||
         fabs( p2.rootInterval( 6, 100 ) - 7 ) > 1e-8 )
        ut.failure( "rootInterval" );
    if ( !approx_equal( roots2[0].real(), 1.0 ) || !approx_equal( roots2[1].real(), 3.0 ) ||
         !approx_equal( roots2[2].real(), 7.0 ) )
        ut.failure( "roots-2" );
    auto roots3_0 = std::complex<double>( -1.287815479557648, -0.857896758328490 );
    auto roots3_1 = std::complex<double>( -1.287815479557648, +0.857896758328490 );
    auto roots3_2 = std::complex<double>( 0.287815479557648, -1.416093080171909 );
    auto roots3_3 = std::complex<double>( 0.287815479557648, +1.416093080171909 );
    if ( !approx_equal( roots3[0], roots3_0 ) || !approx_equal( roots3[1], roots3_1 ) ||
         !approx_equal( roots3[2], roots3_2 ) || !approx_equal( roots3[3], roots3_3 ) )
        ut.failure( "roots-3" );
}


// Test finding roots of simple Wilkinson's like polynomials
void testWilkinsonRoots( AMP::UnitTest &ut )
{
    double roots0[20];
    for ( int i = 0; i < 20; i++ )
        roots0[i] = i + 1;
    for ( int n = 1; n <= 13; n++ ) {
        auto pr    = Polynomial::createFromRoots( n, roots0 );
        auto roots = pr.roots();
        bool pass  = true;
        for ( int i = 0; i < n; i++ ) {
            if ( !approx_equal( roots[i], roots0[i], 1e-10 ) )
                pass = false;
        }
        if ( pass ) {
            ut.passes( "Wilkinson polynomial " + std::to_string( n ) );
        } else if ( n > 10 ) {
            ut.expected_failure( "Wilkinson polynomial " + std::to_string( n ) );
        } else {
            ut.failure( "Wilkinson polynomial " + std::to_string( n ) );
        }
    }
}


// Test finding duplicate
void testDuplicateRoots( AMP::UnitTest &ut )
{
    std::vector<double> roots = { 2, 2, 3, 3, 3 };
    std::sort( roots.begin(), roots.end() );
    // Create the polynomial and get the roots
    auto p = Polynomial::createFromRoots( roots );
    auto r = p.roots();
    // Get the error
    double err = 0;
    for ( size_t i = 0; i < roots.size(); i++ )
        err = std::max( err, std::abs( r[i] - roots[i] ) / roots[i] );
    // Print the results
    if ( err < 1e-5 ) {
        ut.passes( "duplicate roots" );
    } else {
        ut.expected_failure( "duplicate roots" );
    }
}


// Test finding real roots
void testRandomRoots( int N, AMP::UnitTest &ut )
{
    constexpr int N_it = 10;
    // Generate real roots
    static std::random_device dev;
    static std::default_random_engine gen( dev() );
    static std::uniform_real_distribution<double> dist( -1, 1 );
    std::vector<double> roots( N );
    for ( int i = 0; i < N; i++ )
        roots[i] = dist( gen );
    std::sort( roots.begin(), roots.end() );
    // Create the polynomial
    [[maybe_unused]] Polynomial p;
    auto t1 = std::chrono::high_resolution_clock::now();
    for ( int it = 0; it < N_it; it++ )
        p = Polynomial::createFromRoots( roots );
    auto t2 = std::chrono::high_resolution_clock::now();
    int tp  = to_ns( t2 - t1 ) / N_it;
    // Find the roots
    [[maybe_unused]] std::vector<std::complex<double>> r;
    for ( int it = 0; it < N_it; it++ )
        r = p.roots();
    auto t3 = std::chrono::high_resolution_clock::now();
    int tr  = to_us( t3 - t2 ) / N_it;
    // Get the error
    double err = 0;
    bool nans  = false;
    for ( int i = 0; i < N; i++ ) {
        err  = std::max( err, std::abs( r[i] - roots[i] ) );
        nans = nans || r[i] != r[i];
    }
    if ( nans )
        err = std::numeric_limits<double>::quiet_NaN();
    // Print the results
    printf( "Time to create polynomial from %i roots: %i ns\n", N, tp );
    printf( "   Time to find roots: %i us\n", tr );
    printf( "   Maximum error: %e\n", err );
    if ( err < 1e-7 ) {
        ut.passes( "roots-" + std::to_string( N ) );
    } else if ( N >= 20 ) {
        ut.expected_failure( "roots-" + std::to_string( N ) );
    } else {
        ut.failure( "roots-" + std::to_string( N ) );
    }
}


// Test fitting a polynomial
void testFitting( AMP::UnitTest &ut )
{
#ifdef USE_LAPACK
    auto fun             = []( double x ) { return sqrt( 1 + x * x ); };
    auto p_fit           = Polynomial::fit( 4, fun, 0, 1, 10000 );
    const double p_ans[] = { 1.000191501137478,
                             -0.005909874136544,
                             0.542553079181662,
                             -0.114799099354642,
                             -0.007971204746391 };
    bool pass            = true;
    for ( int i = 0; i <= p_fit.order(); i++ )
        pass = pass && fabs( p_fit[i] - p_ans[i] ) < 1e-5;
    AMP::pout << "Error fitting sqrt(1+x^2): " << p_fit.error( fun, 0, 1 ) << std::endl;
    if ( pass )
        ut.passes( "Fit sqrt(1+x^2)" );
    else
        ut.failure( "Fit sqrt(1+x^2)" );
#else
    ut.expected_failure( "Fitting polynomials is disabled without Lapack" );
#endif
}


// Main
int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Run basic tests
    testCreate( ut );
    testPolynomial( ut );

    // Test root finding
    testBasicRoots( ut );
    testWilkinsonRoots( ut );
    testDuplicateRoots( ut );
    testRandomRoots( 4, ut );
    testRandomRoots( 10, ut );
    testRandomRoots( 20, ut );
    testRandomRoots( 50, ut );
    // testRandomRoots( 200, ut );

    // Test the fitting
    testFitting( ut );

    // Finished
    int N_errors = ut.NumFailGlobal();
    ut.report();
    ut.reset();
    AMP::AMPManager::shutdown();
    return N_errors;
}
