#include "AMP/utils/AMPManager.h"
#include "AMP/utils/MathExpr.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include <chrono>
#include <cmath>


void test( const std::string &expression,
           const std::vector<std::string> &vars,
           const std::vector<double> &data,
           const double &ans,
           AMP::UnitTest &ut )
{
    AMP::MathExpr fun( expression, vars );
    auto fun2 = AMP::MathExpr( fun.getExpr(), fun.getVars() );
    auto r1   = fun( data.data() );
    auto r2   = fun2( data.data() );
    bool pass = AMP::Utilities::approx_equal_abs( r1, ans, 1e-12 );
    pass      = pass && AMP::Utilities::approx_equal_abs( r1, r2, 1e-14 );
    if ( pass )
        ut.passes( expression );
    else
        ut.failure( expression );
}


void test_Performance( AMP::UnitTest &ut )
{
    int N1  = 1000;
    int N2  = 100000;
    auto t1 = std::chrono::steady_clock::now();
    AMP::MathExpr fun( "sin(0.1*x)", { "x" } );
    for ( int i = 1; i < N1; i++ ) {
        [[maybe_unused]] AMP::MathExpr fun2( fun.getExpr(), fun.getVars() );
    }
    auto t2   = std::chrono::steady_clock::now();
    double s1 = 0;
    for ( double i = 0; i < N2; i++ )
        s1 += fun( &i );
    auto t3   = std::chrono::steady_clock::now();
    double s2 = 0;
    for ( double i = 0; i < N2; i++ )
        s2 += sin( 0.1 * i );
    auto t4 = std::chrono::steady_clock::now();
    int ns1 = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / N1;
    int ns2 = std::chrono::duration_cast<std::chrono::nanoseconds>( t3 - t2 ).count() / N2;
    int ns3 = std::chrono::duration_cast<std::chrono::nanoseconds>( t4 - t3 ).count() / N2;
    printf( "Time to create: %i ns\n", ns1 );
    printf( "Time to eval: %i ns\n", ns2 );
    printf( "Time for C++: %i ns\n", ns3 );
    if ( fabs( s2 - s1 ) > 1e-14 ) {
        printf( "Error: %0.6f %0.6f %0.2e\n", s1, s2, s2 - s1 );
        ut.failure( "Performance" );
    }
}


int main( int, char *[] )
{
    AMP::UnitTest ut;

    // Test some basic functions
    constexpr double e  = 2.718281828459045;
    constexpr double pi = 3.141592653589793;
    test( "e", {}, {}, e, ut );
    test( "pi", {}, {}, pi, ut );
    test( "5*8", {}, {}, 40, ut );
    test( "2*x", { "x" }, { 3.2 }, 6.4, ut );
    test( "3*x", { "x" }, { 6.5 }, 19.5, ut );
    test( "fac(8)", {}, {}, 40320, ut );
    test( "ncr(52,5)", {}, {}, 2598960, ut );
    test( "npr(52,5)", {}, {}, 311875200, ut );
    test( "exp(2.5)", {}, {}, 12.182493960703473, ut );
    test( "sin(0.8*pi)", {}, {}, sin( 0.8 * pi ), ut );
    test( "sqrt(x^2+y^2)", { "x", "y" }, { 1.5, 3.5 }, std::sqrt( 14.5 ), ut );
    test( "x*exp(y)", { "x", "y" }, { 1.5, 3.5 }, 1.5 * exp( 3.5 ), ut );
    test( "3*(a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p)",
          { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p" },
          { 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16. },
          408,
          ut );

    // Test the performance
    test_Performance( ut );

    // Finished
    ut.report();
    int N_errors = ut.NumFailGlobal();
    return N_errors;
}
