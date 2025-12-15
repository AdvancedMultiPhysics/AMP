#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "AMP/utils/Utilities.h"
#include "AMP/utils/extended_int.h"

inline double time() { return AMP::Utilities::time(); }

using eint64   = AMP::extended::int64N<1>;
using eint128  = AMP::extended::int128_t;
using eint256  = AMP::extended::int256_t;
using eint512  = AMP::extended::int512_t;
using eint1024 = AMP::extended::int1024_t;
using eint2048 = AMP::extended::int2048_t;


// Check the sizes of the different types
static_assert( sizeof( eint64 ) == 8, " eint64 is not 8 bytes" );
static_assert( sizeof( eint128 ) == 16, " eint128 is not 16 bytes" );
static_assert( sizeof( eint256 ) == 32, " eint256 is not 32 bytes" );
static_assert( sizeof( eint512 ) == 64, " eint512 is not 64 bytes" );
static_assert( sizeof( eint1024 ) == 128, " eint1024 is not 128 bytes" );
static_assert( sizeof( eint2048 ) == 256, " eint2048 is not 256 bytes" );


// Functions to time add/multiply
template<class TYPE>
int time_add()
{
    int N_it = 10000;
    int N    = 1000;
    std::default_random_engine gen;
    std::uniform_int_distribution<int> dist( 0, 1000 );
    auto start = std::chrono::high_resolution_clock::now();
    for ( int j = 0; j < N_it; j++ ) {
        [[maybe_unused]] TYPE y1( dist( gen ) );
        [[maybe_unused]] TYPE y2( dist( gen ) );
        for ( int i = 0; i < N; i++ )
            y1 += y2;
    }
    auto stop  = std::chrono::high_resolution_clock::now();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( stop - start ).count();
    return ns / ( N * N_it );
}
template<class TYPE>
int time_mult()
{
    int N_it = 1000;
    int N    = 1000;
    std::default_random_engine gen;
    std::uniform_int_distribution<int> dist( 0, 1000 );
    auto start = std::chrono::high_resolution_clock::now();
    for ( int j = 0; j < N_it; j++ ) {
        [[maybe_unused]] TYPE y1( dist( gen ) % 3 + 1 );
        [[maybe_unused]] TYPE y2( dist( gen ) % 3 + 1 );
        for ( int i = 0; i < N; i++ )
            y1 *= y2;
    }
    auto stop  = std::chrono::high_resolution_clock::now();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( stop - start ).count();
    return ns / ( N * N_it );
}


// Test conversion to/from decimal string
int testDecimal( const std::string &x )
{
    eint1024 y( x.data() );
    auto z = y.decimal();
    if ( x != z ) {
        std::cout << "testDecimal failed:\n";
        std::cout << "   " << x << std::endl;
        std::cout << "   " << z << std::endl;
        return 1;
    }
    return 0;
}


// The main function
int main( int, char *[] )
{
    int N_errors = 0;

    // Print the size of the largest integer
    if ( sizeof( intmax_t ) != 8 )
        printf( "sizeof(intmax_t) = %u, optimization possible\n", (uint8_t) sizeof( intmax_t ) );

    // Check some very basic math
    // Most of these checks are done at compile time in extended_int.cpp
    eint256 a;
    if ( static_cast<double>( a ) != 0 ) {
        printf( "Failed default initialization\n" );
        N_errors++;
    }
    eint512 x1( "-18330084947618492243362509" );
    eint512 x2( "11658926934915909934019286" );
    eint512 x3 = x1 - x2;
    eint512 x4 = x1 + ( -x2 );
    eint512 ans( "-29989011882534402177381795" );
    if ( x3 != ans || x4 != ans ) {
        printf( "Failed basic +-\n" );
        N_errors++;
    }

    // Check converting to/from a long decimal
    N_errors += testDecimal( "15634454661" );
    N_errors += testDecimal( "-6843184314" );
    N_errors +=
        testDecimal( "-26383752273331116085217755810940080809980496163410708066430999326829" );

    // Check the log2
    double x =
        eint1024( "26383752273331116085217755810940080809980496163410708066430999326829" ).log2();
    if ( fabs( x - 223.968832115 ) > 1e-6 ) {
        printf( "log2 = %f\n", x );
        N_errors++;
    }

    // Test the performance of add
    printf( "\nPerformance (add):\n" );
    printf( "   int64_t: %i ns\n", time_add<int64_t>() );
    printf( "   eint128: %i ns\n", time_add<eint128>() );
    printf( "   eint256: %i ns\n", time_add<eint256>() );
    printf( "   eint512: %i ns\n", time_add<eint512>() );
    printf( "\n" );

    // Test the performance of multiply
    printf( "Performance (multiply):\n" );
    printf( "   int64_t: %i ns\n", time_mult<int64_t>() );
    printf( "   eint128: %i ns\n", time_mult<eint128>() );
    printf( "   eint256: %i ns\n", time_mult<eint256>() );
    printf( "   eint512: %i ns\n", time_mult<eint512>() );
    printf( "\n" );

    if ( N_errors == 0 )
        printf( "All tests passed\n" );
    else
        printf( "Some tests failed\n" );
    return N_errors;
}
