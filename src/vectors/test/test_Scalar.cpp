#include "AMP/utils/Utilities.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/Scalar.h"

#include <chrono>
#include <iostream>
#include <string_view>


#define RECORD( X )                                     \
    do {                                                \
        bool test = X;                                  \
        if ( !test ) {                                  \
            std::cout << "Failed: " << #X << std::endl; \
            pass = false;                               \
        }                                               \
    } while ( 0 )


// Test auto creation of a Scalar
bool fun( const AMP::Scalar &x ) { return x.get<double>() != 0.0; }
bool fun2( const std::any &x ) { return std::any_cast<size_t>( x ) != 0; }


// Test the create function
bool testCreate()
{
    int x = 38;
    AMP::Scalar y1( static_cast<double>( x ) );
    auto y2   = y1.create( x );
    auto y3   = y1.create( y1 );
    bool pass = y1.type() == y2.type() && y1.getTypeHash() == y2.getTypeHash() &&
                y1.type() == y3.type() && y1.getTypeHash() == y3.getTypeHash();
    return pass;
}


// Test storing and getting a value (integer)
template<class TYPE>
bool testGet( TYPE x )
{
    // Store the scalar
    AMP::Scalar y( x );
    bool pass = true;
    // Test getting the scalar
    auto z = std::real( x );
    pass   = pass && y.get<int>() == static_cast<int>( z );
    pass   = pass && y.get<uint64_t>() == static_cast<uint64_t>( z );
    pass   = pass && y.get<float>() == static_cast<float>( z );
    pass   = pass && y.get<double>() == static_cast<double>( z );
    pass   = pass && y.get<std::complex<float>>() == std::complex<float>( z, 0.0 );
    pass   = pass && y.get<std::complex<double>>() == std::complex<double>( z, 0.0 );
    return pass;
}


// Test basic arithmetic
template<class TYPE>
bool testArithmetic()
{
    TYPE a = (TYPE) 17.2;
    TYPE b = (TYPE) 3.7;
    AMP::Scalar x( a );
    AMP::Scalar y( b );
    bool pass = true;
    // Test some different operations
    pass = pass && std::abs( ( a - b ) - ( x - y ).get<TYPE>() ) < 1e-8;
    pass = pass && std::abs( ( a + b ) - ( x + y ).get<TYPE>() ) < 1e-8;
    pass = pass && std::abs( ( a * b ) - ( x * y ).get<TYPE>() ) < 1e-8;
    pass = pass && std::abs( ( a / b ) - ( x / y ).get<TYPE>() ) < 1e-8;
    return pass;
}
// Test NaNs/Inf
bool testInf()
{
    bool pass = true;
#ifndef _MSC_VER
    AMP::Scalar x = 1.0;
    AMP::Scalar y = 0.0;
    auto z1       = x / y;
    AMP::Scalar z2( std::numeric_limits<double>::infinity() );
    auto v1 = z1.get<double>();
    auto v2 = z2.get<double>();
    pass    = pass && AMP::Utilities::isInf( v1 );
    pass    = pass && AMP::Utilities::isInf( v2 );
#endif
    return pass;
}
bool testNaN()
{
    bool pass = true;
#ifndef _MSC_VER
    AMP::Scalar x = 0.0;
    AMP::Scalar y = 0.0;
    auto z1       = x / y;
    AMP::Scalar z2( std::numeric_limits<double>::quiet_NaN() );
    auto v1 = z1.get<double>();
    auto v2 = z2.get<double>();
    pass    = pass && AMP::Utilities::isNaN( v1 );
    pass    = pass && AMP::Utilities::isNaN( v2 );
#endif
    return pass;
}


// Test the performance
void testPerformance()
{
    size_t N = 1000000;

    auto start = std::chrono::system_clock::now();
    for ( size_t i = 1; i <= N; i++ ) {
        fun2( i );
    }
    auto stop  = std::chrono::system_clock::now();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( stop - start ).count();
    printf( "Time to create std::any: %i ns\n", static_cast<int>( ns / N ) );

    start = std::chrono::system_clock::now();
    for ( size_t i = 1; i <= N; i++ )
        fun( i );
    stop = std::chrono::system_clock::now();
    ns   = std::chrono::duration_cast<std::chrono::nanoseconds>( stop - start ).count();
    printf( "Time to store/get value: %i ns\n", static_cast<int>( ns / N ) );
}


// Test passing scalar by reference
void passConstRef( const AMP::Scalar &x )
{
    auto y = x; // d_data for x has a bad address according to totalview
    AMP_ASSERT( static_cast<double>( y ) == 1.0 );
}


int main( int, char ** )
{
    bool pass = true;

    // Test some basic types
    RECORD( testGet<char>( 'x' ) );
    RECORD( testGet<int>( 1 ) );
    RECORD( testGet<int64_t>( 2 ) );
    RECORD( testGet<uint64_t>( 3 ) );
    RECORD( testGet<float>( 4 ) );
    RECORD( testGet<double>( 5 ) );
    RECORD( testGet<long double>( 6 ) );
    RECORD( testGet( std::complex<float>( 8.0, 0.0 ) ) );
    RECORD( testGet( std::complex<double>( 9.0, 0.0 ) ) );

    // Test create
    RECORD( testCreate() );

    // Test passing values
    RECORD( fun( 1 ) );
    RECORD( fun( 2u ) );
    RECORD( fun( 3.0f ) );
    RECORD( fun( 4.0 ) );

    // Test complex
    auto c1 = AMP::Scalar( std::complex<float>( 3.0, 1.0 ) );
    RECORD( c1.get<std::complex<float>>() == std::complex<float>( 3.0, 1.0 ) );

    // Test copy
    auto c2 = c1;
    RECORD( c1.get<std::complex<float>>() == c2.get<std::complex<float>>() );

    // Test arithmetic
    RECORD( testArithmetic<double>() );
    RECORD( testArithmetic<int64_t>() );
    RECORD( testArithmetic<std::complex<double>>() );
    RECORD( testInf() );
    RECORD( testNaN() );

    // Test the performance
    testPerformance();

    // Test passing scalar by reference
    passConstRef( 1.0 );

    if ( pass )
        std::cout << "All tests passed\n";
    else
        std::cout << "Some tests failed\n";
    return pass ? 0 : 1;
}
