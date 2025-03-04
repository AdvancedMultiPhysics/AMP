#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Array.hpp"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.hpp"


using namespace AMP;


class TestAllocateClass
{
public:
    TestAllocateClass()
    {
        data = new double[8];
        N_alloc++;
    }
    TestAllocateClass( const TestAllocateClass & )
    {
        data = new double[8];
        N_alloc++;
    }
    TestAllocateClass &operator=( const TestAllocateClass &rhs )
    {
        if ( this != &rhs ) {
            data = new double[8];
            N_alloc++;
        }
        return *this;
    }
    TestAllocateClass( TestAllocateClass &&rhs )
    {
        delete[] data;
        data     = rhs.data;
        rhs.data = nullptr;
    }
    TestAllocateClass &operator=( TestAllocateClass &&rhs )
    {
        if ( this != &rhs ) {
            delete[] data;
            data     = rhs.data;
            rhs.data = nullptr;
        }
        return *this;
    }
    ~TestAllocateClass()
    {
        delete[] data;
        N_alloc--;
    }
    static int get_N_alloc() { return N_alloc; }

private:
    double *data;
    static int N_alloc;
};

int TestAllocateClass::N_alloc = 0;


// Function to test linear interpolation
template<class TYPE>
TYPE fun( double x, double y, double z );
template<>
inline double fun<double>( double x, double y, double z )
{
    return sin( x ) * cos( y ) * exp( -z );
}
template<>
inline int fun<int>( double x, double y, double z )
{
    return static_cast<int>( 100000 * fun<double>( x, y, z ) );
}
template<class TYPE>
void test_interp( UnitTest &ut, const std::vector<size_t> &N )
{
    Array<TYPE> A( N );
    std::vector<size_t> N2( N );
    N2.resize( 3, 1 );
    auto testname = AMP::Utilities::stringf(
        "interp<%s,%i,%i,%i>", typeid( TYPE ).name(), (int) N2[0], (int) N2[1], (int) N2[2] );

    // Fill A
    A.fill( 0 );
    for ( size_t i = 0; i < A.size( 0 ); i++ ) {
        double x = i * 1.0 / std::max<double>( N2[0] - 1, 1 );
        for ( size_t j = 0; j < A.size( 1 ); j++ ) {
            double y = j * 1.0 / std::max<double>( N2[1] - 1, 1 );
            for ( size_t k = 0; k < A.size( 2 ); k++ ) {
                double z     = k * 1.0 / std::max<double>( N2[2] - 1, 1 );
                A( i, j, k ) = fun<TYPE>( x, y, z );
            }
        }
    }

    // Test the input points
    bool pass = true;
    std::vector<double> x( 3 );
    for ( size_t i = 0; i < A.size( 0 ); i++ ) {
        x[0] = i;
        for ( size_t j = 0; j < A.size( 1 ); j++ ) {
            x[1] = j;
            for ( size_t k = 0; k < A.size( 2 ); k++ ) {
                x[2] = k;
                if ( fabs( A( i, j, k ) - A.interp( x ) ) > 1e-12 * A( i, j, k ) )
                    pass = false;
            }
        }
    }
    if ( pass )
        ut.passes( testname + " (input points)" );
    else
        ut.failure( testname + " (input points)" );

    // Test random points
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dis( 0, 1 );
    pass = true;
    std::vector<double> x1( 3, 0 );
    std::vector<double> x2( 3, 0 );
    for ( int i = 0; i < 10000; i++ ) {
        for ( size_t d = 0; d < N.size(); d++ ) {
            x1[d] = dis( gen );
            x2[d] = ( N[d] - 1 ) * x1[d];
        }
        TYPE y1 = fun<TYPE>( x1[0], x1[1], x1[2] );
        TYPE y2 = A.interp( x2 );
        if ( fabs( y1 - y2 ) > 1e-3 * y1 )
            pass = false;
    }
    if ( pass )
        ut.passes( testname + " (random points)" );
    else
        ut.failure( testname + " (random points)" );
}


// Run some basic tests of ArraySize
void testArraySize( UnitTest &ut )
{
    // Test constexpr functions of ArraySize
    constexpr size_t N[2] = { 4, 5 };
    constexpr ArraySize s1( 2, 3 );
    constexpr ArraySize s2( 2, N );
    constexpr ArraySize s3 = { 2, 3, 4 };
    constexpr auto s4      = cat( s1, s2 );
    static_assert( s1.size() == 2 && s1.length() == 6, "Static constructor failed" );
    static_assert( s2.size() == 2 && s2.length() == 20, "Static constructor failed" );
    static_assert( s3.size() == 3 && s3.length() == 24, "Static constructor failed" );
    static_assert( s4.size() == 4 && s4.length() == 120, "Static constructor failed" );
    static_assert( s3.index( 1, 2, 3 ) == 23, "index failed" );
    static_assert( s3.index( { 1, 2, 3 } ) == 23, "index failed" );
    // Test non-const functions in ArraySize (requires C++17 to make them constexpr)
    bool pass = true;
    auto ijk  = s3.ijk( 23 );
    pass      = pass && ijk[0] == 1 && ijk[1] == 2 && ijk[2] == 3 && ijk[3] == 0 && ijk[4] == 0;
    pass      = pass && s3.index( ijk ) == 23;
    if ( pass )
        ut.passes( "ArraySize tests" );
    else
        ut.failure( "ArraySize tests" );
    // Test performance of index and ijk
    pass      = true;
    double t1 = Utilities::time();
    // ArraySize s( 223, 991, 569 );
    ArraySize s( 223, 91, 59 );
    for ( size_t k = 0, index = 0; k < s[2]; k++ ) {
        for ( size_t j = 0; j < s[1]; j++ ) {
            for ( size_t i = 0; i < s[0]; i++, index++ ) {
                pass = pass && index == s.index( i, j, k );
            }
        }
    }
    double t2 = Utilities::time();
    for ( size_t k = 0; k < s[2]; k++ ) {
        for ( size_t j = 0; j < s[1]; j++ ) {
            for ( size_t i = 0; i < s[0]; i++ ) {
                ijk  = s.ijk( s.index( i, j, k ) );
                pass = pass && ijk[0] == i && ijk[1] == j && ijk[2] == k;
            }
        }
    }
    double t3 = Utilities::time();
    printf( "Time to compute index: %0.2f ns\n", 1e9 * ( t2 - t1 ) / s.length() );
    printf( "Time to compute ijk: %0.2f ns\n", 1e9 * ( t3 - t2 ) / s.length() );
    if ( !pass )
        ut.passes( "ArraySize times" );
}


// Test the initializer lists
void testArrayInitializerList( UnitTest &ut )
{
    bool pass        = true;
    Array<double> M1 = { 1, 2, 3 };
    Array<double> M2 = { { 1, 2, 3 }, { 4, 5, 6 } };
    pass             = pass && M1.size() == ArraySize( 3 ) && M2.size() == ArraySize( 2, 3 );
    pass             = pass && M1( 0 ) == 1;
    pass             = pass && M1( 1 ) == 2;
    pass             = pass && M1( 2 ) == 3;
    pass             = pass && M2( 0, 0 ) == 1;
    pass             = pass && M2( 0, 1 ) == 2;
    pass             = pass && M2( 0, 2 ) == 3;
    pass             = pass && M2( 1, 0 ) == 4;
    pass             = pass && M2( 1, 1 ) == 5;
    pass             = pass && M2( 1, 2 ) == 6;
    if ( pass )
        ut.passes( "Array constructor - initializer list" );
    else
        ut.failure( "Array constructor - initializer list" );
}


// Run some basic tests of Array
void testArray( UnitTest &ut )
{
    // Create several matricies
    Array<double> M1, M2( 10, 5 );
    M1.resize( 10, 7 );
    for ( size_t i = 0; i < M2.size( 0 ); i++ ) {
        for ( size_t j = 0; j < M2.size( 1 ); j++ ) {
            M1( i, j ) = i + 10 * j;
            M2( i, j ) = i + 10 * j;
        }
    }
    M1.resize( 10, 5 );
    Array<double> M3( M1 );
    Array<double> M4 = M2;
    Array<double> M5 = M1;
    M5( 0, 0 )       = -1;
    if ( M1 == M2 && M1 == M3 && M1 == M4 && M1 != M5 )
        ut.passes( "Array constructors" );
    else
        ut.failure( "Array constructors" );

    // Test construction based on initializer lists
    testArrayInitializerList( ut );

    // Test range based creation
    Array<double> Mr( Range<double>( 1, 6, 0.5 ) );
    if ( Mr.length() == 11 && Mr.ndim() == 1 && Mr( 2 ) == 2.0 )
        ut.passes( "Array range-based constructor" );
    else
        ut.failure( "Array range-based constructor" );

    // Test std::string
    bool pass = true;
    Array<std::string> S;
    pass = pass && S.length() == 0;
    S.resize( 1 );
    pass   = pass && S.length() == 1;
    pass   = pass && S( 0 ).size() == 0;
    S( 0 ) = std::string( "test" );
    pass   = pass && S( 0 ) == "test";
    if ( pass )
        ut.passes( "Array string" );
    else
        ut.failure( "Array string" );

    // Test a failed allocation
    // Note: testing the allocation failure causes issues on a MAC
    try {
        bool test_alloc = !AMP::Utilities::running_valgrind();
#if defined( __APPLE__ ) || defined( USE_DEVICE )
        test_alloc = false;
#endif
        if ( test_alloc ) {
            size_t N = 10000;
            Array<double> M( N, N, N );
            ut.failure( "Failed allocation succeeded???" );
            AMP_ASSERT( M.length() == N * N * N );
        } else {
            ut.expected_failure( "Skipping failed allocation test (MAC, DEVICE, valgrind)" );
        }
    } catch ( std::logic_error &err ) {
        std::string msg = err.what();
        if ( msg == "Failed to allocate array" )
            ut.passes( "Caught failed allocation" );
        else
            ut.passes( "Caught exception for failed allocation: " + msg );
    } catch ( ... ) {
        ut.failure( "Caught unknown exception for failed allocation" );
    }

    // Test math opertors
    if ( M1.min() == 0 && M1.min( 0 ).min() == 0 )
        ut.passes( "min" );
    else
        ut.failure( "min" );
    if ( M1.max() == 49 && M1.max( 0 ).max() == 49 )
        ut.passes( "max" );
    else
        ut.failure( "max" );
    if ( M1.sum() == 1225 && M1.sum( 0 ).sum() == 1225 )
        ut.passes( "sum" );
    else
        ut.failure( "sum" );
    if ( M1.mean() == 24.5 )
        ut.passes( "mean" );
    else
        ut.failure( "mean" );
    if ( !M1.NaNs() )
        ut.passes( "NaNs" );
    else
        ut.failure( "NaNs" );

    // Test math operators with index subsets
    std::vector<size_t> idx{ 0, 4, 0, 2 };
    if ( M1.min( idx ) == 0 )
        ut.passes( "min on subset" );
    else
        ut.failure( "min on subset" );
    if ( M1.max( idx ) == 24 )
        ut.passes( "max on subset" );
    else
        ut.failure( "max on subset" );
    if ( M1.sum( idx ) == 180 )
        ut.passes( "sum on subset" );
    else {
        ut.failure( "sum on subset" );
    }
    if ( M1.mean( idx ) == 12 )
        ut.passes( "mean on subset" );
    else
        ut.failure( "mean on subset" );

    // Test find
    std::vector<size_t> index = M1.find( 7, []( double a, double b ) { return a == b; } );
    if ( index.size() != 1 )
        ut.failure( "find" );
    else if ( index[0] == 7 )
        ut.passes( "find" );
    else
        ut.failure( "find" );

    // Test subset
    M3 = M1.subset( { 0, 9, 0, 4 } );
    if ( M3 == M1 )
        ut.passes( "full subset" );
    else
        ut.failure( "full subset" );
    M3   = M1.subset( { 3, 7, 1, 3 } );
    pass = true;
    for ( size_t i = 0; i < M3.size( 0 ); i++ ) {
        for ( size_t j = 0; j < M3.size( 1 ); j++ )
            pass = pass && M3( i, j ) == ( i + 3 ) + 10 * ( j + 1 );
    }
    if ( pass )
        ut.passes( "partial subset" );
    else
        ut.failure( "partial subset" );
    M3.scale( 2 );
    M2.copySubset( { 3, 7, 1, 3 }, M3 );
    pass = true;
    for ( size_t i = 0; i < M3.size( 0 ); i++ ) {
        for ( size_t j = 0; j < M3.size( 1 ); j++ )
            pass = pass && M3( i, j ) == M2( i + 3, j + 1 );
    }
    if ( pass )
        ut.passes( "copyFromSubset" );
    else
        ut.failure( "copyFromSubset" );

    // Test the time required to create a view
    [[maybe_unused]] Array<double> M_view;
    double t1 = Utilities::time();
    for ( size_t i = 0; i < 100000; i++ ) {
        M_view.viewRaw( { M1.size( 0 ), M1.size( 1 ) }, M1.data() );
    }
    double t2 = Utilities::time();
    if ( M_view == M1 )
        ut.passes( "view" );
    else
        ut.failure( "view" );
    printf( "Time to create view: %0.2f ns\n", ( t2 - t1 ) * 1e9 / 100000 );
    // Test time to access elements
    {
        Array<double> x( 100000 );
        x.rand();
        double s = 0;
        t1       = Utilities::time();
        for ( size_t i = 0; i < x.length(); i++ )
            s += x( i );
        t2 = Utilities::time();
        AMP_ASSERT( s > 0 );
        printf( "Time to access: %0.2f ns\n", ( t2 - t1 ) * 1e9 / x.length() );
    }
    // Simple tests of +/-
    M2 = M1;
    M2.scale( 2 );
    M3 = M1;
    M3 += M1;
    if ( M1 + M1 == M2 && M3 == M2 )
        ut.passes( "operator+(Array&)" );
    else
        ut.failure( "operator+(Array&)" );
    M3 = M2;
    M3 -= M1;
    if ( M2 - M1 == M1 && M3 == M1 )
        ut.passes( "operator-(Array&)" );
    else
        ut.failure( "operator-(Array&)" );

    M1 += 3;
    pass = true;
    for ( size_t i = 0; i < M1.size( 0 ); i++ ) {
        for ( size_t j = 0; j < M1.size( 1 ); j++ )
            pass = pass && ( M1( i, j ) == i + 3 + 10 * j );
    }
    if ( pass )
        ut.passes( "operator+(scalar)" );
    else
        ut.failure( "operator+(scalar)" );

    M1 -= 3;
    pass = true;
    for ( size_t i = 0; i < M1.size( 0 ); i++ ) {
        for ( size_t j = 0; j < M1.size( 1 ); j++ )
            pass = pass && ( M1( i, j ) == i + 10 * j );
    }
    if ( pass )
        ut.passes( "operator-(scalar)" );
    else
        ut.failure( "operator-(scalar)" );

    // swap test
    auto dA1 = M1.data();
    auto dA2 = M2.data();
    M1.swap( M2 );
    pass = ( ( M1.data() == dA2 ) && ( M2.data() == dA1 ) );
    if ( pass )
        ut.passes( "swap" );
    else
        ut.failure( "swap" );
}


// The main function
int main( int argc, char *argv[] )
{
    // Startup
    AMPManager::startup( argc, argv );
    UnitTest ut;

    // Run basic ArraySize tests
    testArraySize( ut );

    // Run basic Array tests
    testArray( ut );

    // Test sum
    {
        Array<double> x( 1000, 100 );
        x.rand();
        double t1          = Utilities::time();
        double s1          = x.sum();
        double t2          = Utilities::time();
        double s2          = 0;
        const size_t N     = x.length();
        const double *data = x.data();
        for ( size_t i = 0; i < N; i++ )
            s2 += data[i];
        double t3 = Utilities::time();
        if ( fabs( s1 - s2 ) / s1 < 1e-12 )
            ut.passes( "sum" );
        else
            ut.failure( "sum" );
        printf( "Time to perform sum (sum()): %0.2f ns\n", ( t2 - t1 ) * 1e9 / N );
        printf( "Time to perform sum (raw): %0.2f ns\n", ( t3 - t2 ) * 1e9 / N );
    }

    // Test the allocation of a non-trivial type
    {
        bool pass = true;
        std::shared_ptr<TestAllocateClass> ptr;
        {
            Array<TestAllocateClass> x( 3, 4 );
            pass = pass && TestAllocateClass::get_N_alloc() == 12;
            x.resize( 2, 1 );
            pass = pass && TestAllocateClass::get_N_alloc() == 2;
            ptr  = x.getPtr();
        }
        pass = pass && TestAllocateClass::get_N_alloc() == 2;
        ptr.reset();
        pass = pass && TestAllocateClass::get_N_alloc() == 0;
        if ( pass )
            ut.passes( "Allocator" );
        else
            ut.failure( "Allocator" );
    }

    // Test cat
    {
        Array<double> r1( Range<double>( -3.5, -2, 0.1 ) );
        Array<double> r2( Range<double>( -1.96, 0, 0.02 ) );
        Array<double> r3( Range<double>( 0.05, 1, 0.05 ) );
        Array<double> r4( Range<double>( 1.1, 2, 0.1 ) );
        auto tmp = Array<double>::cat( { r1, r2, r3, r4 } );
        if ( tmp.length() == 145 )
            ut.passes( "cat" );
        else
            ut.failure( "cat" );
    }

    // Test string range based constructor
    {
        auto createRange = []( double lb, double ub, double dx ) {
            return AMP::Array<double>( AMP::Range( lb, ub, dx ) );
        };
        bool pass = true;
        auto tmp1 = Array<double>( "[ -3.5:0.1:-2 -1.96:0.02:0 0.05:0.05:1 1.1:0.1:2 ]" );
        pass      = pass && tmp1.ndim() == 1 && tmp1.length() == 145;
        auto tmp2 = Array<int>( "[ 0 1:1:10; 11:20 0 ]" );
        pass      = pass && tmp2.ndim() == 2 && tmp2.size( 0 ) == 2 && tmp2.size( 1 ) == 11;
        auto tmp3 = AMP::Array<double>::cat( { createRange( -3.5, -2, 0.1 ),
                                               createRange( -1.96, 0, 0.02 ),
                                               createRange( 0.05, 1, 0.05 ),
                                               createRange( 1.1, 2, 0.1 ) } );
        pass      = pass && tmp1 == tmp3;
        if ( pass )
            ut.passes( "range-string" );
        else
            ut.failure( "range-string" );
    }

    // Test resize
    {
        Array<double> M1( 10, 10 );
        M1.rand();
        auto M2 = M1;
        M2.resizeDim( 0, 20, -1 );
        bool pass = M2.size( 0 ) == 20 && M2( 19, 0 ) == -1;
        M2.resize( M1.size() );
        pass = pass && M2 == M1;
        if ( pass )
            ut.passes( "resize" );
        else
            ut.failure( "resize" );
    }

    // Test interpolation
    {
        test_interp<double>( ut, { 100 } );
        test_interp<double>( ut, { 50, 50 } );
        test_interp<double>( ut, { 30, 30, 30 } );
    }

    // Test print
    {
        Array<uint16_t> M1( 3, 4 );
        M1.rand();
        std::cout << std::endl;
        M1.print( std::cout, "M1" );
        std::cout << std::endl;
    }

    // Finished
    ut.report( 1 );
    auto num_failed = static_cast<int>( ut.NumFailGlobal() );
    if ( num_failed == 0 )
        std::cout << "All tests passed\n";
    ut.reset();
    AMPManager::shutdown();
    return num_failed;
}
