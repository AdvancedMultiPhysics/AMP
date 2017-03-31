#include <iostream>
#include <math.h>
#include <stdexcept>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <cmath>

#include "utils/AMPManager.h"
#include "utils/Array.h"
#include "utils/PIO.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"


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
        data     = rhs.data;
        rhs.data = NULL;
    }
    TestAllocateClass &operator=( TestAllocateClass &&rhs )
    {
        if ( this != &rhs ) {
            data     = rhs.data;
            rhs.data = NULL;
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
template<class TYPE> TYPE fun( double x, double y, double z );
template<> inline double fun<double>( double x, double y, double z )
{
    return sin(x)*cos(y)*exp(-z);
}
template<> inline int fun<int>( double x, double y, double z )
{
    return static_cast<int>( 100000*fun<double>(x,y,z) );
}
template<class TYPE>
void test_interp( UnitTest& ut, const std::vector<size_t>& N )
{
    Array<TYPE> A(N);
    std::vector<size_t> N2( N );
    N2.resize(3,1);
    char buf[100];
    std::sprintf(buf,"interp<%s,%i,%i,%i>",typeid(TYPE).name(),(int)N2[0],(int)N2[1],(int)N2[2]);
    std::string testname(buf);
    // Fill A
    A.fill( 0 );
    for (size_t i=0; i<A.size(0); i++) {
        double x = i*1.0/std::max<double>(N2[0]-1,1);
        for (size_t j=0; j<A.size(1); j++) {
            double y = j*1.0/std::max<double>(N2[1]-1,1);
            for (size_t k=0; k<A.size(2); k++) {
                double z = k*1.0/std::max<double>(N2[2]-1,1);
                A(i,j,k) = fun<TYPE>(x,y,z);
            }
        }
    }
    // Test the input points
    bool pass = true;
    std::vector<double> x(3);
    for (size_t i=0; i<A.size(0); i++) {
        x[0] = i;
        for (size_t j=0; j<A.size(1); j++) {
            x[1] = j;
            for (size_t k=0; k<A.size(2); k++) {
                x[2] = k;
                if ( fabs( A(i,j,k)-A.interp( x ) ) > 1e-12*A(i,j,k) )
                    pass = false;
            }
        }
    }
    if ( pass )
        ut.passes( testname + " (input points)" );
    else
        ut.failure( testname + " (input points)" );
    // Test random points
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(0,1);
    pass = true;
    std::vector<double> x1(3,0);
    std::vector<double> x2(3,0);
    for (int i=0; i<10000; i++) {
        for (size_t d=0; d<N.size(); d++) {
            x1[d] = dis(gen);
            x2[d] = (N[d]-1)*x1[d];
        }
        TYPE y1 = fun<TYPE>( x1[0], x1[1], x1[2] );
        TYPE y2 = A.interp( x2 );
        if ( fabs(y1-y2) > 1e-3*y1 )
            pass = false;
    }
    if ( pass )
        ut.passes( testname + " (random points)" );
    else
        ut.failure( testname + " (random points)" );
}


// The main function
int main( int argc, char *argv[] )
{
    // Startup
    AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMPManager::startup( argc, argv, startup_properties );
    UnitTest ut;

    // these are currently not defined for AMP
    //    Utilities::setAbortBehavior(true,true,true);
    //    Utilities::setErrorHandlers();

    // Limit the scope of variables
    {
        // Create several matrices
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
        M5( 0, 0 ) = -1;
        if ( M1 == M2 && M1 == M3 && M1 == M4 && M1 != M5 )
            ut.passes( "Array constructors" );
        else
            ut.failure( "Array constructors" );
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
        try {
            size_t N = 10000;
            Array<double> M( N, N, N );
#if defined( __APPLE__ )
            ut.expected_failure( "Failed allocation succeeded (MAC)" );
#else
            ut.failure( "Failed allocation succeeded???" );
#endif
            AMP_ASSERT( M.length() == N * N * N );
        } catch ( ... ) {
            ut.passes( "Caught failed allocation" );
        }
        // Test math operators
        if ( M1.min() == 0 )
            ut.passes( "min" );
        else
            ut.failure( "min" );
        if ( M1.max() == 49 )
            ut.passes( "max" );
        else
            ut.failure( "max" );
        if ( M1.sum() == 1225 )
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
        M3 = M1.subset<double>( { 0, 9, 0, 4 } );
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
        Array<double> M_view;
        double t1 = Utilities::time();
        for ( size_t i = 0; i < 100000; i++ ) {
            M_view.viewRaw( { M1.size( 0 ), M1.size( 1 ) }, M1.data() );
            NULL_USE( M_view );
        }
        double t2 = Utilities::time();
        if ( M_view == M1 )
            ut.passes( "view" );
        else
            ut.failure( "view" );
        pout << "Time to create view: " << ( t2 - t1 ) * 1e9 / 100000 << " ns\n";
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
        
        //swap test
        auto dA1 = M1.data();
        auto dA2 = M2.data();
        M1.swap(M2);
        pass = ((M1.data()==dA2)&&(M2.data()==dA1));
        if ( pass )
            ut.passes( "swap" );
        else
            ut.failure( "swap" );
    }
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
        pout << "Time to perform sum (sum()): " << ( t2 - t1 ) * 1e9 / N << " ns\n";
        pout << "Time to perform sum (raw): " << ( t3 - t2 ) * 1e9 / N << " ns\n";
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
    // Test interpolation
    {
        test_interp<double>( ut, { 100 } );
        test_interp<double>( ut, { 50, 50 } );
        test_interp<double>( ut, { 30, 30, 30 } );
    }

    // Finished
    ut.report(1);
    int num_failed = static_cast<int>( ut.NumFailGlobal() );
    if ( num_failed == 0 )
        pout << "All tests passed\n";
    AMPManager::shutdown();
    return num_failed;
}
