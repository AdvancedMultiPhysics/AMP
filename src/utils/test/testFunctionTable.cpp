#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/FunctionTable.h"
#include "AMP/utils/FunctionTable.hpp"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/typeid.h"

#include <limits>


#define PASS_FAIL( PASS, MSG )                                                        \
    count++;                                                                          \
    if ( !( PASS ) ) {                                                                \
        N_err++;                                                                      \
        printf( "Failed FunctionTable<%s>::%s\n", AMP::getTypeID<TYPE>().name, MSG ); \
    }


template<class T>
double SUM( const std::vector<T> &x )
{
    return AMP::FunctionTable<T>::sum( x.size(), x.data() );
}


template<class TYPE>
int TestFunctionTable()
{
    using FUN = AMP::FunctionTable<TYPE>;
    size_t N  = 10000;
    std::vector<TYPE> x( N );
    int N_err = 0;
    int count = 0;

    // Initialize to random data and test min/max/sum
    FUN::rand( N, x.data() );
    double min = FUN::min( N, x.data() );
    double max = FUN::max( N, x.data() );
    double sum = FUN::sum( N, x.data() );
    PASS_FAIL( FUN::equals( N, x.data(), x.data(), 1e-6 ), "equals" );
    PASS_FAIL( min >= 0 && min < 0.01, "min" );
    PASS_FAIL( max <= 1 && max > 0.99, "max" );
    PASS_FAIL( fabs( sum / N - 0.5 ) <= 0.01, "sum" );

    // Test basic arithmetic tests
    double eps = std::numeric_limits<TYPE>::epsilon();
    double tol = 20 * eps * sum;
    auto y     = x;
    FUN::scale( N, 2, y.data() );
    PASS_FAIL( fabs( SUM( y ) - 2 * sum ) < tol, "scale" );
    y = x;
    FUN::px( N, 0.5, y.data() );
    PASS_FAIL( fabs( SUM( y ) - ( sum + 0.5 * N ) ) < 10 * tol, "px(1)" );
    y = x;
    FUN::px( N, x.data(), y.data() );
    PASS_FAIL( fabs( SUM( y ) - 2 * sum ) < tol, "px(2)" );
    y = x;
    FUN::mx( N, 0.5, y.data() );
    PASS_FAIL( fabs( SUM( y ) - ( sum - 0.5 * N ) ) < 10 * tol, "mx(1)" );
    y = x;
    FUN::scale( N, 2, y.data() );
    FUN::mx( N, x.data(), y.data() );
    PASS_FAIL( fabs( SUM( y ) - sum ) < tol, "mx(2)" );

    // Test axpy
    y = x;
    FUN::axpy( 0.5, N, x.data(), y.data() );
    PASS_FAIL( fabs( SUM( y ) - 1.5 * sum ) < 5 * tol, "axpy" );
    y = x;
    FUN::axpby( 0.5, N, x.data(), 0.25, y.data() );
    PASS_FAIL( fabs( SUM( y ) - 0.75 * sum ) < 5 * tol, "axpby" );

    // Test reduce functions
    double sum2 = FUN::reduce( []( TYPE x, TYPE y ) { return x + y; }, N, x.data(), (TYPE) 0 );
    double sum3 = FUN::reduce(
        []( TYPE x, TYPE y, TYPE z ) { return x + y + z; }, N, x.data(), x.data(), (TYPE) 0 );
    PASS_FAIL( fabs( sum - sum2 ) < tol, "reduce(1)" );
    PASS_FAIL( fabs( 2 * sum - sum3 ) < tol, "reduce(2)" );

    // Test transform functions
    FUN::transform( []( TYPE x ) { return 2 * x; }, N, x.data(), y.data() );
    PASS_FAIL( fabs( SUM( y ) - 2 * sum ) < tol, "transform(1)" );
    FUN::transform( []( TYPE x, TYPE y ) { return x + 2 * y; }, N, x.data(), x.data(), y.data() );
    PASS_FAIL( fabs( SUM( y ) - 3 * sum ) < 10 * tol, "transform(2)" );

    // Everything passed
    if ( N_err == 0 )
        printf( "Passed FunctionTable<%s> (%i tests)\n", AMP::getTypeID<TYPE>().name, count );
    return N_err;
}


int main( int, char *[] )
{
    int N_error = 0;
    N_error += TestFunctionTable<double>();
    N_error += TestFunctionTable<float>();
    return N_error;
}
