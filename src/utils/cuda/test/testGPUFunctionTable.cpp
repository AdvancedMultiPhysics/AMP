#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/FunctionTable.h"
#include "AMP/utils/FunctionTable.hpp"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/cuda/CudaAllocator.h"
#include "AMP/utils/cuda/GPUFunctionTable.h"

#include <cuda.h>


template<class TYPE, class FUN, class ALLOC>
void TestFunctionTable( AMP::UnitTest *ut,
                        AMP::Array<TYPE, FUN, ALLOC> &A,
                        AMP::Array<TYPE, FUN, ALLOC> &B )
{
    bool pass     = true;
    double thresh = 1e-6;
    double val;

    // Tests for sum and equals operators
    size_t n1 = 20;
    size_t n2 = 456079;
    A.resize( n1 );
    B.resize( n2 );
    for ( size_t i = 0; i < n1; i++ ) {
        A.data()[i] = (TYPE) 1.0;
    }
    for ( size_t i = 0; i < n1; i++ ) {
        A.data()[i] = 1.0;
    }
    TYPE r1 = FUN::sum( A );
    TYPE r2 = FUN::sum( B );
    if ( fabs( r1 - (float) n1 ) > thresh || fabs( r2 - (float) n2 > thresh ) ) {
        pass = false;
    }

    A.resize( n2 );
    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = 1.0;
        B.data()[i] = 1.0;
    }
    bool eq = FUN::equals( A, B, thresh );
    if ( !eq ) {
        pass = false;
    }

    int rind       = std::rand() % n2;
    B.data()[rind] = 0.0;
    eq             = FUN::equals( A, B, thresh );
    if ( eq ) {
        pass = false;
    }

    // Tests for transform operators - using same tests as AMPNN
    // ReLU
    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = -1.0;
    }
    FUN::transformReLU( A, B );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( fabs( B.data()[i] ) > thresh ) {
            pass = false;
        }
    }
    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = 1.0;
    }
    FUN::transformReLU( A, B );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( fabs( B.data()[i] - 1.0 ) > thresh ) {
            pass = false;
        }
    }

    // Abs
    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = -1.0;
    }
    FUN::transformAbs( A, B );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( fabs( B.data()[i] - 1.0 ) > thresh ) {
            pass = false;
        }
    }

    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = 1.0;
    }
    FUN::transformAbs( A, B );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( fabs( B.data()[i] - 1.0 ) > thresh ) {
            pass = false;
        }
    }

    // HardTanh
    for ( size_t i = 0; i < n2; i++ ) {
        if ( i % 2 == 0 ) {
            A.data()[i] = 2.0;
        } else {
            A.data()[i] = -2.0;
        }
    }
    FUN::transformHardTanh( A, B );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( i % 2 == 0 ) {
            if ( fabs( B.data()[i] - 1.0 ) > thresh ) {
                pass = false;
            }
        } else {
            if ( fabs( B.data()[i] + 1.0 ) > thresh ) {
                pass = false;
            }
        }
    }

    // Tanh
    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = ( 1.0 / 2.0 ) * log( 1.0 );
    }
    FUN::transformTanh( A, B );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( fabs( B.data()[i] ) > thresh ) {
            pass = false;
        }
    }

    for ( size_t i = 1; i <= n2; i++ ) {
        A.data()[i - 1] = ( 1.0 / 2.0 ) * log( i );
    }
    FUN::transformTanh( A, B );
    for ( size_t i = 1; i <= n2; i++ ) {
        val = ( i - 1.0 ) / ( i + 1.0 );
        if ( fabs( val - B.data()[i - 1] ) > thresh ) {
            pass = false;
        }
    }

    // Sigmoid
    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = ( 1.0 / 2.0 ) * log( 1.0 );
    }
    FUN::transformSigmoid( A, B );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( fabs( B.data()[i] - ( 1.0 / 2.0 ) ) > thresh ) {
            pass = false;
        }
    }

    for ( size_t i = 1; i <= n2; i++ ) {
        A.data()[i - 1] = log( 1.0 / i );
    }
    FUN::transformSigmoid( A, B );
    for ( size_t i = 1; i <= n2; i++ ) {
        val = 1.0 / ( 1.0 + i );
        if ( fabs( val - B.data()[i - 1] ) > thresh ) {
            pass = false;
        }
    }

    // Softplus
    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = log( expm1( 1.0 ) );
    }
    FUN::transformSoftPlus( A, B );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( fabs( 1.0 - B.data()[i] ) > thresh ) {
            pass = false;
        }
    }

    for ( size_t i = 0; i < n2; i++ ) {
        A.data()[i] = -1.0;
    }
    FUN::transformSoftPlus( A, B );
    val = log1p( exp( -1.0 ) );
    for ( size_t i = 0; i < n2; i++ ) {
        if ( fabs( B.data()[i] - val ) > thresh ) {
            pass = false;
        }
    }

    if ( pass ) {
        ut->passes( "Pass" );
    } else {
        ut->failure( "fail" );
    }
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    AMP::Array<double> A;
    AMP::Array<double> B;
    TestFunctionTable( &ut, A, B );

#ifdef AMP_USE_CUDA
    AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> C;
    AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> D;
    TestFunctionTable( &ut, C, D );
#endif

    ut.report();
    AMP::AMPManager::shutdown();
}
