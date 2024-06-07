#include "AMP/utils/hip/testGPUAllocators.hpp"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Array.hpp"
#include "AMP/utils/FunctionTable.h"
#include "AMP/utils/FunctionTable.hpp"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/hip/HipAllocator.h"
#include <hip/hip_runtime_api.h>

#include <memory>

template<typename T, typename Alloc1, typename Alloc2>
void ArrayTestWithAllocators( AMP::UnitTest &ut )
{
    AMP::Array<T, AMP::FunctionTable, Alloc1> A;
    AMP::Array<T, AMP::FunctionTable, Alloc2> B;
    AMP::Array<T> R;

    const size_t n             = 10;
    std::vector<size_t> v1     = { n };
    std::vector<size_t> v2     = { n, n };
    std::vector<size_t> v3     = { n, n, n };
    std::vector<size_t> v4     = { n, n, n, n };
    std::vector<size_t> vEmpty = { 0, 0, 0, 0 };
    A.resize( v1 );
    A.resize( v4 );
    A.resize( v3 );
    A.resize( vEmpty );
    A.resize( v2 );

    // Test to make sure the data is operable
    KernelWrapper<T> K;
    K.setData( A.data(), 4.0, A.length() );
    K.opData( A.data(), A.length() );
    checkHipErrors(hipDeviceSynchronize());

    B.resize( v2 );
    K.setData( B.data(), 4.0, B.length() );
    K.opData( B.data(), B.length() );
    R.resize( v2 );
    checkHipErrors(hipMemcpy( R.data(), B.data(), sizeof( T ) * R.length(), hipMemcpyDeviceToHost ));
    bool pass = true;
    for ( size_t i = 0; i < R.length(); i++ ) {
        if ( A.data()[i] != R.data()[i] ) {
            pass = false;
        }
    }

    if ( pass ) {
        ut.passes( "GPU Allocator Success" );
    } else {
        ut.failure( "GPU Allocator Failure" );
    }
}

template<typename T>
using ReboundManagedAllocator =
    typename std::allocator_traits<AMP::HipManagedAllocator<double>>::template rebind_alloc<T>;

template<typename T>
using ReboundDevAllocator =
    typename std::allocator_traits<AMP::HipDevAllocator<double>>::template rebind_alloc<T>;

int main( int argc, char *argv[] )
{
    // Declare Arrays and utils
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    ArrayTestWithAllocators<double,
                            AMP::HipManagedAllocator<double>,
                            AMP::HipDevAllocator<double>>( ut );

    ArrayTestWithAllocators<float, ReboundManagedAllocator<float>, ReboundDevAllocator<float>>(
        ut );

    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
