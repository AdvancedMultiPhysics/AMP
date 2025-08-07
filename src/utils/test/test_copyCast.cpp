#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"

#include <cstdlib>
#include <memory>
#include <stdio.h>
#ifdef AMP_USE_DEVICE
    #include <thrust/device_ptr.h>
// #include <thrust/host_ptr.h>
#endif

template<typename TypeIn,
         typename TypeOut,
         class AllocatorIn,
         class AllocatorOut,
         typename ptrIn,
         typename ptrOut,
         typename ExecutionSpace,
         AMP::Utilities::Backend backend>
void runTest( AMP::UnitTest &ut, const std::string &pass_msg )
{
    AllocatorIn alloc1;
    TypeIn *v1 = alloc1.allocate( 3 );
    AllocatorOut alloc2;
    TypeOut *v2 = alloc2.allocate( 3 );
    ptrIn v1p( v1 );
    ptrOut v2p( v2 );
    std::srand( 12 );
    for ( int i = 0; i < 3; i++ )
        v1p[i] = static_cast<TypeIn>( std::rand() ) / static_cast<TypeIn>( RAND_MAX );

    // Perform copy-cast
    AMP::Utilities::copyCast<TypeIn, TypeOut, backend, ExecutionSpace>( 3, v1, v2 );
    bool pass = true;
    auto tol  = std::numeric_limits<float>::epsilon();
    for ( int i = 0; i < 3; i++ ) {
        const double err = std::abs( ( v1p[i] - v2p[i] ) / v1p[i] );
        if ( err > tol ) {
            ut.failure( AMP::Utilities::stringf(
                "Cast precision loss %e larger than %e, for entry %d", err, tol, i ) );
            pass = false;
        }
    }
    if ( pass )
        ut.passes( pass_msg );
}


template<class AllocatorIn,
         class AllocatorOut,
         typename ptrIn,
         typename ExecutionSpace,
         AMP::Utilities::Backend backend>
void testOverflow( AMP::UnitTest &ut, const std::string &mem_type )
{
    AllocatorIn alloc1;
    double *v1 = alloc1.allocate( 3 );
    AllocatorOut alloc2;
    float *v2 = alloc2.allocate( 3 );
    ptrIn v1p( v1 );
    std::srand( 12 );
    for ( int i = 0; i < 3; i++ )
        v1p[i] = static_cast<double>( std::rand() ) / static_cast<double>( RAND_MAX );
    v1p[1] = std::numeric_limits<float>::max() * 2;
    // Perform copy-cast
    try {
        AMP::Utilities::copyCast<double, float, backend, ExecutionSpace>( 3, v1, v2 );
        ut.failure( mem_type + " copyCast didn't catch an overflow." );
    } catch ( ... ) {
        ut.passes( mem_type + " overflow test succeeded." );
    }
}

int main( int argc, char *argv[] )
{
    // Start AMP (initializes Kokkos)
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    // Test Host Serial
    runTest<double,
            float,
            AMP::HostAllocator<double>,
            AMP::HostAllocator<float>,
            double *,
            float *,
            AMP::HostAllocator<void>,
            AMP::Utilities::Backend::Serial>( ut, "Serial Host double->float passed" );
    runTest<float,
            double,
            AMP::HostAllocator<float>,
            AMP::HostAllocator<double>,
            float *,
            double *,
            AMP::HostAllocator<void>,
            AMP::Utilities::Backend::Serial>( ut, "Serial Host float->double passed" );
#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    testOverflow<AMP::HostAllocator<double>,
                 AMP::HostAllocator<float>,
                 double *,
                 AMP::HostAllocator<void>,
                 AMP::Utilities::Backend::Serial>( ut, "Serial Host" );
#endif

#ifdef AMP_USE_OPENMP
    // Test Host OpenMP
    runTest<double,
            float,
            AMP::HostAllocator<double>,
            AMP::HostAllocator<float>,
            double *,
            float *,
            AMP::HostAllocator<void>,
            AMP::Utilities::Backend::OpenMP>( ut, "OpenMP Host double->float passed" );
    runTest<float,
            double,
            AMP::HostAllocator<float>,
            AMP::HostAllocator<double>,
            float *,
            double *,
            AMP::HostAllocator<void>,
            AMP::Utilities::Backend::OpenMP>( ut, "OpenMP Host float->double passed" );
    #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    testOverflow<AMP::HostAllocator<double>,
                 AMP::HostAllocator<float>,
                 double *,
                 AMP::HostAllocator<void>,
                 AMP::Utilities::Backend::OpenMP>( ut, "OpenMP Host" );
    #endif
#endif


#ifdef AMP_USE_KOKKOS
    // Test Host Kokkos
    runTest<double,
            float,
            AMP::HostAllocator<double>,
            AMP::HostAllocator<float>,
            double *,
            float *,
            AMP::HostAllocator<void>,
            AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Host double->float passed" );
    runTest<float,
            double,
            AMP::HostAllocator<float>,
            AMP::HostAllocator<double>,
            float *,
            double *,
            AMP::HostAllocator<void>,
            AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Host float->double passed" );
    #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    testOverflow<AMP::HostAllocator<double>,
                 AMP::HostAllocator<float>,
                 double *,
                 AMP::HostAllocator<void>,
                 AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Host" );
    #endif
    #ifdef AMP_USE_DEVICE
    // Test Managed Kokkos
    runTest<double,
            float,
            AMP::ManagedAllocator<double>,
            AMP::ManagedAllocator<float>,
            thrust::device_ptr<double>,
            thrust::device_ptr<float>,
            AMP::ManagedAllocator<void>,
            AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Managed double->float passed" );
    runTest<float,
            double,
            AMP::ManagedAllocator<float>,
            AMP::ManagedAllocator<double>,
            thrust::device_ptr<float>,
            thrust::device_ptr<double>,
            AMP::ManagedAllocator<void>,
            AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Managed float->double passed" );
        #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    testOverflow<AMP::ManagedAllocator<double>,
                 AMP::ManagedAllocator<float>,
                 thrust::device_ptr<double>,
                 AMP::ManagedAllocator<void>,
                 AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Managed" );
        #endif
    // Test Device Kokkos
    runTest<double,
            float,
            AMP::DeviceAllocator<double>,
            AMP::DeviceAllocator<float>,
            thrust::device_ptr<double>,
            thrust::device_ptr<float>,
            AMP::DeviceAllocator<void>,
            AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Device double->float passed" );
    runTest<float,
            double,
            AMP::DeviceAllocator<float>,
            AMP::DeviceAllocator<double>,
            thrust::device_ptr<float>,
            thrust::device_ptr<double>,
            AMP::DeviceAllocator<void>,
            AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Device float->double passed" );
        #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    testOverflow<AMP::DeviceAllocator<double>,
                 AMP::DeviceAllocator<float>,
                 thrust::device_ptr<double>,
                 AMP::DeviceAllocator<void>,
                 AMP::Utilities::Backend::Kokkos>( ut, "Kokkos Device" );
        #endif
    #endif
#endif

#ifdef AMP_USE_DEVICE
    // Test Host
    runTest<double,
            float,
            AMP::HostAllocator<double>,
            AMP::HostAllocator<float>,
            double *,
            float *,
            AMP::HostAllocator<void>,
            AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Host double->float passed" );
    runTest<float,
            double,
            AMP::HostAllocator<float>,
            AMP::HostAllocator<double>,
            float *,
            double *,
            AMP::HostAllocator<void>,
            AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Host float->double passed" );
    #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    testOverflow<AMP::HostAllocator<double>,
                 AMP::HostAllocator<float>,
                 double *,
                 AMP::HostAllocator<void>,
                 AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Host" );
    #endif
    // Test Managed
    runTest<double,
            float,
            AMP::ManagedAllocator<double>,
            AMP::ManagedAllocator<float>,
            thrust::device_ptr<double>,
            thrust::device_ptr<float>,
            AMP::ManagedAllocator<void>,
            AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Managed double->float passed" );
    runTest<float,
            double,
            AMP::ManagedAllocator<float>,
            AMP::ManagedAllocator<double>,
            thrust::device_ptr<float>,
            thrust::device_ptr<double>,
            AMP::ManagedAllocator<void>,
            AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Managed float->double passed" );
    #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    testOverflow<AMP::ManagedAllocator<double>,
                 AMP::ManagedAllocator<float>,
                 thrust::device_ptr<double>,
                 AMP::ManagedAllocator<void>,
                 AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Managed" );
    #endif
    // Test Device
    runTest<double,
            float,
            AMP::DeviceAllocator<double>,
            AMP::DeviceAllocator<float>,
            thrust::device_ptr<double>,
            thrust::device_ptr<float>,
            AMP::DeviceAllocator<void>,
            AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Device double->float passed" );
    runTest<float,
            double,
            AMP::DeviceAllocator<float>,
            AMP::DeviceAllocator<double>,
            thrust::device_ptr<float>,
            thrust::device_ptr<double>,
            AMP::DeviceAllocator<void>,
            AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Device float->double passed" );
    #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    testOverflow<AMP::DeviceAllocator<double>,
                 AMP::DeviceAllocator<float>,
                 thrust::device_ptr<double>,
                 AMP::DeviceAllocator<void>,
                 AMP::Utilities::Backend::Hip_Cuda>( ut, "Hip_Cuda Device" );
    #endif
#endif

    int N_errors = ut.NumFailGlobal();
    ut.report();
    ut.reset();
    AMP::AMPManager::shutdown();
    return N_errors;
}
