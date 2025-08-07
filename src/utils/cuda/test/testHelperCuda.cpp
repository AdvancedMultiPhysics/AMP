#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/cuda/CudaAllocator.h"
#include "AMP/utils/cuda/helper_cuda.h"
#include "AMP/utils/memory.h"

#include <iostream>
#include <memory>

#ifdef AMP_USE_KOKKOS
    #include <Kokkos_Core.hpp>
    #include <Kokkos_Macros.hpp>
#endif


static inline std::string getMemorySpace( void *ptr )
{
    return AMP::Utilities::getString( AMP::Utilities::getMemoryType( ptr ) );
}


template<class KokkosSpace>
void testKokkosMemorySpace( const char *str )
{
    size_t N = 100;
    KokkosSpace space;
    auto ptr = space.allocate( N * sizeof( double ) );
    std::cout << "Kokkos " << str << ": " << getMemorySpace( ptr ) << std::endl;
    space.deallocate( ptr, N * sizeof( double ) );
}


int main( int argc, char *argv[] )
{
    // Start AMP (initializes Kokkos)
    AMP::AMPManager::startup( argc, argv );

    // cudaError_t
    std::cout << "cudaSuccess: " << cudaGetName( cudaSuccess ) << std::endl;
    std::cout << "cudaErrorNoDevice: " << cudaGetName( cudaErrorNoDevice ) << std::endl;
    std::cout << "cudaErrorMemoryAllocation: " << cudaGetName( cudaErrorMemoryAllocation )
              << std::endl;
    std::cout << "cudaErrorMissingConfiguration: " << cudaGetName( cudaErrorMissingConfiguration )
              << std::endl;

    // CUresult
    std::cout << "CUDA_SUCCESS: " << cudaGetName( CUDA_SUCCESS ) << std::endl;
    std::cout << "CUDA_ERROR_NO_DEVICE: " << cudaGetName( CUDA_ERROR_NO_DEVICE ) << std::endl;
    std::cout << "CUDA_ERROR_OUT_OF_MEMORY: " << cudaGetName( CUDA_ERROR_OUT_OF_MEMORY )
              << std::endl;

    // Memory pointer type
    AMP::CudaDevAllocator<double> devAllocator;
    AMP::CudaManagedAllocator<double> managedAllocator;
    AMP::HostAllocator<double> hostAllocator;
    size_t N     = 100;
    auto device  = devAllocator.allocate( N );
    auto managed = managedAllocator.allocate( N );
    auto host    = hostAllocator.allocate( N );
    std::cout << std::endl;
    std::cout << "Device: " << getMemorySpace( device ) << std::endl;
    std::cout << "Managed: " << getMemorySpace( managed ) << std::endl;
    std::cout << "Host: " << getMemorySpace( host ) << std::endl;
    devAllocator.deallocate( device, N );
    managedAllocator.deallocate( managed, N );
    hostAllocator.deallocate( host, N );

    // Check Kokkos memory pointers
#ifdef AMP_USE_KOKKOS
    std::cout << std::endl;
    testKokkosMemorySpace<Kokkos::HostSpace>( "HostSpace" );
    #ifdef KOKKOS_ENABLE_CUDA
    testKokkosMemorySpace<Kokkos::CudaSpace>( "CudaSpace" );
    #endif
    #ifdef KOKKOS_ENABLE_OPENMPTARGET
    testKokkosMemorySpace<Kokkos::OpenMPTargetSpace>( " OpenMPTargetSpace" );
    #endif
#endif

    AMP::AMPManager::shutdown();
    return 0;
}
