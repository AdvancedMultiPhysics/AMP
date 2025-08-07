#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/hip/HipAllocator.h"
#include "AMP/utils/hip/helper_hip.h"
#include "AMP/utils/memory.h"

#include <iostream>
#include <memory>

#ifdef AMP_USE_KOKKOS
    #include <Kokkos_Core.hpp>
    #include <Kokkos_Macros.hpp>
#endif


static inline std::string_view getMemorySpace( void *ptr )
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

    // hipError_t
    std::cout << "hipSuccess: " << hipGetName( hipSuccess ) << std::endl;
    std::cout << "hipErrorNoDevice: " << hipGetName( hipErrorNoDevice ) << std::endl;
    std::cout << "hipErrorMemoryAllocation: " << hipGetName( hipErrorMemoryAllocation )
              << std::endl;
    std::cout << "hipErrorMissingConfiguration: " << hipGetName( hipErrorMissingConfiguration )
              << std::endl;

    // Memory pointer type
    AMP::HipDevAllocator<double> devAllocator;
    AMP::HipManagedAllocator<double> managedAllocator;
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
    #ifdef KOKKOS_ENABLE_HIP
    testKokkosMemorySpace<Kokkos::HIPSpace>( "HIPSpace" );
    #endif
    #ifdef KOKKOS_ENABLE_OPENMPTARGET
    testKokkosMemorySpace<Kokkos::OpenMPTargetSpace>( " OpenMPTargetSpace" );
    #endif
#endif

    AMP::AMPManager::shutdown();
    return 0;
}
