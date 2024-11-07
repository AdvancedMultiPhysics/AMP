#ifndef included_AMP_MEMORY
#define included_AMP_MEMORY

#include <type_traits>

#ifdef USE_CUDA
    #include "AMP/utils/cuda/CudaAllocator.h"
#endif
#ifdef USE_HIP
    #include "AMP/utils/hip/HipAllocator.h"
#endif
#include "AMP/utils/UtilityMacros.h"


namespace AMP::Utilities {

//! Enum to store pointer type
enum class MemoryType : uint8_t { unregistered = 0, host = 1, managed = 2, device = 3 };

//! Return the pointer type
MemoryType getMemoryType( const void *ptr );

//! Return a string for the memory type
std::string getString( MemoryType );

} // namespace AMP::Utilities


namespace AMP {
// managed allocators
#ifdef USE_CUDA
template<typename TYPE>
using ManagedAllocator = AMP::CudaManagedAllocator<TYPE>;
#elif defined( USE_HIP )
template<typename TYPE>
using ManagedAllocator = AMP::HipManagedAllocator<TYPE>;
#endif

// device allocators
#ifdef USE_CUDA
template<typename TYPE>
using DeviceAllocator = AMP::CudaDevAllocator<TYPE>;
#elif defined( USE_HIP )
template<typename TYPE>
using DeviceAllocator = AMP::HipDevAllocator<TYPE>;
#endif

// host allocator
template<typename TYPE>
using HostAllocator = std::allocator<TYPE>;

} // namespace AMP

template<typename ALLOC>
AMP::Utilities::MemoryType getAllocatorMemoryType()
{
    using intAllocator = typename std::allocator_traits<ALLOC>::template rebind_alloc<int>;
    if ( std::is_same_v<intAllocator, std::allocator<int>> ) {
        return AMP::Utilities::MemoryType::host;
#ifdef USE_CUDA
    } else if ( std::is_same_v<intAllocator, AMP::CudaManagedAllocator<int>> ) {
        return AMP::Utilities::MemoryType::managed;
    } else if ( std::is_same_v<intAllocator, AMP::CudaDevAllocator<int>> ) {
        return AMP::Utilities::MemoryType::device;
#endif
#ifdef USE_HIP
    } else if ( std::is_same_v<intAllocator, AMP::HipManagedAllocator<int>> ) {
        return AMP::Utilities::MemoryType::managed;
    } else if ( std::is_same_v<intAllocator, AMP::HipDevAllocator<int>> ) {
        return AMP::Utilities::MemoryType::device;
#endif
    } else {
        AMP_ERROR( "Unknown Allocator" );
    }
}


#endif
