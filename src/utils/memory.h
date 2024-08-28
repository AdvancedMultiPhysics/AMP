#ifndef included_AMP_MEMORY
#define included_AMP_MEMORY

#include <type_traits>

#ifdef USE_CUDA
    #include "AMP/utils/cuda/CudaAllocator.h"
#endif
#ifdef USE_HIP
    #include "AMP/utils/hip/HipAllocator.h"
#endif
#include "AMP/utils/Utilities.h"


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

// host allocators
#ifdef USE_CUDA
template<typename TYPE>
using HostAllocator = AMP::CudaHostAllocator<TYPE>;
#elif defined( USE_HIP )
template<typename TYPE>
using HostAllocator = AMP::HipHostAllocator<TYPE>;
#else // no device
template<typename TYPE>
using HostAllocator = std::allocator<TYPE>;
#endif

template<typename ALLOC>
AMP::Utilities::MemoryType getAllocatorMemoryType()
{
    using intAllocator = typename std::allocator_traits<ALLOC>::template rebind_alloc<int>;
    if ( std::is_same<intAllocator, std::allocator<int>>::value ) {
        return AMP::Utilities::MemoryType::unregistered;
#ifdef USE_CUDA
    } else if ( std::is_same<intAllocator, AMP::CudaManagedAllocator<int>>::value ) {
        return AMP::Utilities::MemoryType::managed;
    } else if ( std::is_same<intAllocator, AMP::CudaDevAllocator<int>>::value ) {
        return AMP::Utilities::MemoryType::device;
#endif
#ifdef USE_HIP
    } else if ( std::is_same<intAllocator, AMP::HipManagedAllocator<int>>::value ) {
        return AMP::Utilities::MemoryType::managed;
    } else if ( std::is_same<intAllocator, AMP::HipDevAllocator<int>>::value ) {
        return AMP::Utilities::MemoryType::device;
#endif
    } else {
        AMP_ERROR( "Unknown Allocator" );
    }
}

} // namespace AMP
#endif
