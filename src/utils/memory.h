#ifndef included_AMP_MEMORY
#define included_AMP_MEMORY

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/UtilityMacros.h"

#ifdef AMP_USE_CUDA
    #include "AMP/utils/cuda/CudaAllocator.h"
#endif
#ifdef AMP_USE_HIP
    #include "AMP/utils/hip/HipAllocator.h"
#endif

#include <string_view>
#include <type_traits>


namespace AMP::Utilities {

//! Enum to store pointer type
enum class MemoryType : int8_t { none = -1, unregistered = 0, host = 1, managed = 2, device = 3 };

//! Return the pointer type
MemoryType getMemoryType( const void *ptr );

//! Return a string for the memory type
std::string_view getString( MemoryType );

//! Return the memory type from a string
MemoryType memoryLocationFromString( std::string_view name );

//! Perform memory-memory copy (pointer may be in any memory space)
void memcpy( void *dest, const void *src, std::size_t count );

//! Perform memory set (pointer may be in any memory space)
void memset( void *dest, int ch, std::size_t count );

//! Perform memory zero (pointer may be in any memory space)
void zero( void *dest, std::size_t count );

//! Perform copy with conversion if necessary
template<class T1, class T2>
void copy( size_t N, const T1 *src, T2 *dst );


} // namespace AMP::Utilities


namespace AMP {
// managed allocators
#ifdef AMP_USE_CUDA
template<typename TYPE>
using ManagedAllocator = AMP::CudaManagedAllocator<TYPE>;
#elif defined( AMP_USE_HIP )
template<typename TYPE>
using ManagedAllocator = AMP::HipManagedAllocator<TYPE>;
#endif

// device allocators
#ifdef AMP_USE_CUDA
template<typename TYPE>
using DeviceAllocator = AMP::CudaDevAllocator<TYPE>;
#elif defined( AMP_USE_HIP )
template<typename TYPE>
using DeviceAllocator = AMP::HipDevAllocator<TYPE>;
#endif

// host allocator
template<typename TYPE>
using HostAllocator = std::allocator<TYPE>;

} // namespace AMP


namespace AMP::Utilities {
template<typename ALLOC>
constexpr AMP::Utilities::MemoryType getAllocatorMemoryType()
{
    using intAllocator = typename std::allocator_traits<ALLOC>::template rebind_alloc<int>;
    if ( std::is_same_v<intAllocator, std::allocator<int>> ) {
        return AMP::Utilities::MemoryType::host;
#ifdef AMP_USE_CUDA
    } else if ( std::is_same_v<intAllocator, AMP::CudaManagedAllocator<int>> ) {
        return AMP::Utilities::MemoryType::managed;
    } else if ( std::is_same_v<intAllocator, AMP::CudaDevAllocator<int>> ) {
        return AMP::Utilities::MemoryType::device;
#endif
#ifdef AMP_USE_HIP
    } else if ( std::is_same_v<intAllocator, AMP::HipManagedAllocator<int>> ) {
        return AMP::Utilities::MemoryType::managed;
    } else if ( std::is_same_v<intAllocator, AMP::HipDevAllocator<int>> ) {
        return AMP::Utilities::MemoryType::device;
#endif
    } else {
        AMP_ERROR( "Unknown Allocator" );
    }
}

} // namespace AMP::Utilities

#endif
