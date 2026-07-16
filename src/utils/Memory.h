#ifndef included_AMP_MEMORY
#define included_AMP_MEMORY

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/device/Device.h"

#ifdef AMP_USE_CUDA
    #include "AMP/utils/cuda/CudaAllocator.h"
#endif
#ifdef AMP_USE_HIP
    #include "AMP/utils/hip/HipAllocator.h"
#endif

#include <string_view>
#include <tuple>
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

//! Check if MemoryType is device accessible and not unregistered
std::tuple<bool, bool> memoryLocationsDeviceAccessible( const MemoryType t );

//! Check if MemoryTypes are all compatible, registered, and device-accessible
std::tuple<bool, bool> memoryLocationsDeviceAccessible( const MemoryType t1,
                                                        const MemoryType t2,
                                                        const bool check_strict = false );

//! Check if MemoryTypes are all compatible, registered, and device-accessible
std::tuple<bool, bool> memoryLocationsDeviceAccessible( const MemoryType t1,
                                                        const MemoryType t2,
                                                        const MemoryType t3,
                                                        const bool check_strict = false );

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

// host allocators
#ifdef AMP_USE_CUDA
template<typename TYPE>
using HostAllocator = AMP::CudaHostAllocator<TYPE>;
#elif defined( AMP_USE_HIP )
template<typename TYPE>
using HostAllocator = AMP::HipHostAllocator<TYPE>;
#else
template<typename T>
class HostAllocator
{
public:
    using value_type = T;

    std::allocator<T> a;

    T *allocate( size_t n ) { return a.allocate( n ); }
    T *allocate( size_t n, AMP::Utilities::ComputeStream ) { return a.allocate( n ); }
    void deallocate( T *p, size_t n ) { a.deallocate( p, n ); }
    void deallocate( T *p, size_t n, AMP::Utilities::ComputeStream ) { a.deallocate( p, n ); }
    {
        a.deallocate( p, n );
    }
};
#endif

// template<typename TYPE>
// using HostAllocator = std::allocator<TYPE>;

} // namespace AMP


namespace AMP::Utilities {
template<typename ALLOC>
constexpr AMP::Utilities::MemoryType getAllocatorMemoryType()
{
    using intAllocator = typename std::allocator_traits<ALLOC>::template rebind_alloc<int>;
    if ( std::is_same_v<intAllocator, HostAllocator<int>> ) {
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
