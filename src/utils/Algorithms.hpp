#ifndef included_AMP_Algorithms_hpp
#define included_AMP_Algorithms_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/extrema.h>
    #include <thrust/for_each.h>
    #include <thrust/inner_product.h>
    #include <thrust/scan.h>
#else
    #define deviceMemcpy( ... ) AMP_ERROR( "Device memcpy without device" )
#endif

#include <algorithm>
#include <cstring>
#include <numeric>

namespace AMP {
namespace Utilities {

template<typename TYPE>
void Algorithms<TYPE>::fill_n( TYPE *x, const size_t N, const TYPE alpha )
{
    if ( getMemoryType( x ) < MemoryType::device ) {
        std::fill( x, x + N, alpha );
    } else {
#ifdef AMP_USE_DEVICE
        thrust::fill_n( thrust::device, x, N, alpha );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
}

template<typename TYPE>
void Algorithms<TYPE>::copy_n( const TYPE *x, const size_t N, TYPE *y )
{
    const auto xmtype = getMemoryType( x );
    const auto ymtype = getMemoryType( y );
    AMP_DEBUG_ASSERT( xmtype != MemoryType::none && ymtype != MemoryType::none );
    if ( xmtype == MemoryType::managed && ymtype == MemoryType::managed ) {
        // managed-managed operations can use device or CPU
#ifdef AMP_USE_DEVICE
        deviceMemcpy( y, x, N * sizeof( TYPE ), deviceMemcpyDeviceToDevice );
#else
        memcpy( y, x, N * sizeof( TYPE ) );
#endif
    } else if ( xmtype <= MemoryType::managed && ymtype <= MemoryType::managed ) {
        // host-host
        memcpy( y, x, N * sizeof( TYPE ) );
    } else if ( ymtype <= MemoryType::host ) {
        // device to host
        deviceMemcpy( y, x, N * sizeof( TYPE ), deviceMemcpyDeviceToHost );
    } else if ( xmtype <= MemoryType::host ) {
        // host to device
        deviceMemcpy( y, x, N * sizeof( TYPE ), deviceMemcpyHostToDevice );
    } else {
        // device to device
        deviceMemcpy( y, x, N * sizeof( TYPE ), deviceMemcpyDeviceToDevice );
    }
}

template<typename TYPE>
void Algorithms<TYPE>::inclusive_scan( TYPE *x, const size_t N, TYPE *y )
{
    if ( getMemoryType( x ) < MemoryType::device ) {
        std::inclusive_scan( x, x + N, y );
    } else {
#ifdef AMP_USE_DEVICE
        thrust::inclusive_scan( thrust::device, x, x + N, y );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
}

template<typename TYPE>
void Algorithms<TYPE>::exclusive_scan( TYPE *x, const size_t N, TYPE *y, TYPE alpha )
{
    if ( getMemoryType( x ) < MemoryType::device ) {
        std::exclusive_scan( x, x + N, y, alpha );
    } else {
#ifdef AMP_USE_DEVICE
        thrust::exclusive_scan( thrust::device, x, x + N, y, alpha );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
}

template<typename TYPE>
TYPE Algorithms<TYPE>::max_element( TYPE *x, const size_t N )
{
    if ( getMemoryType( x ) < MemoryType::device ) {
        return *std::max_element( x, x + N );
    } else {
#ifdef AMP_USE_DEVICE
        return *thrust::max_element( thrust::device, x, x + N );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
    return TYPE{ 0 };
}

template<typename TYPE>
TYPE Algorithms<TYPE>::accumulate( TYPE *x, const size_t N, TYPE alpha )
{
    if ( getMemoryType( x ) < MemoryType::device ) {
        return std::accumulate( x, x + N, alpha );
    } else {
#ifdef AMP_USE_DEVICE
        return thrust::reduce( thrust::device, x, x + N, alpha, thrust::plus<TYPE>() );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
    return TYPE{ 0 };
}

} // namespace Utilities
} // namespace AMP

#endif
