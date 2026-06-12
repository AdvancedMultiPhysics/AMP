#ifndef included_AMP_Algorithms_hpp
#define included_AMP_Algorithms_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/extrema.h>
    #include <thrust/fill.h>
    #include <thrust/scan.h>
    #include <thrust/sort.h>
    #include <thrust/transform.h>
    #include <thrust/unique.h>
#else
    #define deviceMemcpy( ... ) AMP_ERROR( "Device memcpy without device" )
#endif

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>

namespace AMP {
namespace Utilities {

template<typename TYPE>
void Algorithms::fill_n( TYPE *x, const size_t N, const TYPE alpha, const MemoryType mem_loc )
{
    if ( N > 0 ) {
        if ( mem_loc <= MemoryType::host ) {
            std::fill_n( x, N, alpha );
        } else {
#ifdef AMP_USE_DEVICE
            thrust::fill_n( thrust::device, x, N, alpha );
#else
            AMP_ERROR( "Invalid memory type" );
#endif
        }
    }
}

template<typename TYPE>
void Algorithms::zero_n( TYPE *x, const size_t N, const MemoryType mem_loc )
{
    fill_n<TYPE>( x, N, 0, mem_loc );
}

template<typename TYPE>
void Algorithms::copy_n( TYPE *dst, const TYPE *src, const size_t N, const MemoryType mem_loc )
{
    static_assert( std::is_trivially_copyable_v<TYPE> );
    if ( mem_loc <= MemoryType::host ) {
        std::copy_n( src, N, dst );
    } else {
#ifdef AMP_USE_DEVICE
        deviceMemcpy( dst, src, N * sizeof( TYPE ), deviceMemcpyDeviceToDevice );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
}

template<typename TYPE>
void Algorithms::copy_n(
    TYPE *dst, const MemoryType dst_loc, const TYPE *src, const MemoryType src_loc, const size_t N )
{
    static_assert( std::is_trivially_copyable_v<TYPE> );

    // call single space version if possible
    if ( dst_loc == src_loc ) {
        copy_n<TYPE>( dst, src, N, src_loc );
        return;
    }

#ifdef AMP_USE_DEVICE
    if ( src_loc >= MemoryType::managed && dst_loc >= MemoryType::managed ) {
        // mixture of device and managed, do device copy
        deviceMemcpy( dst, src, N * sizeof( TYPE ), deviceMemcpyDeviceToDevice );
        return;
    } else if ( src_loc <= MemoryType::host ) {
        // src host, and by above dst must be device
        deviceMemcpy( dst, src, N * sizeof( TYPE ), deviceMemcpyHostToDevice );
        return;
    } else if ( dst_loc <= MemoryType::host ) {
        // dst host, and by above src must be device
        deviceMemcpy( dst, src, N * sizeof( TYPE ), deviceMemcpyDeviceToHost );
        return;
    }
#endif

    if ( src_loc <= MemoryType::managed && dst_loc <= MemoryType::managed ) {
        // mixture of host and managed, do host copy
        std::copy_n( src, N, dst );
        return;
    }

    AMP_ERROR( "Algorithms::copy_n: un-copyable memory locations" );
}

template<class TDst, class TSrc>
void Algorithms::copyCast(
    TDst *dst, const MemoryType dst_loc, const TSrc *src, const MemoryType src_loc, size_t N )
{
    static_assert( std::is_trivially_copyable_v<TSrc> );
    static_assert( std::is_trivially_copyable_v<TDst> );

    AMP_INSIST( src && dst, "copyCast bad buffer before" );
    if constexpr ( std::is_same_v<TSrc, TDst> ) {
        // The types are the same, fall back to simpler copy
        copy_n<TDst>( static_cast<TSrc *>( dst ), dst_loc, src, src_loc, N );
    } else if ( dst_loc <= MemoryType::host && src_loc <= MemoryType::host ) {
        // both on host, use std::transform to convert
        std::transform(
            src, src + N, dst, []( const TSrc in ) -> TDst { return static_cast<TDst>( in ); } );
    } else {
#ifdef AMP_USE_DEVICE
        // at least one is on device, ensure both available there and use thrust transform
        TDst *dst_cpy = dst;
        if ( dst_loc <= MemoryType::host ) {
            deviceMalloc( &dst_cpy, N * sizeof( TDst ) );
            copy_n<TDst>( dst_cpy, MemoryType::device, dst, MemoryType::host, N );
        }
        TSrc *src_cpy = const_cast<TSrc *>( src );
        if ( src_loc <= MemoryType::host ) {
            deviceMalloc( &src_cpy, N * sizeof( TSrc ) );
            copy_n<TSrc>( src_cpy, MemoryType::device, src, MemoryType::host, N );
        }

        // transform
        thrust::transform(
            thrust::device, src_cpy, src_cpy + N, dst_cpy, []( const TSrc in ) -> TDst {
                return static_cast<TDst>( in );
            } );

        // free if needed
        if ( dst_loc <= MemoryType::host ) {
            deviceFree( dst_cpy );
        }
        if ( src_loc <= MemoryType::host ) {
            deviceFree( src_cpy );
        }
#else
        AMP_ERROR( "Algorithms::copy_n: un-copyable memory locations" );
#endif
    }
    AMP_INSIST( src && dst, "copyCast bad buffer after" );
}

template<typename TYPE>
void Algorithms::inclusive_scan( const TYPE *x, const size_t N, TYPE *y, const MemoryType mem_loc )
{
    if ( mem_loc <= MemoryType::host ) {
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
void Algorithms::exclusive_scan(
    const TYPE *x, const size_t N, TYPE *y, TYPE alpha, const MemoryType mem_loc )
{
    if ( mem_loc <= MemoryType::host ) {
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
void Algorithms::sort( TYPE *x, const size_t N, const MemoryType mem_loc )
{
    if ( mem_loc <= MemoryType::host ) {
        std::sort( x, x + N );
    } else {
#ifdef AMP_USE_DEVICE
        thrust::sort( thrust::device, x, x + N );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
}

template<typename TYPE>
size_t Algorithms::unique( TYPE *x, const size_t N, const MemoryType mem_loc )
{
    TYPE *last = nullptr;
    if ( mem_loc <= MemoryType::host ) {
        last = std::unique( x, x + N );
    } else {
#ifdef AMP_USE_DEVICE
        last = thrust::unique( thrust::device, x, x + N );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
    std::ptrdiff_t diff = last - x;
    AMP_DEBUG_ASSERT( diff > 0 );
    return static_cast<size_t>( diff );
}

template<typename TYPE>
TYPE Algorithms::min_element( const TYPE *x, const size_t N, const MemoryType mem_loc )
{
    if ( mem_loc <= MemoryType::host ) {
        return *std::min_element( x, x + N );
    } else {
#ifdef AMP_USE_DEVICE
        return *thrust::min_element( thrust::device,
                                     thrust::device_pointer_cast( x ),
                                     thrust::device_pointer_cast( x ) + N );
#else
        AMP_ERROR( "Invalid memory type" );
        return TYPE{ 0 };
#endif
    }
}

template<typename TYPE>
TYPE Algorithms::max_element( const TYPE *x, const size_t N, const MemoryType mem_loc )
{
    if ( mem_loc <= MemoryType::host ) {
        return *std::max_element( x, x + N );
    } else {
#ifdef AMP_USE_DEVICE
        return *thrust::max_element( thrust::device,
                                     thrust::device_pointer_cast( x ),
                                     thrust::device_pointer_cast( x ) + N );
#else
        AMP_ERROR( "Invalid memory type" );
        return TYPE{ 0 };
#endif
    }
}

template<typename TYPE>
TYPE Algorithms::accumulate( const TYPE *x, const size_t N, TYPE alpha, const MemoryType mem_loc )
{
    if ( mem_loc <= MemoryType::host ) {
        return std::accumulate( x, x + N, alpha );
    } else {
#ifdef AMP_USE_DEVICE
        return thrust::reduce( thrust::device, x, x + N, alpha, thrust::plus<TYPE>() );
#else
        AMP_ERROR( "Invalid memory type" );
        return TYPE{ 0 };
#endif
    }
}

} // namespace Utilities
} // namespace AMP

#endif
