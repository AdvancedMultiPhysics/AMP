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
void Algorithms<TYPE>::fill_n( TYPE *x, const size_t N, const TYPE alpha )
{
    if ( N > 0 ) {
        if ( getMemoryType( x ) <= MemoryType::host ) {
            std::fill( x, x + N, alpha );
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
void Algorithms<TYPE>::copy_n( const TYPE *x, const size_t N, TYPE *y )
{
    static_assert( std::is_trivially_copyable_v<TYPE> );
    AMP::Utilities::memcpy( y, x, N * sizeof( TYPE ) );
}

template<typename TYPE>
void Algorithms<TYPE>::inclusive_scan( const TYPE *x, const size_t N, TYPE *y )
{
    if ( getMemoryType( x ) <= MemoryType::host ) {
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
void Algorithms<TYPE>::exclusive_scan( const TYPE *x, const size_t N, TYPE *y, TYPE alpha )
{
    if ( getMemoryType( x ) <= MemoryType::host ) {
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
void Algorithms<TYPE>::sort( TYPE *x, const size_t N )
{
#ifndef AMP_USE_DEVICE
    std::sort( x, x + N );
#else
    if ( getMemoryType( x ) <= MemoryType::host ) {
        std::sort( x, x + N );
    } else {
        thrust::sort( thrust::device, x, x + N );
    }
#endif
}

template<typename TYPE>
size_t Algorithms<TYPE>::unique( TYPE *x, const size_t N )
{
    TYPE *last = nullptr;
#ifndef AMP_USE_DEVICE
    last = std::unique( x, x + N );
#else
    if ( getMemoryType( x ) <= MemoryType::host ) {
        last = std::unique( x, x + N );
    } else {
        last = thrust::unique( thrust::device, x, x + N );
    }
#endif
    std::ptrdiff_t diff = last - x;
    AMP_DEBUG_ASSERT( diff > 0 );
    return static_cast<size_t>( diff );
}

template<typename TYPE>
TYPE Algorithms<TYPE>::min_element( const TYPE *x, const size_t N )
{
    if ( getMemoryType( x ) <= MemoryType::host ) {
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
TYPE Algorithms<TYPE>::max_element( const TYPE *x, const size_t N )
{
    if ( getMemoryType( x ) <= MemoryType::host ) {
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
TYPE Algorithms<TYPE>::accumulate( const TYPE *x, const size_t N, TYPE alpha )
{
    if ( getMemoryType( x ) <= MemoryType::host ) {
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
