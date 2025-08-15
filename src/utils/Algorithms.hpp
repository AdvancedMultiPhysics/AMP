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
    if ( N > 0 ) {
        if ( getMemoryType( x ) < MemoryType::managed ) {
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
    if ( getMemoryType( x ) < MemoryType::managed ) {
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
    if ( getMemoryType( x ) < MemoryType::managed ) {
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
TYPE Algorithms<TYPE>::max_element( const TYPE *x, const size_t N )
{
    if ( getMemoryType( x ) < MemoryType::managed ) {
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
TYPE Algorithms<TYPE>::accumulate( const TYPE *x, const size_t N, TYPE alpha )
{
    if ( getMemoryType( x ) < MemoryType::managed ) {
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
