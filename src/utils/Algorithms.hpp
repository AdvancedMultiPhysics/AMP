#ifndef included_AMP_Algorithms_hpp
#define included_AMP_Algorithms_hpp

#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"

#ifdef USE_DEVICE
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/extrema.h>
    #include <thrust/for_each.h>
    #include <thrust/inner_product.h>
    #include <thrust/scan.h>
#endif

#include <algorithm>
#include <numeric>

namespace AMP {
namespace Utilities {

template<typename TYPE>
void Algorithms<TYPE>::fill_n( TYPE *x, const size_t N, const TYPE alpha )
{
    if ( getMemoryType( x ) < MemoryType::device ) {
        std::fill( x, x + N, alpha );
    } else {
#ifdef USE_DEVICE
        thrust::fill_n( thrust::device, x, N, alpha );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
}

template<typename TYPE>
void Algorithms<TYPE>::copy_n( TYPE *x, const size_t N, TYPE *y )
{
    if ( getMemoryType( x ) < MemoryType::device ) {
        std::copy( x, x + N, y );
    } else {
#ifdef USE_DEVICE
        thrust::copy_n( thrust::device, x, N, y );
#else
        AMP_ERROR( "Invalid memory type" );
#endif
    }
}

template<typename TYPE>
void Algorithms<TYPE>::inclusive_scan( TYPE *x, const size_t N, TYPE *y )
{
    if ( getMemoryType( x ) < MemoryType::device ) {
        std::inclusive_scan( x, x + N, y );
    } else {
#ifdef USE_DEVICE
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
#ifdef USE_DEVICE
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
#ifdef USE_DEVICE
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
#ifdef USE_DEVICE
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
