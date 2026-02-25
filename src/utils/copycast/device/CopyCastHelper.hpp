#ifndef included_AMP_DevCopyCast_HPP_
#define included_AMP_DevCopyCast_HPP_

#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"
#include <memory>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/mr/allocator.h>
#include <thrust/transform_reduce.h>

#include <iostream>

namespace AMP::Utilities {

template<typename T1, typename T2>
struct copyCast_<T1, T2, AMP::Utilities::Backend::Hip_Cuda, AMP::HostAllocator<void>> {
    void static apply( const size_t len, const T1 *vec_in, T2 *vec_out )
    {
        auto lambda = [] __host__ __device__( T1 x ) { return static_cast<T2>( x ); };
        thrust::transform( thrust::host, vec_in, vec_in + len, vec_out, lambda );
    }
};

template<typename T1, typename T2>
struct copyCast_<T1, T2, AMP::Utilities::Backend::Hip_Cuda, AMP::ManagedAllocator<void>> {
    void static apply( const size_t len, const T1 *vec_in, T2 *vec_out )
    {
        auto lambda = [] __host__ __device__( T1 x ) { return static_cast<T2>( x ); };
        thrust::transform( thrust::device, vec_in, vec_in + len, vec_out, lambda );
    }
};

template<typename T1, typename T2>
struct copyCast_<T1, T2, AMP::Utilities::Backend::Hip_Cuda, AMP::DeviceAllocator<void>> {
    void static apply( const size_t len, const T1 *vec_in, T2 *vec_out )
    {
        auto lambda = [] __host__ __device__( T1 x ) { return static_cast<T2>( x ); };
        thrust::transform( thrust::device, vec_in, vec_in + len, vec_out, lambda );
    }
};

} // namespace AMP::Utilities

// #include "CopyCast.hpp"

#endif
