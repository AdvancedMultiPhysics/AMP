#ifndef included_AMP_CudaOperationsHelpers_hpp
#define included_AMP_CudaOperationsHelpers_hpp

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/inner_product.h>

namespace AMP {
namespace LinearAlgebra {


template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::setToScalar( TYPE alpha, size_t N, TYPE *x )
{
    thrust::fill_n( thrust::device, x, N, alpha );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::copy( size_t N, const TYPE *x, TYPE *y )
{
    thrust::copy_n( thrust::device, x, N, y );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::scale( TYPE alpha, size_t N, TYPE *x )
{
    auto lambda = [alpha] __host__ __device__( TYPE y ) { return y * alpha; };
    thrust::transform( thrust::device, x, x + N, x, lambda );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::scale( TYPE alpha, size_t N, const TYPE *x, TYPE *y )
{
    auto lambda = [alpha] __host__ __device__( TYPE x ) { return x * alpha; };
    thrust::transform( thrust::device, x, x + N, y, lambda );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::add( size_t N, const TYPE *x, const TYPE *y, TYPE *z )
{
    thrust::transform( thrust::device, x, x + N, y, z, thrust::plus<TYPE>() );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::subtract( size_t N, const TYPE *x, const TYPE *y, TYPE *z )
{
    thrust::transform( thrust::device, x, x + N, y, z, thrust::minus<TYPE>() );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::multiply( size_t N, const TYPE *x, const TYPE *y, TYPE *z )
{
    thrust::transform( thrust::device, x, x + N, y, z, thrust::multiplies<TYPE>() );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::divide( size_t N, const TYPE *x, const TYPE *y, TYPE *z )
{
    thrust::transform( thrust::device, x, x + N, y, z, thrust::divides<TYPE>() );
}


template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::reciprocal( size_t N, const TYPE *x, TYPE *y )
{
    auto lambda = [] __host__ __device__( TYPE x ) { return (TYPE) 1 / x; };
    thrust::transform( thrust::device, x, x + N, y, lambda );
}


template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::linearSum(
    TYPE alpha, size_t N, const TYPE *x, TYPE beta, const TYPE *y, TYPE *z )
{
    auto lambda = [alpha, beta] __host__ __device__( TYPE x, TYPE y ) {
        return alpha * x + beta * y;
    };
    thrust::transform( thrust::device, x, x + N, y, z, lambda );
}


template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::abs( size_t N, const TYPE *x, TYPE *y )
{
    auto lambda = [] __host__ __device__( TYPE x ) { return x < 0 ? -x : x; };
    thrust::transform( thrust::device, x, x + N, y, lambda );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::addScalar( size_t N, const TYPE *x, TYPE alpha, TYPE *y )
{
    auto lambda = [alpha] __host__ __device__( TYPE x ) { return x + alpha; };
    thrust::transform( thrust::device, x, x + N, y, lambda );
}

template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localMin( size_t N, const TYPE *x )
{
    return thrust::reduce(
        thrust::device, x, x + N, std::numeric_limits<TYPE>::max(), thrust::minimum<TYPE>() );
}

template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localMax( size_t N, const TYPE *x )
{
    auto lambda = [=] __host__ __device__( TYPE x ) { return x; };
    return thrust::transform_reduce(
        thrust::device, x, x + N, lambda, (TYPE) 0, thrust::maximum<TYPE>() );
}


template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localSum( size_t N, const TYPE *x )
{
    return thrust::reduce( thrust::device, x, x + N, 0, thrust::plus<TYPE>() );
}

template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localL1Norm( size_t N, const TYPE *x )
{
    auto lambda = [=] __host__ __device__( TYPE x ) { return x < 0 ? -x : x; };
    return thrust::transform_reduce(
        thrust::device, x, x + N, lambda, (TYPE) 0, thrust::plus<TYPE>() );
}

template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localL2Norm( size_t N, const TYPE *x )
{
    auto lambda = [=] __host__ __device__( TYPE x ) { return x * x; };
    auto result = thrust::transform_reduce(
        thrust::device, x, x + N, lambda, (TYPE) 0, thrust::plus<TYPE>() );
    return sqrt( result );
}

template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localMaxNorm( size_t N, const TYPE *x )
{
    auto lambda = [=] __host__ __device__( TYPE x ) { return x < 0 ? -x : x; };
    return thrust::transform_reduce(
        thrust::device, x, x + N, lambda, (TYPE) 0, thrust::maximum<TYPE>() );
}

template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localDot( size_t N, const TYPE *x, const TYPE *y )
{
    return thrust::inner_product( thrust::device, x, x + N, y, (TYPE) 0 );
}

template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localMinQuotient( size_t N, const TYPE *x, const TYPE *y )
{
    return thrust::inner_product( thrust::device,
                                  x,
                                  x + N,
                                  y,
                                  std::numeric_limits<TYPE>::max(),
                                  thrust::minimum<TYPE>(),
                                  thrust::divides<TYPE>() );
}
template<typename T>
struct thrust_wrs {
    typedef T first_argument_type;
    typedef T second_argument_type;
    typedef T result_type;
    __host__ __device__ T operator()( const T &x, const T &y ) const { return x * x * y * y; }
};

template<typename TYPE>
TYPE DeviceOperationsHelpers<TYPE>::localWrmsNorm( size_t N, const TYPE *x, const TYPE *y )
{
    return thrust::inner_product(
        thrust::device, x, x + N, y, 0, thrust::plus<TYPE>(), thrust_wrs<TYPE>() );
}


} // namespace LinearAlgebra
} // namespace AMP

#endif
