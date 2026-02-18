#ifndef included_AMP_DeviceOperationsHelpers_hpp
#define included_AMP_DeviceOperationsHelpers_hpp

#include "AMP/utils/device/Device.h"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

namespace AMP {
namespace LinearAlgebra {

// the random value part suggested from google AI
struct GenRand {
    __host__ __device__ float operator()( const size_t idx ) const
    {
        thrust::random::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist( 0.0f, 1.0f ); // Example: range [0.0, 1.0)

        // Use the index to seed the engine for unique sequences
        randEng.discard( idx );
        return uniDist( randEng );
    }
};


template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::setRandomValues( size_t N, TYPE *x )
{
    thrust::counting_iterator<unsigned int> index_begin( 0 );
    thrust::transform( thrust::device, index_begin, index_begin + N, x, GenRand() );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::scale( TYPE alpha, size_t N, TYPE *x )
{
    thrust::transform( thrust::device, x, x + N, x, alpha * thrust::placeholders::_1 );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::scale( TYPE alpha, size_t N, const TYPE *x, TYPE *y )
{
    thrust::transform( thrust::device, x, x + N, y, alpha * thrust::placeholders::_1 );
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
    thrust::transform(
        thrust::device, x, x + N, y, static_cast<TYPE>( 1.0 ) / thrust::placeholders::_1 );
}


template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::linearSum(
    const TYPE alpha, size_t N, const TYPE *x, const TYPE beta, const TYPE *y, TYPE *z )
{
    if ( alpha == 1.0 && beta == 1.0 ) {
        thrust::transform( thrust::device, x, x + N, y, z, thrust::plus() );
    } else if ( alpha == 1.0 ) {
        thrust::transform( thrust::device,
                           x,
                           x + N,
                           y,
                           z,
                           thrust::placeholders::_1 + beta * thrust::placeholders::_2 );
    } else if ( beta == 1.0 ) {
        thrust::transform( thrust::device,
                           x,
                           x + N,
                           y,
                           z,
                           alpha * thrust::placeholders::_1 + thrust::placeholders::_2 );
    } else {
        thrust::transform( thrust::device,
                           x,
                           x + N,
                           y,
                           z,
                           alpha * thrust::placeholders::_1 + beta * thrust::placeholders::_2 );
    }
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
    thrust::transform( thrust::device, x, x + N, y, thrust::placeholders::_1 + alpha );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::setMin( size_t N, TYPE alpha, TYPE *x )
{
    auto lambda = [alpha] __host__ __device__( TYPE x ) { return x < alpha ? alpha : x; };
    thrust::transform( thrust::device, x, x + N, x, lambda );
}

template<typename TYPE>
void DeviceOperationsHelpers<TYPE>::setMax( size_t N, TYPE alpha, TYPE *x )
{
    auto lambda = [alpha] __host__ __device__( TYPE x ) { return x > alpha ? alpha : x; };
    thrust::transform( thrust::device, x, x + N, x, lambda );
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
    return thrust::reduce( thrust::device, x, x + N, (TYPE) 0, thrust::plus<TYPE>() );
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
    return std::sqrt( result );
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
        thrust::device, x, x + N, y, (TYPE) 0, thrust::plus<TYPE>(), thrust_wrs<TYPE>() );
}


} // namespace LinearAlgebra
} // namespace AMP

#endif
