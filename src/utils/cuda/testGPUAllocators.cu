#include "AMP/utils/cuda/testGPUAllocators.hpp"
#include <cuda.h>


template<typename T>
__global__ void operateDataKernel( T *d_A )
{
    d_A[threadIdx.x] += 1;
}

template<typename T>
__global__ void setDataKernel( T *d_A, T v )
{
    d_A[threadIdx.x] = v;
}

template<typename T>
void setDataW( T *d_A, T v, size_t n )
{
    setDataKernel<T><<<1, n>>>( d_A, v );
}

template<typename T>
void opDataW( T *d_A, size_t n )
{
    operateDataKernel<T><<<1, n>>>( d_A );
}

template void opDataW<double>( double *d_A, size_t n );
template void setDataW<double>( double *d_A, double v, size_t n );

template void opDataW<float>( float *d_A, size_t n );
template void setDataW<float>( float *d_A, float v, size_t n );
