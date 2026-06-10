#ifndef included_AMP_GPUDevAllocator
#define included_AMP_GPUDevAllocator


#include <cuda.h>
#include <cuda_runtime.h>

#include "AMP/utils/cuda/Helper_Cuda.h"


namespace AMP {

/**
 * \class  CudaDevAllocator
 * @brief  Allocator based on cudaMalloc
 */
template<typename T>
class CudaDevAllocator
{
public:
    using value_type = T;

    T *allocate( size_t n )
    {
        T *ptr;
        auto err = cudaMalloc( &ptr, n * sizeof( T ) );
        checkCudaErrors( err );
        return ptr;
    }

    T *allocate( size_t n, cudaStream_t stream )
    {
        T *ptr;
        auto err = cudaMallocAsync( &ptr, n * sizeof( T ), stream );
        checkCudaErrors( err );
        return ptr;
    }

    void deallocate( T *p, size_t )
    {
        auto err = cudaFree( p );
        checkCudaErrors( err );
    }

    void deallocate( T *p, size_t, cudaStream_t stream )
    {
        auto err = cudaFreeAsync( p, stream );
        checkCudaErrors( err );
    }
};


/**
 * \class  CudaManagedAllocator
 * @brief  Allocator based on cudaMallocManaged
 */
template<typename T>
class CudaManagedAllocator
{
public:
    using value_type = T;

    T *allocate( size_t n )
    {
        T *ptr;
        auto err = cudaMallocManaged( &ptr, n * sizeof( T ), cudaMemAttachGlobal );
        checkCudaErrors( err );
        return ptr;
    }

    T *allocate( size_t n, cudaStream_t ) { return allocate( n ); }

    void deallocate( T *p, size_t )
    {
        auto err = cudaFree( p );
        checkCudaErrors( err );
    }

    void deallocate( T *p, size_t, cudaStream_t ) { deallocate( p, 0 ); }
};

} // namespace AMP


#endif
