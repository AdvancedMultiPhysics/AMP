#ifndef included_AMP_GPUDevAllocator
#define included_AMP_GPUDevAllocator


#include <cuda.h>
#include <cuda_runtime.h>

#include "AMP/utils/cuda/Helper_Cuda.h"


namespace AMP {

/**
 * \class  CudaHostAllocator
 * @brief  Allocator based on cudaMallocHost
 */
template<typename T>
class CudaHostAllocator
{
public:
    using value_type = T;

    T *allocate( size_t n )
    {
        AMP_WARNING( "non-stream aware: dev alloc" );
        T *ptr;
        auto err = cudaMallocHost( &ptr, n * sizeof( T ) );
        checkCudaErrors( err );
        return ptr;
    }

    T *allocate( size_t n, cudaStream_t ) { return allocate( n ); }

    T *allocate( size_t n, const AMP::Utilities::AccelerationContext & ) { return allocate( n ); }

    void deallocate( T *p, size_t )
    {
        AMP_WARNING( "non-stream aware: dev dealloc" );
        auto err = cudaFreeHost( p );
        checkCudaErrors( err );
    }

    void deallocate( T *p, size_t n, cudaStream_t ) { deallocate( p, n ); }

    void deallocate( T *p, size_t n, const AMP::Utilities::AccelerationContext & )
    {
        deallocate( p, n );
    }
};

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
        AMP_WARNING( "non-stream aware: dev alloc" );
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

    T *allocate( size_t n, const AMP::Utilities::AccelerationContext &ctx )
    {
        return allocate( n, ctx.getStream() );
    }

    void deallocate( T *p, size_t )
    {
        AMP_WARNING( "non-stream aware: dev dealloc" );
        auto err = cudaFree( p );
        checkCudaErrors( err );
    }

    void deallocate( T *p, size_t, cudaStream_t stream )
    {
        auto err = cudaFreeAsync( p, stream );
        checkCudaErrors( err );
    }

    void deallocate( T *p, size_t n, const AMP::Utilities::AccelerationContext &ctx )
    {
        deallocate( p, n, ctx.getStream() );
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
        AMP_WARNING( "non-stream aware: managed alloc" );
        T *ptr;
        auto err = cudaMallocManaged( &ptr, n * sizeof( T ), cudaMemAttachGlobal );
        checkCudaErrors( err );
        return ptr;
    }

    T *allocate( size_t n, cudaStream_t )
    {
        T *ptr;
        auto err = cudaMallocManaged( &ptr, n * sizeof( T ), cudaMemAttachGlobal );
        checkCudaErrors( err );
        // following will be needed some day, but is not currently functional
        // err =
        //     cudaStreamAttachMemAsync( stream, (void *) ptr, n * sizeof( T ), cudaMemAttachSingle
        //     );
        // checkCudaErrors( err );
        return ptr;
    }

    T *allocate( size_t n, const AMP::Utilities::AccelerationContext &ctx )
    {
        return allocate( n, ctx.getStream() );
    }

    void deallocate( T *p, size_t )
    {
        AMP_WARNING( "non-stream aware: managed dealloc" );
        auto err = cudaFree( p );
        checkCudaErrors( err );
    }

    void deallocate( T *p, size_t, cudaStream_t )
    {
        auto err = cudaFree( p );
        checkCudaErrors( err );
    }

    void deallocate( T *p, size_t n, const AMP::Utilities::AccelerationContext &ctx )
    {
        deallocate( p, n, ctx.getStream() );
    }
};

} // namespace AMP


#endif
