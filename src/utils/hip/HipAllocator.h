#ifndef included_AMP_GPUDevAllocator
#define included_AMP_GPUDevAllocator

#include "AMP/utils/AccelerationContext.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/hip/Helper_Hip.h"

DISABLE_WARNINGS
#include <hip/hip_runtime.h>
ENABLE_WARNINGS

namespace AMP {

/**
 * \class  HipHostAllocator
 * @brief  Allocator based on hipMallocHost
 */
template<typename T>
class HipHostAllocator
{
public:
    using value_type = T;

    T *allocate( size_t n )
    {
        T *ptr;
        auto err = hipHostMalloc( (void **) &ptr, n * sizeof( T ) );
        checkHipErrors( err );
        return ptr;
    }

    T *allocate( size_t n, hipStream_t ) { return allocate( n ); }

    T *allocate( size_t n, const AMP::Utilities::AccelerationContext & ) { return allocate( n ); }

    void deallocate( T *p, size_t )
    {
        auto err = hipFreeHost( (void *) p );
        checkHipErrors( err );
    }

    void deallocate( T *p, size_t n, hipStream_t ) { deallocate( p, n ); }

    void deallocate( T *p, size_t n, const AMP::Utilities::AccelerationContext & )
    {
        deallocate( p, n );
    }
};

/**
 * \class  HipDevAllocator
 * @brief  Allocator based on hipMalloc
 */
template<typename T>
class HipDevAllocator
{
public:
    using value_type = T;

    T *allocate( size_t n )
    {
        AMP_WARNING( "non-stream aware: dev alloc" );
        T *ptr;
        auto err = hipMalloc( (void **) &ptr, n * sizeof( T ) );
        checkHipErrors( err );
        return ptr;
    }

    T *allocate( size_t n, hipStream_t stream )
    {
        T *ptr;
        auto err = hipMallocAsync( (void **) &ptr, n * sizeof( T ), stream );
        checkHipErrors( err );
        return ptr;
    }

    T *allocate( size_t n, const AMP::Utilities::AccelerationContext &ctx )
    {
        return allocate( n, ctx.getStream() );
    }

    void deallocate( T *p, size_t )
    {
        AMP_WARNING( "non-stream aware: dev dealloc" );
        auto err = hipFree( (void *) p );
        checkHipErrors( err );
    }

    void deallocate( T *p, size_t, hipStream_t stream )
    {
        AMP_ASSERT( p );
        auto err = hipFreeAsync( (void *) p, stream );
        checkHipErrors( err );
    }

    void deallocate( T *p, size_t n, const AMP::Utilities::AccelerationContext &ctx )
    {
        deallocate( p, n, ctx.getStream() );
    }
};

/**
 * \class  HipManagedAllocator
 * @brief  Allocator based on hipMallocManaged
 */
template<typename T>
class HipManagedAllocator
{
public:
    using value_type = T;

    T *allocate( size_t n )
    {
        AMP_WARNING( "non-stream aware: managed alloc" );
        T *ptr;
        auto err = hipMallocManaged( (void **) &ptr, n * sizeof( T ), hipMemAttachGlobal );
        checkHipErrors( err );
        return ptr;
    }

    T *allocate( size_t n, hipStream_t )
    {
        T *ptr;
        auto err = hipMallocManaged( (void **) &ptr, n * sizeof( T ), hipMemAttachGlobal );
        checkHipErrors( err );
        // following will be needed some day, but is not currently functional
        // err = hipStreamAttachMemAsync( stream, (void *) ptr, n * sizeof( T ), hipMemAttachSingle
        // );
        // checkHipErrors( err );
        return ptr;
    }

    T *allocate( size_t n, const AMP::Utilities::AccelerationContext &ctx )
    {
        return allocate( n, ctx.getStream() );
    }

    void deallocate( T *p, size_t )
    {
        AMP_WARNING( "non-stream aware: managed dealloc" );
        auto err = hipFree( (void *) p );
        checkHipErrors( err );
    }

    void deallocate( T *p, size_t, hipStream_t )
    {
        auto err = hipFree( (void *) p );
        checkHipErrors( err );
    }

    void deallocate( T *p, size_t n, const AMP::Utilities::AccelerationContext &ctx )
    {
        deallocate( p, n, ctx.getStream() );
    }
};

} // namespace AMP

#endif
