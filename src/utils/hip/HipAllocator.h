#ifndef included_AMP_GPUDevAllocator
#define included_AMP_GPUDevAllocator

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
        auto err = hipHostMalloc( &ptr, n * sizeof( T ) );
        checkHipErrors( err );
        return ptr;
    }

    T *allocate( size_t n, hipStream_t ) { return allocate( n ); }

    void deallocate( T *p, size_t )
    {
        auto err = hipFreeHost( p );
        checkHipErrors( err );
    }

    void deallocate( T *p, size_t, hipStream_t )
    {
        auto err = hipFreeHost( p );
        checkHipErrors( err );
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
        T *ptr;
        auto err = hipMalloc( &ptr, n * sizeof( T ) );
        checkHipErrors( err );
        return ptr;
    }

    T *allocate( size_t n, hipStream_t stream )
    {
        T *ptr;
        auto err = hipMallocAsync( &ptr, n * sizeof( T ), stream );
        checkHipErrors( err );
        return ptr;
    }

    void deallocate( T *p, size_t )
    {
        auto err = hipFree( p );
        checkHipErrors( err );
    }

    void deallocate( T *p, size_t, hipStream_t stream )
    {
        auto err = hipFreeAsync( p, stream );
        checkHipErrors( err );
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
        T *ptr;
        auto err = hipMallocManaged( &ptr, n * sizeof( T ), hipMemAttachGlobal );
        checkHipErrors( err );
        return ptr;
    }

    T *allocate( size_t n, hipStream_t )
    {
        T *ptr;
        auto err = hipMallocManaged( &ptr, n * sizeof( T ), hipMemAttachGlobal );
        checkHipErrors( err );
        // following will be needed some day, but is not currently functional
        // err = hipStreamAttachMemAsync( stream, (void *) ptr, n * sizeof( T ), hipMemAttachSingle
        // );
        // checkHipErrors( err );
        return ptr;
    }

    void deallocate( T *p, size_t )
    {
        auto err = hipFree( p );
        checkHipErrors( err );
    }

    void deallocate( T *p, size_t, hipStream_t )
    {
        auto err = hipFree( p );
        checkHipErrors( err );
    }
};

} // namespace AMP

#endif
