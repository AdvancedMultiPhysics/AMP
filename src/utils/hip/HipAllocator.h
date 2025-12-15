#ifndef included_AMP_GPUDevAllocator
#define included_AMP_GPUDevAllocator

#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/hip/Helper_Hip.h"

DISABLE_WARNINGS
#include <hip/hip_runtime.h>
ENABLE_WARNINGS

namespace AMP {

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

    void deallocate( T *p, size_t )
    {
        auto err = hipFree( p );
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

    void deallocate( T *p, size_t )
    {
        auto err = hipFree( p );
        checkHipErrors( err );
    }
};

} // namespace AMP

#endif
