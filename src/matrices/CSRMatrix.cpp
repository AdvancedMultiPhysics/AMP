#include "AMP/matrices/CSRMatrix.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRPolicy.h"
#include "AMP/utils/memory.h"

#define INSTANTIATE_FULL( policy, allocator ) \
    template class AMP::LinearAlgebra::CSRMatrix<policy, allocator>;

// Check if device based allocators are needed
#ifdef USE_DEVICE
    #define INSTANTIATE_ALLOCS( policy )                        \
        INSTANTIATE_FULL( policy, AMP::HostAllocator<void> )    \
        INSTANTIATE_FULL( policy, AMP::ManagedAllocator<void> ) \
        INSTANTIATE_FULL( policy, AMP::DeviceAllocator<void> )
#else
    #define INSTANTIATE_ALLOCS( policy ) INSTANTIATE_FULL( policy, AMP::HostAllocator<void> )
#endif

using CSRPolicyDouble = AMP::LinearAlgebra::CSRPolicy<size_t, int, double>;
using CSRPolicyFloat  = AMP::LinearAlgebra::CSRPolicy<size_t, int, float>;
INSTANTIATE_ALLOCS( CSRPolicyDouble )
INSTANTIATE_ALLOCS( CSRPolicyFloat )
// Check if hypre is present
#if defined( AMP_USE_HYPRE )
    #include "HYPRE_utilities.h"
using HYPRECSRPolicyFloat  = AMP::LinearAlgebra::CSRPolicy<HYPRE_BigInt, HYPRE_Int, double>;
using HYPRECSRPolicyDouble = AMP::LinearAlgebra::CSRPolicy<HYPRE_BigInt, HYPRE_Int, float>;
INSTANTIATE_ALLOCS( HYPRECSRPolicyDouble )
INSTANTIATE_ALLOCS( HYPRECSRPolicyFloat )
#endif
