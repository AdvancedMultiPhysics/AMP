#include "AMP/matrices/operations/default/CSRMatrixOperationsDefault.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRPolicy.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/operations/default/CSRLocalMatrixOperationsDefault.hpp"
#include "AMP/matrices/operations/default/spgemm/CSRMatrixSpGEMMDefault.hpp"
#include "AMP/utils/memory.h"

#define INSTANTIATE_FULL( policy, allocator )                           \
    template class AMP::LinearAlgebra::CSRLocalMatrixOperationsDefault< \
        policy,                                                         \
        allocator,                                                      \
        AMP::LinearAlgebra::CSRLocalMatrixData<policy, allocator>>;     \
    template class AMP::LinearAlgebra::CSRMatrixOperationsDefault<      \
        policy,                                                         \
        allocator,                                                      \
        AMP::LinearAlgebra::CSRLocalMatrixData<policy, allocator>>;     \
    template class AMP::LinearAlgebra::CSRMatrixSpGEMMHelperDefault<    \
        policy,                                                         \
        allocator,                                                      \
        AMP::LinearAlgebra::CSRLocalMatrixData<policy, allocator>>;

// Check if device based allocators are needed
#ifdef USE_DEVICE
    #define INSTANTIATE_ALLOCS( policy )                        \
        INSTANTIATE_FULL( policy, AMP::HostAllocator<void> )    \
        INSTANTIATE_FULL( policy, AMP::ManagedAllocator<void> ) \
        INSTANTIATE_FULL( policy, AMP::DeviceAllocator<void> )
#else
    #define INSTANTIATE_ALLOCS( policy ) INSTANTIATE_FULL( policy, AMP::HostAllocator<void> )
#endif

using CSRPolicyFloat  = AMP::LinearAlgebra::CSRPolicy<size_t, int, float>;
using CSRPolicyDouble = AMP::LinearAlgebra::CSRPolicy<size_t, int, double>;
INSTANTIATE_ALLOCS( CSRPolicyDouble )
INSTANTIATE_ALLOCS( CSRPolicyFloat )
// Check if hyzpre is present
#if defined( AMP_USE_HYPRE )
    #include "HYPRE_utilities.h"
using HYPRECSRPolicyFloat  = AMP::LinearAlgebra::CSRPolicy<HYPRE_BigInt, HYPRE_Int, double>;
using HYPRECSRPolicyDouble = AMP::LinearAlgebra::CSRPolicy<HYPRE_BigInt, HYPRE_Int, float>;
INSTANTIATE_ALLOCS( HYPRECSRPolicyDouble )
INSTANTIATE_ALLOCS( HYPRECSRPolicyFloat )
#endif


// This chain of macros is best read from bottom to top

// Full instantiator macro is final one in the chain
// this gives all template arguments
#define INSTANTIATE_CC_FULL( policy, policyIn, allocator )                                  \
    template void AMP::LinearAlgebra::CSRMatrixOperationsDefault<                           \
        policy,                                                                             \
        allocator,                                                                          \
        AMP::LinearAlgebra::CSRLocalMatrixData<policy, allocator>>::                        \
        copyCast<policyIn>(                                                                 \
            CSRMatrixData<policyIn,                                                         \
                          allocator,                                                        \
                          AMP::LinearAlgebra::CSRLocalMatrixData<policyIn, allocator>> *    \
                X,                                                                          \
            CSRMatrixData<policy,                                                           \
                          allocator,                                                        \
                          AMP::LinearAlgebra::CSRLocalMatrixData<policy, allocator>> *      \
                Y );                                                                        \
    template void AMP::LinearAlgebra::CSRLocalMatrixOperationsDefault<                      \
        policy,                                                                             \
        allocator,                                                                          \
        AMP::LinearAlgebra::CSRLocalMatrixData<policy, allocator>>::                        \
        copyCast<policyIn>(                                                                 \
            std::shared_ptr<AMP::LinearAlgebra::CSRLocalMatrixData<policyIn, allocator>> X, \
            std::shared_ptr<AMP::LinearAlgebra::CSRLocalMatrixData<policy, allocator>> Y );

// Allocator instatiator is responsible for adding all supported
// allocator types for a given policy and forwarding along
#ifdef USE_DEVICE
    #define INSTANTIATE_CC_ALLOCS( policy, policyIn )                        \
        INSTANTIATE_CC_FULL( policy, policyIn, AMP::HostAllocator<void> )    \
        INSTANTIATE_CC_FULL( policy, policyIn, AMP::ManagedAllocator<void> ) \
        INSTANTIATE_CC_FULL( policy, policyIn, AMP::DeviceAllocator<void> )
#else
    #define INSTANTIATE_CC_ALLOCS( policy, policyIn ) \
        INSTANTIATE_CC_FULL( policy, policyIn, AMP::HostAllocator<void> )
#endif

// Policy instantiator starts the chain and forwards
// all known policies to the next instantiator
INSTANTIATE_CC_ALLOCS( CSRPolicyFloat, CSRPolicyDouble )
INSTANTIATE_CC_ALLOCS( CSRPolicyDouble, CSRPolicyFloat )
INSTANTIATE_CC_ALLOCS( CSRPolicyFloat, CSRPolicyFloat )
INSTANTIATE_CC_ALLOCS( CSRPolicyDouble, CSRPolicyDouble )
#ifdef AMP_USE_HYPRE
INSTANTIATE_CC_ALLOCS( HYPRECSRPolicyFloat, HYPRECSRPolicyDouble )
INSTANTIATE_CC_ALLOCS( HYPRECSRPolicyDouble, HYPRECSRPolicyFloat )
INSTANTIATE_CC_ALLOCS( HYPRECSRPolicyDouble, HYPRECSRPolicyDouble )
INSTANTIATE_CC_ALLOCS( HYPRECSRPolicyFloat, HYPRECSRPolicyFloat )
#endif
