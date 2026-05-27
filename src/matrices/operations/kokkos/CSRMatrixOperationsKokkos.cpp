#include "AMP/matrices/operations/kokkos/CSRMatrixOperationsKokkos.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/operations/kokkos/CSRLocalMatrixOperationsKokkos.hpp"
#include "AMP/utils/Memory.h"

#ifdef AMP_USE_KOKKOS

namespace AMP::LinearAlgebra {

    #define KOKKOS_INST( mode, execspace, viewspace )                                             \
        template class CSRLocalMatrixOperationsKokkos<config_mode_t<mode>, execspace, viewspace>; \
        template class CSRMatrixOperationsKokkos<config_mode_t<mode>, execspace, viewspace>;

    #ifdef AMP_USE_DEVICE
        #ifdef AMP_USE_CUDA
            #define CSR_INST( mode )                                                      \
                KOKKOS_INST( mode, Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace ) \
                KOKKOS_INST( mode, Kokkos::Cuda, Kokkos::CudaUVMSpace )                   \
                KOKKOS_INST( mode, Kokkos::Cuda, Kokkos::CudaSpace )
        #else
            #define CSR_INST( mode )                                                      \
                KOKKOS_INST( mode, Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace ) \
                KOKKOS_INST( mode, Kokkos::HIP, Kokkos::HIPManagedSpace )                 \
                KOKKOS_INST( mode, Kokkos::HIP, Kokkos::HIPSpace )
        #endif
CSR_CONFIG_FORALL( CSR_INST )
    #else
        #define CSR_INST( mode ) \
            KOKKOS_INST( mode, Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace )
CSR_CONFIG_FORALL( CSR_INST )
    #endif

    #define KOKKOS_CC_INST( mode, mode_in, execspace, viewspace )                                 \
        template void CSRMatrixOperationsKokkos<config_mode_t<mode>, execspace, viewspace>::      \
            copyCast<config_mode_t<mode_in>>( CSRMatrixData<config_mode_t<mode_in>> *,            \
                                              CSRMatrixData<config_mode_t<mode>> * );             \
        template void CSRLocalMatrixOperationsKokkos<config_mode_t<mode>, execspace, viewspace>:: \
            copyCast<config_mode_t<mode_in>>(                                                     \
                std::shared_ptr<CSRLocalMatrixData<config_mode_t<mode_in>>>,                      \
                std::shared_ptr<CSRLocalMatrixData<config_mode_t<mode>>> );

    #ifdef AMP_USE_DEVICE
        #ifdef AMP_USE_CUDA
            #define CC_INST( mode, mode_in )                                              \
                KOKKOS_CC_INST(                                                           \
                    mode, mode_in, Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace ) \
                KOKKOS_CC_INST( mode, mode_in, Kokkos::Cuda, Kokkos::CudaUVMSpace )       \
                KOKKOS_CC_INST( mode, mode_in, Kokkos::Cuda, Kokkos::CudaSpace )
        #else
            #define CC_INST( mode, mode_in )                                              \
                KOKKOS_CC_INST(                                                           \
                    mode, mode_in, Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace ) \
                KOKKOS_CC_INST( mode, mode_in, Kokkos::HIP, Kokkos::HIPManagedSpace )     \
                KOKKOS_CC_INST( mode, mode_in, Kokkos::HIP, Kokkos::HIPSpace )
        #endif
CSR_CONFIG_CC_FORALL( CC_INST )
    #else
        #define CC_INST( mode, mode_in ) \
            KOKKOS_CC_INST( mode, mode_in, Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace )
CSR_CONFIG_CC_FORALL( CC_INST )
    #endif

} // namespace AMP::LinearAlgebra
#endif
