#include "AMP/matrices/operations/kokkos/CSRMatrixOperationsKokkos.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/operations/kokkos/CSRLocalMatrixOperationsKokkos.hpp"
#include "AMP/utils/Memory.h"

#ifdef AMP_USE_KOKKOS

namespace AMP::LinearAlgebra {

    #define KOKKOS_INST( mode, execspace )                                             \
        template class CSRLocalMatrixOperationsKokkos<config_mode_t<mode>, execspace>; \
        template class CSRMatrixOperationsKokkos<config_mode_t<mode>, execspace>;

    #ifdef AMP_USE_DEVICE
        #ifdef AMP_USE_CUDA
            #define CSR_INST( mode )                                   \
                KOKKOS_INST( mode, Kokkos::DefaultHostExecutionSpace ) \
                KOKKOS_INST( mode, Kokkos::Cuda )
        #else
            #define CSR_INST( mode )                                   \
                KOKKOS_INST( mode, Kokkos::DefaultHostExecutionSpace ) \
                KOKKOS_INST( mode, Kokkos::HIP )
        #endif
CSR_CONFIG_FORALL( CSR_INST )
    #else
        #define CSR_INST( mode ) KOKKOS_INST( mode, Kokkos::DefaultHostExecutionSpace )
CSR_CONFIG_FORALL( CSR_INST )
    #endif

    #define KOKKOS_CC_INST( mode, mode_in, execspace )                                            \
        template void CSRMatrixOperationsKokkos<config_mode_t<mode>, execspace>::copyCast<        \
            config_mode_t<mode_in>>( CSRMatrixData<config_mode_t<mode_in>> *,                     \
                                     CSRMatrixData<config_mode_t<mode>> * );                      \
        template void CSRLocalMatrixOperationsKokkos<config_mode_t<mode>, execspace>::copyCast<   \
            config_mode_t<mode_in>>( std::shared_ptr<CSRLocalMatrixData<config_mode_t<mode_in>>>, \
                                     std::shared_ptr<CSRLocalMatrixData<config_mode_t<mode>>> );

    #ifdef AMP_USE_DEVICE
        #ifdef AMP_USE_CUDA
            #define CC_INST( mode, mode_in )                                       \
                KOKKOS_CC_INST( mode, mode_in, Kokkos::DefaultHostExecutionSpace ) \
                KOKKOS_CC_INST( mode, mode_in, Kokkos::Cuda )
        #else
            #define CC_INST( mode, mode_in )                                       \
                KOKKOS_CC_INST( mode, mode_in, Kokkos::DefaultHostExecutionSpace ) \
                KOKKOS_CC_INST( mode, mode_in, Kokkos::HIP )
        #endif
CSR_CONFIG_CC_FORALL( CC_INST )
    #else
        #define CC_INST( mode, mode_in ) \
            KOKKOS_CC_INST( mode, mode_in, Kokkos::DefaultHostExecutionSpace )
CSR_CONFIG_CC_FORALL( CC_INST )
    #endif

} // namespace AMP::LinearAlgebra
#endif
