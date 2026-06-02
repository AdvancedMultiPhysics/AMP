#include "AMP/matrices/operations/kokkos/CSRMatrixOperationsKokkos.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/operations/kokkos/CSRLocalMatrixOperationsKokkos.hpp"
#include "AMP/utils/Memory.h"

#ifdef AMP_USE_KOKKOS

namespace AMP::LinearAlgebra {

    #define KOKKOS_INST( mode )                                             \
        template class CSRLocalMatrixOperationsKokkos<config_mode_t<mode>>; \
        template class CSRMatrixOperationsKokkos<config_mode_t<mode>>;

    #define CSR_INST( mode ) KOKKOS_INST( mode )
CSR_CONFIG_FORALL( CSR_INST )

    #define KOKKOS_CC_INST( mode, mode_in )                                                        \
        template void                                                                              \
        CSRMatrixOperationsKokkos<config_mode_t<mode>>::copyCast<config_mode_t<mode_in>>(          \
            CSRMatrixData<config_mode_t<mode_in>> *, CSRMatrixData<config_mode_t<mode>> * );       \
        template void                                                                              \
            CSRLocalMatrixOperationsKokkos<config_mode_t<mode>>::copyCast<config_mode_t<mode_in>>( \
                std::shared_ptr<CSRLocalMatrixData<config_mode_t<mode_in>>>,                       \
                std::shared_ptr<CSRLocalMatrixData<config_mode_t<mode>>> );

    #define CC_INST( mode, mode_in ) KOKKOS_CC_INST( mode, mode_in )
CSR_CONFIG_CC_FORALL( CC_INST )

} // namespace AMP::LinearAlgebra
#endif
