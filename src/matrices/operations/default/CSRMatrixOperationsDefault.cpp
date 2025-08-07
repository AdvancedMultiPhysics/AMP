#include "AMP/matrices/operations/default/CSRMatrixOperationsDefault.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/operations/default/CSRLocalMatrixOperationsDefault.hpp"
#include "AMP/matrices/operations/default/spgemm/CSRMatrixSpGEMMDefault.hpp"
#include "AMP/utils/Memory.h"

namespace AMP::LinearAlgebra {
#define CSR_INST( mode )                                                 \
    template class CSRLocalMatrixOperationsDefault<config_mode_t<mode>>; \
    template class CSRMatrixOperationsDefault<config_mode_t<mode>>;      \
    template class CSRMatrixSpGEMMHelperDefault<config_mode_t<mode>>;

CSR_CONFIG_FORALL( CSR_INST )
#undef CSR_INST

#define CC_INST( mode, mode_in )                                                                \
    template void                                                                               \
    CSRMatrixOperationsDefault<config_mode_t<mode>>::copyCast<config_mode_t<mode_in>>(          \
        CSRMatrixData<                                                                          \
            typename config_mode_t<mode_in>::set_alloc_t<config_mode_t<mode>::allocator>> *,    \
        CSRMatrixData<config_mode_t<mode>> * );                                                 \
    template void                                                                               \
        CSRLocalMatrixOperationsDefault<config_mode_t<mode>>::copyCast<config_mode_t<mode_in>>( \
            std::shared_ptr<CSRLocalMatrixData<                                                 \
                typename config_mode_t<mode_in>::set_alloc_t<config_mode_t<mode>::allocator>>>, \
            std::shared_ptr<CSRLocalMatrixData<config_mode_t<mode>>> );

CSR_CONFIG_CC_FORALL( CC_INST )
} // namespace AMP::LinearAlgebra
