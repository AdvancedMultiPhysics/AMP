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
    template class CSRMatrixSpGEMMDefault<config_mode_t<mode>>;

CSR_CONFIG_FORALL( CSR_INST )
#undef CSR_INST

} // namespace AMP::LinearAlgebra
