#include "AMP/matrices/operations/spgemm/CSRMatrixSpGEMMCommon.hpp"
#include "AMP/matrices/CSRConfig.h"

namespace AMP::LinearAlgebra {
#define CSR_INST( mode ) template class CSRMatrixSpGEMMCommon<config_mode_t<mode>>;
CSR_CONFIG_FORALL( CSR_INST )
#undef CSR_INST

} // namespace AMP::LinearAlgebra
