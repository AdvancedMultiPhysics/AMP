#include "AMP/matrices/CSRMatrix.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/utils/Memory.h"

namespace AMP::LinearAlgebra {
#define CSR_INST( mode ) template class CSRMatrix<config_mode_t<mode>>;
CSR_CONFIG_FORALL( CSR_INST )
} // namespace AMP::LinearAlgebra
