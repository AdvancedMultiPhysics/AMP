#include "AMP/matrices/data/CSRMatrixDataHelpers.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/utils/Memory.h"

namespace AMP::LinearAlgebra {
#define CSR_INST( mode ) template class CSRMatrixDataHelpers<config_mode_t<mode>>;
CSR_CONFIG_FORALL( CSR_INST )
} // namespace AMP::LinearAlgebra
