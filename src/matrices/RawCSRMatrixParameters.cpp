#include "AMP/matrices/RawCSRMatrixParameters.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"

namespace AMP::LinearAlgebra {

#define CSR_INST( mode ) template class RawCSRMatrixParameters<config_mode_t<mode>>;

CSR_CONFIG_FORALL( CSR_INST )

} // namespace AMP::LinearAlgebra
