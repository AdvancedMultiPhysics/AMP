#include "AMP/matrices/AMPCSRMatrixParameters.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"

namespace AMP::LinearAlgebra {

#define CSR_INST( mode ) template class AMPCSRMatrixParameters<config_mode_t<mode>>;
CSR_CONFIG_FORALL( CSR_INST )

} // namespace AMP::LinearAlgebra
