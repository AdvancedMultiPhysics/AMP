#include "AMP/matrices/data/CSRMatrixData.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRLocalMatrixData.hpp"
#include "AMP/matrices/data/CSRMatrixCommunicator.hpp"
#include "AMP/utils/Memory.h"

namespace AMP::LinearAlgebra {
#define CSR_INST( mode )                                       \
    template class CSRLocalMatrixData<config_mode_t<mode>>;    \
    template class CSRMatrixCommunicator<config_mode_t<mode>>; \
    template class CSRMatrixData<config_mode_t<mode>>;
CSR_CONFIG_FORALL( CSR_INST )
} // namespace AMP::LinearAlgebra
