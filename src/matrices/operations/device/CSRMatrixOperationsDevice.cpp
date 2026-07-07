#include "AMP/matrices/operations/device/CSRMatrixOperationsDevice.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/operations/device/CSRLocalMatrixOperationsDevice.hpp"
#include "AMP/matrices/operations/device/spgemm/CSRMatrixSpGEMMDevice.hpp"
#include "AMP/utils/Memory.h"


namespace AMP::LinearAlgebra {
#define CSR_INST( mode )                                                \
    template class CSRLocalMatrixOperationsDevice<config_mode_t<mode>>; \
    template class CSRMatrixOperationsDevice<config_mode_t<mode>>;      \
    template class CSRMatrixSpGEMMDevice<config_mode_t<mode>>;
CSR_CONFIG_FORALL( CSR_INST )

} // namespace AMP::LinearAlgebra
