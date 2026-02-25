#include "AMP/matrices/CSRMatrix.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/utils/Memory.h"

namespace AMP::LinearAlgebra {
#define CSR_INST( mode ) template class CSRMatrix<config_mode_t<mode>>;
CSR_CONFIG_FORALL( CSR_INST )
#undef CSR_INST

#define CC_INST( mode, mode_in )                                             \
    template std::shared_ptr<Matrix>                                         \
    CSRMatrix<config_mode_t<mode_in>>::migrate<config_mode_t<mode>>() const; \
    template std::shared_ptr<Matrix>                                         \
    CSRMatrix<config_mode_t<mode_in>>::migrate<config_mode_t<mode>>(         \
        AMP::Utilities::Backend backend ) const;

CSR_INOUT_CONFIG_MIGRATE( CC_INST )
#undef CC_INST

} // namespace AMP::LinearAlgebra
