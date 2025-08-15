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
#undef CSR_INST

#define CSR_INST( mode )                                                       \
    template std::shared_ptr<                                                  \
        CSRMatrixData<typename config_mode_t<mode>::set_alloc_t<alloc::host>>> \
    CSRMatrixData<config_mode_t<mode>>::migrate( AMP::Utilities::Backend backend ) const;

CSR_CONFIG_FORALL( CSR_INST )
#undef CSR_INST

#ifdef AMP_USE_DEVICE
    #define CSR_INST( mode )                                                                  \
        template std::shared_ptr<                                                             \
            CSRMatrixData<typename config_mode_t<mode>::set_alloc_t<alloc::managed>>>         \
        CSRMatrixData<config_mode_t<mode>>::migrate( AMP::Utilities::Backend backend ) const; \
        template std::shared_ptr<                                                             \
            CSRMatrixData<typename config_mode_t<mode>::set_alloc_t<alloc::device>>>          \
        CSRMatrixData<config_mode_t<mode>>::migrate( AMP::Utilities::Backend backend ) const;

CSR_CONFIG_FORALL( CSR_INST )
    #undef CSR_INST
#endif
} // namespace AMP::LinearAlgebra
