#ifndef included_AMP_CSRMatrixSpGEMMDevice
#define included_AMP_CSRMatrixSpGEMMDevice

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/data/CSRMatrixCommunicator.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/spgemm/CSRMatrixSpGEMMCommon.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Memory.h"

#ifdef AMP_USE_CUDA
    #include "AMP/matrices/operations/device/spgemm/cuda/SpGEMM_Cuda.h"
#endif

#ifdef AMP_USE_HIP
    #include "AMP/matrices/operations/device/spgemm/hip/SpGEMM_Hip.h"
#endif

#include <map>
#include <memory>
#include <type_traits>
#include <vector>

namespace AMP::LinearAlgebra {

template<typename Config>
class CSRMatrixSpGEMMDevice : public CSRMatrixSpGEMMCommon<Config>
{
public:
    using allocator_type    = typename Config::allocator_type;
    using config_type       = Config;
    using matrixdata_t      = CSRMatrixData<Config>;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;
    using lidx_t            = typename Config::lidx_t;
    using gidx_t            = typename Config::gidx_t;
    using scalar_t          = typename Config::scalar_t;

    CSRMatrixSpGEMMDevice() = default;
    CSRMatrixSpGEMMDevice( std::shared_ptr<matrixdata_t> A_,
                           std::shared_ptr<matrixdata_t> B_,
                           std::shared_ptr<matrixdata_t> C_ )
        : CSRMatrixSpGEMMCommon<Config>( A_, B_, C_ )
    {
    }

    ~CSRMatrixSpGEMMDevice() = default;

    virtual void multiplyLocal( std::shared_ptr<localmatrixdata_t> A_data,
                                std::shared_ptr<localmatrixdata_t> B_data,
                                std::shared_ptr<localmatrixdata_t> C_data ) override;
};

} // namespace AMP::LinearAlgebra

#endif
