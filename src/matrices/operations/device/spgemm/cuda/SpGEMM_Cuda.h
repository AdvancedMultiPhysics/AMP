#ifndef included_AMP_SpGEMM_Cuda
#define included_AMP_SpGEMM_Cuda

#include "AMP/utils/cuda/Helper_Cuda.h"

#include <cusparse.h>

#include <cstdint>
#include <type_traits>

namespace AMP::LinearAlgebra {

// This class wraps the operations in the cusparse-spgemm example found at
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spgemm/spgemm_example.c
template<typename rowidx_t, typename colidx_t, typename scalar_t>
class VendorSpGEMM
{
    // Only signed int 32's and 64's are supported for index types
    // Only floats and doubles supported for value types
    static_assert( std::is_same_v<rowidx_t, int> || std::is_same_v<rowidx_t, long long> );
    static_assert( std::is_same_v<colidx_t, int> || std::is_same_v<colidx_t, long long> );
    static_assert( std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double> );

    // 64 bit row pointers with 32 bit column indices is invalid
    static_assert( std::is_same_v<rowidx_t, int> ||
                   (std::is_same_v<rowidx_t, long long> && std::is_same_v<colidx_t, long long>) );

public:
    VendorSpGEMM( const int64_t M_,
                  const int64_t N_,
                  const int64_t K_,
                  const int64_t A_nnz,
                  rowidx_t *A_rs,
                  colidx_t *A_cols,
                  scalar_t *A_vals,
                  const int64_t B_nnz,
                  rowidx_t *B_rs,
                  colidx_t *B_cols,
                  scalar_t *B_vals,
                  rowidx_t *C_rs );

    ~VendorSpGEMM();

    int64_t getCnnz();

    void compute( rowidx_t *C_rs, colidx_t *C_cols, scalar_t *C_vals );

private:
    const int64_t M;
    const int64_t N;
    const int64_t K;

    scalar_t alpha;
    scalar_t beta;
};

} // namespace AMP::LinearAlgebra


#endif
