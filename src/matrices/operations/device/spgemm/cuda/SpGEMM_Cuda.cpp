#include "AMP/matrices/operations/device/spgemm/cuda/SpGEMM_Cuda.h"

namespace AMP::LinearAlgebra {

template<typename rowidx_t, typename colidx_t, typename scalar_t>
VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::VendorSpGEMM( const int64_t M_,
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
                                                          rowidx_t *C_rs )
    : M( M_ ), N( N_ ), K( K_ ), alpha( 1.0 ), beta( 0.0 )
{
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::~VendorSpGEMM()
{
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
int64_t VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::getCnnz()
{
    return 0;
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
void VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::compute( rowidx_t *C_rs,
                                                          colidx_t *C_cols,
                                                          scalar_t *C_vals )
{
}


// explicit instantiations, only two index types and two scalar types supported
// note that 64 row pointers with 32 column indices is invalid,
// so only 6 combinations instead of 8
template class VendorSpGEMM<int, int, float>;
template class VendorSpGEMM<int, long long, float>;
template class VendorSpGEMM<long long, long long, float>;
template class VendorSpGEMM<int, int, double>;
template class VendorSpGEMM<int, long long, double>;
template class VendorSpGEMM<long long, long long, double>;

} // namespace AMP::LinearAlgebra
