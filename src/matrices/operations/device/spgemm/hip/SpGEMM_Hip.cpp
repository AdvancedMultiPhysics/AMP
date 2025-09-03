#include "SpGEMM_Hip.h"

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
    // set index and scalar types
    itype = std::is_same_v<rowidx_t, int> ? rocsparse_indextype_i32 : rocsparse_indextype_i64;
    jtype = std::is_same_v<colidx_t, int> ? rocsparse_indextype_i32 : rocsparse_indextype_i64;
    ttype = std::is_same_v<scalar_t, float> ? rocsparse_datatype_f32_r : rocsparse_datatype_f64_r;

    // create handle and matrix descriptions
    rocsparse_create_handle( &handle );
    rocsparse_create_csr_descr(
        &matA, M, K, A_nnz, A_rs, A_cols, A_vals, itype, jtype, rocsparse_index_base_zero, ttype );
    rocsparse_create_csr_descr(
        &matB, K, N, B_nnz, B_rs, B_cols, B_vals, itype, jtype, rocsparse_index_base_zero, ttype );
    rocsparse_create_csr_descr(
        &matC, M, N, 0, C_rs, nullptr, nullptr, itype, jtype, rocsparse_index_base_zero, ttype );
    rocsparse_create_csr_descr(
        &matD, 0, 0, 0, nullptr, nullptr, nullptr, itype, jtype, rocsparse_index_base_zero, ttype );

    // get temporary buffer size and allocate
    rocsparse_spgemm( handle,
                      rocsparse_operation_none,
                      rocsparse_operation_none,
                      &alpha,
                      matA,
                      matB,
                      &beta,
                      matD,
                      matC,
                      ttype,
                      rocsparse_spgemm_alg_default,
                      rocsparse_spgemm_stage_buffer_size,
                      &buffer_size,
                      nullptr );

    deviceMalloc( &temp_buffer, buffer_size );
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::~VendorSpGEMM()
{
    // destroy held handle and matrix descriptors
    rocsparse_destroy_spmat_descr( matA );
    rocsparse_destroy_spmat_descr( matB );
    rocsparse_destroy_spmat_descr( matC );
    rocsparse_destroy_spmat_descr( matD );
    rocsparse_destroy_handle( handle );

    // free workspace buffer
    deviceFree( temp_buffer );
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
int64_t VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::getCnnz()
{
    rocsparse_spgemm( handle,
                      rocsparse_operation_none,
                      rocsparse_operation_none,
                      &alpha,
                      matA,
                      matB,
                      &beta,
                      matD,
                      matC,
                      ttype,
                      rocsparse_spgemm_alg_default,
                      rocsparse_spgemm_stage_nnz,
                      &buffer_size,
                      temp_buffer );
    int64_t C_rows;
    int64_t C_cols;
    int64_t C_nnz;
    rocsparse_spmat_get_size( matC, &C_rows, &C_cols, &C_nnz );

    return C_nnz;
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
void VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::compute( rowidx_t *C_rs,
                                                          colidx_t *C_cols,
                                                          scalar_t *C_vals )
{
    // associate externally managed storage with C decscription
    rocsparse_csr_set_pointers( matC, C_rs, C_cols, C_vals );

    // do the actual computation
    rocsparse_spgemm( handle,
                      rocsparse_operation_none,
                      rocsparse_operation_none,
                      &alpha,
                      matA,
                      matB,
                      &beta,
                      matD,
                      matC,
                      ttype,
                      rocsparse_spgemm_alg_default,
                      rocsparse_spgemm_stage_compute,
                      &buffer_size,
                      temp_buffer );
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
