#include "AMP/matrices/operations/device/spgemm/cuda/SpGEMM_Cuda.h"

namespace AMP::LinearAlgebra {

#define CHECK_CUSPARSE( func )                     \
    do {                                           \
        cusparseStatus_t status = ( func );        \
        if ( status != CUSPARSE_STATUS_SUCCESS ) { \
            AMP_ERROR( "cusparse failure" );       \
        }                                          \
    } while ( 0 )

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
    : M( M_ ),
      N( N_ ),
      K( K_ ),
      alpha( 1.0 ),
      beta( 0.0 ),
      itype( std::is_same_v<rowidx_t, int> ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I ),
      jtype( std::is_same_v<colidx_t, int> ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I ),
      computeType( std::is_same_v<scalar_t, float> ? CUDA_R_32F : CUDA_R_64F ),
      opA( CUSPARSE_OPERATION_NON_TRANSPOSE ),
      opB( CUSPARSE_OPERATION_NON_TRANSPOSE ),
      alg( CUSPARSE_SPGEMM_ALG3 )
{
    // create cusparse handle and spgemm context
    CHECK_CUSPARSE( cusparseCreate( &handle ) );
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr( &spgemmDesc ) );

    // Create csr descriptions
    auto base_zero = CUSPARSE_INDEX_BASE_ZERO;
    CHECK_CUSPARSE( cusparseCreateCsr(
        &matA, M, K, A_nnz, A_rs, A_cols, A_vals, itype, jtype, base_zero, computeType ) );
    CHECK_CUSPARSE( cusparseCreateCsr(
        &matB, K, N, B_nnz, B_rs, B_cols, B_vals, itype, jtype, base_zero, computeType ) );
    CHECK_CUSPARSE( cusparseCreateCsr(
        &matC, M, N, 0, C_rs, nullptr, nullptr, itype, jtype, base_zero, computeType ) );

    // estimate workspace sizes and allocate
    float chunk_fraction = 0.2;
    CHECK_CUSPARSE( cusparseSpGEMM_workEstimation( handle,
                                                   opA,
                                                   opB,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   computeType,
                                                   alg,
                                                   spgemmDesc,
                                                   &bufferSize1,
                                                   nullptr ) );
    deviceMalloc( &dBuffer1, bufferSize1 );

    int64_t num_prods;
    CHECK_CUSPARSE( cusparseSpGEMM_workEstimation( handle,
                                                   opA,
                                                   opB,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   computeType,
                                                   alg,
                                                   spgemmDesc,
                                                   &bufferSize1,
                                                   dBuffer1 ) );
    CHECK_CUSPARSE( cusparseSpGEMM_getNumProducts( spgemmDesc, &num_prods ) );

    size_t buffer_tmp_size;
    void *buffer_tmp;
    CHECK_CUSPARSE( cusparseSpGEMM_estimateMemory( handle,
                                                   opA,
                                                   opB,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   computeType,
                                                   alg,
                                                   spgemmDesc,
                                                   chunk_fraction,
                                                   &buffer_tmp_size,
                                                   nullptr,
                                                   nullptr ) );
    deviceMalloc( &buffer_tmp, buffer_tmp_size );

    CHECK_CUSPARSE( cusparseSpGEMM_estimateMemory( handle,
                                                   opA,
                                                   opB,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   computeType,
                                                   alg,
                                                   spgemmDesc,
                                                   chunk_fraction,
                                                   &buffer_tmp_size,
                                                   buffer_tmp,
                                                   &bufferSize2 ) );
    deviceFree( buffer_tmp );
    deviceMalloc( &dBuffer2, bufferSize2 );
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::~VendorSpGEMM()
{
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr( spgemmDesc ) );
    CHECK_CUSPARSE( cusparseDestroySpMat( matA ) );
    CHECK_CUSPARSE( cusparseDestroySpMat( matB ) );
    CHECK_CUSPARSE( cusparseDestroySpMat( matC ) );
    CHECK_CUSPARSE( cusparseDestroy( handle ) );

    // free temporary buffers
    deviceFree( dBuffer1 );
    deviceFree( dBuffer2 );
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
int64_t VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::getCnnz()
{
    // intermediate product
    CHECK_CUSPARSE( cusparseSpGEMM_compute( handle,
                                            opA,
                                            opB,
                                            &alpha,
                                            matA,
                                            matB,
                                            &beta,
                                            matC,
                                            computeType,
                                            alg,
                                            spgemmDesc,
                                            &bufferSize2,
                                            dBuffer2 ) );

    // query and return size
    int64_t C_nrows, C_ncols, C_nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize( matC, &C_nrows, &C_ncols, &C_nnz ) );
    return C_nnz;
}

template<typename rowidx_t, typename colidx_t, typename scalar_t>
void VendorSpGEMM<rowidx_t, colidx_t, scalar_t>::compute( rowidx_t *C_rs,
                                                          colidx_t *C_cols,
                                                          scalar_t *C_vals )
{
    // update matC description with externally allocated buffers
    CHECK_CUSPARSE( cusparseCsrSetPointers( matC, C_rs, C_cols, C_vals ) );
    CHECK_CUSPARSE( cusparseSpGEMM_copy(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, alg, spgemmDesc ) );
}

// explicit instantiations, only two index types and two scalar types supported
template class VendorSpGEMM<int, int, float>;
// template class VendorSpGEMM<int, long long, float>;
template class VendorSpGEMM<long long, long long, float>;
template class VendorSpGEMM<int, int, double>;
// template class VendorSpGEMM<int, long long, double>;
template class VendorSpGEMM<long long, long long, double>;

} // namespace AMP::LinearAlgebra
