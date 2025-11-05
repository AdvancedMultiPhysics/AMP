#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/Aggregator.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/CommunicationList.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_ptr.h>
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/transform.h>
#endif

#include <cstdint>
#include <limits>
#include <numeric>

// DEBUG
#include <fstream>
#include <iostream>

namespace AMP::Solver::AMG {

#ifdef AMP_USE_DEVICE
template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void fill_p_diag( const lidx_t *agg_ids,
                             const lidx_t *P_rs,
                             const lidx_t num_rows,
                             const gidx_t begin_col,
                             gidx_t *P_cols,
                             lidx_t *P_cols_loc,
                             scalar_t *P_coeffs )
{
    for ( int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows;
          row += blockDim.x * gridDim.x ) {
        const auto agg = agg_ids[row];
        if ( agg >= 0 ) {
            const auto rs  = P_rs[row];
            P_cols[rs]     = begin_col + static_cast<gidx_t>( agg );
            P_cols_loc[rs] = agg;
            P_coeffs[rs]   = 1.0;
        }
    }
}
#endif

std::shared_ptr<LinearAlgebra::Matrix>
Aggregator::getAggregateMatrix( std::shared_ptr<LinearAlgebra::Matrix> A,
                                std::shared_ptr<LinearAlgebra::MatrixParameters> matParams )
{
    return LinearAlgebra::csrVisit( A, [this, matParams]( auto csr_ptr ) {
        return this->getAggregateMatrix( csr_ptr, matParams );
    } );
}

template<typename Config>
std::shared_ptr<LinearAlgebra::Matrix>
Aggregator::getAggregateMatrix( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                std::shared_ptr<LinearAlgebra::MatrixParameters> matParams )
{
    using gidx_t            = typename Config::gidx_t;
    using lidx_t            = typename Config::lidx_t;
    using matrix_t          = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t      = typename matrix_t::matrixdata_t;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;

    // get aggregates
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    auto agg_ids       = localmatrixdata_t::makeLidxArray( A_nrows );
    const auto num_agg = assignLocalAggregates( A, agg_ids.get() );

    auto A_data = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );

    // if there is no parameters object passed in create one matching usual
    // purpose of a (tentative) prolongator
    if ( matParams.get() == nullptr ) {
        auto leftDOFs = A_data->getRightDOFManager(); // inner dof manager for A*P
        auto rightDOFs =
            std::make_shared<AMP::Discretization::DOFManager>( num_agg, A_data->getComm() );
        auto leftClParams     = std::make_shared<AMP::LinearAlgebra::CommunicationListParameters>();
        auto rightClParams    = std::make_shared<AMP::LinearAlgebra::CommunicationListParameters>();
        leftClParams->d_comm  = A_data->getComm();
        rightClParams->d_comm = A_data->getComm();
        leftClParams->d_localsize    = leftDOFs->numLocalDOF();
        rightClParams->d_localsize   = rightDOFs->numLocalDOF();
        leftClParams->d_remote_DOFs  = leftDOFs->getRemoteDOFs();
        rightClParams->d_remote_DOFs = rightDOFs->getRemoteDOFs();

        matParams = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
            leftDOFs,
            rightDOFs,
            A->getComm(),
            A_data->getLeftVariable(),
            A_data->getLeftVariable(),
            std::function<std::vector<size_t>( size_t )>() );
    }
    auto P = std::make_shared<matrixdata_t>( matParams );

    // non-zeros only in diag block and at most one per row
    auto diag_nnz = localmatrixdata_t::makeLidxArray( A_nrows );
    auto offd_nnz = localmatrixdata_t::makeLidxArray( A_nrows );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        std::transform( agg_ids.get(),
                        agg_ids.get() + A_nrows,
                        diag_nnz.get(),
                        []( const lidx_t lbl ) -> lidx_t { return lbl >= 0 ? 1 : 0; } );
    } else {
#ifdef AMP_USE_DEVICE
        thrust::transform(
            thrust::device,
            agg_ids.get(),
            agg_ids.get() + A_nrows,
            diag_nnz.get(),
            [] __device__( const lidx_t lbl ) -> lidx_t { return lbl >= 0 ? 1 : 0; } );
        deviceSynchronize();
        getLastDeviceError( "Aggregator::getAggregateMatrix" );
#else
        AMP_ERROR( "Aggregator::getAggregateMatrix Undefined memory location" );
#endif
    }
    AMP::Utilities::Algorithms<lidx_t>::fill_n( offd_nnz.get(), A_nrows, 0 );
    P->setNNZ( diag_nnz.get(), offd_nnz.get() );

    // fill in data (diag block only) using aggregates from above
    auto P_diag                               = P->getDiagMatrix();
    auto [P_rs, P_cols, P_cols_loc, P_coeffs] = P_diag->getDataFields();
    const auto begin_col                      = static_cast<gidx_t>( P->beginCol() );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t row = 0; row < A_nrows; ++row ) {
            const auto agg = agg_ids[row];
            if ( agg >= 0 ) {
                const auto rs  = P_rs[row];
                P_cols[rs]     = begin_col + static_cast<gidx_t>( agg );
                P_cols_loc[rs] = agg;
                P_coeffs[rs]   = 1.0;
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( A_nrows, BlockDim, GridDim );
        fill_p_diag<<<GridDim, BlockDim>>>(
            agg_ids.get(), P_rs, A_nrows, begin_col, P_cols, P_cols_loc, P_coeffs );
        getLastDeviceError( "Aggregator::getAggregateMatrix" );
#else
        AMP_ERROR( "Aggregator::getAggregateMatrix Undefined memory location" );
#endif
    }

    // reset dof managers and return matrix
    P->assemble();
    return std::make_shared<matrix_t>( P );
}

} // namespace AMP::Solver::AMG
