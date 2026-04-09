#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/Aggregator.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/CommunicationList.h"
#include "AMP/vectors/VectorBuilder.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_ptr.h>
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/transform.h>
#endif

#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>

namespace AMP::Solver::AMG {

// First are versions that use simple constant near-null vectors on all levels
std::shared_ptr<LinearAlgebra::Matrix>
Aggregator::getAggregateMatrix( std::shared_ptr<LinearAlgebra::Matrix> A,
                                std::shared_ptr<LinearAlgebra::MatrixParameters> matParams )
{
    return LinearAlgebra::csrVisit( A, [this, matParams]( auto csr_ptr ) {
        return this->getAggregateMatrix( csr_ptr, matParams );
    } );
}

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


template<typename Config>
std::shared_ptr<LinearAlgebra::Matrix>
Aggregator::getAggregateMatrix( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                std::shared_ptr<LinearAlgebra::MatrixParameters> matParams )
{
    std::shared_ptr<LinearAlgebra::Matrix> P;
    std::tie( P, std::ignore ) = getAggregateMatrix( A, nullptr, matParams );
    return P; // no input null vector, so don't return one
}

#ifdef AMP_USE_DEVICE
template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void fill_p_diag_scatter( const lidx_t *agg_ids,
                                     const lidx_t *P_rs,
                                     const scalar_t *null_vals,
                                     const scalar_t *coarse_null_vals,
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
            P_coeffs[rs]   = null_vals[row] / coarse_null_vals[agg];
        }
    }
}

template<typename lidx_t, typename scalar_t>
__global__ void acc_coarse_null( const lidx_t *agg_ids,
                                 const scalar_t *null_vals,
                                 const lidx_t num_rows,
                                 scalar_t *coarse_null_vals )
{
    for ( int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows;
          row += blockDim.x * gridDim.x ) {
        const auto agg = agg_ids[row];
        if ( agg >= 0 ) {
            const auto nv_sq = null_vals[row] * null_vals[row];
            atomicAdd( &coarse_null_vals[agg], nv_sq );
        }
    }
}
#endif

std::tuple<std::shared_ptr<LinearAlgebra::Matrix>, std::shared_ptr<LinearAlgebra::Vector>>
Aggregator::getAggregateMatrix( std::shared_ptr<LinearAlgebra::Matrix> A,
                                std::shared_ptr<const LinearAlgebra::Vector> nearNullVec,
                                std::shared_ptr<LinearAlgebra::MatrixParameters> matParams )
{
    return LinearAlgebra::csrVisit( A, [this, nearNullVec, matParams]( auto csr_ptr ) {
        return this->getAggregateMatrix( csr_ptr, nearNullVec, matParams );
    } );
}

template<typename Config>
std::tuple<std::shared_ptr<LinearAlgebra::Matrix>, std::shared_ptr<LinearAlgebra::Vector>>
Aggregator::getAggregateMatrix( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                std::shared_ptr<const LinearAlgebra::Vector> nearNullVec,
                                std::shared_ptr<LinearAlgebra::MatrixParameters> matParams )
{
    using gidx_t            = typename Config::gidx_t;
    using lidx_t            = typename Config::lidx_t;
    using scalar_t          = typename Config::scalar_t;
    using matrix_t          = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t      = typename matrix_t::matrixdata_t;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;

    auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );

    // get aggregates
    auto agg_ids       = localmatrixdata_t::template sharedArrayBuilder<int>( A_nrows );
    const auto num_agg = assignLocalAggregates( A, agg_ids.get() );

    // if there is no parameters object passed in create one matching usual
    // purpose of a (tentative) prolongator
    std::shared_ptr<AMP::Discretization::DOFManager> rightDOFs;
    if ( matParams.get() == nullptr ) {
        auto leftDOFs = A_data->getRightDOFManager(); // inner dof manager for A*P
        rightDOFs = std::make_shared<AMP::Discretization::DOFManager>( num_agg, A_data->getComm() );
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
            A_data->getBackend(),
            std::function<std::vector<size_t>( size_t )>() );
    } else {
        rightDOFs = matParams->getRightDOFManager();
    }

    // create and fill matrix data
    auto P = std::make_shared<matrixdata_t>( matParams );
    // non-zeros only in diag block and at most one per row
    auto diag_nnz = localmatrixdata_t::makeLidxArray( A_nrows );
    auto offd_nnz = localmatrixdata_t::makeLidxArray( A_nrows );
    AMP::Utilities::Algorithms<lidx_t>::fill_n( offd_nnz.get(), A_nrows, 0 );
    if constexpr ( !AMP::LinearAlgebra::alloc_info<Config::allocator>::device_accessible ) {
        std::transform( agg_ids.get(),
                        agg_ids.get() + A_nrows,
                        diag_nnz.get(),
                        []( const lidx_t lbl ) -> lidx_t { return lbl >= 0; } );
    } else {
#ifdef AMP_USE_DEVICE
        thrust::transform(
            thrust::device,
            agg_ids.get(),
            agg_ids.get() + A_nrows,
            diag_nnz.get(),
            [] __device__( const lidx_t lbl ) -> lidx_t { return lbl >= 0 ? 1 : 0; } );
        getLastDeviceError( "Aggregator::getAggregateMatrix" );
#else
        AMP_ERROR( "Aggregator::getAggregateMatrix Undefined memory location" );
#endif
    }
    P->setNNZ( diag_nnz.get(), offd_nnz.get() );

    // Pull values out of near null vector for
    // accumulation while aggregates are being written
    std::shared_ptr<LinearAlgebra::Vector> coarseNearNullVec;
    if ( nearNullVec ) {
        std::shared_ptr<scalar_t[]> null_ones; // only allocated if no nullvec passed
        const scalar_t *null_vals =
            nearNullVec->getVectorData()->template getRawDataBlock<const scalar_t>( 0 );

        // Make storage for the induced near-nullspace vector on the coarse space
        coarseNearNullVec = createVector( rightDOFs,
                                          A_data->getLeftVariable(),
                                          true,
                                          A_data->getMemoryLocation(),
                                          A_data->getBackend() );
        coarseNearNullVec->setNoGhosts();
        coarseNearNullVec->setToScalar( 0.0 );
        scalar_t *coarse_null_vals =
            coarseNearNullVec->getVectorData()->template getRawDataBlock<scalar_t>( 0 );

        // accumulate null_vals over each aggegrate into coarse_null vals
        // first pass sums squared values, second pass takes sqrt to get
        // local norms
        if constexpr ( !AMP::LinearAlgebra::alloc_info<Config::allocator>::device_accessible ) {
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                const auto agg = agg_ids[row];
                if ( agg >= 0 ) {
                    const auto nv_sq = null_vals[row] * null_vals[row];
                    coarse_null_vals[agg] += nv_sq;
                }
            }
            std::transform(
                coarse_null_vals,
                coarse_null_vals + num_agg,
                coarse_null_vals,
                []( const scalar_t cnv_sq ) -> scalar_t { return std::sqrt( cnv_sq ); } );
        } else {
#ifdef AMP_USE_DEVICE
            dim3 BlockDim;
            dim3 GridDim;
            setKernelDims( A_nrows, BlockDim, GridDim );
            acc_coarse_null<<<GridDim, BlockDim>>>(
                agg_ids.get(), null_vals, A_nrows, coarse_null_vals );
            getLastDeviceError( "Aggregator::getAggregateMatrix" );
            thrust::transform(
                thrust::device,
                coarse_null_vals,
                coarse_null_vals + num_agg,
                coarse_null_vals,
                [] __device__( const scalar_t cnv_sq ) -> scalar_t { return sqrt( cnv_sq ); } );
            getLastDeviceError( "Aggregator::getAggregateMatrix" );
#else
            AMP_ERROR( "Aggregator::getAggregateMatrix Undefined memory location" );
#endif
        }
        coarseNearNullVec->getVectorData()->setUpdateStatus(
            AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED );
        coarseNearNullVec->makeConsistent(); // no ghosts, so should be no-op

        // fill in data (diag block only) using aggregates from above
        auto P_diag                               = P->getDiagMatrix();
        auto [P_rs, P_cols, P_cols_loc, P_coeffs] = P_diag->getDataFields();
        const auto begin_col                      = static_cast<gidx_t>( P->beginCol() );
        if constexpr ( !AMP::LinearAlgebra::alloc_info<Config::allocator>::device_accessible ) {
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                const auto agg = agg_ids[row];
                if ( agg >= 0 ) {
                    const auto rs  = P_rs[row];
                    P_cols[rs]     = begin_col + static_cast<gidx_t>( agg );
                    P_cols_loc[rs] = agg;
                    P_coeffs[rs]   = null_vals[row] / coarse_null_vals[agg];
                }
            }
        } else {
#ifdef AMP_USE_DEVICE
            {
                dim3 BlockDim;
                dim3 GridDim;
                setKernelDims( A_nrows, BlockDim, GridDim );
                fill_p_diag_scatter<<<GridDim, BlockDim>>>( agg_ids.get(),
                                                            P_rs,
                                                            null_vals,
                                                            coarse_null_vals,
                                                            A_nrows,
                                                            begin_col,
                                                            P_cols,
                                                            P_cols_loc,
                                                            P_coeffs );
                getLastDeviceError( "Aggregator::getAggregateMatrix" );
            }
#else
            AMP_ERROR( "Aggregator::getAggregateMatrix Undefined memory location" );
#endif
        }
    } else {
        // don't do anything with coarse null vector because we don't
        // have an input null vector
        // just fill P with ones
        auto P_diag                               = P->getDiagMatrix();
        auto [P_rs, P_cols, P_cols_loc, P_coeffs] = P_diag->getDataFields();
        const auto begin_col                      = static_cast<gidx_t>( P->beginCol() );
        if constexpr ( !AMP::LinearAlgebra::alloc_info<Config::allocator>::device_accessible ) {
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
            {
                dim3 BlockDim;
                dim3 GridDim;
                setKernelDims( A_nrows, BlockDim, GridDim );
                fill_p_diag<<<GridDim, BlockDim>>>(
                    agg_ids.get(), P_rs, A_nrows, begin_col, P_cols, P_cols_loc, P_coeffs );
                getLastDeviceError( "Aggregator::getAggregateMatrix" );
            }
#else
            AMP_ERROR( "Aggregator::getAggregateMatrix Undefined memory location" );
#endif
        }
    }

    // reset dof managers and return matrix
    P->assemble();
    return std::make_tuple( std::make_shared<matrix_t>( P ), coarseNearNullVec );
}

} // namespace AMP::Solver::AMG
