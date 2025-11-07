#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/Aggregator.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/vectors/CommunicationList.h"

#include <cstdint>
#include <limits>
#include <numeric>

// DEBUG
#include <fstream>
#include <iostream>

namespace AMP::Solver::AMG {

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
    using gidx_t       = typename Config::gidx_t;
    using lidx_t       = typename Config::lidx_t;
    using matrix_t     = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t = typename matrix_t::matrixdata_t;

    // get aggregates
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    std::vector<int> agg_ids( A_nrows, 0 );
    const auto num_agg = assignLocalAggregates( A, agg_ids.data() );

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
    std::vector<lidx_t> diag_nnz( A_nrows );
    std::transform( agg_ids.begin(),
                    agg_ids.end(),
                    diag_nnz.begin(),
                    []( const int lbl ) -> lidx_t { return lbl >= 0 ? 1 : 0; } );
    P->setNNZ( diag_nnz, std::vector<lidx_t>( A_nrows, 0 ) );

    // fill in data (diag block only) using aggregates from above
    auto P_diag                               = P->getDiagMatrix();
    auto [P_rs, P_cols, P_cols_loc, P_coeffs] = P_diag->getDataFields();
    const auto begin_col                      = static_cast<gidx_t>( P->beginCol() );
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto agg = agg_ids[row];
        if ( agg >= 0 ) {
            const auto rs  = P_rs[row];
            P_cols[rs]     = begin_col + static_cast<gidx_t>( agg );
            P_cols_loc[rs] = agg;
            P_coeffs[rs]   = 1.0;
        }
    }

    // reset dof managers and return matrix
    P->resetDOFManagers();
    return std::make_shared<matrix_t>( P );
}

} // namespace AMP::Solver::AMG
