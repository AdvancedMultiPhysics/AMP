#include <tuple>
#include <type_traits>

#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/amg/Aggregation.hpp"

namespace AMP::Solver::AMG {

coarse_ops_type pairwise_coarsen( std::shared_ptr<Operator::Operator> fine,
                                  const AMG::PairwiseCoarsenSettings &settings )
{
    auto linop = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( fine );
    AMP_INSIST( linop, "UASolver: operator must be linear" );
    auto mat = linop->getMatrix();
    AMP_INSIST( mat, "matrix cannot be NULL" );

    return LinearAlgebra::csrVisit(
        mat, [&]( auto csr_ptr ) { return pairwise_coarsen( *csr_ptr, settings ); } );
}


} // namespace AMP::Solver::AMG
