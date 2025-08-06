#ifndef included_AMP_SimpleAggregator_H_
#define included_AMP_SimpleAggregator_H_

#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Aggregator.h"

namespace AMP::Solver::AMG {

struct SimpleAggregator : Aggregator {
    SimpleAggregator() = default;
    SimpleAggregator( float wt_ ) : Aggregator( wt_ ){};
    SimpleAggregator( int ms_, float wt_ ) : Aggregator( ms_, wt_ ){};

    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A, int *agg_ids ) override;

    template<typename Config>
    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A, int *agg_ids );
};

} // namespace AMP::Solver::AMG

#endif
