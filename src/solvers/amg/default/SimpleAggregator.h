#ifndef included_AMP_SimpleAggregator_H_
#define included_AMP_SimpleAggregator_H_

#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Aggregator.h"

namespace AMP::Solver::AMG {

struct SimpleAggregator : Aggregator {
    SimpleAggregator( const float strength_threshold ) : Aggregator( strength_threshold ) {}

    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A, int *agg_ids ) override;

    template<typename Config>
    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A, int *agg_ids );
};

} // namespace AMP::Solver::AMG

#endif
