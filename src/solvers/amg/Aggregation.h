#ifndef included_AMP_AMG_Aggregation
#define included_AMP_AMG_Aggregation

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/Operator.h"
#include "AMP/solvers/amg/AggregationSettings.h"
#include "AMP/solvers/amg/Aggregator.h"

namespace AMP::Solver::AMG {

using coarse_ops_type = std::tuple<std::shared_ptr<AMP::Operator::Operator>,
                                   std::shared_ptr<AMP::Operator::LinearOperator>,
                                   std::shared_ptr<AMP::Operator::Operator>>;

coarse_ops_type pairwise_coarsen( std::shared_ptr<AMP::Operator::Operator> fine,
                                  const PairwiseCoarsenSettings &settings );

coarse_ops_type aggregator_coarsen( std::shared_ptr<AMP::Operator::Operator> fine,
                                    Aggregator &aggregator );

struct PairwiseAggregator : Aggregator {
    PairwiseAggregator( const PairwiseCoarsenSettings &settings )
        : Aggregator( settings ), d_settings( settings )
    {
    }

    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A, int *agg_ids ) override;
    template<class Config>
    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A, int *agg_ids );

private:
    PairwiseCoarsenSettings d_settings;
};

} // namespace AMP::Solver::AMG
#endif
