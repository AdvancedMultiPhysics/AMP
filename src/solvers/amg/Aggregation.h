#ifndef included_AMP_AMG_Aggregation
#define included_AMP_AMG_Aggregation

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/LinearOperator.h"

namespace AMP::Solver::AMG {
struct CoarsenSettings {
	float strength_threshold;
	size_t redist_coarsen_factor;
	size_t min_local_coarse;
	size_t min_coarse;
	size_t pairwise_passes;
};
struct PairwiseCoarsenSettings : CoarsenSettings {
	size_t pairwise_passes;
	bool checkdd;
};

using coarse_ops_type = std::tuple<
	std::shared_ptr<AMP::Operator::Operator>,
	std::shared_ptr<AMP::Operator::LinearOperator>,
	std::shared_ptr<AMP::Operator::Operator>>;

coarse_ops_type
pairwise_coarsen( std::shared_ptr<AMP::Operator::Operator> fine,
                  const PairwiseCoarsenSettings & settings );

template<class Config>
coarse_ops_type
pairwise_coarsen( const LinearAlgebra::CSRMatrix<Config> & fine,
                  const PairwiseCoarsenSettings & settings );

}
#endif
