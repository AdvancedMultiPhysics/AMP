#ifndef included_AMP_VectorHelpers
#define included_AMP_VectorHelpers

#include "AMP/utils/GroupedRedistributionPlan.h"
#include "AMP/vectors/Vector.h"


namespace AMP::LinearAlgebra::VectorHelpers {

using GroupedRedistributionContext = AMP::Utilities::GroupedRedistributionPlan;

std::shared_ptr<Vector> redistribute( std::shared_ptr<const Vector> vec, int new_nprocs );

std::shared_ptr<Vector> redistribute( std::shared_ptr<const Vector> vec,
                                      const GroupedRedistributionContext &context );

void redistribute( const GroupedRedistributionContext &context,
                   std::shared_ptr<const Vector> src,
                   std::shared_ptr<Vector> dst );

void scatterRedistributed( const GroupedRedistributionContext &context,
                           std::shared_ptr<const Vector> src,
                           std::shared_ptr<Vector> dst );


//! Perform multiple L1 norms on vector subsets
std::vector<Scalar> L1Norm( std::shared_ptr<const Vector> vec,
                            const std::vector<std::string> &names );

//! Perform multiple L2 norms on vector subsets
std::vector<Scalar> L2Norm( std::shared_ptr<const Vector> vec,
                            const std::vector<std::string> &names );

//! Perform multiple max norms on vector subsets
std::vector<Scalar> maxNorm( std::shared_ptr<const Vector> vec,
                             const std::vector<std::string> &names );

//! Perform multiple local L1 norms on vector subsets
std::vector<Scalar> localL1Norm( std::shared_ptr<const Vector> vec,
                                 const std::vector<std::string> &names );

//! Perform multiple local L2 norms on vector subsets
std::vector<Scalar> localL2Norm( std::shared_ptr<const Vector> vec,
                                 const std::vector<std::string> &names );

//! Perform multiple local max norms on vector subsets
std::vector<Scalar> localMaxNorm( std::shared_ptr<const Vector> vec,
                                  const std::vector<std::string> &names );


} // namespace AMP::LinearAlgebra::VectorHelpers

#endif
