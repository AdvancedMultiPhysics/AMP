#ifndef included_AMP_testDiffusionFDHelper
#define included_AMP_testDiffusionFDHelper

#include "AMP/vectors/Vector.h"

// Compute discrete L^p norms of vector u, for p = 1,2,inf on a regular box-shaped mesh with mesh
// spacings h in each dimension
std::array<double, 3> getDiscreteNorms( const std::vector<double> &h,
                                        std::shared_ptr<const AMP::LinearAlgebra::Vector> u );

#endif