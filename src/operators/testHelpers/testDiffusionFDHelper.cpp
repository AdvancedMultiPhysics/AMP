#include "testDiffusionFDHelper.h"


// Compute discrete L^p norms of vector u, for p = 1,2,inf on a regular box-shaped mesh with mesh
// spacings h in each dimension
std::array<double, 3> getDiscreteNorms( const std::vector<double> &h,
                                        std::shared_ptr<const AMP::LinearAlgebra::Vector> u )
{

    double vol = 1.0;
    for ( auto hk : h ) {
        vol *= hk;
    }

    // Compute norms
    double uL1Norm  = static_cast<double>( u->L1Norm() ) * vol;
    double uL2Norm  = static_cast<double>( u->L2Norm() ) * std::pow( vol, 0.5 );
    double uMaxNorm = static_cast<double>( u->maxNorm() );

    std::array<double, 3> unorms = { uL1Norm, uL2Norm, uMaxNorm };
    return unorms;
}