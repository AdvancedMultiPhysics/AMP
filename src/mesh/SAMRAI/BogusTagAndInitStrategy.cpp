#include "AMP/mesh/SAMRAI/BogusTagAndInitStrategy.h"

#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/PatchLevel.h"


namespace AMP::Mesh {

BogusTagAndInitStrategy::BogusTagAndInitStrategy() = default;

BogusTagAndInitStrategy::~BogusTagAndInitStrategy() = default;

void BogusTagAndInitStrategy::initializeLevelData(
    const std::shared_ptr<SAMRAI::hier::PatchHierarchy> &,
    const int,
    const double,
    const bool,
    const bool,
    const std::shared_ptr<SAMRAI::hier::PatchLevel> &,
    const bool )
{
}

void BogusTagAndInitStrategy::resetHierarchyConfiguration(
    const std::shared_ptr<SAMRAI::hier::PatchHierarchy> &, const int, const int )
{
}

void BogusTagAndInitStrategy::applyGradientDetector(
    const std::shared_ptr<SAMRAI::hier::PatchHierarchy> &,
    const int,
    const double,
    const int,
    const bool,
    const bool )
{
}

} // namespace AMP::Mesh
