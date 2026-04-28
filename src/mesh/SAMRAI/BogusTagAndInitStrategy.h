#ifndef included_AMP_BogusTagAndInitStrategy
#define included_AMP_BogusTagAndInitStrategy

#include "SAMRAI/mesh/StandardTagAndInitStrategy.h"


namespace SAMRAI::hier {
class PatchLevel;
class PatchHierarchy;
} // namespace SAMRAI::hier


namespace AMP::Mesh {


class BogusTagAndInitStrategy : public SAMRAI::mesh::StandardTagAndInitStrategy
{
public:
    BogusTagAndInitStrategy();

    ~BogusTagAndInitStrategy();

    virtual void
    initializeLevelData( const std::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                         const int level_number,
                         const double time,
                         const bool can_be_refined,
                         const bool initial_time,
                         const std::shared_ptr<SAMRAI::hier::PatchLevel> &old_level = {},
                         const bool allocate_data = true ) override;

    virtual void
    resetHierarchyConfiguration( const std::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                                 const int coarsest_level,
                                 const int finest_level ) override;

    virtual void
    applyGradientDetector( const std::shared_ptr<SAMRAI::hier::PatchHierarchy> &hierarchy,
                           const int level_number,
                           const double time,
                           const int tag_index,
                           const bool initial_time,
                           const bool uses_richardson_extrapolation_too ) override;
};


} // namespace AMP::Mesh

#endif
