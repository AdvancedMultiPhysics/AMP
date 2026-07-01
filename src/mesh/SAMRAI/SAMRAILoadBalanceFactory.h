#ifndef included_AMP_Mesh_SAMRAILoadBalanceFactory
#define included_AMP_Mesh_SAMRAILoadBalanceFactory

#include "AMP/utils/FactoryStrategy.hpp"

#include "SAMRAI/mesh/LoadBalanceStrategy.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/Dimension.h"

#include <memory>
#include <string>


namespace AMP::Mesh {


//! Operator factory class
class SAMRAILoadBalanceFactory :
    public FactoryStrategy<SAMRAI::mesh::LoadBalanceStrategy,
                           SAMRAI::tbox::Dimension,
                           std::string,
                           std::shared_ptr<SAMRAI::tbox::Database>>
{
public:
    static std::unique_ptr<SAMRAI::mesh::LoadBalanceStrategy>
    create( SAMRAI::tbox::Dimension dim,
            const std::string &name,
            std::shared_ptr<const AMP::Database> input_db = {} );
    [[deprecated]] static std::unique_ptr<SAMRAI::mesh::LoadBalanceStrategy>
    create( SAMRAI::tbox::Dimension dim,
            const std::string &name,
            std::shared_ptr<SAMRAI::tbox::Database> input_db );
    using FactoryStrategy::create;
};

} // namespace AMP::Mesh

#endif
