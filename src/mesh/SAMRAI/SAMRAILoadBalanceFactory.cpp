#include "AMP/mesh/SAMRAI/SAMRAILoadBalanceFactory.h"
#include "AMP/mesh/SAMRAI/SimpleLoadBalancer.h"
#include "AMP/utils/Database.h"

#include "SAMRAI/mesh/CascadePartitioner.h"
#include "SAMRAI/mesh/ChopAndPackLoadBalancer.h"
#include "SAMRAI/mesh/TreeLoadBalancer.h"


template<typename TYPE>
static std::unique_ptr<SAMRAI::mesh::LoadBalanceStrategy>
createInstance( SAMRAI::tbox::Dimension dim,
                std::string name,
                std::shared_ptr<SAMRAI::tbox::Database> input_db )
{
    return std::make_unique<TYPE>( dim, name, input_db );
}


template<>
void AMP::FactoryStrategy<SAMRAI::mesh::LoadBalanceStrategy,
                          SAMRAI::tbox::Dimension,
                          std::string,
                          std::shared_ptr<SAMRAI::tbox::Database>>::registerDefault()
{
    d_factories["SimpleLoadBalancer"]      = createInstance<AMP::Mesh::SimpleLoadBalancer>;
    d_factories["TreeLoadBalancer"]        = createInstance<SAMRAI::mesh::TreeLoadBalancer>;
    d_factories["CascadePartitioner"]      = createInstance<SAMRAI::mesh::CascadePartitioner>;
    d_factories["ChopAndPackLoadBalancer"] = createInstance<SAMRAI::mesh::ChopAndPackLoadBalancer>;
}
std::unique_ptr<SAMRAI::mesh::LoadBalanceStrategy>
AMP::Mesh::SAMRAILoadBalanceFactory::create( SAMRAI::tbox::Dimension dim,
                                             const std::string &name,
                                             std::shared_ptr<SAMRAI::tbox::Database> input_db )
{
    return FactoryStrategy::create( name, dim, name, input_db );
}
std::unique_ptr<SAMRAI::mesh::LoadBalanceStrategy>
AMP::Mesh::SAMRAILoadBalanceFactory::create( SAMRAI::tbox::Dimension dim,
                                             const std::string &name,
                                             std::shared_ptr<const AMP::Database> input_db )
{
    if ( !input_db )
        input_db = std::make_shared<AMP::Database>( name );
    return FactoryStrategy::create( name, dim, name, input_db->cloneToSAMRAI() );
}
