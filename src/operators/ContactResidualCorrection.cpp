#include "AMP/operators/ContactResidualCorrection.h"
#include "AMP/discretization/DOF_Manager.h"

namespace AMP::Operator {

void ContactResidualCorrection::apply( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                       AMP::LinearAlgebra::Vector::shared_ptr r )
{
    auto rMaster = r->subsetVectorForVariable( d_masterVariable );
    auto rSlave  = r->subsetVectorForVariable( d_slaveVariable );

    auto master_dof_map = rMaster->getDOFManager();
    auto slave_dof_map  = rSlave->getDOFManager();

    for ( size_t i = 0; i < d_masterNodes.size(); i++ ) {
        std::vector<size_t> masterGlobalIds;
        std::vector<size_t> slaveGlobalIds;
        master_dof_map->getDOFs( d_masterNodes[i], masterGlobalIds );
        slave_dof_map->getDOFs( d_slaveNodes[i], slaveGlobalIds );
        const double zero = 0.0;
        for ( auto &elem : d_dofs[i] ) {
            double slaveVal = rSlave->getLocalValueByGlobalID( slaveGlobalIds[elem] );
            rMaster->addValuesByGlobalID( 1, &masterGlobalIds[elem], &slaveVal );
            rSlave->setValuesByGlobalID( 1, &slaveGlobalIds[elem], &zero );
            slaveVal = rSlave->getLocalValueByGlobalID( slaveGlobalIds[elem] );
        } // end for j
    } // end for i
}
} // namespace AMP::Operator
