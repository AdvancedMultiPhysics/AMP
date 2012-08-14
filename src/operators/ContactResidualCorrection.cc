
#include "ContactResidualCorrection.h"

namespace AMP {
  namespace Operator {

    void ContactResidualCorrection :: apply( AMP::LinearAlgebra::Vector::const_shared_ptr, AMP::LinearAlgebra::Vector::const_shared_ptr,
        AMP::LinearAlgebra::Vector::shared_ptr r, const double , const double ) {
      AMP::LinearAlgebra::Vector::shared_ptr rMaster = r->subsetVectorForVariable(d_masterVariable);
      AMP::LinearAlgebra::Vector::shared_ptr rSlave = r->subsetVectorForVariable(d_slaveVariable);

      AMP::Discretization::DOFManager::shared_ptr master_dof_map = rMaster->getDOFManager();
      AMP::Discretization::DOFManager::shared_ptr slave_dof_map = rSlave->getDOFManager();

      for(size_t i = 0; i < d_masterNodes.size(); i++) {
        std::vector<size_t> masterGlobalIds;
        std::vector<size_t> slaveGlobalIds;
        master_dof_map->getDOFs(d_masterNodes[i], masterGlobalIds);
        slave_dof_map->getDOFs(d_slaveNodes[i], slaveGlobalIds);
        for(size_t j = 0; j < d_dofs[i].size(); j++) {
          double slaveVal = rSlave->getLocalValueByGlobalID( slaveGlobalIds[d_dofs[i][j]] );
          rMaster->addLocalValueByGlobalID(masterGlobalIds[d_dofs[i][j]], slaveVal);
          rSlave->setLocalValueByGlobalID(slaveGlobalIds[d_dofs[i][j]], 0.0);
          slaveVal = rSlave->getLocalValueByGlobalID( slaveGlobalIds[d_dofs[i][j]] );
        }//end for j
      }//end for i
    }

  }
}




