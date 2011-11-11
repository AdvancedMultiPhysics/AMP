
#include "ContactSolver.h"
#include "operators/ContactResidualCorrection.h"

namespace AMP {
  namespace Solver {

    void ContactSolver :: solve(boost::shared_ptr<AMP::LinearAlgebra::Vector>, boost::shared_ptr<AMP::LinearAlgebra::Vector>  u) {
AMP_ERROR("ContactSolver is not converted yet");
/*
      boost::shared_ptr<AMP::Operator::ContactResidualCorrection> op = boost::dynamic_pointer_cast<
        AMP::Operator::ContactResidualCorrection>(d_pOperator);

      AMP::LinearAlgebra::Variable::shared_ptr masterVariable = op->getMasterVariable();
      AMP::LinearAlgebra::Variable::shared_ptr slaveVariable = op->getSlaveVariable();

      AMP::Mesh::Mesh::shared_ptr masterMesh = op->getMasterMesh();
      AMP::Mesh::Mesh::shared_ptr slaveMesh = op->getSlaveMesh();

      std::vector<unsigned int> masterNodes = op->getMasterNodes();
      std::vector<unsigned int> slaveNodes = op->getSlaveNodes();

      std::vector<std::vector<unsigned int> > dofs = op->getDofs();

      AMP::LinearAlgebra::Vector::shared_ptr uMaster = u->subsetVectorForVariable(masterVariable);
      AMP::LinearAlgebra::Vector::shared_ptr uSlave = u->subsetVectorForVariable(slaveVariable);

      AMP::Mesh::DOFMap::shared_ptr master_dof_map = masterMesh->getDOFMap(masterVariable);
      AMP::Mesh::DOFMap::shared_ptr slave_dof_map = slaveMesh->getDOFMap(slaveVariable);

      for(size_t i = 0; i < masterNodes.size(); i++) {
        AMP::Mesh::LibMeshNode masterNd = masterMesh->getNode( masterNodes[i] );
        AMP::Mesh::LibMeshNode slaveNd = slaveMesh->getNode( slaveNodes[i] );
        std::vector<unsigned int> masterGlobalIds;
        std::vector<unsigned int> slaveGlobalIds;
        master_dof_map->getDOFs(masterNd, masterGlobalIds, dofs[i]);
        slave_dof_map->getDOFs(slaveNd, slaveGlobalIds, dofs[i]);
        for(size_t j = 0; j < dofs[i].size(); j++) {
          double masterVal = uMaster->getLocalValueByGlobalID( masterGlobalIds[j] );
          uSlave->setLocalValueByGlobalID(slaveGlobalIds[j], masterVal);
        }//end for j
      }//end for i
*/
    }

  }
}

