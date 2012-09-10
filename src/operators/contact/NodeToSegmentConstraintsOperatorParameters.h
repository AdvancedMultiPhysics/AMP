
#ifndef included_AMP_NodeToSegmentConstraintsOperatorParameters
#define included_AMP_NodeToSegmentConstraintsOperatorParameters

#include "operators/OperatorParameters.h"
#include "discretization/DOF_Manager.h"

namespace AMP {
  namespace Operator {

    class NodeToSegmentConstraintsOperatorParameters : public OperatorParameters {
      public :

        NodeToSegmentConstraintsOperatorParameters (const boost::shared_ptr<AMP::Database> &db)
          : OperatorParameters(db) {  }

        virtual ~NodeToSegmentConstraintsOperatorParameters () { }

        AMP::AMP_MPI d_GlobalComm;

        size_t d_DOFsPerNode;
        AMP::Discretization::DOFManager::shared_ptr d_DOFManager;

        AMP::Mesh::MeshID d_MasterMeshID;
        AMP::Mesh::MeshID d_SlaveMeshID;

        int d_MasterBoundaryID;
        int d_SlaveBoundaryID;
 
      protected :

      private :

    };

  }
}

#endif

