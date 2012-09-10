
#ifndef included_AMP_SubchannelParameters
#define included_AMP_SubchannelParameters

#include "operators/OperatorParameters.h"
#include "operators/subchannel/SubchannelPhysicsModel.h"
#include "operators/boundary/RobinPhysicsModel.h"
#include "discretization/DOF_Manager.h"

namespace AMP {
namespace Operator {

  /**
    Parameter class to provide parameters to all subchannel classes
    */
  class SubchannelOperatorParameters : public OperatorParameters {
    public :

      /**
        Constructor
        */
      SubchannelOperatorParameters(const boost::shared_ptr<AMP::Database> &db)
        : OperatorParameters(db) {  }

      /**
        Destructor
        */
      ~SubchannelOperatorParameters() { }

      // pointer to subchannel physics model
      boost::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;

      boost::shared_ptr<RobinPhysicsModel> d_dittusBoelterCoefficient;

      AMP::LinearAlgebra::Vector::shared_ptr d_frozenSolution;

      boost::shared_ptr<AMP::Discretization::DOFManager> d_dofMap;
  };

}
}

#endif
