
#ifndef included_AMP_NeumannVectorCorrectionParameters
#define included_AMP_NeumannVectorCorrectionParameters

#include "operators/LinearBoundaryOperatorParameters.h"
#include "operators/OperatorParameters.h"
#include "RobinPhysicsModel.h"

namespace AMP {
namespace Operator {

  class NeumannVectorCorrectionParameters : public OperatorParameters {
    public :

      NeumannVectorCorrectionParameters(const boost::shared_ptr<AMP::Database> &db)
        : OperatorParameters(db) {  }

      ~NeumannVectorCorrectionParameters() { }

      AMP::LinearAlgebra::Variable::shared_ptr d_variable;
 
      AMP::LinearAlgebra::Vector::shared_ptr d_variableFlux;

      boost::shared_ptr<RobinPhysicsModel> d_robinPhysicsModel;

  };

}
}

#endif

