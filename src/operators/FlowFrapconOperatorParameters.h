
#ifndef included_AMP_FlowFrapconOperatorParameters
#define included_AMP_FlowFrapconOperatorParameters

#include "operators/OperatorParameters.h"
#include "FlowFrapconJacobianParameters.h"

namespace AMP {
namespace Operator {

  class FlowFrapconOperatorParameters : public OperatorParameters {
    public :

      FlowFrapconOperatorParameters(const boost::shared_ptr<AMP::Database> &db)
        : OperatorParameters(db) {  }

      ~FlowFrapconOperatorParameters() { }

      AMP::LinearAlgebra::Variable::shared_ptr d_variable;

  };

}
}

#endif

