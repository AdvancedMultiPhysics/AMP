
#ifndef included_AMP_DirichletVectorCorrectionParameters
#define included_AMP_DirichletVectorCorrectionParameters

#include "operators/OperatorParameters.h"

namespace AMP {
namespace Operator {

  class DirichletVectorCorrectionParameters : public OperatorParameters {
    public :

      DirichletVectorCorrectionParameters(const boost::shared_ptr<AMP::Database> &db)
        : OperatorParameters(db) {  }

      ~DirichletVectorCorrectionParameters() { }

      //This must be a simple variable not a dual or multivariable
      AMP::LinearAlgebra::Variable::shared_ptr d_variable;

  };

}
}

#endif

