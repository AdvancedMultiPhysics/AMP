
#ifndef included_AMP_ConsMomentumGalWFLinearFEOperatorParameters
#define included_AMP_ConsMomentumGalWFLinearFEOperatorParameters

#include "LinearFEOperatorParameters.h"
#include "FlowTransportModel.h"

#include <vector>

namespace AMP {
namespace Operator {

  class ConsMomentumGalWFLinearFEOperatorParameters : public LinearFEOperatorParameters {
    public :

      ConsMomentumGalWFLinearFEOperatorParameters(const boost::shared_ptr<AMP::Database> &db)
        : LinearFEOperatorParameters(db) {  }

      virtual ~ConsMomentumGalWFLinearFEOperatorParameters() { }

      boost::shared_ptr<FlowTransportModel> d_transportModel; 

      std::vector<AMP::LinearAlgebra::Vector::shared_ptr> d_frozenVec; 

    protected :

    private :

  };

}
}

#endif


