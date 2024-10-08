
#ifndef included_AMP_NavierStokesLinearFEOperatorParameters
#define included_AMP_NavierStokesLinearFEOperatorParameters

#include "AMP/operators/flow/FlowTransportModel.h"
#include "AMP/operators/flow/NavierStokesConstants.h"
#include "AMP/operators/libmesh/LinearFEOperatorParameters.h"
#include "AMP/vectors/Vector.h"

namespace AMP::Operator {

class NavierStokesLinearFEOperatorParameters : public LinearFEOperatorParameters
{
public:
    explicit NavierStokesLinearFEOperatorParameters( std::shared_ptr<AMP::Database> db )
        : LinearFEOperatorParameters( db )
    {
    }

    virtual ~NavierStokesLinearFEOperatorParameters() {}

    //      AMP::LinearAlgebra::Vector::shared_ptr
    //      d_frozenVec[NavierStokes::TOTAL_NUMBER_OF_VARIABLES];
    AMP::LinearAlgebra::Vector::shared_ptr d_frozenVec;

    std::shared_ptr<FlowTransportModel> d_transportModel;

protected:
private:
};
} // namespace AMP::Operator

#endif
