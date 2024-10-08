
#ifndef included_AMP_NavierStokesLSWFFEOperatorParameters
#define included_AMP_NavierStokesLSWFFEOperatorParameters

#include "AMP/operators/flow/FlowTransportModel.h"
#include "AMP/operators/flow/NavierStokesConstants.h"
#include "AMP/operators/libmesh/FEOperatorParameters.h"
#include "AMP/vectors/Vector.h"

namespace AMP::Operator {

class NavierStokesLSWFFEOperatorParameters : public FEOperatorParameters
{
public:
    explicit NavierStokesLSWFFEOperatorParameters( std::shared_ptr<AMP::Database> db )
        : FEOperatorParameters( db )
    {
    }

    virtual ~NavierStokesLSWFFEOperatorParameters() {}

    //      std::shared_ptr<AMP::Discretization::DOFManager>
    //      d_dofMap[NavierStokes::TOTAL_NUMBER_OF_VARIABLES];
    //      AMP::LinearAlgebra::Vector::shared_ptr
    //      d_frozenVec[NavierStokes::TOTAL_NUMBER_OF_VARIABLES];

    std::shared_ptr<AMP::Discretization::DOFManager> d_dofMap;
    AMP::LinearAlgebra::Vector::shared_ptr d_frozenVec;

    std::shared_ptr<FlowTransportModel> d_transportModel;

protected:
private:
};
} // namespace AMP::Operator

#endif
