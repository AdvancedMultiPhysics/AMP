#ifndef included_AMP_MassLinearFEOperatorParameters
#define included_AMP_MassLinearFEOperatorParameters

#include "AMP/operators/libmesh/LinearFEOperatorParameters.h"

#include "AMP/operators/libmesh/MassDensityModel.h"

namespace AMP {
namespace Operator {

class MassLinearFEOperatorParameters : public LinearFEOperatorParameters
{
public:
    explicit MassLinearFEOperatorParameters( AMP::shared_ptr<AMP::Database> db )
        : LinearFEOperatorParameters( db )
    {
    }

    virtual ~MassLinearFEOperatorParameters() {}

    AMP::shared_ptr<MassDensityModel> d_densityModel;

protected:
private:
};
} // namespace Operator
} // namespace AMP

#endif
