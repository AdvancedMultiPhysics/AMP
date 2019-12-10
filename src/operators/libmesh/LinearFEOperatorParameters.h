
#ifndef included_AMP_LinearFEOperatorParameters
#define included_AMP_LinearFEOperatorParameters

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/operators/libmesh/FEOperatorParameters.h"

namespace AMP {
namespace Operator {

class LinearFEOperatorParameters : public FEOperatorParameters
{
public:
    /**
      Constructor.
      */
    explicit LinearFEOperatorParameters( AMP::shared_ptr<AMP::Database> db )
        : FEOperatorParameters( db )
    {
    }

    /**
      Destructor.
      */
    virtual ~LinearFEOperatorParameters() {}

    AMP::shared_ptr<AMP::Discretization::DOFManager> d_inDofMap;
    AMP::shared_ptr<AMP::Discretization::DOFManager> d_outDofMap;

protected:
private:
};
} // namespace Operator
} // namespace AMP

#endif
