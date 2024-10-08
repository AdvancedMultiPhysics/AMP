
#ifndef included_AMP_MechanicsLinearFEOperatorParameters
#define included_AMP_MechanicsLinearFEOperatorParameters

#include "AMP/operators/libmesh/LinearFEOperatorParameters.h"
#include "AMP/operators/mechanics/MechanicsMaterialModel.h"

#include <vector>

namespace AMP::Operator {

/**
 * This class encapsulates parameters used to initialize or reset the
 MechanicsLinearFEOperator.
 @see MechanicsLinearFEOperator
 */
class MechanicsLinearFEOperatorParameters : public LinearFEOperatorParameters
{
public:
    /**
      Constructor.
      */
    explicit MechanicsLinearFEOperatorParameters( std::shared_ptr<AMP::Database> db )
        : LinearFEOperatorParameters( db )
    {
    }

    /**
      Destructor.
      */
    virtual ~MechanicsLinearFEOperatorParameters() {}

    std::shared_ptr<MechanicsMaterialModel> d_materialModel; /**< Material model. */

    AMP::LinearAlgebra::Vector::shared_ptr
        d_dispVec; /**< Displacement vector, which is passed from
                     MechanicsNonlinearFEOperator to MechanicsLinearFEOperator. */

protected:
private:
};
} // namespace AMP::Operator

#endif
