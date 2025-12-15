#ifndef included_AMP_RobinVectorCorrection
#define included_AMP_RobinVectorCorrection

#include "AMP/operators/boundary/BoundaryOperator.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"

// Libmesh files
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/quadrature.h"

#include <string>

namespace AMP::Operator {

typedef NeumannVectorCorrectionParameters RobinVectorCorrectionParameters;


/**
  A class to impose Robin Boundary conditions for a nonlinear operator.
  This can be written as:
     \f$\alpha k(u)*\frac{\partial u}{\partial n} + \beta h*u = \gamma*c \f$.
  Imposing this condition would involve evaluating the expression and adding the
  contribution to the residual vector. This class is derived from NeumannVectorCorrection
  as it implements similar functionality.
*/
class RobinVectorCorrection : public NeumannVectorCorrection
{
public:
    /**
      Constructor. This function reads all the parameters required for Robin boundary
      conditions. Since it is derived from NeumannVectorCorrection, its constructor
      will be called to read the required parameters.
      */
    explicit RobinVectorCorrection( std::shared_ptr<const OperatorParameters> params );

    virtual ~RobinVectorCorrection() {}

    //! Return the name of the operator
    std::string type() const override { return "RobinVectorCorrection"; }

    /**
      Sets Robin values into the appropriate locations of the output vector (r).
      */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    /**
      This function can be used to change the Robin boundary conditions i.e., change the
      RHS flux values.
      */
    void reset( std::shared_ptr<const OperatorParameters> params ) override;

protected:
    /**
      This function returns a parameter object that can be used to reset the corresponding
      RobinMatrixCorrection operator.
      */
    std::shared_ptr<OperatorParameters>
        getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr ) override;

    // input variable for the unkown rhs
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_srcVariable;

    double d_hef; // Convective Coefficient

    double d_alpha;
    double d_beta;
    double d_gamma;

    bool d_skipParams;

    std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr> d_elementInputVec;


private:
    int d_InstanceID;
};
} // namespace AMP::Operator

#endif
