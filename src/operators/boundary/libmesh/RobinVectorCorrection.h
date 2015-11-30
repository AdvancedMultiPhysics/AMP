
#ifndef included_AMP_RobinVectorCorrection
#define included_AMP_RobinVectorCorrection

#include "operators/boundary/BoundaryOperator.h"
#include "operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"

/* Libmesh files */
#include "libmesh/fe_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/elem.h"
#include "libmesh/quadrature.h"

#include "libmesh/enum_order.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/auto_ptr.h"
#include "libmesh/string_to_enum.h"

#include <string>

namespace AMP {
namespace Operator {

  typedef NeumannVectorCorrectionParameters RobinVectorCorrectionParameters;

  /**
    A class to impose Robin Boundary conditions for a nonlinear operator. This can 
    be written as \f$\alpha k(u)*\frac{\partial u}{\partial n} + \beta h*u = \gamma*c \f$.
    Imposing this condition would involve evaluating the expression and adding the 
    contribution to the residual vector. This class is derived from NeumannVectorCorrection
    as it implements similar functionality.
    */

  class RobinVectorCorrection : public NeumannVectorCorrection 
  {
    public :

      /**
        Constructor. This function reads all the parameters required for Robin boundary 
        conditions. Since it is derived from NeumannVectorCorrection, its constructor
        will be called to read the required parameters.
        */
      RobinVectorCorrection(const AMP::shared_ptr<NeumannVectorCorrectionParameters> & params)
        : NeumannVectorCorrection (params)
      {
          reset(params);
          d_InstanceID = d_iInstance_id;
      }

      virtual ~RobinVectorCorrection() { }

      /**
        Sets Robin values into the appropriate locations of the output vector (r). 
        */
      void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
		  AMP::LinearAlgebra::Vector::shared_ptr f) override;

      /**
        This function can be used to change the Robin boundary conditions i.e., change the
        RHS flux values.
        */
      void reset(const AMP::shared_ptr<OperatorParameters>& params);

    protected :

      /**
        This function returns a parameter object that can be used to reset the corresponding
        RobinMatrixCorrection operator.
        */
      AMP::shared_ptr<OperatorParameters> getJacobianParameters(AMP::LinearAlgebra::Vector::const_shared_ptr ) override;

      // input variable for the unkown rhs
      AMP::LinearAlgebra::Variable::shared_ptr d_srcVariable;

      double d_hef;  //Convective Coefficient

      double d_alpha;
      double d_beta;
      double d_gamma;

      bool d_skipParams; 

      std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr> d_elementInputVec;


    private :

      int d_InstanceID;
 
  };

}
}

#endif

