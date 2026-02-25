#ifndef included_AMP_MassMatrixCorrection
#define included_AMP_MassMatrixCorrection

#include "AMP/operators/boundary/BoundaryOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrectionParameters.h"
#include "AMP/vectors/Variable.h"


namespace AMP::Operator {


typedef DirichletMatrixCorrectionParameters MassMatrixCorrectionParameters;


/**
  A class used to impose HOMOGENEOUS Dirichlet boundary conditions for a mass operator. This is a
  minor modification of
  the class
  DirichletMatrixCorrection. A feature to handle nonhomogeneous boundary condition needs to be
  implemented soon.
 */
class MassMatrixCorrection : public BoundaryOperator
{
public:
    //! Constructor
    explicit MassMatrixCorrection( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~MassMatrixCorrection() {}

    //! Return the name of the operator
    std::string type() const override { return "MassMatrixCorrection"; }

    /**
      Set the variable for the vector that will used with this operator.
      */
    void setVariable( const std::shared_ptr<AMP::LinearAlgebra::Variable> &var )
    {
        d_variable = var;
    }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr,
                AMP::LinearAlgebra::Vector::shared_ptr ) override
    {
        // Do Nothing
    }

    void resetBoundaryIds( std::shared_ptr<const MassMatrixCorrectionParameters> );

    /**
      This function modifies the entries of the matrix formed by the volume operator
      in order to impose Dirichlet boundary conditions. This function can also be used
      to change the Dirichlet boundary conditions, if required.
     */
    void reset( std::shared_ptr<const OperatorParameters> ) override;

protected:
    // This must be a simple variable not a dual or multivariable
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_variable;

    bool d_bSetIdentityOnDiagonal;

    std::vector<short int> d_boundaryIds;

    std::vector<std::vector<unsigned int>> d_dofIds;


private:
};
} // namespace AMP::Operator

#endif
