
#ifndef included_AMP_DirichletVectorCorrection
#define included_AMP_DirichletVectorCorrection

#include "BoundaryOperator.h"
#include "DirichletVectorCorrectionParameters.h"

namespace AMP {
namespace Operator {

/**
  A class used to impose Dirichlet boundary conditions for a nonlinear operator. This involves the
  following steps:
  1) Make the intial guess vector for the nonlinear problem satisfy the specified Dirichlet boundary
  conditions.
  2) Make the entries corresponding to Dirichlet boundary conditions in the residual vector always
  be equal to zero.
  3) Make the entries corresponding to Dirichlet boundary conditions in the vector containing
  updates to the
  solution always be equal to zero.
  (1) and (3) together ensure that the solution always satisfies the specified Dirichlet boundary
  conditions.
  This can also be used to easily form a RHS vector corresponding to point forces (Dirac delta
  functions).
  */
class DirichletVectorCorrection : public BoundaryOperator
{
public:
    /**
      Constructor
      */
    explicit DirichletVectorCorrection(
        const AMP::shared_ptr<DirichletVectorCorrectionParameters> &params )
        : BoundaryOperator( params )
    {
        d_variable = params->d_variable;
        reset( params );
    }

    /**
      Destructor
      */
    virtual ~DirichletVectorCorrection() {}

    /**
      Set the variable for the vector that will used with this operator
      */
    void setVariable( const AMP::LinearAlgebra::Variable::shared_ptr &var ) { d_variable = var; }

    /**
     * Function to pass a vector of dirichlet values.
     */
    void setDirichletValues( AMP::LinearAlgebra::Vector::shared_ptr vals )
    {
        d_dirichletValues2 = mySubsetVector( vals, d_variable );
    }

    /**
      Sets Dirichlet values into the appropriate locations of the output vector (r). This does not
      affect
      the remaining values in that vector. u is not used.
      */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    void applyZeroValues( AMP::LinearAlgebra::Vector::shared_ptr r );

    void applyNonZeroValues( AMP::LinearAlgebra::Vector::shared_ptr r );

    void applyResidual( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                        AMP::LinearAlgebra::Vector::shared_ptr r );

    /**
      This function can be used to change the Dirichlet boundary conditions, if required.
      */
    void reset( const AMP::shared_ptr<OperatorParameters> &params );

    void setRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs )
    {
        this->applyZeroValues( rhs );
    }

    void modifyInitialSolutionVector( AMP::LinearAlgebra::Vector::shared_ptr sol )
    {
        if ( !d_setResidual ) {
            this->applyNonZeroValues( sol );
        }
    }

    std::vector<short int> getBoundaryIds() { return d_boundaryIds; }

    std::vector<std::vector<size_t>> getDofIds() { return d_dofIds; }

protected:
    AMP::LinearAlgebra::Vector::shared_ptr
    mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                    AMP::LinearAlgebra::Variable::shared_ptr var );

    AMP::LinearAlgebra::Vector::const_shared_ptr
    mySubsetVector( AMP::LinearAlgebra::Vector::const_shared_ptr vec,
                    AMP::LinearAlgebra::Variable::shared_ptr var );

    /**
      This function returns a parameter object that can be used to reset the corresponding
      DirichletMatrixCorrection operator.
      */
    AMP::shared_ptr<OperatorParameters>
        getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr ) override;

    std::vector<short int> d_boundaryIds;

    std::vector<std::vector<size_t>> d_dofIds;

    std::vector<std::vector<double>> d_dirichletValues1;

    AMP::LinearAlgebra::Vector::shared_ptr d_dirichletValues2;

    // This must be a simple variable not a dual or multivariable
    AMP::LinearAlgebra::Variable::shared_ptr d_variable;

    bool d_isAttachedToVolumeOperator;

    bool d_setResidual;

    int d_valuesType;

    double d_scalingFactor;

private:
};
}
}

#endif
