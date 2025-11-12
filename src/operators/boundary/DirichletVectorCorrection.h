#ifndef included_AMP_DirichletVectorCorrection
#define included_AMP_DirichletVectorCorrection

#include "AMP/operators/boundary/BoundaryOperator.h"
#include "AMP/operators/boundary/DirichletVectorCorrectionParameters.h"

namespace AMP::Operator {

/**
  A class used to impose Dirichlet boundary conditions for a nonlinear operator. This involves the
  following steps:
  1) Make the initial guess vector for the nonlinear problem satisfy the specified Dirichlet
  boundary conditions. 2) Make the entries corresponding to Dirichlet boundary conditions in the
  residual vector always be equal to zero. 3) Make the entries corresponding to Dirichlet boundary
  conditions in the vector containing updates to the solution always be equal to zero. (1) and (3)
  together ensure that the solution always satisfies the specified Dirichlet boundary conditions.
  This can also be used to easily form a RHS vector corresponding to point forces (Dirac delta
  functions).
  */
class DirichletVectorCorrection : public BoundaryOperator
{
public:
    /**
      Constructor
      */
    explicit DirichletVectorCorrection( std::shared_ptr<const OperatorParameters> params );

    /**
      Destructor
      */
    virtual ~DirichletVectorCorrection() {}

    //! Return the name of the operator
    std::string type() const override { return "DirichletVectorCorrection"; }

    /**
      Set the variable for the vector that will used with this operator
      */
    void setVariable( const std::shared_ptr<AMP::LinearAlgebra::Variable> &var )
    {
        d_variable = var;
    }

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
    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    void setRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs ) override
    {
        this->applyZeroValues( rhs );
    }

    void modifyInitialSolutionVector( AMP::LinearAlgebra::Vector::shared_ptr sol ) override
    {
        if ( !d_setResidual ) {
            this->applyNonZeroValues( sol );
        }
    }

    const auto &getBoundaryIds() const { return d_boundaryIds; }

    const auto &getDofIds() const { return d_dofIds; }

protected:
    AMP::LinearAlgebra::Vector::shared_ptr
    mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                    std::shared_ptr<AMP::LinearAlgebra::Variable> var );

    AMP::LinearAlgebra::Vector::const_shared_ptr
    mySubsetVector( AMP::LinearAlgebra::Vector::const_shared_ptr vec,
                    std::shared_ptr<AMP::LinearAlgebra::Variable> var );

    /**
      This function returns a parameter object that can be used to reset the corresponding
      DirichletMatrixCorrection operator.
      */
    std::shared_ptr<OperatorParameters>
        getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr ) override;

    std::vector<short int> d_boundaryIds;

    std::vector<std::vector<size_t>> d_dofIds;

    std::vector<std::vector<double>> d_dirichletValues1;

    AMP::LinearAlgebra::Vector::shared_ptr d_dirichletValues2;

    // This must be a simple variable not a dual or multivariable
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_variable;

    bool d_isAttachedToVolumeOperator = false;

    bool d_setResidual = false;

    bool d_skipParams = false;

    int d_valuesType = 0;

    double d_scalingFactor = 0;

private:
};
} // namespace AMP::Operator

#endif
