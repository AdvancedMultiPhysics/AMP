
#ifndef included_AMP_DirichletMatrixCorrection
#define included_AMP_DirichletMatrixCorrection

#include "AMP/operators/boundary/BoundaryOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrectionParameters.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/vectors/Variable.h"

namespace AMP::Operator {

/**
 *  A class used to impose Dirichlet boundary conditions for a linear operator. For a linear
 * operator, imposing
 *  Dirichlet boundary conditions involves the following steps:
 *  1) Modify the entries of the matrix appropriately.
 *  2) Add a vector of corrections to the RHS vector
 *  3) Set the Dirichlet values at the appropriate locations in the RHS vector.
 */
class DirichletMatrixCorrection : public BoundaryOperator
{
public:
    //! Constructor
    explicit DirichletMatrixCorrection( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~DirichletMatrixCorrection() {}

    //! Return the name of the operator
    std::string type() const override { return "DirichletMatrixCorrection"; }

    //! Set the variable for the vector that will used with this operator.
    void setVariable( const std::shared_ptr<AMP::LinearAlgebra::Variable> &var )
    {
        d_variable = var;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_variable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_variable;
    }

    virtual void apply( AMP::LinearAlgebra::Vector::const_shared_ptr,
                        AMP::LinearAlgebra::Vector::shared_ptr ) override
    {
        // Do Nothing
    }

    /**
      This function modifies the entries of the matrix formed by the volume operator
      in order to impose Dirichlet boundary conditions. This function can also be used
      to change the Dirichlet boundary conditions, if required.
      */
    void reset( std::shared_ptr<const OperatorParameters> ) override;

    /**
      Adds a vector to the RHS vector. This is one of the steps for imposing Dirichlet boundary
      conditions.
      This step can be skipped if the Dirichlet boundary conditons are homogeneous.
      */
    void addRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs ) override;

    /**
      Sets the Dirichlet values at the appropriate locations in the RHS vector. This is one
      of the steps for imposing Dirichlet boundary conditions.
      */
    void setRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs ) override;

    const auto &getBoundaryIds() { return d_boundaryIds; }

    const auto &getDofIds() { return d_dofIds; }


protected:
    void parseParams( std::shared_ptr<const DirichletMatrixCorrectionParameters> );


protected:
    // This must be a simple variable not a dual or multivariable
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_variable;

    std::vector<short int> d_boundaryIds;

    std::vector<std::vector<double>> d_dirichletValues;

    std::vector<std::vector<unsigned int>> d_dofIds;

    AMP::LinearAlgebra::Vector::shared_ptr d_rhsCorrectionAdd;

    AMP::LinearAlgebra::Vector::shared_ptr d_dispVals;

    std::shared_ptr<DirichletVectorCorrection> d_rhsCorrectionSet;

    bool d_symmetricCorrection = false;

    bool d_zeroDirichletBlock = false;

    bool d_skipRHSaddCorrection = false;

    bool d_skipRHSsetCorrection = false;

    bool d_computedAddRHScorrection = false;

    bool d_initialized = false;

    void initRhsCorrectionSet();

    void initRhsCorrectionAdd( AMP::LinearAlgebra::Vector::shared_ptr rhs );

    void applyMatrixCorrection();

    bool d_applyMatrixCorrectionWasCalled;

    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_inputMatrix;

private:
};
} // namespace AMP::Operator

#endif
