
#ifndef included_AMP_LinearOperator
#define included_AMP_LinearOperator

#include "AMP/matrices/Matrix.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/vectors/Vector.h"
#include <memory>

namespace AMP::Operator {


/**
 * An abstract base class for representing a linear operator. This class
 * stores the matrix representation of the linear operator. It provides
 * an implementation of the apply() function.
 * @see Operator
 */
class LinearOperator : public Operator
{
public:
    /**
     * Constructor. This resets the matrix shared pointer.
     * @param [in] params
     */
    explicit LinearOperator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~LinearOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "LinearOperator"; }

    /**
     * The apply function for this operator, A, performs the following operation:
     * f = A(u)
     * Here, A(u) is simply a Matrix-Vector multiplication.
     * @param [in] u input vector.
     * @param [out] f residual/output vector.
     */
    virtual void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                        AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    /**
     * @return The matrix representation of this linear operator.
     */
    std::shared_ptr<AMP::LinearAlgebra::Matrix> getMatrix();

    /**
     * Copies the shared pointer for the matrix representation of this linear operator.
     *  @param [in] in_mat The matrix representation of this linear operator.
     */
    virtual void setMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> in_mat );

    virtual void setVariables( std::shared_ptr<AMP::LinearAlgebra::Variable> in,
                               std::shared_ptr<AMP::LinearAlgebra::Variable> out )
    {
        d_inputVariable  = in;
        d_outputVariable = out;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_inputVariable;
    }
    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outputVariable;
    }

    /** \brief Get a right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{x}\f$ is a
     * right vector ) \return  A newly created right vector
     */
    virtual std::shared_ptr<AMP::LinearAlgebra::Vector> getInputVector() const;

    /** \brief Get a left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{y}\f$ is a left
     * vector )
     * \return  A newly created left vector
     */
    virtual std::shared_ptr<AMP::LinearAlgebra::Vector> getOutputVector() const;

protected:
    //! Empty constructor
    LinearOperator();

    // input and output variables
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inputVariable;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outputVariable;

    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_matrix; // The matrix shared pointer

private:
};
} // namespace AMP::Operator

#endif
