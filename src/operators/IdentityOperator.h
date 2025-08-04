#ifndef included_AMP_IdentityOperator
#define included_AMP_IdentityOperator

#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/vectors/Vector.h"

namespace AMP::Operator {


/**
 * Class IdentityOperator is the identity operator A(u) = u
 */
class IdentityOperator : public LinearOperator
{
public:
    //! Constructor. This resets the matrix shared pointer.
    IdentityOperator();

    /**
     * Constructor. This resets the matrix shared pointer.
     * @param [in] params
     */
    explicit IdentityOperator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~IdentityOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "IdentityOperator"; }

    /**
     * The apply function for this operator, A, performs the following operation:
     * r = A(u)
     * Here, A(u) is simply a Matrix-Vector multiplication.
     * @param [in] u input vector.
     * @param [out] f output vector.
     */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    /**
     * This function is useful for re-initializing/updating an operator
     * \param params
     *    parameter object containing parameters to change
     */
    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    std::shared_ptr<OperatorParameters>
    getParameters( const std::string &type,
                   std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                   std::shared_ptr<OperatorParameters> params = nullptr ) override;
    /**
     * Copies the shared pointer for the matrix representation of this linear operator.
     *  @param [in] in_mat The matrix representation of this linear operator.
     */
    void setMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> in_mat ) override;

    //! Set the input variable
    virtual void setInputVariable( std::shared_ptr<AMP::LinearAlgebra::Variable> var )
    {
        d_inputVariable = var;
    }

    //! Set the output variable
    virtual void setOutputVariable( std::shared_ptr<AMP::LinearAlgebra::Variable> var )
    {
        d_outputVariable = var;
    }

    /** \brief Get a right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{x}\f$ is a
     * right vector ) \return  A newly created right vector
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override;

    /** \brief Get a left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{y}\f$ is a left
     * vector )
     * \return  A newly created left vector
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> createOutputVector() const override;

private:
    size_t d_localSize;
};
} // namespace AMP::Operator

#endif
