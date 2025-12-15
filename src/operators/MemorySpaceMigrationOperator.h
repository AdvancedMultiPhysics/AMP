#ifndef included_MemorySpaceMigrationOperator_H_
#define included_MemorySpaceMigrationOperator_H_

#include "AMP/operators/Operator.h"

namespace AMP::Operator {
/**
 * This operator serves as an adaptor for an existing operator, Op,
 * It accepts vectors potentially in another memory address space,
 * copies them to the address space of Op so that Op can be used without
 * modification. A version of this Operator derived from LinearOperator
 * also exists for use with LinearOperators
 */
class MemorySpaceMigrationOperator : public Operator
{
public:
    //! Default constructor
    MemorySpaceMigrationOperator( void );

    //! Constructor
    explicit MemorySpaceMigrationOperator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~MemorySpaceMigrationOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "MemorySpaceMigrationOperator"; };

    /**
     * This function is useful for re-initializing/updating an operator
     * \param params
     *    parameter object containing parameters to change
     */
    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    /**
      This base class can not give a meaningful definition of apply. See the derived classes for
      how they define apply. Each operator is free to define apply in a way that is appropriate
      for that operator.
      \param u: shared pointer to const input vector u
      \param f: shared pointer to output vector storing result of applying this operator
      */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                std::shared_ptr<AMP::LinearAlgebra::Vector> f ) override;

    /**
     * Default base class implementation of the residual: f-L(u)
     * \param f: shared pointer to const vector rhs
     * \param u: shared pointer to const vector u
     * \param r: shared pointer to vector residual
     */
    void residual( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                   std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                   std::shared_ptr<AMP::LinearAlgebra::Vector> r ) override;

    /**
     * This function returns a OperatorParameters object
     * constructed by the operator which contains parameters from
     * which new operators can be created. Returning
     * a parameter object instead of an Operator itself is meant
     * to give users more flexibility. Examples of how this functionality
     * might be used would be the construction of Jacobian, frozen Jacobian,
     * preconditioner approximations to the Jacobian, adjoint operators etc
     * \param type: std:string specifying type of return operator parameters
     *      being requested. Currently the valid option is Jacobian
     * \param u: const pointer to current solution vector
     * \param params: pointer to additional parameters that might be required
     *      to construct the return parameters
     */
    std::shared_ptr<OperatorParameters>
    getParameters( const std::string &type,
                   std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                   std::shared_ptr<OperatorParameters> params = nullptr ) override;

    /**
     * Specify level of diagnostic information printed during iterations.
     * \param level
     *    zero prints none or minimal information, higher numbers provide increasingly
     *    verbose debugging information.
     */
    void setDebugPrintInfoLevel( int level ) override;

    //! Return the output variable
    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    //! Return the input variable
    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

    /** \brief Get a input vector ( For \f$\mathbf{A(x)}\f$, \f$\mathbf{x}\f$ is a
     * input vector ) \return  A newly created input vector
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> createInputVector() const override;

    /** \brief Get an output vector ( For \f$\mathbf{y=A(x)}\f$, \f$\mathbf{y}\f$ is an output
     * vector )
     * \return  A newly created output vector
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> createOutputVector() const override;

    //! Return the selector for output vectors
    std::shared_ptr<AMP::LinearAlgebra::VectorSelector> selectOutputVector() const override;

    //! Return the selector for input vectors
    std::shared_ptr<AMP::LinearAlgebra::VectorSelector> selectInputVector() const override;

    //! given a vector return whether it is valid or not
    // default behavior is to return true;
    bool isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> ) override;

    /**
     * interface used to make a vector consistent in an operator defined
     * way. An example of where an operator is required to make a vector consistent is
     * in the context of AMR where ghost values on coarse-fine interfaces are filled
     * in an operator dependent way. The default implementation is to simply call the
     * vector makeConsistent(SET)
     */
    void makeConsistent( std::shared_ptr<AMP::LinearAlgebra::Vector> vec ) override;

    //! re-initialize a vector, e.g. after a regrid operation has happened.
    //! This is useful for example when numerical
    //! overshoots or undershoots have happened due to interpolation for example
    //! The default is a null op
    void reInitializeVector( std::shared_ptr<AMP::LinearAlgebra::Vector> ) override;

    bool d_migrate_data;

    //! scratch space for input vectors
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_inputVec;
    //! scratch space for output vectors
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_outputVec;
    //! scratch space for residuals
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_resVec;

    //! Operator being adapted for memory operations
    std::shared_ptr<Operator> d_pOperator;
};

} // namespace AMP::Operator
#endif
