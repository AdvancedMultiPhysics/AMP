#ifndef included_AMP_TimeOperator
#define included_AMP_TimeOperator

#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include <memory>


namespace AMP::TimeIntegrator {

/*!
  @brief base class for operator class associated with ImplicitTimeIntegrator

  Class ImplicitTimeOperator is a base class derived from Operator. It
  is the operator class associated with a ImplicitTimeIntegrator. The solver associated
  with the ImplicitTimeIntegrator will register this object.
  For implicit time integrators solving a problem M(u_t) = f(u), at each time step
  a nonlinear or linear problem, F(u) = 0, where F(u) = M(u_t)-f(u) discretized, needs to be solved.
  The TimeOperator class is meant to represent this operator. Internally, two operators are used to
  represent
  F, a mass operator and a rhs operator. M corresponds to the mass operator and f() corresponds to
  the rhs operator.
  The mass operator is NULL in the case of FVM, FD discretizations.

  @see ImplicitTimeIntegrator
  @see Operator
  @see SolverStrategy
  @see TimeOperatorParameters
*/
class TimeOperator : public AMP::Operator::Operator
{
public:
    /**
     * Main constructor meant for use by users.
     @param [in] params : parameter object that must be of type
     TimeOperatorParameters. The database object contained in the parameter
     object contains the following fields:

     1. name: bLinearMassOperator
        description: boolean to indicate whether the mass operator is a linear operator, currently
        either both mass and rhs operators have to be linear or both have to be nonlinear
        type: bool
        default value: FALSE
        valid values: (TRUE, FALSE)
        optional field: yes

     2. name: bLinearRhsOperator
        description: boolean to indicate whether the rhs operator is a linear operator, currently
        either both mass and rhs operators have to be linear or both have to be nonlinear
        type: bool
        default value: FALSE
        valid values: (TRUE, FALSE)
        optional field: yes

     3. name: bAlgebraicComponent
        description: for a DAE system this TimeOperator would be one of several components. This
     field
        indicates if this component is an algebraic component, ie, having no time derivative
        type: bool
        default value: FALSE
        valid values: (TRUE, FALSE)
        optional field: yes

     4. name: CurrentDt
        description: current time step value
        type: double
        default value: none
        valid values: (positve real values)
        optional field: no

    */
    explicit TimeOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params );

    /**
     * virtual destructor
     */
    virtual ~TimeOperator();

    /**
     * This function is useful for re-initializing an operator
     * \param params
     *        parameter object containing parameters to change
     */
    void reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params ) override;

    /**
     * This function registers a rhs operator with the TimeOperator class
     @param [in] op : shared pointer to Operator, cannot be another TimeOperator
     */
    void registerRhsOperator( std::shared_ptr<AMP::Operator::Operator> op ) { d_pRhsOperator = op; }

    /**
     * This function registers a mass operator with the TimeOperator class. Not necessary
     * for FD or FVM discretizations
     @param [in] op : shared pointer to Operator, cannot be another TimeOperator
     */
    void registerMassOperator( std::shared_ptr<AMP::Operator::Operator> op )
    {
        d_pMassOperator = op;
    }

    /**
     * register a variable as being an algebraic component. Deprecated.
     */
    void registerAlgebraicVariable( std::shared_ptr<AMP::LinearAlgebra::Variable> var )
    {
        d_pAlgebraicVariable = var;
    }

    /**
     * return a shared pointer to the rhs operator
     */
    std::shared_ptr<AMP::Operator::Operator> getRhsOperator( void ) { return d_pRhsOperator; }

    /**
     * return a shared pointer to the mass operator
     */
    std::shared_ptr<AMP::Operator::Operator> getMassOperator( void ) { return d_pMassOperator; }

    /**
     * set the value of the current time step
     @param [in] dt : value of current timestep
     */
    void setDt( double dt ) { d_dCurrentDt = dt; }

    /**
     * returns a Variable object corresponding to the output variable for the TimeOperator.
     * Currently
     * this is the output variable associated with the rhs operator.
     */
    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() override
    {
        return d_pRhsOperator->getOutputVariable();
    }

protected:
    TimeOperator();

    void getFromInput( std::shared_ptr<AMP::Database> db );

    bool d_bLinearMassOperator;

    bool d_bLinearRhsOperator;

    /**
     * set to true if this operator corresponds to an algebraic component
     */
    bool d_bAlgebraicComponent;

    double d_dCurrentDt;

    /**
     * pointer to rhs operator
     */
    std::shared_ptr<AMP::Operator::Operator> d_pRhsOperator;

    /**
     * pointer to mass operator
     */
    std::shared_ptr<AMP::Operator::Operator> d_pMassOperator;

    /**
     * algebraic variable
     */
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_pAlgebraicVariable;

    /**
     * scratch vector for internal use
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pScratchVector;

    /**
     * vector containing source terms if any
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pSourceTerm;

private:
};
} // namespace AMP::TimeIntegrator

#endif
