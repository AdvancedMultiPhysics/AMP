#ifndef included_AMP_TimeOperatorParameters
#define included_AMP_TimeOperatorParameters

#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include <memory>

namespace AMP::TimeIntegrator {

class TimeOperatorParameters : public AMP::Operator::OperatorParameters
{
public:
    /**
     * Construct and initialize a parameter list according to input
     * data.  Guess what the required and optional keywords are.
     */
    explicit TimeOperatorParameters( std::shared_ptr<AMP::Database> db );
    /**
     * Destructor.
     */
    virtual ~TimeOperatorParameters();

    /**
     * Right hand side operator when time operator is written as: u_t = f(u)+g
     * This pointer should be NULL
     * (1) if the parameter object is being used for a reset and not for construction
     */

    std::shared_ptr<AMP::Operator::Operator> d_pRhsOperator;

    /**
     * Mass operator which may or may not be present (should be present for FEM)
     * This pointer should be NULL
     * (1) if the parameter object is being used for a reset and not for construction
     */
    std::shared_ptr<AMP::Operator::Operator> d_pMassOperator;

    /**
     * Source/sink term as well as term containing boundary corrections from mass and rhs operators
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pSourceTerm;

    /**
     * Parameters to reset the rhs operator, this pointer should be NULL only in two cases
     * (1) if we have a linear rhs operator,
     * (2) during construction phase when a non NULL d_pRhsOperator should be supplied
     */
    std::shared_ptr<AMP::Operator::OperatorParameters> d_pRhsOperatorParameters;

    /**
     * Parameters to reset the lhs mass operator, this pointer should be NULL only in two cases
     * (1) if we have a linear mass operator,
     * (2) during construction phase when a non NULL d_pMassOperator should be supplied
     */
    std::shared_ptr<AMP::Operator::OperatorParameters> d_pMassOperatorParameters;

    /**
     * algebraic variable
     */
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_pAlgebraicVariable;
};
} // namespace AMP::TimeIntegrator

#endif
