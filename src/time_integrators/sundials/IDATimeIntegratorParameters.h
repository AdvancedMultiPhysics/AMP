#ifndef included_AMP_IDATimeIntegratorParameters
#define included_AMP_IDATimeIntegratorParameters


#ifndef included_AMP_TimeIntegratorParameters
    #include "AMP/time_integrators/TimeIntegratorParameters.h"
#endif

#ifndef included_AMP_IDATimeOperator
    #include "AMP/time_integrators/sundials/IDATimeOperator.h"
#endif

#ifndef included_AMP_LinearTimeOperator
    #include "AMP/time_integrators/LinearTimeOperator.h"
#endif

// BC : Altered this to get a compile going..
//#ifndef included_AMP_MassMatrix
//#include "AMP/operators/MassMatrix.h"
//#endif
#include "AMP/operators/Operator.h"

#ifndef included_AMP_SolverStrategy
    #include "AMP/solvers/SolverStrategy.h"
#endif

namespace AMP::TimeIntegrator {


/*!
 @brief TimeIntegratorParameters is a base class for providing
 parameters for the TimeIntegrator's. The Database object contained
 must contain the following entries:

 Required input keys and data types:
 @param initial_time double value for the initial simulation time.
 @param final_time double value for the final simulation time.
 @param max_integrator_steps integer value for the maximum number
 of timesteps allowed.

 All input data items described above, except for initial_time,
 may be overwritten by new input values when continuing from restart.

 */

class IDATimeIntegratorParameters : public TimeIntegratorParameters
{
public:
    explicit IDATimeIntegratorParameters( std::shared_ptr<AMP::Database> db );

    virtual ~IDATimeIntegratorParameters();


    std::shared_ptr<AMP::LinearAlgebra::Vector> d_ic_vector_prime;

    // Needs to be fixed - JL
    std::shared_ptr<IDATimeOperator> d_pIDATimeOperator;
    std::shared_ptr<LinearTimeOperator> d_pLinearTimeOperator;
    std::shared_ptr<TimeOperatorParameters> d_pLinearTimeOperatorParameters;

    std::shared_ptr<AMP::Solver::SolverStrategy> d_pNestedSolver;
    std::shared_ptr<AMP::Operator::LinearOperator> d_pLinearOperator;


    // source term
    // temp operators
    std::shared_ptr<AMP::Operator::Operator> d_temp_operator_1;
    std::shared_ptr<AMP::Operator::Operator> d_temp_operator_2;

protected:
private:
    explicit IDATimeIntegratorParameters() = delete;
    explicit IDATimeIntegratorParameters( const IDATimeIntegratorParameters & );
    void operator=( const IDATimeIntegratorParameters & );
};
} // namespace AMP::TimeIntegrator

#endif
