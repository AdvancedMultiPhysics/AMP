#ifndef included_AMP_IDATimeIntegrator
#define included_AMP_IDATimeIntegrator

#include <string>


#include "AMP/operators/LinearOperator.h"
#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/sundials/IDATimeIntegratorParameters.h"
#include "AMP/time_integrators/sundials/IDATimeOperator.h"
#include "AMP/vectors/sundials/SundialsVector.h"

#ifdef AMP_USE_SUNDIALS
extern "C" {
    #include "ida/ida.h"
    #include "ida/ida_spgmr.h"
    #include "sundials/sundials_nvector.h"
    #include "sundials/sundials_types.h"
}


namespace AMP::TimeIntegrator {


/** \class IDATimeIntegrator
 *
 * Class IDATimeIntegrator is a concrete time integrator
 * that provides an interface to the IDA time integrator within the SUNDIALS library.
 * It is derived from the TimeIntegrator class. This class is not derived from the
 * ImplicitTimeIntegrator class as it is more an interface or adaptor rather than
 * an implicit time integrator within the AMP package. As such, it uses the nonlinear
 * solvers provided by IDA and cannot leverage the solvers that AMP provides currently
 * other than within preconditioning.

 @see TimeIntegrator

 */
class IDATimeIntegrator : public TimeIntegrator
{
public:
    /**
     * Main constructor that accepts parameters object. The parameter object must be
     * a TimeIntegratorParameters object.The database object within the parameter
     * object expects the following fields in addition to any fields expected by the
     * TimeIntegrator base class.

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

     3. name: relative_tolerance
     description: relative tolerance for solves in IDA
     type: double
     default value: none
     valid values: see IDA manual
     optional field: no

     4. name: absolute_tolerance
     description: absolute tolerance for IDA solves
     type: double
     default value: none
     valid values: see IDA manual
     optional field: no

     5. name: CallCalcIC
     description: indicate whether to ask IDA to calculate initial conditions or not
     type: bool
     default value: TRUE
     valid values: (TRUE, FALSE)
     optional field: yes

     6. name: usePreconditioner
     description: whether to use preconditioner or not
     type: bool
     default value: TRUE
     valid values: (TRUE, FALSE)
     optional field: yes

     7. name:
     description:
     type:
     default value:
     valid values:
     optional field:

     8. name: createLinearTimeOperatorInternally
     description: indicate whether the linear operators required by the IDA preconditioner should be
     constructed
     internally or whether the user will supply
     type: bool
     default value: FALSE
     valid values: (TRUE, FALSE)
     optional field: yes

     9. name: bManufacturedProblem
     description: added by Gary, need to discuss whether appropriate mechanism or not
     type:
     default value:
     valid values:
     optional field:

    */

    explicit IDATimeIntegrator( std::shared_ptr<TimeIntegratorParameters> parameters );

    /**
     * Destructor.
     */
    virtual ~IDATimeIntegrator();

    /**
     * Initialize from parameter list.
     */
    void initialize( std::shared_ptr<TimeIntegratorParameters> parameters ) override;

    /**
     * Resets the internal state of the time integrator as needed.
     * A parameter argument is passed to allow for general flexibility
     * in determining what needs to be reset Typically used after a regrid.
     */
    void reset( std::shared_ptr<const TimeIntegratorParameters> parameters ) override;

    /**
     * Specify next time step to use.
     */
    double getNextDt( const bool good_solution ) override;

    /**
     * Set an initial guess for the time advanced solution.
     */
    void setInitialGuess( const bool first_step,
                          const double current_time,
                          const double current_dt,
                          const double old_dt );

    int advanceSolution( const double dt,
                         const bool first_step,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> out ) override;

    /**
     * Update state of the solution.
     */
    void updateSolution( void ) override;

    void updateSourceTerm( void );

    /**
     * Check time advanced solution to determine whether it is acceptable.
     * Return true if the solution is acceptable; return false otherwise.
     * The integer argument is the return code generated by the call to the
     * nonlinear solver "solve" routine.   The meaning of this value depends
     * on the particular nonlinear solver in use and must be intepreted
     * properly by the user-supplied solution checking routine.
     */
    bool checkNewSolution( void ) override;

    /**
     * return a pointer to the IDA time operator, deprecated
     */
    std::shared_ptr<IDATimeOperator> getIDATimeOperator() const; // BP, can go

    /**
     * return a pointer to the linear time operator used by the preconditioner
     */
    std::shared_ptr<LinearTimeOperator> getLinearTimeOperator() const;

    /**
     * return a shared pointer to the residual vector
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> getResidualVector() const;

    /**
     * added by Gary, need discussion whether to deprecate
     */
    bool getBoolManufacturedProblem( void ) { return d_bManufacturedProblem; }

    /**
     * return a shared pointer to the source term at the current time step
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> getSourceTerm() override;

    /**
     * return a shared pointer to the preconditioner being used
     */
    inline std::shared_ptr<AMP::Solver::SolverStrategy> getPreconditioner( void )
    {
        return d_pNestedSolver;
    }

    /**
     * return a void * pointer to the IDA_mem data structure used by IDA
     */
    inline void *getIDAMem( void ) { return d_ida_mem; }

    std::string type() const override { return "IDATimeIntegrator"; }

    /**
     * not sure why this variable exists and is public, BP
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_residual;

private:
    /**
     * Constructor.
     */
    IDATimeIntegrator();

    void initializeIDA();

    /**
     * Read data from input database.
     */
    void getFromInput( std::shared_ptr<AMP::Database> input_db ); // note that this was "protected"
                                                                  // (and not virtual) in
                                                                  // TimeIntegrator.h
    // and BackwardEulerTimeIntegrator has own its implementation of getFromInput - overridden...?

    // we definitely need this
    void setupVectors( void );

    static int IDAResTrial( realtype tt, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data );

    static int IDAPrecSetup( realtype tt,
                             N_Vector yy,
                             N_Vector yp,
                             N_Vector rr,
                             realtype cj,
                             void *user_data,
                             N_Vector tmp1,
                             N_Vector tmp2,
                             N_Vector tmp3 );

    static int IDAPrecSolve( realtype tt,
                             N_Vector yy,
                             N_Vector yp,
                             N_Vector rr,
                             N_Vector rvec,
                             N_Vector zvec,
                             realtype cj,
                             realtype delta,
                             void *user_data,
                             N_Vector tmp );

    void *d_ida_mem;

    double d_relative_tolerance;
    double d_absolute_tolerance;

    int d_linear_solver_type;

    // flag to decide whether to call the IDA consistent IC calculation (default=false)
    bool d_bCallCalcIC;

    // flag to decide whether to create the linear operator internally
    bool d_createLinearOperatorInternally;

    // flag to use preconditioning
    bool d_bUsePreconditioner;

    bool d_bLinearMassOperator;
    bool d_bLinearRhsOperator;
    bool d_bManufacturedProblem;

    std::shared_ptr<IDATimeOperator> d_pIDATimeOperator; // BP, can go, but need to be careful
    std::shared_ptr<LinearTimeOperator> d_pLinearTimeOperator;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_solution_prime;

    std::shared_ptr<AMP::Solver::SolverStrategy> d_pNestedSolver;
};
} // namespace AMP::TimeIntegrator

#endif
#endif
