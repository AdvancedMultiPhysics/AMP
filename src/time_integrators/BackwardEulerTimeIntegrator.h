#ifndef included_AMP_BackwardEulerTimeIntegrator
#define included_AMP_BackwardEulerTimeIntegrator

#include <string>

#ifndef included_AMP_ImplicitTimeIntegrator
#include "ImplicitTimeIntegrator.h"
#endif

namespace AMP {
namespace TimeIntegrator {

/** \class BackwardEulerTimeIntegrator
 *
 * Class BackwardEulerTimeIntegrator is a concrete time integrator
 * that implements the backward Euler method.
 */
class BackwardEulerTimeIntegrator : public ImplicitTimeIntegrator
{
public:
    /**
     * Constructor that accepts parameter list.
     */
    explicit BackwardEulerTimeIntegrator( std::shared_ptr<TimeIntegratorParameters> parameters );

    /**
     * Destructor.
     */
    virtual ~BackwardEulerTimeIntegrator();

    /**
     * Initialize from parameter list.
     */
    void initialize( std::shared_ptr<TimeIntegratorParameters> parameters ) override;

    /**
     * Resets the internal state of the time integrator as needed.
     * A parameter argument is passed to allow for general flexibility
     * in determining what needs to be reset Typically used after a regrid.
     */
    void reset( std::shared_ptr<TimeIntegratorParameters> parameters ) override;

    /**
     * Specify initial time step.
     */
    double getInitialDt();

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
                          const double old_dt ) override;

    /**
     * Update state of the solution.
     */
    void updateSolution( void ) override;

    /**
     * Check time advanced solution to determine whether it is acceptable.
     * Return true if the solution is acceptable; return false otherwise.
     * The integer argument is the return code generated by the call to the
     * nonlinear solver "solve" routine.   The meaning of this value depends
     * on the particular nonlinear solver in use and must be intepreted
     * properly by the user-supplied solution checking routine.
     */
    bool checkNewSolution( void ) const override;

protected:
    void initializeTimeOperator( std::shared_ptr<TimeIntegratorParameters> parameters ) override;

private:
    /**
     * Constructor.
     */
    BackwardEulerTimeIntegrator();

    /**
     * Read data from input database.
     */
    void getFromInput( std::shared_ptr<AMP::Database> input_db );

    /**
     * setup the vectors used by BE
     */
    void setupVectors( void );

    double d_initial_dt;
};
} // namespace TimeIntegrator
} // namespace AMP

#endif
