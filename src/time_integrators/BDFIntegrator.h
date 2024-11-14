#ifndef included_BDFIntegrator_h_
#define included_BDFIntegrator_h_

#include <limits>
#include <list>

#include "AMP/time_integrators/ImplicitIntegrator.h"

#ifdef ENABLE_RESTART
class RestartData;
#endif

namespace AMP {
class Database;
namespace Operator {
class Operator;
}

namespace TimeIntegrator {

using DataManagerCallBack = std::function<void( std::shared_ptr<AMP::LinearAlgebra::Vector> )>;

class BDFIntegrator : public AMP::TimeIntegrator::ImplicitIntegrator
{
public:
    explicit BDFIntegrator( AMP::TimeIntegrator::TimeIntegratorParameters::shared_ptr );
    ~BDFIntegrator();

    static std::unique_ptr<AMP::TimeIntegrator::TimeIntegrator> createTimeIntegrator(
        std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters );

    //! returns the total number of timestep rejections
    int getNumberOfStepRejections( void ) { return d_total_steprejects; }

    /**
     * Return time increment for advancing the solution at the first timestep.
     */
    using AMP::TimeIntegrator::TimeIntegrator::getInitialDt;
    double getInitialDt() override;

    //! return the factor d_current_dt*getGamma() used to scale the rhs operator
    double getTimeOperatorScaling( void );

    /**
     * Set the initial guess for the time advanced solution at the start
     * of the nonlinear iteration.  The boolean argument first_step
     * indicates whether we are at the first step on the current mesh
     * configuration.  This is true when the mesh is constructed
     * initially and after regridding.  In these cases, setting the initial
     * iterate using extrapolation, for example, may not be possible.
     */
    void setInitialGuess( const bool first_step,
                          const double current_time,
                          const double current_dt,
                          const double old_dt ) override;

    void setIsNewTimeStep( bool bval ) { d_is_new_timestep = bval; }

    /**
     * registering and allocation of variables
     */
    void initializeVariables( bool is_from_restart );

    /**
     * Set flag to indicate whether regridding is taking place.
     */
    void setRegridStatus( bool is_after_regrid );

    void setIterationCounts( const int nli, const int li );

    int getFunctionEvaluationCount( void ) { return d_evaluatefunction_count; }
    /**
     * Reset cached information that depends on the mesh configuration.
     *
     */
    void reset(
        std::shared_ptr<const AMP::TimeIntegrator::TimeIntegratorParameters> parameters ) override;

    //! print the statistics on the implicit equations
    void printStatistics( std::ostream &os = AMP::pout ) override;

    int
    integratorSpecificAdvanceSolution( const double dt,
                                       const bool first_step,
                                       std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                                       std::shared_ptr<AMP::LinearAlgebra::Vector> out ) override;

    void registerDataManagerCallback( DataManagerCallBack callBackFn )
    {
        d_registerVectorForManagement = callBackFn;
    }

    void registerVectorsForMemoryManagement( void );

    std::vector<double> getTimeHistoryScalings( void ) override;

    size_t sizeOfTimeHistory() const override { return d_max_integrator_index + 1; }

    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> getTimeHistoryVectors()
    {
        return d_prev_solutions;
    }

    std::shared_ptr<AMP::LinearAlgebra::Vector> getTimeHistorySourceTerm() override
    {
        return d_integrator_source_vector;
    }

    std::string type() const override { return "BDFIntegrator"; }

public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    void writeRestart( int64_t fid ) const override;

    /**
     * \brief    Read restart data to file
     * \details  This function will create a variable from the restart file
     * \param fid    File identifier to write
     * \param manager   Restart manager
     */
    BDFIntegrator( int64_t fid, AMP::IO::RestartManager *manager );

protected:
    /*
     * Helper functions.
     */

    /**
     * read in parameters from input database
     */
    void getFromInput( std::shared_ptr<AMP::Database> input_db, bool is_from_restart );

    /**
     * Return the next time increment through which to advance the solution.
     * The good_solution is the value returned by a call to checkNewSolution(),
     * which determines whether the computed solution is acceptable or not.
     * The integer solver_retcode is the return code generated by the
     * nonlinear solver.   This value must be interpreted in a manner
     * consistant with the solver in use.
     */
    double integratorSpecificGetNextDt( const bool good_solution,
                                        const int solver_retcode ) override;

    double getNextDtTruncationError( const bool good_solution, const int solver_retcode );
    double getNextDtPredefined( const bool good_solution, const int solver_retcode );
    double getNextDtConstant( const bool good_solution, const int solver_retcode );
    double getNextDtFinalConstant( const bool good_solution, const int solver_retcode );

    void integratorSpecificInitialize( void ) override;

    //  int integratorSpecificAdvanceSolution( const double dt, const bool first_step ) override;

    /**
     * Check the computed solution and return true if it is acceptable;
     * otherwise return false.  The integer solver_retcode is the return
     * code generated by the nonlinear solver.  This value must be
     * interpreted in a manner consistent with the solver in use.
     */
    bool integratorSpecificCheckNewSolution( const int solver_retcode ) override;

    /**
     * Update solution storage and dependent quantities after computing an
     * acceptable time advanced solution.   The new_time value is the new
     * solution time.
     *
     * Function overloaded from algs::ImplicitEquationStrategy.
     */
    void integratorSpecificUpdateSolution( const double new_time ) override;

    //! calculates the linear weighted sum of previous time solutions that forms a source
    //! term for the nonlinear equations at each timestep
    void computeIntegratorSourceTerm( void );

    //! estimate timestep based on relative change in temperature and energy
    double estimateDynamicalTimeScale( double current_dt );

    //! estimate timestep based on truncation error estimates
    double estimateDtWithTruncationErrorEstimates( double current_dt, bool good_solution );

    //! estimate the speed of the Marshak wave front, used as a heuristic within some timestep
    //! calculations
    void estimateFrontSpeed();

    //! evaluates the predictor and stores in the predictor energy and temperature variables
    void evaluatePredictor( void );

    //! evaluates the AB2 predictor and stores in the predictor energy and temperature variables
    void evaluateAB2Predictor( void );

    //! evaluates the leap frog predictor and stores in the predictor energy and temperature
    //! variables
    void evaluateLeapFrogPredictor( void );

    //! evaluates a predictor based on an interpolating polynomial
    void evaluateBDFInterpolantPredictor( void );

    //! evaluates the forward Euler predictor and stores in the predictor energy and temperature
    //! variables
    void evaluateForwardEulerPredictor( void );

    //! estimates the BDF2 time derivative using the approach in Gresho and Sani
    void estimateBDF2TimeDerivative( void );

    //! estimates the CN time derivative using the approach in Gresho and Sani
    void estimateCNTimeDerivative( void );

    //! estimates the BE time derivative using the approach in Gresho and Sani
    void estimateBETimeDerivative( void );

    //! wrapper for estimating the time derivative after a succesful step
    void estimateTimeDerivative( void );

    /**
     * Utility function that calculates the scaling term for the local truncation error (LTE)
     * expressions obtained by doing Taylor series expansions at a given timestep
     */
    double calculateLTEScalingFactor( void );

    /**
     * Utility function to calculate the scaled norm of the LTE error.
     */
    void calculateScaledLTENorm( std::shared_ptr<AMP::LinearAlgebra::Vector> x,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> y,
                                 std::vector<double> &norms );
    /**
     * routine to calculate the current truncation error
     */
    void calculateTemporalTruncationError( void );

    /**
     * prints out norms of vector components calculated individually, prefix and postfix are strings
     * that the user can customize to make the message more informational
     */
    void printVectorComponentNorms( const std::shared_ptr<AMP::LinearAlgebra::Vector> &vec,
                                    const std::string &prefix,
                                    const std::string &postfix,
                                    const std::string &norm );

    /**
     * Returns on the max timestep that the predictor can use without energy or temperature going
     * negative
     */
    double getPredictorTimestepBound( void );

    void setTimeHistoryScalings() override;

    /**
     * Set solution and function scalings for multi-physics scalings automatically
     */
    void setMultiPhysicsScalings( void );

    DataManagerCallBack d_registerVectorForManagement;

#ifdef ENABLE_RESTART
    std::shared_ptr<RestartData> d_restart_data;
#endif

    //! stores the solution vectors at previous time levels, only the number required
    //! for the time integrator to function are stored
    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> d_prev_solutions;

    //! stores linear weighted sum of previous time solutions that becomes a source term
    //! for the nonlinear system of equations
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_integrator_source_vector = nullptr;

    //! the next vector will store the predictor for the next timestep
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_predictor_vector = nullptr;

    //! the next vector will store the current estimate for the time derivative
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_timederivative_vector = nullptr;

    //! the next vector is used for scratch or intermediate storage
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_scratch_vector = nullptr;

    //! the next vector is used for scratch or intermediate storage for rhs
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_scratch_function_vector = nullptr;

    //! the next vector will store the rhs at the current time level
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_current_function_vector = nullptr;

    //! the next vector will store the old rhs at the previous time level
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_prev_function_vector = nullptr;

    //! the next vector will store the old time derivative at previous time level
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_old_td_vector = nullptr;

    //! vector to keep track of the number of nonlinear iterations
    std::vector<int> d_nonlinearIterations;

    //! vector to keep track of the number of linear iterations
    std::vector<int> d_linearIterations;

    //! vector to keep track of whether the step is accepted
    std::vector<int> d_step_accepted;

    //! vector of vectors to store the local truncation errors for components
    std::vector<std::vector<double>> d_LTE;

    //! vector to store the timesteps
    std::vector<double> d_timesteps;

    //! vector to the store the simulation times
    std::vector<double> d_times;

    //! truncation error order for time integrators
    std::vector<double> d_integrator_order;

    //! vector of previous timesteps
    std::vector<double> d_integrator_steps;

    //! names of time integrators
    std::vector<std::string> d_integrator_names;

    //! names of variables to associate with different vectors
    std::vector<std::string> d_vector_names;

    //! names of variables to associate with different components of a vector
    std::vector<std::string> d_var_names;

    //! time history scalings
    std::vector<double> d_a;

    //! norm type used for truncation error calculation
    std::string d_timeTruncationErrorNormType = "l2Norm";

    //! string for the time integrator being used, options are "Backward Euler", "BDF2", or "CN"
    std::string d_implicit_integrator = "BDF2";

    //! the one step method used to start up BDF2, options are "Backward Euler" or "CN"
    std::string d_bdf_starting_integrator = "CN";

    //! timestep strategy to use, options are "truncationErrorStrategy"
    //!  "predefined", "constant", "final_constant", "Rider-Knoll", "limit relative change"
    std::string d_timestep_strategy = "truncationErrorStrategy";

    //! type of PI controller, current options are "H211b", "PI.4.7", and "Deadbeat"
    std::string d_pi_controller_type = "PC.4.7";

    //! error scaling used for time local error calculations, options are "fixed_scaling" or
    //! "fixed_resolution"
    std::string d_time_error_scaling = "fixed_scaling";

    //! predictor to use with predictor corrector approach
    std::string d_predictor_type = "leapfrog";

    std::list<double> d_l2errorNorms_E;
    std::list<double> d_l2errorNorms_T;

    //! fixed used defined scales for time error scaling
    std::vector<double> d_problem_scales;

    //! double containing t_{n+1}
    double d_new_time = std::numeric_limits<double>::signaling_NaN();

    //! ratio of current to previous timestep
    double d_alpha = 1.0;

    //! max factor to cut timestep by when a timestep is rejected
    double d_DtCutLowerBound = std::numeric_limits<double>::signaling_NaN();

    //! upper bound on factor by which timestep can be increased
    double d_DtGrowthUpperBound = std::numeric_limits<double>::signaling_NaN();

    //! absolute tolerance for the truncation error based strategy
    double d_time_atol = std::numeric_limits<double>::signaling_NaN();

    //! relative tolerance for the truncation error based strategy
    double d_time_rtol = std::numeric_limits<double>::signaling_NaN();

    //! truncation error estimate at previous step
    double d_prevTimeTruncationErrorEstimate = 1.0;

    //! truncation error estimate in time over both E and T
    double d_timeTruncationErrorEstimate = 1.0;

    //! truncation error ratio to previous step
    double d_timeErrorEstimateRatio = 1.0;

    //! iterations becomes too large, default is 0.75
    //! target fractional change in energy for each timestep
    double d_target_relative_change = 0.1;

    //! factor to multiply rhs by
    double d_gamma = 0.0;

    //! index into d_integrator_names that shows what is the max BDF order
    size_t d_max_integrator_index = -1;

    //! index into d_integrator_names showing what is the current integrator
    size_t d_integrator_index = 0;

    //! keeps track of the number of timesteps after the last regrid
    int d_timesteps_after_regrid = 0;

    //! the number of timesteps after a regrid before the pi controller is enabled
    int d_enable_picontrol_regrid_steps = 0;

    //! number of steps to use bdf1 error estimator after regrid
    int d_bdf1_eps_regrid_steps = 0;

    //! keeps track of successive step rejections at a given timestep
    int d_current_steprejects = 0;

    //! keeps track of successive step acceptances
    int d_current_stepaccepts = 0;

    //! convergence reason returned by solver
    int d_solver_converged_reason = -1;

    //! records number of nonlinear iterations at current timestep
    int d_nonlinear_iterations = 0;

    //! records number of linear iterations at current timestep
    int d_linear_iterations = 0;

    //! counter to track number of nonlinear function calls
    int d_evaluatefunction_count = 0;

    //! used only with the final constant timestep scheme, number of
    //! time intervals before the final constant timestep is attained
    int d_number_of_time_intervals = -1;

    //! used only with the final constant timestep scheme, number of
    //! initial steps to take with d_initial_dt
    int d_number_initial_fixed_steps = 0;

    //! enable auto scaling if true
    bool d_auto_component_scaling = false;

    //! used only with the final constant timestep scheme, counter
    //! to keep track of current step
    int d_final_constant_timestep_current_step = 1;

    //! whether to log statistics about time integrator
    bool d_log_statistics = true;

    //! state of simulation, whether it is immediately after a regrid
    bool d_is_after_regrid = false;

    //! boolean flag to determine if a new timestep has started
    bool d_is_new_timestep = true;

    //! boolean flag to combine different time error estimators
    bool d_combine_timestep_estimators = false;

    //! boolean to minimize the number of timestep changes in the truncation error strategy
    bool d_control_timestep_variation = false;

    //! boolean flag to use a predictor for the initial guess at each timestep
    bool d_use_predictor = true;

    //! boolean flag to incrementally add in predictor for very first step using forward Euler or
    //! AB2
    bool d_use_initial_predictor = true;

    //! use an error estimator based on BDF1 for the first 3 steps during and immediately after
    //! regrid
    bool d_use_bdf1_estimator_on_regrid = true;

    //! boolean flag to keep track of successive time step rejections
    bool d_prevSuccessiveRejects = false;

    //! boolean flag used to do constant time interpolation, used after regrids
    bool d_use_constant_time_interpolation = false;

    //! boolean flag to enable or disable the use of a PI based timestep controller
    bool d_use_pi_controller = true;

    //! calculate time truncation error if enabled
    bool d_calculateTimeTruncError = false;

    //! boolean flag to detect a call to reset immediately following restart
    bool d_reset_after_restart = false;

    //! boolean to ensure registerVectorsForMemoryManagement is called once
    bool d_vectors_registered_for_mgmt = false;
};

} // namespace TimeIntegrator
} // namespace AMP

#endif
