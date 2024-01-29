#ifndef included_AMP_ImplicitIntegrator
#define included_AMP_ImplicitIntegrator

#include <string>

#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/time_integrators/TimeOperator.h"
#include "AMP/vectors/Scalar.h"

// Forward declares
namespace AMP {
class Database;
}
namespace AMP::Solver {
class SolverStrategy;
}

namespace AMP::TimeIntegrator {

/*!
 * @brief Manage implicit time integration
 *
 * Class ImplicitIntegrator manages implicit time integration
 * over  a mesh.  It maintains references
 * to an Operator and SolverStrategy
 * objects, which provide operations describing the implicit equations
 * and solving the problem at each time step, respectively.
 *
 * <b> Input Parameters </b>
 *
 * <b> Definitions: </b>
 *    - \b initial_time
 *       initial simulation time.
 *    - \b final_time
 *       final simulation time.
 *    - \b max_integrator_steps
 *       maximum number of timesteps performed on the coarsest hierarchy level
 *       during the simulation.
 *
 * All input data items described above, except for initial_time, may be
 * overridden by new input values when continuing from restart.
 *
 * <b> Details: </b> <br>
 * <table>
 *   <tr>
 *     <th>parameter</th>
 *     <th>type</th>
 *     <th>default</th>
 *     <th>range</th>
 *     <th>opt/req</th>
 *     <th>behavior on restart</th>
 *   </tr>
 *   <tr>
 *     <td>initial_time</td>
 *     <td>double</td>
 *     <td>none</td>
 *     <td>>=0</td>
 *     <td>req</td>
 *     <td>May not be modified by input db on restart</td>
 *   </tr>
 *   <tr>
 *     <td>final_time</td>
 *     <td>double</td>
 *     <td>none</td>
 *     <td>final_time >= initial_time</td>
 *     <td>req</td>
 *     <td>Parameter read from restart db may be overridden by input db</td>
 *   </tr>
 *   <tr>
 *     <td>max_integrator_steps</td>
 *     <td>int</td>
 *     <td>none</td>
 *     <td>>=0</td>
 *     <td>req</td>
 *     <td>Parameter read from restart db may be overridden by input db</td>
 *   </tr>
 * </table>
 *
 * A sample input file entry might look like:
 *
 * @code
 * initial_time = 0.0
 * final_time   = 1.0
 * max_integrator_steps = 100
 * @endcode
 *
 * @see AMP::Operator::Operator
 * @see AMP::Solver::SolverStrategy
 */
class ImplicitIntegrator : public AMP::TimeIntegrator::TimeIntegrator
{
public:
    /**
     * The constructor for ImplicitIntegrator initializes the
     * default state of the integrator.  The integrator is configured with
     * the concrete strategy objects in the argument list that provide
     * operations related to the nonlinear solver and implicit equations
     * to solve.  Data members are initialized from the input and restart
     * databases.
     *
     * Note that no vectors are created in the constructor.  Vectors are
     * created and the nonlinear solver is initialized in the initialize()
     * member function.
     *
     * @pre !object_name.empty()
     * @pre implicit_equations != 0
     * @pre nonlinear_solver != 0
     * @pre hierarchy
     */
    explicit ImplicitIntegrator(
        std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> prm );

    ImplicitIntegrator() = delete;

    // The following are not implemented:
    ImplicitIntegrator( const ImplicitIntegrator & ) = delete;
    void operator=( const ImplicitIntegrator & ) = delete;

    /**
     * Empty destructor for ImplicitIntegrator
     */
    virtual ~ImplicitIntegrator();

    /**
     * Initialize state of time integrator.  This includes creating
     * solution vector and initializing solver components.
     */
    void initialize( void );

    /**
     * reset parameters so that the time integrator is ready to start at the
     * initial time
     */
    void reset( void );

    /**
     * Resets the internal state of the time integrator as needed.
     * A parameter argument is passed to allow for general flexibility
     * in determining what needs to be reset Typically used after a regrid.
     */
    void reset(
        std::shared_ptr<const AMP::TimeIntegrator::TimeIntegratorParameters> parameters ) override;

    /*!
     * @brief Integrate entire patch hierarchy through the
     * specified time increment.
     *
     * Integrate entire patch hierarchy through the specified time
     * increment.  The time advance assumes the use of a nonlinear
     * solver to implicitly integrate the discrete equations.  The integer
     * return value is the return code generated by the particular solver
     * package in use.  It is the user's responsibility to interpret this
     * code in a manner consistent with the solver they are using.
     *
     * The boolean first_step argument is true when this is the very
     * first call to the advance function or if the call occurs immediately
     * after the hierarchy has changed due to regridding.  Otherwise it is
     * false.  Note that, when the argument is true, the use of extrapolation
     * to construct the initial guess for the advanced solution may not be
     * possible.
     *
     *
     * @param dt Time step size
     * @param first_step Whether this is the first step after grid change
     *
     */
    int advanceSolution( const double dt,
                         const bool first_step,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> out ) override;

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
     * Update solution (e.g., reset pointers for solution data, update
     * dependent variables, etc.) after time advance.  It is assumed that
     * when this routine is invoked, an acceptable new solution has been
     * computed.  The double return value is the simulation time corresponding
     * to the advanced solution.
     */
    void updateSolution( void ) override;

    /**
     * Return time increment for next solution advance.  Timestep selection
     * is generally based on whether the nonlinear solution iteration
     * converged and, if so, whether the solution meets some user-defined
     * criteria.  This routine assumes that, before it is called, the
     * routine checkNewSolution() was called.  The boolean argument is the
     * return value from that call.  The integer argument is the return code
     * generated by the nonlinear solver package that computed the solution.
     */
    virtual double getNextDt( const bool good_solution ) override;

    /**
     * Return true if the number of integration steps performed by the
     * integrator has not reached the specified maximum; return false
     * otherwise.
     */
    bool stepsRemaining() const override { return d_integrator_step < d_max_integrator_steps; }

    /**
     * Print out all members of integrator instance to given output stream.
     */
    virtual void printClassData( std::ostream &os ) const;

    /**
     * Returns the object name.
     */
    const std::string &getObjectName() const { return d_object_name; }

    std::shared_ptr<AMP::Solver::SolverStrategy> getSolver( void ) { return d_solver; }

    using TimeIntegrator::initialize;

    //! print the statistics on the solver
    void printStatistics( std::ostream &os = AMP::pout ) override;

    //! for multiphysics problems it may be necessary to scale the solution
    // and nonlinear function for correct solution of the implicit problem
    // each timestep. The first vector is for solution scaling, the second for function
    void setComponentScalings( std::shared_ptr<AMP::LinearAlgebra::Vector> s,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> f )
    {
        d_solution_scaling = s;
        d_function_scaling = f;
        AMP_INSIST( d_operator,
                    "Operator must be registered prior to calling setComponentScalings" );
        auto timeOperator =
            std::dynamic_pointer_cast<AMP::TimeIntegrator::TimeOperator>( d_operator );
        AMP_INSIST( timeOperator, "setComponentScalings only works with TimeOperator" );
        timeOperator->setComponentScalings( s, f );
    }

    void setTimeScalingFunction( std::function<void( AMP::Scalar )> fnPtr )
    {
        d_fTimeScalingFnPtr = fnPtr;
    }

    double getGamma( void ) const override;

protected:
    /**
     * Set the initial guess for the time advanced solution at the start
     * of the nonlinear iteration.  The boolean argument first_step
     * indicates whether we are at the first step on the current hierarchy
     * configuration.  This is true when the hierarchy is constructed
     * initially and after regridding.  In these cases, setting the initial
     * iterate using extrapolation, for example, may not be possible.
     *
     * Function overloaded from algs::ImplicitEquationStrategy.
     */
    virtual void setInitialGuess( const bool first_step,
                                  const double current_time,
                                  const double current_dt,
                                  const double old_dt );

    virtual void setTimeHistoryScalings() {}

    /*
     * Pointers to implicit equation and solver strategy objects and patch
     * hierarchy.  The strategies provide nonlinear equation and solver
     * routines for treating the nonlinear problem on the hierarchy.
     */
    std::shared_ptr<AMP::Solver::SolverStrategy> d_solver = nullptr;

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_solution_scaling;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_function_scaling;

    /*
     * Data members representing integrator times, time increments,
     * and step count information.
     */

    int d_solver_retcode = 0; //! the return code from the implicit solver at the
                              //! current timestep

    bool d_first_step = true; //! whether it is the first step


    //! allows for specialization of advanceSolution for classes of time
    //! integrators
    virtual int
    integratorSpecificAdvanceSolution( const double dt,
                                       const bool first_step,
                                       std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                                       std::shared_ptr<AMP::LinearAlgebra::Vector> out );

    std::function<void( AMP::Scalar )> d_fTimeScalingFnPtr;

private:
    //! allows for specialization of the checkNewSolution for classes of time
    //! integrators
    virtual bool integratorSpecificCheckNewSolution( const int solver_retcode );

    //! allows for specialization of updateSolution for classes of time
    //! integrators
    virtual void integratorSpecificUpdateSolution( double time );

    //! allows for specialization of getNextDt for classes of time integrators
    virtual double integratorSpecificGetNextDt( const bool good_solution,
                                                const int solver_retcode );

    //! allows for specialization of the initialization routine for classes of
    //! time integrators
    virtual void integratorSpecificInitialize( void );
};

} // namespace AMP::TimeIntegrator

#endif
