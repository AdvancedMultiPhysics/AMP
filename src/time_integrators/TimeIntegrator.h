#ifndef included_AMP_TimeIntegrator
#define included_AMP_TimeIntegrator

#include "AMP/operators/Operator.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Writer.h"
#include "AMP/vectors/Vector.h"
#include <memory>

#include <string>


namespace AMP {
namespace TimeIntegrator {

/*!
  @brief Abstract base class for time integration

  Class TimeIntegrator is an abstract base class for managing
  time integration

  Initialization of an TimeIntegrator object is performed through
  a TimeIntegratorParameters object

 */
class TimeIntegrator
{
public:
    //! Convience typedef
    typedef std::shared_ptr<AMP::TimeIntegrator::TimeIntegrator> shared_ptr;

    /**
     * The constructor for TimeIntegrator initializes the
     * default state of the integrator. Data members are
     * initialized from the input and restart databases.
     *
     * Note that no vectors are created in the constructor.  Vectors are
     * created and initialized in the initialize()
     * member function.
     *
     */
    explicit TimeIntegrator( std::shared_ptr<TimeIntegratorParameters> parameters );

    /**
     * Empty destructor for TimeIntegrator
     */
    virtual ~TimeIntegrator();

    /**
     * @brief  Initialize state of time integrator.
     * @details  Initialize state of time integrator.  This includes
     * creating solution vector and initializing solver components.
     */
    virtual void initialize( std::shared_ptr<TimeIntegratorParameters> parameters );

    /**
     * @brief  Resets the internal state of the time integrator.
     * @details Resets the internal state of the time integrator as needed.
     * A parameter argument is passed to allow for general flexibility
     * in determining what needs to be reset.
     */
    virtual void reset( std::shared_ptr<TimeIntegratorParameters> parameters ) = 0;

    /*!
     * @brief Integrate through the specified time increment.
     *
     * @details  Integrate through the specified time increment.
     *
     * The boolean first_step argument is true when this is the very
     * first call to the advance function.  Otherwise it is false.
     * Note that, when the argument is true, the use of extrapolation
     * to construct the initial guess for the advanced solution may not be
     * possible.
     *
     * This function may, or may not, copy the new solution into the
     * solution vector at the previous time step (updateSolution).
     * If it is copied, this can reduce the memory requirements, but
     * restricts the implementation to those functions that will not be
     * called multiple times within a time step.  If it is not copied,
     * the update solution function must be called after each time step.
     * The concrete implementation needs to make this clear.
     *
     * @param dt Time step size
     * @param first_step Whether this is the first step
     *
     * @return value is the return code generated by the particular solver
     * package in use
     */
    virtual int advanceSolution( const double dt, const bool first_step ) = 0;

    /**
     * Check time advanced solution to determine whether it is acceptable.
     * Return true if the solution is acceptable; return false otherwise.
     * The meaning of this value must be intepreted
     * properly by the user-supplied solution checking routine.
     */
    virtual bool checkNewSolution( void ) const = 0;

    /**
     * @brief Update solution after time advance.
     *
     * @details  Update solution (e.g., reset pointers for solution data, update
     * dependent variables, etc.) after time advance.  It is assumed that
     * when this routine is invoked, an acceptable new solution has been
     * computed.  The double return value is the simulation time corresponding
     * to the advanced solution.
     *
     * In particular, this is designed to copy the solution vector
     * advanced during the time integration process to the solution
     * vector at the previous time and assumes that the new solution
     * has already been checked with the checkNewSolution function.
     */
    virtual void updateSolution( void ) = 0;

    /**
     * Retrieve the current solution.
     */
    virtual std::shared_ptr<AMP::LinearAlgebra::Vector> getCurrentSolution( void )
    {
        return d_solution;
    }

    /**
     * @brief  Return time increment for next solution advance.
     * @details Return time increment for next solution advance.  Timestep selection
     * is generally based on whether the solution meets some user-defined
     * criteria.  This routine assumes that, before it is called, the
     * routine checkNewSolution() was called.  The boolean argument is the
     * return value from that call.
     */
    virtual double getNextDt( const bool good_solution ) = 0;

    /**
     * @brief  Return initial integration time.
     * @details  Return initial integration time.
     */
    virtual double getInitialTime() const;

    /**
     * @brief  Return final integration time.
     * @details Return final integration time.
     */
    virtual double getFinalTime() const;

    /**
     * @brief  Return final integration time.
     * @details Return current integration time.
     */
    virtual double getCurrentTime() const;

    /**
     * @brief  Return current timestep.
     * @details Return current timestep.
     */
    virtual double getCurrentDt() const;

    /**
     * @brief  Return current integration step number.
     * @details Return current integration step number.
     */
    virtual int getIntegratorStep() const;

    /**
     * @brief  Return maximum number of integration steps.
     * @details Return maximum number of integration steps.
     */
    virtual int getMaxIntegratorSteps() const;

    /**
     * @brief Have the number of integration steps reached the maximum.
     * @details  Return true if the number of integration steps performed by the
     * integrator has not reached the specified maximum; return false
     * otherwise.
     */
    virtual bool stepsRemaining() const;

    /**
     * \brief  Append the vectors of interest to the solution vector
     * \details  This function will append the necessary vectors that this solver
     *  owns to the global vector provided.  Note that each solver may own any number
     *  of vectors, but no vector may be owned by multiple solvers.
     * \param vec   The vector to append
     */
    virtual void appendSolutionVector( AMP::LinearAlgebra::Vector::shared_ptr vec )
    {
        NULL_USE( vec );
    }

    /**
     * \brief  Append the vectors of interest to the rhs vector
     * \details  This function will append the necessary vectors that this solver
     *  owns to the global vector provided.  Note that each solver may own any number
     *  of vectors, but no vector may be owned by multiple solvers.
     * \param vec   The vector to append
     */
    virtual void appendRhsVector( AMP::LinearAlgebra::Vector::shared_ptr vec ) { NULL_USE( vec ); }

    /**
     * \brief  Registers a writer with the solver
     * \details  This function will register a writer with the solver.  The solver
     *  may then register any vector components it "owns" with the writer.
     * \param writer   The writer to register
     */
    virtual void registerWriter( std::shared_ptr<AMP::Utilities::Writer> writer )
    {
        d_writer = writer;
    }

    /**
     * Print out all members of integrator instance to given output stream.
     */
    void printClassData( std::ostream &os ) const;

    /**
     * Write out state of object to given database.
     *
     * When assertion checking is active, the database pointer must be non-null.
     */
    void putToDatabase( std::shared_ptr<AMP::Database> db );

    void registerOperator( std::shared_ptr<AMP::Operator::Operator> op ) { d_operator = op; }

    std::shared_ptr<AMP::Operator::Operator> getOperator( void ) { return d_operator; }

protected:
    /*
     * Read input data from specified database and initialize class members.
     * If run is from restart, a subset of the restart values may be replaced
     * with those read from input.
     *
     * When assertion checking is active, the database pointer must be non-null.
     */
    void getFromInput( const std::shared_ptr<AMP::Database> db );

    /*
     * Read object state from restart database and initialize class members.
     * Check that class and restart version numbers are the same.
     */
    void getFromRestart();

    /*
     * String used to identify specific class instantiation.
     */
    std::string d_object_name;

    /*
     * Solution vector advanced during the time integration process.
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_solution;

    /*
     * Solution vector at previous time
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pPreviousTimeSolution;

    /**
     * The operator is the right hand side operator for an explicit integrator when the time
     * integration problem is :
     * u_t = f(u)
     * but in the case of implicit time integrators the operator represents u_t-f(u)
     */
    std::shared_ptr<AMP::Operator::Operator> d_operator;

    /**
     * The operator is the left hand side mass operator (for FEM formulations)
     */
    std::shared_ptr<AMP::Operator::Operator> d_pMassOperator;

    /*
     * Source-sink vector, g,  when the time integration problem is of the form u_t = f(u)+g
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pSourceTerm;

    /*
     * Data members representing integrator times, time increments,
     * and step count information.
     */
    double d_initial_time;
    double d_final_time;
    double d_current_time;

    double d_current_dt;
    double d_old_dt;
    double d_min_dt;
    double d_max_dt;
    double d_initial_dt;

    int d_integrator_step;
    int d_max_integrator_steps;

    // Writer for internal data
    std::shared_ptr<AMP::Utilities::Writer> d_writer;

    // declare the default constructor to be private
    TimeIntegrator()
    {
        // initialize member data
        d_initial_time         = 0;
        d_final_time           = 0;
        d_current_time         = 0;
        d_current_dt           = 0;
        d_old_dt               = 0;
        d_min_dt               = 0;
        d_max_dt               = 0;
        d_initial_dt           = 0;
        d_integrator_step      = 0;
        d_max_integrator_steps = 0;
    };

private:
    // The following are not implemented:
    explicit TimeIntegrator( const TimeIntegrator & );
    void operator=( const TimeIntegrator & );
};
} // namespace TimeIntegrator
} // namespace AMP

#endif

#include "TimeIntegrator.I"
