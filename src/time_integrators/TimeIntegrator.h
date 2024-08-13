#ifndef included_AMP_TimeIntegrator
    #define included_AMP_TimeIntegrator

    #include "AMP/operators/Operator.h"
    #include "AMP/time_integrators/TimeIntegratorParameters.h"
    #include "AMP/utils/Database.h"
    #include "AMP/vectors/Vector.h"

    #include <cstdint>
    #include <memory>
    #include <string>


// Declare some classes
namespace AMP::IO {
class Writer;
class RestartManager;
} // namespace AMP::IO


namespace AMP::TimeIntegrator {

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

    /** \brief Return the name of the TimeIntegrator
     */
    virtual std::string type() const;

    //! Get a unique id hash for the vector
    uint64_t getID() const;

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
    virtual void reset( std::shared_ptr<const TimeIntegratorParameters> parameters ) = 0;

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
     * @param dt            Time step size
     * @param first_step    Whether this is the first step
     * @param in            Input vector
     * @param out           Output vector
     *
     * @return value is the return code generated by the particular solver
     * package in use
     */
    virtual int advanceSolution( const double dt,
                                 const bool first_step,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> out ) = 0;

    /**
     * Check time advanced solution to determine whether it is acceptable.
     * Return true if the solution is acceptable; return false otherwise.
     * The meaning of this value must be intepreted
     * properly by the user-supplied solution checking routine.
     */
    virtual bool checkNewSolution( void ) = 0;

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
    virtual std::shared_ptr<AMP::LinearAlgebra::Vector> getSolution( void )
    {
        return d_solution_vector;
    }

    /**
     * @brief  Return time increment for next solution advance.
     * @details Return time increment for next solution advance.  Timestep selection
     * is generally based on whether the solution meets some user-defined
     * criteria.  This routine assumes that, before it is called, the
     * routine checkNewSolution() was called.  The boolean argument is the
     * return value from that call.
     */
    virtual double getNextDt( const bool good_solution );

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
     * get initial time step.
     */
    virtual double getInitialDt() { return d_initial_dt; };

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
    virtual void registerWriter( std::shared_ptr<AMP::IO::Writer> writer ) { d_writer = writer; }

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

    virtual void registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
    {
        d_operator = op;
    }

    std::shared_ptr<AMP::Operator::Operator> getOperator( void ) { return d_operator; }

    virtual int getTotalRejectedSteps() const { return d_total_steprejects; }

    virtual void setCurrentDt( const double dt ) { d_current_dt = dt; }

    virtual void setInitialDt( const double dt ) { d_initial_dt = dt; }

    virtual void setCurrentTime( const double t ) { d_current_time = t; }

    virtual void setInitialTime( const double t ) { d_initial_time = t; }

    virtual void setFinalTime( const double t ) { d_final_time = t; }

    virtual void printStatistics( std::ostream &os = AMP::pout ) { NULL_USE( os ); }

    virtual void setSourceTerm( AMP::LinearAlgebra::Vector::shared_ptr src )
    {
        d_pSourceTerm = src;
    }
    virtual AMP::LinearAlgebra::Vector::shared_ptr getSourceTerm( void ) { return d_pSourceTerm; }

    virtual double getGamma( void ) { return d_current_dt; }

    /**
     * \brief  Scaling factors for previous time history vectors
     * \details For RK methods the previous time vector is always scaled by 1. For
     * multistep methods this vector can consist of multiple scalings with the first
     * entry being the scaling for $y^{n-1}$, the next being for $y^{n-2}$ etc
     */
    virtual std::vector<double> getTimeHistoryScalings( void )
    {
        return std::vector<double>( 1, 1.0 );
    }

    /*
     * Returns the number of previous solutions that are stored
     */
    virtual size_t sizeOfTimeHistory() const { return 1; }

protected:
    /*
     * Read input data from specified database and initialize class members.
     * If run is from restart, a subset of the restart values may be replaced
     * with those read from input.
     *
     * When assertion checking is active, the database pointer must be non-null.
     */
    void getFromInput( std::shared_ptr<const AMP::Database> db );

    /*
     * Read object state from restart database and initialize class members.
     * Check that class and restart version numbers are the same.
     */
    void getFromRestart();

    /*
     * String used to identify specific class instantiation.
     */
    std::string d_object_name;

    /**
     * pointer to parameters
     */
    std::shared_ptr<TimeIntegratorParameters> d_pParameters;

    /*
     * Initial conditions vector
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_ic_vector = nullptr;

    /*
     * Solution vector advanced during the time integration process.
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_solution_vector;

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
    double d_initial_time = std::numeric_limits<double>::signaling_NaN();
    double d_final_time   = std::numeric_limits<double>::signaling_NaN();
    double d_current_time = std::numeric_limits<double>::signaling_NaN();

    //! initial time increment
    double d_initial_dt = std::numeric_limits<double>::signaling_NaN();
    double d_current_dt = std::numeric_limits<double>::signaling_NaN();
    double d_old_dt     = std::numeric_limits<double>::signaling_NaN();
    double d_min_dt     = std::numeric_limits<double>::signaling_NaN();

    //! maximum allowable timestep (user defined)
    double d_max_dt = std::numeric_limits<double>::max();

    int d_iDebugPrintInfoLevel = 0;
    int d_integrator_step      = 0;
    int d_max_integrator_steps = 0;

    int d_total_steprejects = 0; //! keeps track of total number of step rejections

    // Writer for internal data
    std::shared_ptr<AMP::IO::Writer> d_writer;

    TimeIntegrator() = default;

public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    virtual void writeRestart( int64_t fid ) const;

    /**
     * \brief    Read restart data to file
     * \details  This function will create a variable from the restart file
     * \param fid    File identifier to write
     * \param manager   Restart manager
     */
    TimeIntegrator( int64_t fid, AMP::IO::RestartManager *manager );

private:
    // The following are not implemented:
    explicit TimeIntegrator( const TimeIntegrator & ) = delete;
    void operator=( const TimeIntegrator & )          = delete;
};

} // namespace AMP::TimeIntegrator

#endif

#include "AMP/time_integrators/TimeIntegrator.I"
