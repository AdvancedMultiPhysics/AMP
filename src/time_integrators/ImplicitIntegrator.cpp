#include <fstream>
#include <stdio.h>
#include <stdlib.h>

#include "AMP/time_integrators/ImplicitIntegrator.h"

#include "AMP/IO/PIO.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/time_integrators/TimeOperator.h"
#include "AMP/time_integrators/TimeOperatorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/vectors/Vector.h"

namespace AMP::TimeIntegrator {

ImplicitIntegrator::ImplicitIntegrator(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> params )
    : AMP::TimeIntegrator::TimeIntegrator( params )
{
    d_initialized = false;
    AMP_ASSERT( params );
    auto db                      = d_pParameters->d_db;
    d_user_managed_time_operator = db->getWithDefault<bool>( "user_managed_time_operator", false );
    registerOperator( d_operator );
    d_initialized = true;
}

ImplicitIntegrator::~ImplicitIntegrator() = default;

/*
 *************************************************************************
 *
 * Initialize integrator and nonlinear solver:
 *
 * (1) Create vector containing solution state advanced in time.
 *
 * (2) Equation class registers data components with solution vector.
 *
 * (3) Initialize nonlinear solver.
 *
 *************************************************************************
 */

void ImplicitIntegrator::initialize() { integratorSpecificInitialize(); }

// default implementation
void ImplicitIntegrator::integratorSpecificInitialize()
{
    AMP_ERROR( "This should be defined in the derived class!!" );
}

void ImplicitIntegrator::createSolver( void )
{
    auto timeIntegratorDB = d_pParameters->d_db;
    AMP_ASSERT( timeIntegratorDB );

    auto solverName = timeIntegratorDB->getWithDefault<std::string>( "solver_name", "Solver" );

    std::shared_ptr<AMP::Database> solverDB;
    if ( timeIntegratorDB->keyExists( solverName ) ) {
        solverDB = timeIntegratorDB->getDatabase( solverName );
    } else {
        auto globalDB = d_pParameters->d_global_db;
        AMP_INSIST( globalDB, "Solver database not specified" );
        solverDB = globalDB->getDatabase( solverName );
    }

    // unless explicitly set, let solver use non-zero guess
    if ( !solverDB->keyExists( "zero_initial_guess" ) ) {
        solverDB->putScalar<bool>( "zero_initial_guess", false );
    }

    solverDB->print( AMP::plog );
    auto solver_params = std::make_shared<AMP::Solver::SolverStrategyParameters>( solverDB );

    solver_params->d_pOperator     = d_operator;
    solver_params->d_global_db     = d_pParameters->d_global_db;
    solver_params->d_pInitialGuess = d_solution_vector;
    d_solver                       = AMP::Solver::SolverFactory::create( solver_params );
}


void ImplicitIntegrator::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    d_operator = op;
    AMP_INSIST( d_operator, "Operator is null" );

    if ( d_user_managed_time_operator ) {
        d_operator = op;
    } else {
        // user supplied operators cannot be TimeOperators for now
        AMP_ASSERT( std::dynamic_pointer_cast<TimeOperator>( d_operator ) == nullptr );

        auto timeOperator_db = std::make_shared<AMP::Database>( "TimeOperatorDatabase" );
        timeOperator_db->putScalar( "name", "TimeOperator" );
        timeOperator_db->putScalar( "print_info_level", d_iDebugPrintInfoLevel );

        auto timeOperatorParameters =
            std::make_shared<AMP::TimeIntegrator::TimeOperatorParameters>( timeOperator_db );
        timeOperatorParameters->d_pSourceTerm  = d_pSourceTerm;
        timeOperatorParameters->d_pRhsOperator = d_operator;

        d_operator = std::make_shared<TimeOperator>( timeOperatorParameters );
    }
}

/*
 *************************************************************************
 *
 * Integrate solution through given time increment:
 *
 * (1) If number of levels in hierarchy has changed since last advance
 *     due to regridding, the range of levels in the vectors is reset.
 *
 * (2) Construct initial guess at new solution by extrapolation.
 *
 * (3) Call the equation advance set up routine.
 *
 * (4) Compute the new solution using the nonlinear solver.
 *
 * (5) Return integer return code define by nonlinear solver.
 *
 *************************************************************************
 */

int ImplicitIntegrator::advanceSolution( const double dt,
                                         const bool first_step,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> out )
{
    if ( d_iDebugPrintInfoLevel == 1 ) {
        AMP::printp(
            "Starting timestep %i, dt = %e, time = %e\n", d_integrator_step, dt, d_current_time );
    } else if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "\n++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        AMP::pout << "Begin of timestep # " << d_integrator_step << std::endl;
        AMP::pout << "Time increment is " << dt << std::endl;
        AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
    }

    d_current_dt = dt;

    // this call must be before the call to setInitialGuess as the computed
    // coefficients are used there
    setTimeHistoryScalings();

    auto rval = integratorSpecificAdvanceSolution( dt, first_step, in, out );

    d_time_history_initialized = false;
    return rval;
}

// provide a default implementation
int ImplicitIntegrator::integratorSpecificAdvanceSolution(
    const double dt,
    const bool first_step,
    std::shared_ptr<AMP::LinearAlgebra::Vector> in,
    std::shared_ptr<AMP::LinearAlgebra::Vector> out )
{

    AMP_ASSERT( stepsRemaining() && ( d_current_time < d_final_time ) );

    d_solution_vector = in;
    d_current_dt      = dt;

    if ( first_step ) {
        d_solution_vector->getVectorData()->reset();
        auto params = std::make_shared<AMP::Solver::SolverStrategyParameters>();
        params->d_vectors.push_back( d_solution_vector );

        //        d_solver->initialize( params );
    }

    setInitialGuess( first_step, d_current_time, d_current_dt, d_old_dt );

    // set a NULL rhs
    std::shared_ptr<AMP::LinearAlgebra::Vector> rhs = nullptr;
    d_solver->apply( rhs, d_solution_vector );
    const auto convStatus = d_solver->getConvergenceStatus();
    d_solver_retcode =
        convStatus <= AMP::Solver::SolverStrategy::SolverStatus::ConvergedUserCondition ? 1 : 0;

    out->copyVector( d_solution_vector );

    return d_solver_retcode;
}

void ImplicitIntegrator::setInitialGuess( const bool, const double, const double, const double )
{
    // this should probably be pure virtual eventually based on use cases
    AMP_ERROR( "Should be implemented in derived" );
}

/*
 *************************************************************************
 *
 * Get next dt from user-supplied equation class.  Timestep selection
 * is generally based on whether the nonlinear solution iteration
 * converged and, if so, whether the solution meets some user-defined
 * criteria.  It is assumed that, before this routine is called, the
 * routine checkNewSolution() has been called.  The boolean argument
 * is the return value from that call.  The integer argument is
 * that which is returned by the particular nonlinear solver package
 * that generated the solution.
 *
 *************************************************************************
 */

double ImplicitIntegrator::getNextDt( const bool good_solution )
{
    double dt_next = integratorSpecificGetNextDt( good_solution, d_solver_retcode );

    dt_next = std::min( dt_next, d_final_time - d_current_time );

    return dt_next;
}

// provide a default implementation
double ImplicitIntegrator::integratorSpecificGetNextDt( const bool, const int )
{
    // this should probably be pure virtual eventually base don use cases
    AMP_ERROR( "Should be implemented in derived" );
    return 0.0;
}

/*
 *************************************************************************
 *
 * Check whether time advanced solution is acceptable.  Note that the
 * user-supplied solution checking routine must interpret the integer
 * return code generated by the nonlinear solver correctly.
 *
 *************************************************************************
 */

bool ImplicitIntegrator::checkNewSolution()
{
    bool good_solution = integratorSpecificCheckNewSolution( d_solver_retcode );

    if ( ( d_iDebugPrintInfoLevel == 1 ) && ( !good_solution ) ) {
        AMP::pout << "Failed to advance solution past " << d_current_time << std::endl;
    } else if ( ( d_iDebugPrintInfoLevel > 1 ) && ( !good_solution ) ) {
        AMP::pout << "\n++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        AMP::pout << "End of timestep # " << d_integrator_step << std::endl;
        AMP::pout << "Failed to advance solution past " << d_current_time << std::endl;
        AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
    }

    return good_solution;
}

// default base class implementation
bool ImplicitIntegrator::integratorSpecificCheckNewSolution( const int )
{
    // this should probably be pure virtual eventually base don use cases
    AMP_ERROR( "Should be implemented in derived" );
    return false;
}

/*
 *************************************************************************
 *
 * Assuming an acceptable time advanced solution is found, update
 * solution quantities and time information state of integrator.
 * Return the current simulation time.
 *
 *************************************************************************
 */

void ImplicitIntegrator::updateSolution()
{
    d_current_time += d_current_dt;

    integratorSpecificUpdateSolution( d_current_time );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "\n++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        AMP::pout << "End of timestep # " << d_integrator_step << std::endl;
        AMP::pout << "Simulation time is " << d_current_time << std::endl;
        AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
    }

    d_integrator_step++;
}

// provide a default implementation
void ImplicitIntegrator::integratorSpecificUpdateSolution( double )
{
    // this should probably be pure virtual eventually base don use cases
    AMP_ERROR( "Should be implemented in derived" );
}

void ImplicitIntegrator::reset(
    std::shared_ptr<const AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
{
    if ( parameters ) {
        d_pParameters =
            std::const_pointer_cast<AMP::TimeIntegrator::TimeIntegratorParameters>( parameters );
        AMP_ASSERT( parameters->d_db );
        TimeIntegrator::getFromInput( parameters->d_db );
    }
}

void ImplicitIntegrator::reset()
{
    d_current_dt      = getInitialDt();
    d_old_dt          = 0.0;
    d_current_time    = d_initial_time;
    d_integrator_step = 0;
}

/*
 *************************************************************************
 *
 * Print class data members to given output stream.
 *
 *************************************************************************
 */

void ImplicitIntegrator::printClassData( std::ostream &os ) const
{
    os << "\nImplicitIntegrator::printClassData..." << std::endl;
    os << "ImplicitIntegrator: this = " << this << std::endl;
    os << "d_object_name = " << d_object_name << std::endl;
    d_solver->print( os );
    d_solution_vector->getVectorData()->print( os );
}


//! print the statistics on the solver
void ImplicitIntegrator::printStatistics( std::ostream &os )
{
    os << std::endl << "Printing integrator statistics" << std::endl;
    d_solver->printStatistics( os );
}

/*
 *************************************************************************
 *
 * Expose some time operator parameters
 *
 *************************************************************************
 */

void ImplicitIntegrator::setComponentScalings( std::shared_ptr<AMP::LinearAlgebra::Vector> s,
                                               std::shared_ptr<AMP::LinearAlgebra::Vector> f )
{
    d_solution_scaling = s;
    d_function_scaling = f;
    AMP_INSIST( d_operator, "Operator must be registered prior to calling setComponentScalings" );
    auto timeOperator = std::dynamic_pointer_cast<AMP::TimeIntegrator::TimeOperator>( d_operator );
    if ( timeOperator ) {
        timeOperator->setComponentScalings( s, f );
    } else {
        if ( d_fComponentScalingFnPtr )
            d_fComponentScalingFnPtr( s, f );
    }

    if ( d_solver )
        d_solver->setComponentScalings( s, f );
}


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
void ImplicitIntegrator::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    TimeIntegrator::registerChildObjects( manager );
}
void ImplicitIntegrator::writeRestart( int64_t fid ) const { TimeIntegrator::writeRestart( fid ); }

ImplicitIntegrator::ImplicitIntegrator( int64_t fid, AMP::IO::RestartManager *manager )
    : TimeIntegrator( fid, manager )
{
}

} // namespace AMP::TimeIntegrator
