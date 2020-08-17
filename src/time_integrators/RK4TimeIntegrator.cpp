#include "AMP/time_integrators/RK4TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/Utilities.h"

namespace AMP {
namespace TimeIntegrator {

/*
************************************************************************
*                                                                      *
*  Constructor.                                                        *
*                                                                      *
************************************************************************
*/
RK4TimeIntegrator::RK4TimeIntegrator( std::shared_ptr<TimeIntegratorParameters> parameters )
    : TimeIntegrator( parameters )
{
    initialize( parameters );
}

/*
************************************************************************
*                                                                      *
*  Destructor.                                                         *
*                                                                      *
************************************************************************
*/
RK4TimeIntegrator::~RK4TimeIntegrator() = default;

/*
************************************************************************
*                                                                      *
* Initialize.                                                          *
*                                                                      *
************************************************************************
*/
void RK4TimeIntegrator::initialize( std::shared_ptr<TimeIntegratorParameters> parameters )
{
    AMP_ASSERT( parameters.get() != (TimeIntegratorParameters *) nullptr );

    TimeIntegrator::initialize( parameters );

    setupVectors();

    /*
     * Initialize data members from input.
     */
    getFromInput( parameters->d_db );
}

void RK4TimeIntegrator::reset( std::shared_ptr<TimeIntegratorParameters> parameters )
{
    AMP_ASSERT( parameters.get() != (TimeIntegratorParameters *) nullptr );

    AMP_ERROR( "Not Finished" );
}

void RK4TimeIntegrator::setupVectors()
{

    // clone vectors so they have the same data layout as d_solution
    d_new_solution = d_solution->cloneVector( "new solution" );
    d_k1_vec       = d_solution->cloneVector( "k1 term" );
    d_k2_vec       = d_solution->cloneVector( "k2 term" );
    d_k3_vec       = d_solution->cloneVector( "k3 term" );
    d_k4_vec       = d_solution->cloneVector( "k4 term" );

    /* allocateVectorData no longer necessary
    d_new_solution->allocateVectorData();
    d_k1_vec->allocateVectorData();
    d_k2_vec->allocateVectorData();
    d_k3_vec->allocateVectorData();
    d_k4_vec->allocateVectorData();
    */

    /*
     * Set initial value of vectors to 0.
     */
    d_new_solution->setToScalar( (double) 0.0 );
    d_k1_vec->setToScalar( (double) 0.0 );
    d_k2_vec->setToScalar( (double) 0.0 );
    d_k3_vec->setToScalar( (double) 0.0 );
    d_k4_vec->setToScalar( (double) 0.0 );
}

int RK4TimeIntegrator::advanceSolution( const double dt, const bool )
{
    std::shared_ptr<AMP::LinearAlgebra::Vector> f;

    // k1 = f(tn,un)
    d_operator->apply( d_solution, d_k1_vec );
    // u* = un+k1*dt/2
    d_new_solution->axpy( dt / 2.0, *d_k1_vec, *d_solution );
    // k2 = f(t+dt/2, u*)
    d_operator->apply( d_new_solution, d_k2_vec );
    // u* = un+k2*dt/2
    d_new_solution->axpy( dt / 2.0, *d_k2_vec, *d_solution );
    // k3 = f(t+dt/2, u*)
    d_operator->apply( d_new_solution, d_k3_vec );
    // u* = un+k3*dt
    d_new_solution->axpy( dt, *d_k3_vec, *d_solution );
    // k4 = f(t+dt, u*)
    d_operator->apply( d_new_solution, d_k4_vec );
    // u_new = un+ dt*(k1+2*k2+2*k3+k4)/6
    d_k1_vec->add( *d_k1_vec, *d_k4_vec );
    d_k2_vec->add( *d_k2_vec, *d_k3_vec );
    d_k2_vec->scale( 2.0, *d_k2_vec );
    d_k1_vec->add( *d_k1_vec, *d_k2_vec );

    d_new_solution->axpy( dt / 6.0, *d_k1_vec, *d_solution );

    return ( 0 );
}

/*
************************************************************************
*                                                                      *
*  Check whether time advanced solution is acceptable.                 *
*                                                                      *
************************************************************************
*/
bool RK4TimeIntegrator::checkNewSolution() const
{
    /*
     * Ordinarily we would check the actual error in the solution
     * (proportional to the size of d_corrector) against a specified
     * tolerance.  For now, accept everything.
     */
    return ( true );
}

/*
************************************************************************
*                                                                      *
*  Update internal state to reflect time advanced solution.            *
*                                                                      *
************************************************************************
*/
void RK4TimeIntegrator::updateSolution()
{
    d_current_time += d_current_dt;
    d_solution->swapVectors( *d_new_solution );
}

/*
************************************************************************
*                                                                      *
* Read input from database.                                            *
*                                                                      *
************************************************************************
*/
void RK4TimeIntegrator::getFromInput( std::shared_ptr<AMP::Database> input_db )
{
    if ( input_db->keyExists( "initial_timestep" ) ) {
        d_initial_dt = input_db->getScalar<double>( "initial_timestep" );
    } else {
        AMP_ERROR( d_object_name << " -- Key data `initial_timestep'"
                                 << " missing in input." );
    }
}

double RK4TimeIntegrator::getNextDt( const bool ) { return d_current_dt; }
} // namespace TimeIntegrator
} // namespace AMP