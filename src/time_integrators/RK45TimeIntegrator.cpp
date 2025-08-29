//
// $Id: RK45TimeIntegrator.C,v 1.2 2006/02/07 17:36:01 philipb Exp $
// $Revision: 1.2 $
// $Date: 2006/02/07 17:36:01 $
//
// File:  RK45TimeIntegrator.h
// Copyright:  (c) 2005 The Regents of the University of California
// Description:  Concrete time integrator using Runge-Kutta method of order 5 -
// Runge-Kutta-Fehlberg RK45 method
//


#include "AMP/vectors/Vector.h"

#include "AMP/time_integrators/RK45TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/AMPManager.h"

#include "ProfilerApp.h"

namespace AMP::TimeIntegrator {

/*
************************************************************************
*                                                                      *
*  Constructor.                                                        *
*                                                                      *
************************************************************************
*/
RK45TimeIntegrator::RK45TimeIntegrator(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
    : AMP::TimeIntegrator::TimeIntegrator( parameters )
{
    d_initialized = false;
    initialize( parameters );
    d_initialized = true;
}

/*
************************************************************************
*                                                                      *
*  Destructor.                                                         *
*                                                                      *
************************************************************************
*/
RK45TimeIntegrator::~RK45TimeIntegrator() = default;

/*
************************************************************************
*                                                                      *
* Initialize.                                                          *
*                                                                      *
************************************************************************
*/
void RK45TimeIntegrator::initialize(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
{

    AMP::TimeIntegrator::TimeIntegrator::initialize( parameters );

    setupVectors();

    /*
     * Initialize data members from input.
     */
    getFromInput( parameters->d_db );
}

void RK45TimeIntegrator::reset(
    std::shared_ptr<const AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
{
    if ( parameters ) {
        TimeIntegrator::getFromInput( parameters->d_db );
        d_pParameters =
            std::const_pointer_cast<AMP::TimeIntegrator::TimeIntegratorParameters>( parameters );
        AMP_ASSERT( parameters->d_db );
        getFromInput( parameters->d_db );
    }

    d_new_solution->reset();
    d_k1_vec->reset();
    d_k2_vec->reset();
    d_k3_vec->reset();
    d_k4_vec->reset();
    d_k5_vec->reset();
    d_k6_vec->reset();
    d_z_vec->reset();
}

void RK45TimeIntegrator::setupVectors()
{

    // clone vectors so they have the same data layout as d_solution_vector
    d_new_solution = d_solution_vector->clone();
    d_k1_vec       = d_solution_vector->clone();
    d_k2_vec       = d_solution_vector->clone();
    d_k3_vec       = d_solution_vector->clone();
    d_k4_vec       = d_solution_vector->clone();
    d_k5_vec       = d_solution_vector->clone();
    d_k6_vec       = d_solution_vector->clone();
    d_z_vec        = d_solution_vector->clone();

    /*
     * Set initial value of vectors to 0.
     */
    d_new_solution->zero();
    d_k1_vec->zero();
    d_k2_vec->zero();
    d_k3_vec->zero();
    d_k4_vec->zero();
    d_k5_vec->zero();
    d_k6_vec->zero();
    d_z_vec->zero();
}

int RK45TimeIntegrator::advanceSolution( const double dt,
                                         const bool,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> out )
{
    PROFILE( "advanceSolution" );

    d_solution_vector->copyVector( in );
    d_current_dt = dt;

    // k1 = f(tn,un)
    d_operator->apply( d_solution_vector, d_k1_vec );
    if ( d_pSourceTerm )
        d_k1_vec->add( *d_k1_vec, *d_pSourceTerm );

    // u* = un+k1/4
    d_new_solution->axpy( 0.25 * dt, *d_k1_vec, *d_solution_vector );

    // k2 = f(t+dt/4, u*)
    d_operator->apply( d_new_solution, d_k2_vec );
    if ( d_pSourceTerm )
        d_k2_vec->add( *d_k2_vec, *d_pSourceTerm );

    // u* = un+3*k1*dt/32+9*k2*dt/32
    d_new_solution->axpy( 3.0 * dt / 32.0, *d_k1_vec, *d_solution_vector );
    d_new_solution->axpy( 9.0 * dt / 32.0, *d_k2_vec, *d_new_solution );

    // k3 = f(t+3*dt/8, u*)
    d_operator->apply( d_new_solution, d_k3_vec );
    if ( d_pSourceTerm )
        d_k3_vec->add( *d_k3_vec, *d_pSourceTerm );

    // u* = un + 1932*dt*k1/2197 - 7200*dt*k2/2197 + 7296*dt*k3/2197
    d_new_solution->axpy( 1932.0 * dt / 2197.0, *d_k1_vec, *d_solution_vector );
    d_new_solution->axpy( -7200.0 * dt / 2197.0, *d_k2_vec, *d_new_solution );
    d_new_solution->axpy( 7296.0 * dt / 2197.0, *d_k3_vec, *d_new_solution );

    // k4 = f(t+12*dt/13, u*)
    d_operator->apply( d_new_solution, d_k4_vec );
    if ( d_pSourceTerm )
        d_k4_vec->add( *d_k4_vec, *d_pSourceTerm );

    // u* = un + 439*dt*k1/216 - 8*dt*k2 + 3680*dt*k3/513 - 845*dt*k4/4104
    d_new_solution->axpy( 439.0 * dt / 216.0, *d_k1_vec, *d_solution_vector );
    d_new_solution->axpy( -8.0 * dt, *d_k2_vec, *d_new_solution );
    d_new_solution->axpy( 3680.0 * dt / 513.0, *d_k3_vec, *d_new_solution );
    d_new_solution->axpy( -845.0 * dt / 4104.0, *d_k4_vec, *d_new_solution );

    // k5 = f(t+dt, u*)
    d_operator->apply( d_new_solution, d_k5_vec );
    if ( d_pSourceTerm )
        d_k5_vec->add( *d_k5_vec, *d_pSourceTerm );

    // u* = un - 8*dt*k1/27 + 2*dt*k2 - 3544*dt*k3/2565 + 1859*dt*k4/4104 - 11*dt*k5/40
    d_new_solution->axpy( -8.0 * dt / 27.0, *d_k1_vec, *d_solution_vector );
    d_new_solution->axpy( 2.0 * dt, *d_k2_vec, *d_new_solution );
    d_new_solution->axpy( -3544.0 * dt / 2565.0, *d_k3_vec, *d_new_solution );
    d_new_solution->axpy( 1859.0 * dt / 4104.0, *d_k4_vec, *d_new_solution );
    d_new_solution->axpy( -11.0 * dt / 40.0, *d_k5_vec, *d_new_solution );

    // k6 = f(t+dt, u*)
    d_operator->apply( d_new_solution, d_k6_vec );
    if ( d_pSourceTerm )
        d_k6_vec->add( *d_k6_vec, *d_pSourceTerm );

    // z_new = un + ...
    d_z_vec->axpy( 25.0 * dt / 216.0, *d_k1_vec, *d_solution_vector );
    d_z_vec->axpy( 1408.0 * dt / 2565.0, *d_k3_vec, *d_z_vec );
    d_z_vec->axpy( 2197.0 * dt / 4104.0, *d_k4_vec, *d_z_vec );
    d_z_vec->axpy( -0.2 * dt, *d_k5_vec, *d_z_vec );

    // u_new = un + ...
    d_new_solution->axpy( 16.0 * dt / 135.0, *d_k1_vec, *d_solution_vector );
    d_new_solution->axpy( 6656.0 * dt / 12825.0, *d_k3_vec, *d_new_solution );
    d_new_solution->axpy( 28561.0 * dt / 56430.0, *d_k4_vec, *d_new_solution );
    d_new_solution->axpy( -9.0 * dt / 50.0, *d_k5_vec, *d_new_solution );
    d_new_solution->axpy( 2.0 * dt / 55.0, *d_k6_vec, *d_new_solution );

    // store the difference in d_z_vec
    d_z_vec->subtract( *d_new_solution, *d_z_vec );

    d_k1_vec->zero();
    d_k2_vec->zero();
    d_k3_vec->zero();
    d_k4_vec->zero();
    d_k5_vec->zero();
    d_k6_vec->zero();

    out->copyVector( d_new_solution );

    return ( 1 );
}

/*
************************************************************************
*                                                                      *
*  Check whether time advanced solution is acceptable.                 *
*                                                                      *
************************************************************************
*/
bool RK45TimeIntegrator::checkNewSolution()
{
    bool retcode = false;

    auto l2NormOfEstimatedError = static_cast<double>( d_z_vec->L2Norm() );

    // we flag the solution as being acceptable if the l2 norm of the error
    // is less than the required tolerance or we are at the minimum time step
    if ( ( l2NormOfEstimatedError < d_atol ) || ( fabs( d_current_dt - d_min_dt ) < 1.0e-10 ) ) {
        retcode = true;
    }

    if ( ( d_iDebugPrintInfoLevel > 0 ) && ( !retcode ) ) {
        AMP::pout << "\n++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        AMP::pout << "End of timestep # " << d_integrator_step << std::endl;
        AMP::pout << "Failed to advance solution past " << d_current_time << std::endl;
        AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
    }

    return ( retcode );
}

/*
************************************************************************
*                                                                      *
*  Update internal state to reflect time advanced solution.            *
*                                                                      *
************************************************************************
*/
void RK45TimeIntegrator::updateSolution()
{
    // instead of swap we are doing this manually so that the d_solution_vector
    // object is not changed, which otherwise leads to the wrong vector being
    // written at restart
    d_k1_vec->copyVector( d_solution_vector );
    d_solution_vector->copyVector( d_new_solution );
    d_new_solution->copyVector( d_k1_vec );

    d_current_time += d_current_dt;
    ++d_integrator_step;

    if ( d_iDebugPrintInfoLevel > 0 ) {

        AMP::pout << "\n++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        AMP::pout << "End of timestep # " << d_integrator_step << std::endl;
        AMP::pout << "Simulation time is " << d_current_time << std::endl;
        AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
    }
}

/*
************************************************************************
*                                                                      *
* Read input from database.                                            *
*                                                                      *
************************************************************************
*/
void RK45TimeIntegrator::getFromInput( std::shared_ptr<AMP::Database> input_db )
{

    d_safety_factor = input_db->getWithDefault<double>( "safety_factor", 0.9 );
    d_atol          = input_db->getWithDefault<double>( "absolute_tolerance", 1.0e-09 );
    d_use_fixed_dt  = input_db->getWithDefault<bool>( "use_fixed_dt", false );
}

double RK45TimeIntegrator::getNextDt( const bool good_solution )
{
    double next_dt;

    if ( d_use_fixed_dt ) {
        next_dt = std::min( d_current_dt, d_final_time - d_current_time );
    } else {

        if ( good_solution ) {
            auto l2NormOfEstimatedError = d_z_vec->L2Norm().get<double>();

            next_dt =
                0.84 * d_current_dt * std::pow( ( d_atol / l2NormOfEstimatedError ), 1.0 / 5.0 );

            // check to make sure the timestep is not too small or large
            next_dt = std::min( std::max( next_dt, d_min_dt ), d_max_dt );
            // check to make sure we don't step past final time
            next_dt = std::min( next_dt, d_final_time - d_current_time );

            if ( d_iDebugPrintInfoLevel > 0 ) {
                AMP::pout << "Timestep # " << d_integrator_step << ", dt: " << next_dt << std::endl;
                AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
            }
        } else {
            next_dt = d_safety_factor * d_current_dt;
            ++d_total_steprejects;
            if ( d_iDebugPrintInfoLevel > 0 ) {
                AMP::pout << "Failed to advance timestep # " << d_integrator_step
                          << ", new dt: " << next_dt << std::endl;
                AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
            }
        }
    }

    return next_dt;
}

/********************************************************
 *  Restart operations                                   *
 ********************************************************/
void RK45TimeIntegrator::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    TimeIntegrator::registerChildObjects( manager );
}
void RK45TimeIntegrator::writeRestart( int64_t fid ) const { TimeIntegrator::writeRestart( fid ); }

RK45TimeIntegrator::RK45TimeIntegrator( int64_t fid, AMP::IO::RestartManager *manager )
    : TimeIntegrator( fid, manager )
{
    d_initialized = false;
    RK45TimeIntegrator::initialize( d_pParameters );
    d_initialized = true;
}
} // namespace AMP::TimeIntegrator
