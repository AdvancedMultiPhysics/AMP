//
// $Id: RK23TimeIntegrator.C,v 1.2 2006/02/07 17:36:01 philipb Exp $
// $Revision: 1.2 $
// $Date: 2006/02/07 17:36:01 $
//
// File:  RK23TimeIntegrator.h
// Copyright:  (c) 2005 The Regents of the University of California
// Description:  Concrete time integrator using Runge-Kutta method of order 3 -
// RK23 method
//
#include "AMP/time_integrators/RK23TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"

#include "AMP/vectors/Vector.h"

#include "ProfilerApp.h"

namespace AMP::TimeIntegrator {

/*
************************************************************************
*                                                                      *
*  Constructor.                                                        *
*                                                                      *
************************************************************************
*/
RK23TimeIntegrator::RK23TimeIntegrator(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
    : AMP::TimeIntegrator::TimeIntegrator( parameters )
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
RK23TimeIntegrator::~RK23TimeIntegrator() = default;

/*
************************************************************************
*                                                                      *
* Initialize.                                                          *
*                                                                      *
************************************************************************
*/
void RK23TimeIntegrator::initialize(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
{

    AMP::TimeIntegrator::TimeIntegrator::initialize( parameters );

    setupVectors();

    /*
     * Initialize data members from input.
     */
    getFromInput( parameters->d_db );
}

void RK23TimeIntegrator::reset(
    std::shared_ptr<const AMP::TimeIntegrator::TimeIntegratorParameters> )
{
    // AMP_ASSERT(parameters!=nullptr);
    d_new_solution->getVectorData()->reset();
    d_k1_vec->getVectorData()->reset();
    d_k2_vec->getVectorData()->reset();
    d_k3_vec->getVectorData()->reset();
    d_k4_vec->getVectorData()->reset();
    d_z_vec->getVectorData()->reset();
}

void RK23TimeIntegrator::setupVectors()
{

    // clone vectors so they have the same data layout as d_solution_vector
    d_new_solution = d_solution_vector->cloneVector();
    d_k1_vec       = d_solution_vector->cloneVector();
    d_k2_vec       = d_solution_vector->cloneVector();
    d_k3_vec       = d_solution_vector->cloneVector();
    d_k4_vec       = d_solution_vector->cloneVector();
    d_z_vec        = d_solution_vector->cloneVector();
    /*
     * Set initial value of vectors to 0.
     */
    d_new_solution->setToScalar( (double) 0.0 );
    d_k1_vec->setToScalar( (double) 0.0 );
    d_k2_vec->setToScalar( (double) 0.0 );
    d_k3_vec->setToScalar( (double) 0.0 );
    d_k4_vec->setToScalar( (double) 0.0 );
    d_z_vec->setToScalar( (double) 0.0 );
}

int RK23TimeIntegrator::advanceSolution( const double dt,
                                         const bool,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> out )
{
    PROFILE_START( "advanceSolution" );

    d_solution_vector = in;
    d_current_dt      = dt;

    if ( d_iDebugPrintInfoLevel > 5 ) {
        AMP::pout << "*****************************************" << std::endl;
        AMP::pout << "Current solution vector: " << std::endl;
        d_solution_vector->getVectorData()->print( AMP::pout );
        AMP::pout << "*****************************************" << std::endl;
    }

    // while we could swap with k4 this does not work if a step is
    // rejected - could be fixed
    // k1 = f(tn,un)
    d_operator->apply( d_solution_vector, d_k1_vec );

    // u* = un+k1*dt/2
    d_new_solution->axpy( 0.5 * dt, *d_k1_vec, *d_solution_vector );

    // k2 = f(t+dt/2, u*)
    d_operator->apply( d_new_solution, d_k2_vec );

    // u* = un+0.75*k2*dt
    d_new_solution->axpy( 0.75 * dt, *d_k2_vec, *d_solution_vector );

    // k3 = f(t+0.75dt, u*)
    d_operator->apply( d_new_solution, d_k3_vec );

    // first we calculate the 3rd order solution in d_new_solution
    // u* = un+k1*2dt/9+k2*dt/3+k3*4dt/9
    d_new_solution->linearSum( 2.0, *d_k1_vec, 3.0, *d_k2_vec );
    d_new_solution->axpy( 4.0, *d_k3_vec, *d_new_solution );
    d_new_solution->axpy( dt / 9.0, *d_new_solution, *d_solution_vector );

    // k4 = f(t+dt, u*)
    d_operator->apply( d_new_solution, d_k4_vec );

    // now we calculate the estimated error in d_z_vec for adapting the
    // timestep
    // adapted from Cleve Molers post
    d_z_vec->linearSum( -5.0, *d_k1_vec, 6.0, *d_k2_vec );
    d_z_vec->axpy( 8.0, *d_k3_vec, *d_z_vec );
    d_z_vec->axpy( -9.0, *d_k4_vec, *d_z_vec );
    d_z_vec->scale( dt / 72.0, *d_z_vec );

    if ( d_iDebugPrintInfoLevel > 4 ) {
        std::cout << "L2 norm of (z-u*) " << d_z_vec->L2Norm().get<double>() << std::endl;
    }

    out->copyVector( d_new_solution );

    PROFILE_STOP( "advanceSolution" );
    return ( 1 );
}

/*
************************************************************************
*                                                                      *
*  Check whether time advanced solution is acceptable.                 *
*                                                                      *
************************************************************************
*/
bool RK23TimeIntegrator::checkNewSolution()
{
    bool retcode = false;

    auto l2Norm                 = d_z_vec->L2Norm();
    auto l2NormOfEstimatedError = l2Norm.get<double>();

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
void RK23TimeIntegrator::updateSolution()
{
    d_solution_vector->swapVectors( d_new_solution );
    d_current_time += d_current_dt;
    ++d_integrator_step;

    if ( d_iDebugPrintInfoLevel > 0 ) {

        AMP::pout << "\n++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        AMP::pout << "End of timestep # " << d_integrator_step - 1 << std::endl;
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
void RK23TimeIntegrator::getFromInput( std::shared_ptr<AMP::Database> input_db )
{

    d_safety_factor = input_db->getWithDefault<double>( "safety_factor", 0.9 );

    d_atol = input_db->getWithDefault<double>( "absolute_tolerance", 1.0e-09 );

    d_use_fixed_dt = input_db->getWithDefault<bool>( "use_fixed_dt", false );
}

double RK23TimeIntegrator::getNextDt( const bool good_solution )
{
    double next_dt;

    if ( d_use_fixed_dt ) {
        next_dt = std::min( d_current_dt, d_final_time - d_current_time );
    } else {

        auto l2NormOfEstimatedError = d_z_vec->L2Norm().get<double>();

        next_dt =
            d_safety_factor * d_current_dt * pow( d_atol / l2NormOfEstimatedError, 1.0 / 3.0 );

        // check to make sure the timestep is not too small or large
        next_dt = std::min( std::max( next_dt, d_min_dt ), d_max_dt );
        // check to make sure we don't step past final time
        next_dt = std::min( next_dt, d_final_time - d_current_time );
        if ( good_solution ) {
            if ( d_iDebugPrintInfoLevel > 0 ) {
                AMP::pout << "Timestep # " << d_integrator_step << ", dt: " << next_dt << std::endl;
                AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
            }
        } else {
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

} // namespace AMP::TimeIntegrator
