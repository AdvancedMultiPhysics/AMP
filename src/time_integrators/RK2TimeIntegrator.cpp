//
// $Id: RK2TimeIntegrator.C,v 1.2 2006/02/07 17:36:01 philipb Exp $
// $Revision: 1.2 $
// $Date: 2006/02/07 17:36:01 $
//
// File:  RK2TimeIntegrator.h
// Copyright:  (c) 2005 The Regents of the University of California
// Description:  Concrete time integrator using Runge-Kutta method of order 2 -
// RK2 method
//
#include "AMP/time_integrators/RK2TimeIntegrator.h"

#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/AMPManager.h"
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
RK2TimeIntegrator::RK2TimeIntegrator(
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
RK2TimeIntegrator::~RK2TimeIntegrator() = default;

/*
************************************************************************
*                                                                      *
* Initialize.                                                          *
*                                                                      *
************************************************************************
*/
void RK2TimeIntegrator::initialize(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
{

    AMP::TimeIntegrator::TimeIntegrator::initialize( parameters );

    setupVectors();

    /*
     * Initialize data members from input.
     */
    getFromInput( parameters->d_db );
}

void RK2TimeIntegrator::reset(
    std::shared_ptr<const AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
{
    if ( parameters ) {
        d_pParameters =
            std::const_pointer_cast<AMP::TimeIntegrator::TimeIntegratorParameters>( parameters );
        AMP_ASSERT( parameters->d_db );
        getFromInput( parameters->d_db );
    }
}

void RK2TimeIntegrator::setupVectors()
{

    // clone vectors so they have the same data layout as d_solution_vector
    d_new_solution = d_solution_vector->clone();
    d_k1_vec       = d_solution_vector->clone();
    d_k2_vec       = d_solution_vector->clone();

    /*
     * Set initial value of vectors to 0.
     */
    d_new_solution->zero();
    d_k1_vec->zero();
    d_k2_vec->zero();
}

int RK2TimeIntegrator::advanceSolution( const double dt,
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
    // u* = un+dt*k1
    d_new_solution->axpy( dt, *d_k1_vec, *d_solution_vector );
    // k2 = f(t+dt, u*)
    d_operator->apply( d_new_solution, d_k2_vec );
    if ( d_pSourceTerm )
        d_k2_vec->add( *d_k2_vec, *d_pSourceTerm );
    // u_new = un+ dt*(k1+k2)/2
    d_k2_vec->add( *d_k1_vec, *d_k2_vec );
    d_new_solution->axpy( dt / 2.0, *d_k2_vec, *d_solution_vector );

    out->copyVector( d_new_solution );
    d_k1_vec->zero();
    d_k2_vec->zero();
    return ( 1 );
}

/*
************************************************************************
*                                                                      *
*  Check whether time advanced solution is acceptable.                 *
*                                                                      *
************************************************************************
*/
bool RK2TimeIntegrator::checkNewSolution()
{
    bool retcode = true;

    /*
     * Ordinarily we would check the actual error in the solution
     * (proportional to the size of d_corrector) against a specified
     * tolerance.  For now, accept everything.
    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "\n++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        AMP::pout << "End of timestep # " << d_integrator_step << std::endl;
        AMP::pout << "Failed to advance solution past " << d_current_time << std::endl;
        AMP::pout << "++++++++++++++++++++++++++++++++++++++++++++++++\n" << std::endl;
    }
     */

    return ( retcode );
}

/*
************************************************************************
*                                                                      *
*  Update internal state to reflect time advanced solution.            *
*                                                                      *
************************************************************************
*/
void RK2TimeIntegrator::updateSolution()
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

/********************************************************
 *  Restart operations                                   *
 ********************************************************/
void RK2TimeIntegrator::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    TimeIntegrator::registerChildObjects( manager );
}
void RK2TimeIntegrator::writeRestart( int64_t fid ) const { TimeIntegrator::writeRestart( fid ); }

RK2TimeIntegrator::RK2TimeIntegrator( int64_t fid, AMP::IO::RestartManager *manager )
    : TimeIntegrator( fid, manager )
{
    d_initialized = false;
    RK2TimeIntegrator::initialize( d_pParameters );
    d_initialized = true;
}

} // namespace AMP::TimeIntegrator
