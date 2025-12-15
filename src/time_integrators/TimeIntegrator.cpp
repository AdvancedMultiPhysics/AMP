#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/operators/Operator.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>


namespace AMP::TimeIntegrator {

/************************************************************************
 *                                                                       *
 * Constructor and destructor for TimeIntegrator.  The                   *
 * constructor sets default values for data members, then overrides      *
 * them with values read from input or restart.  The destructor does     *
 * nothing interesting.                                      s            *
 *                                                                       *
 ************************************************************************/

TimeIntegrator::TimeIntegrator(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
    : d_pParameters( parameters )
{
    AMPManager::incrementResource( "TimeIntegrator" );
    AMP_INSIST( d_pParameters, "Null parameter" );

    initialize( d_pParameters );

    d_initialized = true;
}

TimeIntegrator::~TimeIntegrator() { AMPManager::decrementResource( "TimeIntegrator" ); }

/****************************************************************
 * Get the default name                                          *
 ****************************************************************/
std::string TimeIntegrator::type() const { return "TimeIntegrator"; }

uint64_t TimeIntegrator::getID() const
{
    // The approach followed below assumes
    // the initial condition vector has been set
    // Not the greatest, but will do for now
    AMP_ASSERT( d_ic_vector );
    return d_ic_vector->getComm().bcast( reinterpret_cast<uint64_t>( this ), 0 );
}

/*
*************************************************************************
*                                                                       *
* Initialize integrator and nonlinear solver:                           *
*                                                                       *
* (1) Create vector containing solution state advanced in time.         *
*                                                                       *
* (2) Equation class registers data components with solution vector.    *
*                                                                       *
* (3) Initialize nonlinear solver.                                      *
*                                                                       *
*************************************************************************
*/

void TimeIntegrator::initialize( std::shared_ptr<TimeIntegratorParameters> parameters )
{
    d_ic_vector = parameters->d_ic_vector;
    AMP_ASSERT( d_ic_vector );

    // for now the solution is set to the initial conditions
    d_solution_vector = d_ic_vector->clone();
    d_solution_vector->copyVector( d_ic_vector );

    d_pSourceTerm = parameters->d_pSourceTerm;

    if ( d_solution_vector.get() == nullptr ) {
        AMP_ERROR( "TimeIntegrator::TimeIntegrator()::TimeIntegrators must be initialized with non "
                   "null initial "
                   "condition_vectors" );
    }

    // initialize the mass operator
    d_pMassOperator = parameters->d_pMassOperator;
    // initialize the rhs operator
    d_operator = parameters->d_operator;

    getFromInput( parameters->d_db );

    d_current_time = d_initial_time;
}

/*
*************************************************************************
*                                                                       *
* Get next dt from user-supplied equation class.  Timestep selection    *
* is generally based on whether the nonlinear solution iteration        *
* converged and, if so, whether the solution meets some user-defined    *
* criteria.  It is assumed that, before this routine is called, the     *
* routine checkNewSolution() has been called.  The boolean argument     *
* is the return value from that call.  The integer argument is          *
* that which is returned by the particular nonlinear solver package     *
* that generated the solution.                                          *
*                                                                       *
*************************************************************************
*/

double TimeIntegrator::getNextDt( const bool )
{
    return std::min( std::min( d_current_dt, d_max_dt ), d_final_time - d_current_time );
}

/*
*************************************************************************
*                                                                       *
* If simulation is not from restart, read data from input database.     *
* Otherwise, override restart values for a subset of the data members   *
* with those found in input.                                            *
*                                                                       *
*************************************************************************
*/

void TimeIntegrator::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    AMP_ASSERT( db );

    if ( db->keyExists( "name" ) ) {
        d_object_name = db->getString( "name" );
    } else {
        AMP_ERROR( " -- Key data `name' missing in input." );
    }

    if ( !d_initialized ) {
        if ( db->keyExists( "initial_time" ) ) {
            d_initial_time = db->getScalar<double>( "initial_time" );
        } else {
            AMP_ERROR( d_object_name + " -- Key data `initial_time' missing in input" );
        }

        d_integrator_step = db->getWithDefault<int>( "integrator_step", 0 );
        d_initial_dt      = db->getWithDefault<double>( "initial_dt", 0.0 );
    }

    if ( db->keyExists( "final_time" ) ) {
        d_final_time = db->getScalar<double>( "final_time" );
        if ( d_final_time < d_initial_time ) {
            AMP_ERROR( d_object_name + " -- Error in input data  final_time < initial_time" );
        }
    } else {
        AMP_ERROR( d_object_name + " -- Key data `final_time' missing in input." );
    }

    if ( db->keyExists( "max_integrator_steps" ) ) {
        d_max_integrator_steps = db->getScalar<int>( "max_integrator_steps" );
        if ( d_max_integrator_steps < 0 ) {
            AMP_ERROR( d_object_name + " -- Error in input data max_integrator_steps < 0" );
        }
    } else {
        AMP_ERROR( d_object_name + " -- Key data `max_integrator_steps' missing in input" );
    }

    d_max_dt = db->getWithDefault<double>( "max_dt", std::numeric_limits<double>::max() );

    if ( d_max_dt < 0.0 ) {
        AMP_ERROR( d_object_name + " -- Error in input data max_dt < 0." );
    }

    d_min_dt = db->getWithDefault<double>( "min_dt", std::numeric_limits<double>::min() );

    if ( d_min_dt < 0.0 ) {
        AMP_ERROR( d_object_name + " -- Error in input data min_dt < 0." );
    }

    d_iDebugPrintInfoLevel = db->getWithDefault<int>( "print_info_level", 0 );

    if ( !d_initialized ) {
        d_current_dt = d_initial_dt;
        d_old_dt     = d_initial_dt;
    }
}

/*
*************************************************************************
*                                                                       *
* Write out class version number and data members to database.          *
*                                                                       *
*************************************************************************
*/

void TimeIntegrator::putToDatabase( std::shared_ptr<AMP::Database> db )
{
    AMP_ASSERT( !db.use_count() );

    db->putScalar( "d_initial_time", d_initial_time );
    db->putScalar( "d_final_time", d_final_time );
    db->putScalar( "d_current_time", d_current_time );
    db->putScalar( "d_current_dt", d_current_dt );
    db->putScalar( "d_old_dt", d_old_dt );
    db->putScalar( "d_integrator_step", d_integrator_step );
    db->putScalar( "d_max_integrator_steps", d_max_integrator_steps );
}
/*
*************************************************************************
*                                                                       *
* Print class data members to given output stream.                      *
*                                                                       *
*************************************************************************
*/

void TimeIntegrator::printClassData( std::ostream &os ) const
{
    os << "\nTimeIntegrator::printClassData..." << std::endl;
    os << "TimeIntegrator: this = " << this << std::endl;
    os << "d_object_name = " << d_object_name << std::endl;
    os << "d_initial_time = " << d_initial_time << std::endl;
    os << "d_final_time = " << d_final_time << std::endl;
    os << "d_current_time = " << d_current_time << std::endl;
    os << "d_current_dt = " << d_current_dt << std::endl;
    os << "d_old_dt = " << d_old_dt << std::endl;
    os << "d_integrator_step = " << d_integrator_step << std::endl;
    os << "d_max_integrator_steps = " << d_max_integrator_steps << std::endl;
}


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
void TimeIntegrator::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    manager->registerObject( d_solution_vector );
}
void TimeIntegrator::writeRestart( int64_t fid ) const
{
    d_pParameters->d_db->putScalar<double>(
        "initial_time", d_current_time, {}, AMP::Database::Check::Overwrite );
    d_pParameters->d_db->putScalar<double>(
        "initial_dt", d_current_dt, {}, AMP::Database::Check::Overwrite );
    d_pParameters->d_db->putScalar<double>(
        "integrator_step", d_integrator_step, {}, AMP::Database::Check::Overwrite );
    IO::writeHDF5( fid, "ti_db", *( d_pParameters->d_db ) );
    IO::writeHDF5( fid, "global_db", *( d_pParameters->d_global_db ) );
    uint64_t vecID = d_solution_vector->getID();
    IO::writeHDF5( fid, "ic_vec", vecID );
}
TimeIntegrator::TimeIntegrator( int64_t fid, AMP::IO::RestartManager *manager )
{
    AMPManager::incrementResource( "TimeIntegrator" );
    uint64_t vecID;
    AMP::IO::readHDF5( fid, "ic_vec", vecID );
    auto ic_vector = manager->getData<AMP::LinearAlgebra::Vector>( vecID );

    AMP::Database db, db_global;
    IO::readHDF5( fid, "ti_db", db );
    IO::readHDF5( fid, "global_db", db_global );

    d_pParameters =
        std::make_shared<TimeIntegratorParameters>( std::make_shared<AMP::Database>( db ) );
    d_pParameters->d_global_db = std::make_shared<AMP::Database>( db_global );
    d_pParameters->d_ic_vector = ic_vector;
    TimeIntegrator::initialize( d_pParameters );

    d_initialized = true;
}


/********************************************************
 *  Simple functions                                     *
 ********************************************************/
void TimeIntegrator::printStatistics( std::ostream & ) {}


} // namespace AMP::TimeIntegrator


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::TimeIntegrator::TimeIntegrator>::DataStoreType(
    std::shared_ptr<const AMP::TimeIntegrator::TimeIntegrator> ti, RestartManager *manager )
    : d_data( ti )
{
    d_hash = ti->getID();
    d_data->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::TimeIntegrator::TimeIntegrator>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = IO::createGroup( fid, name );
    IO::writeHDF5( gid, "type", d_data->type() );
    d_data->writeRestart( gid );
    IO::closeGroup( gid );
}
template<>
std::shared_ptr<AMP::TimeIntegrator::TimeIntegrator>
AMP::IO::RestartManager::DataStoreType<AMP::TimeIntegrator::TimeIntegrator>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid       = IO::openGroup( fid, name );
    auto integrator = AMP::TimeIntegrator::TimeIntegratorFactory::create( gid, manager );
    IO::closeGroup( gid );
    return integrator;
}
