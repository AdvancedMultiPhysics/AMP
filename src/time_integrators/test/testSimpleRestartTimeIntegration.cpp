#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/operators/NullOperator.h"
#include "AMP/operators/Operator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"


class FunctionOperator : public AMP::Operator::Operator
{
public:
    FunctionOperator( std::function<double( double )> f ) : d_f( f ) {}
    std::string type() const override { return "FunctionOperator"; }
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> r ) override
    {
        auto it_f = f->begin();
        auto it_r = r->begin();
        for ( size_t i = 0; i < it_f.size(); ++i, ++it_f, ++it_r )
            *it_r = d_f( *it_f );
    }

private:
    std::function<double( double )> d_f;
};

bool is_implicit( const std::string &name )
{
    // Check the final constant dt is properly computed
    auto imp_ti = { "CN", "Backward Euler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    return std::find( imp_ti.begin(), imp_ti.end(), name ) != imp_ti.end();
}


void testIntegratorRestart( const std::string &name,
                            const std::string &test,
                            std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> params,
                            AMP::UnitTest &ut,
                            bool from_restart )
{
    AMP::pout << "Testing " << name << " with " << test << " operator" << std::endl;
    // Create the time integrator
    auto var      = std::make_shared<AMP::LinearAlgebra::Variable>( "x" );
    auto solution = AMP::LinearAlgebra::createSimpleVector<double>( 1, var, AMP_COMM_WORLD );

    AMP::IO::RestartManager restartManager;
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegrator> timeIntegrator;
    double finalTime;
    std::vector<double> dts;
    if ( from_restart ) {
        restartManager.load( name + "Restart" );
        timeIntegrator = restartManager.getData<AMP::TimeIntegrator::TimeIntegrator>( "TI" );
        auto ic_vec    = timeIntegrator->getSolution();
        AMP_ASSERT( ic_vec );
        params->d_ic_vector = ic_vec;
        timeIntegrator->reset( params );
        timeIntegrator->registerOperator( params->d_operator );
        timeIntegrator->setSourceTerm( params->d_pSourceTerm );
        finalTime = timeIntegrator->getFinalTime();
        solution->copy( *ic_vec );
        // build expected dt vector
        if ( is_implicit( name ) ) {
            dts.resize( 6 );
            std::iota( dts.begin(), dts.end(), 9 );
            std::transform( dts.begin(), dts.end(), dts.begin(), []( double i ) {
                return 0.0001 + i / 20 * 0.0001;
            } );
            dts[5] = 0.000145;
        } else {
            dts.resize( 10 );
            std::fill( dts.begin(), dts.end(), 0.0001 );
        }
    } else {
        timeIntegrator = AMP::TimeIntegrator::TimeIntegratorFactory::create( params );
        restartManager.registerData( timeIntegrator, "TI" );
        finalTime = timeIntegrator->getFinalTime() / 2;
        solution->setToScalar( 1.0 );
        // build expected dt vector
        if ( is_implicit( name ) ) {
            dts.resize( 9 );
            std::iota( dts.begin(), dts.end(), 0 );
            std::transform( dts.begin(), dts.end(), dts.begin(), []( double i ) {
                return 0.0001 + i / 20 * 0.0001;
            } );
        } else {
            dts.resize( 10 );
            std::fill( dts.begin(), dts.end(), 0.0001 );
        }
    }

    auto x = solution->clone();
    x->copy( *solution );

    // Advance the solution
    double T      = timeIntegrator->getCurrentTime();
    double dt     = timeIntegrator->getCurrentDt();
    bool wrong_dt = false;
    int i         = 0;
    while ( T < finalTime ) {
        timeIntegrator->advanceSolution( dt, T == 0, solution, x );
        if ( timeIntegrator->checkNewSolution() ) {
            timeIntegrator->updateSolution();
            solution->copyVector( x );
            if ( std::abs( dt - dts[i] ) > 1e-10 )
                wrong_dt = true;
            T += dt;
            i += 1;
            dt = timeIntegrator->getNextDt( true );
        } else {
            AMP_ERROR( "Solution didn't converge" );
        }
    }

    if ( !from_restart )
        restartManager.write( name + "Restart" );

    // Check the final constant dt is properly computed
    if ( wrong_dt )
        ut.failure( name + " - " + test + ": dt lengths not as expected" );
    else
        ut.passes( name + " - " + test + ": correct dt lengths" );


    // Check the answer
    double ans = 0;
    if ( test == "du/dt=3" ) {
        ans = 1.0 + 3 * T;
    } else if ( test == "du/dt=-3u" ) {
        ans = std::exp( -3.0 * T );
    } else if ( test == "du/dt=-3u+3" ) {
        ans = 1.0;
    }
    double ans2 = static_cast<double>( solution->max() );
    if ( AMP::Utilities::approx_equal( ans2, ans, 5.0e-10 ) )
        ut.passes( name + " - " + test );
    else
        ut.failure(
            AMP::Utilities::stringf( "%s - %s (%0.16f)", name.data(), test.data(), ans - ans2 ) );
}

void updateDatabaseIfImplicitRestart( std::unique_ptr<AMP::Database> &db )
{
    AMP_ASSERT( db );
    auto name = db->getScalar<std::string>( "name" );
    if ( is_implicit( name ) ) {

        //        db->putScalar<std::string>( "name", "ImplicitIntegrator" );
        db->putScalar<std::string>( "implicit_integrator", name );
        db->putScalar<std::string>( "solver_name", "Solver" );
        db->putScalar<std::string>( "timestep_selection_strategy", "final constant" );
        db->putScalar<bool>( "use_predictor", false );
        auto solver_db = AMP::Database::create( "name",
                                                "CGSolver",
                                                "print_info_level",
                                                2,
                                                "max_iterations",
                                                100,
                                                "absolute_tolerance",
                                                1.0e-14,
                                                "relative_tolerance",
                                                1.0e-14,
                                                "zero_initial_guess",
                                                false );
        db->putDatabase( "Solver", std::move( solver_db ) );
    }
}

std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters>
createTimeIntegratorParameters( const std::string &name,
                                const std::shared_ptr<AMP::LinearAlgebra::Vector> ic,
                                const double finalTime )
{
    auto db = AMP::Database::create( "name",
                                     name,
                                     "initial_time",
                                     0.0,
                                     "final_time",
                                     finalTime,
                                     "initial_dt",
                                     0.0001,
                                     "min_dt",
                                     0.0001,
                                     "max_dt",
                                     0.0002,
                                     "max_integrator_steps",
                                     20,
                                     "number_of_time_intervals",
                                     20,
                                     "print_info_level",
                                     2 );

    updateDatabaseIfImplicitRestart( db );

    // Create global db
    auto global_db = std::make_shared<AMP::Database>( "GlobalDatabase" );
    global_db->putDatabase( "TimeIntegrator", std::move( db ) );

    auto params = std::make_shared<AMP::TimeIntegrator::TimeIntegratorParameters>(
        global_db->getDatabase( "TimeIntegrator" ) );

    params->d_ic_vector = ic;
    params->d_global_db = global_db;
    return params;
}

void runBasicIntegratorTestsRestart( const std::string &name, AMP::UnitTest &ut )
{
    double finalTime = 0.002;
    // Create the vectors
    auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "x" );
    auto ic  = AMP::LinearAlgebra::createSimpleVector<double>( 1, var, AMP_COMM_WORLD );
    ic->setToScalar( 1.0 );

    // Test creating Create the time integrator
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> params;
    params = createTimeIntegratorParameters( name, ic, finalTime );

    params->d_operator = std::make_shared<AMP::Operator::NullOperator>();
    try {
        auto timeIntegrator = AMP::TimeIntegrator::TimeIntegratorFactory::create( params );
        ut.passes( name + " - created" );
    } catch ( ... ) {
        ut.failure( name + " - created" );
        return;
    }

    // Test with a fixed source and null operator
    auto source = AMP::LinearAlgebra::createSimpleVector<double>( 1, var, AMP_COMM_WORLD );
    source->setToScalar( 3.0 );
    params->d_pSourceTerm = source;
    params->d_operator    = std::make_shared<AMP::Operator::NullOperator>();
    testIntegratorRestart( name, "du/dt=3", params, ut, false );
    testIntegratorRestart( name, "du/dt=3", params, ut, true );


    // Test with no source and constant operator
    params = createTimeIntegratorParameters( name, ic, finalTime );

    params->d_pSourceTerm = nullptr;
    params->d_operator = std::make_shared<FunctionOperator>( []( double x ) { return -3.0 * x; } );
    testIntegratorRestart( name, "du/dt=-3u", params, ut, false );
    testIntegratorRestart( name, "du/dt=-3u", params, ut, true );

    // Test with fixed source and constant operator
    params = createTimeIntegratorParameters( name, ic, finalTime );

    params->d_pSourceTerm = source;
    params->d_operator = std::make_shared<FunctionOperator>( []( double x ) { return -3.0 * x; } );
    testIntegratorRestart( name, "du/dt=-3u+3", params, ut, false );
    testIntegratorRestart( name, "du/dt=-3u+3", params, ut, true );
}


int testSimpleRestartTimeIntegration( int argc, char *argv[] )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // List of integrators
    auto integrators = { "RK2", "RK4", "CN", "BDF2", "BDF3", "BDF4", "BDF5" };
    // Run the tests
    for ( auto tmp : integrators )
        runBasicIntegratorTestsRestart( tmp, ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
