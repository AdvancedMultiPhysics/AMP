#include "AMP/IO/PIO.h"
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


void testIntegrator( const std::string &name,
                     const std::string &test,
                     std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> params,
                     double ans,
                     AMP::UnitTest &ut )
{
    AMP::pout << "Testing " << name << " with " << test << " operator" << std::endl;
    // Create the time integrator
    auto var            = std::make_shared<AMP::LinearAlgebra::Variable>( "x" );
    auto solution       = AMP::LinearAlgebra::createSimpleVector<double>( 1, var, AMP_COMM_WORLD );
    auto timeIntegrator = AMP::TimeIntegrator::TimeIntegratorFactory::create( params );

    auto x = solution->clone();
    solution->setToScalar( 1.0 );
    x->copy( *solution );

    // Advance the solution
    double T         = 0;
    double dt        = 0.0001;
    double finalTime = timeIntegrator->getFinalTime();
    timeIntegrator->setInitialDt( dt );
    while ( T < finalTime ) {
        timeIntegrator->advanceSolution( dt, T == 0, solution, x );
        if ( timeIntegrator->checkNewSolution() ) {
            timeIntegrator->updateSolution();
            solution->copyVector( x );
        } else {
            AMP_ERROR( "Solution didn't converge" );
        }
        T += dt;
    }

    // Check the answer
    double ans2 = static_cast<double>( solution->max() );
    double tol  = 5.0e-10;
    if ( name == "ExplicitEuler" || name == "BDF1" )
        tol = 5.0e-05;
    if ( AMP::Utilities::approx_equal( ans2, ans, tol ) )
        ut.passes( name + " - " + test );
    else
        ut.failure(
            AMP::Utilities::stringf( "%s - %s (%0.16f)", name.data(), test.data(), ans - ans2 ) );
}

void updateDatabaseIfImplicit( std::shared_ptr<AMP::Database> db )
{
    AMP_ASSERT( db );
    auto imp_ti = { "CN", "Backward Euler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    auto name   = db->getScalar<std::string>( "name" );
    AMP::pout << name << std::endl;
    auto is_implicit = ( std::find( imp_ti.begin(), imp_ti.end(), name ) != imp_ti.end() );
    if ( is_implicit ) {

        //        db->putScalar<std::string>( "name", "ImplicitIntegrator" );
        db->putScalar<std::string>( "implicit_integrator", name );
        db->putScalar<std::string>( "solver_name", "Solver" );
        db->putScalar<std::string>( "timestep_selection_strategy", "constant" );
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

void runBasicIntegratorTests( const std::string &name, AMP::UnitTest &ut )
{
    double finalTime = 0.001;
    // Create the vectors
    auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "x" );
    auto ic  = AMP::LinearAlgebra::createSimpleVector<double>( 1, var, AMP_COMM_WORLD );
    ic->setToScalar( 1.0 );

    // Test creating Create the time integrator
    std::shared_ptr<AMP::Database> db = AMP::Database::create( "name",
                                                               name,
                                                               "initial_time",
                                                               0.0,
                                                               "final_time",
                                                               finalTime,
                                                               "max_integrator_steps",
                                                               1000000,
                                                               "print_info_level",
                                                               2 );
    updateDatabaseIfImplicit( db );

    auto params         = std::make_shared<AMP::TimeIntegrator::TimeIntegratorParameters>( db );
    params->d_ic_vector = ic;

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
    testIntegrator( name, "du/dt=3", params, 1.0 + 3 * finalTime, ut );

    // Test with no source and constant operator
    params->d_pSourceTerm = nullptr;
    params->d_operator = std::make_shared<FunctionOperator>( []( double x ) { return -3.0 * x; } );
    testIntegrator( name, "du/dt=-3u", params, std::exp( -3.0 * finalTime ), ut );

    // Test with fixed source and constant operator
    params->d_pSourceTerm = source;
    params->d_operator = std::make_shared<FunctionOperator>( []( double x ) { return -3.0 * x; } );
    testIntegrator( name, "du/dt=-3u+3", params, 1.0, ut );
}


int testSimpleTimeIntegration( int argc, char *argv[] )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // List of integrators
    // We need to look at the errors for the first order -- whether they are acceptable
    // auto integrators = { "ExplicitEuler", "Backward Euler" "CN",
    //    "RK2", "RK23", "RK34", "RK4", "RK45", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5" };
    auto integrators = { "ExplicitEuler", "RK2",  "RK4",  "RK12", "RK23", "RK34", "RK45", "CN",
                         "BDF1",          "BDF2", "BDF3", "BDF4", "BDF5" };

    // Run the tests
    for ( auto tmp : integrators )
        runBasicIntegratorTests( tmp, ut );

    ut.report( 3 );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
