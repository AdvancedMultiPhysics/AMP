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
                     double tol,
                     AMP::UnitTest &ut )
{
    // Create the time integrator
    auto var            = std::make_shared<AMP::LinearAlgebra::Variable>( "x" );
    auto solution       = AMP::LinearAlgebra::createSimpleVector<double>( 1, var, AMP_COMM_WORLD );
    auto timeIntegrator = AMP::TimeIntegrator::TimeIntegratorFactory::create( params );

    auto x = solution->clone();
    //    solution->setToScalar( 1.0 );
    solution->copyVector( params->d_ic_vector );
    x->copy( *solution );

    // Advance the solution
    double dt          = timeIntegrator->getInitialDt();
    double finalTime   = timeIntegrator->getFinalTime();
    double currentTime = 0.0;
    bool good_solution;
    //    timeIntegrator->setInitialDt( dt );
    while ( ( currentTime < finalTime ) && timeIntegrator->stepsRemaining() && ( dt > 0.0 ) ) {
        timeIntegrator->advanceSolution( dt, currentTime == 0, solution, x );
        good_solution = timeIntegrator->checkNewSolution();
        if ( good_solution ) {
            timeIntegrator->updateSolution();
            solution->copyVector( x );
        }
        dt          = timeIntegrator->getNextDt( good_solution );
        currentTime = timeIntegrator->getCurrentTime();
    }

    // Check the answer
    double ans2 = static_cast<double>( solution->max() );
    //    double tol  = 5.0e-10;
    if ( name == "ExplicitEuler" || name == "BDF1" )
        tol = std::sqrt( tol );
    if ( AMP::Utilities::approx_equal( ans2, ans, tol ) )
        ut.passes( name + " - " + test );
    else
        ut.failure( AMP::Utilities::stringf( "%s - %s (%0.16f) (%0.16f) (%0.16f)",
                                             name.data(),
                                             test.data(),
                                             ans,
                                             ans2,
                                             ans - ans2 ) );
}

bool isImplicitTI( std::shared_ptr<const AMP::Database> db )
{
    AMP_ASSERT( db );
    const auto imp_ti = { "CN", "Backward Euler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    const auto name   = db->getScalar<std::string>( "name" );
    AMP::pout << name << std::endl;
    return ( std::find( imp_ti.begin(), imp_ti.end(), name ) != imp_ti.end() );
}

bool isAdaptiveRK( std::shared_ptr<const AMP::Database> db )
{
    AMP_ASSERT( db );
    const auto adaptive_rk = { "RK12", "RK23", "RK34", "RK45" };
    const auto name        = db->getScalar<std::string>( "name" );
    AMP::pout << name << std::endl;
    return ( std::find( adaptive_rk.begin(), adaptive_rk.end(), name ) != adaptive_rk.end() );
}

void updateDatabaseIfImplicit( std::shared_ptr<AMP::Database> db )
{
    if ( isImplicitTI( db ) ) {

        //        db->putScalar<std::string>( "name", "ImplicitIntegrator" );
        const auto name = db->getScalar<std::string>( "name" );
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

// 3 test cases to run
void runTestCases( const std::string &name,
                   std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> params,
                   const std::array<bool, 3> &run_test,
                   const double icval,
                   double tol,
                   double finalTime,
                   AMP::UnitTest &ut )
{
    std::string test;
    auto var    = std::make_shared<AMP::LinearAlgebra::Variable>( "x" );
    auto source = AMP::LinearAlgebra::createSimpleVector<double>( 1, var, AMP_COMM_WORLD );
    source->setToScalar( 3.0 );

    if ( run_test[0] ) {
        // Test with a fixed source and null operator
        test = "du/dt=3";
        AMP::pout << "Testing " << name << " with " << test
                  << " no operator, fixed source and fixed timestep" << std::endl;

        params->d_pSourceTerm = source;
        params->d_operator    = std::make_shared<AMP::Operator::NullOperator>();
        testIntegrator( name, test, params, icval + 3 * finalTime, tol, ut );
    }

    if ( run_test[1] ) {
        // Test with no source and constant operator
        test = "du/dt=-3u";
        AMP::pout << "Testing " << name << " with " << test
                  << " constant operator, no source and fixed timestep" << std::endl;
        params->d_pSourceTerm = nullptr;
        params->d_operator =
            std::make_shared<FunctionOperator>( []( double x ) { return -3.0 * x; } );
        testIntegrator( name, test, params, icval * std::exp( -3.0 * finalTime ), tol, ut );
    }
    if ( run_test[2] ) {
        // Test with fixed source and constant operator
        test = "du/dt=-3u+3";
        AMP::pout << "Testing " << name << " with " << test
                  << " constant operator, fixed source and fixed timestep" << std::endl;
        params->d_pSourceTerm = source;
        params->d_operator =
            std::make_shared<FunctionOperator>( []( double x ) { return -3.0 * x; } );
        testIntegrator( name,
                        test,
                        params,
                        icval * std::exp( -3.0 * finalTime ) +
                            ( 1.0 - std::exp( -3.0 * finalTime ) ),
                        tol,
                        ut );
    }
}

void runBasicIntegratorTests( const std::string &name, AMP::UnitTest &ut )
{
    double finalTime   = 0.001;
    const double icval = 10.0;
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
    params->d_operator  = std::make_shared<AMP::Operator::NullOperator>();
    try {
        auto timeIntegrator = AMP::TimeIntegrator::TimeIntegratorFactory::create( params );
        ut.passes( name + " - created" );
    } catch ( ... ) {
        ut.failure( name + " - created" );
        return;
    }

    db->putScalar<double>( "initial_dt", 0.0001 );
    runTestCases( name, params, { true, true, true }, 1.0, 5.0e-10, finalTime, ut );

    finalTime  = 0.01;
    double tol = 8.0e-08;
    ic->setToScalar( icval );

    if ( isAdaptiveRK( db ) ) {

        db->putScalar<bool>( "use_fixed_dt", false );
        db->putScalar<double>( "initial_dt", 0.0005 );
        db->putScalar<double>( "final_time", finalTime );

        runTestCases( name, params, { false, true, true }, icval, tol, finalTime, ut );
    }

    if ( isImplicitTI( db ) ) {

        //==============================================================================
        std::string timestep_strategy = "constant";
        db->putScalar<std::string>( "timestep_selection_strategy", timestep_strategy );
        db->putScalar<bool>( "use_predictor", true );
        db->putScalar<bool>( "auto_component_scaling", false );
        db->putScalar<double>( "initial_dt", 0.0005 );
        db->putScalar<double>( "final_time", finalTime );

        runTestCases( name, params, { false, true, true }, icval, tol, finalTime, ut );

        //==============================================================================
        timestep_strategy = "final constant";
        db->putScalar<std::string>( "timestep_selection_strategy", timestep_strategy );
        db->putScalar<double>( "max_dt", 0.001 );
        db->putScalar<int>( "number_of_time_intervals", 25 );
        db->putScalar<int>( "number_initial_fixed_steps", 0 );

        runTestCases( name, params, { false, true, true }, icval, tol, finalTime, ut );

        //==============================================================================
        // vary the PI controller
        tol = 2.0e-07;
        std::string pi_controller{ "PC.4.7" };
        timestep_strategy = "truncationErrorStrategy";
        db->putScalar<std::string>( "timestep_selection_strategy", timestep_strategy );
        db->putVector<double>( "problem_fixed_scaling", { icval } );
        db->putScalar<double>( "initial_dt", 1.0e-06 );
        runTestCases( name, params, { false, true, true }, icval, tol, finalTime, ut );

        //==============================================================================
        pi_controller = "H211b";
        db->putScalar<std::string>( "pi_controller_type", pi_controller );
        runTestCases( name, params, { false, true, true }, icval, tol, finalTime, ut );

        //==============================================================================
        pi_controller = "PC11";
        db->putScalar<std::string>( "pi_controller_type", pi_controller );
        runTestCases( name, params, { false, true, true }, icval, tol, finalTime, ut );

        //==============================================================================
        pi_controller = "Deadbeat";
        db->putScalar<std::string>( "pi_controller_type", pi_controller );
        runTestCases( name, params, { false, true, true }, icval, tol, finalTime, ut );

        //==============================================================================
        // turn off the controller
        db->putScalar<bool>( "use_pi_controller", false );
        runTestCases( name, params, { false, true, true }, icval, tol, finalTime, ut );
    }
}


int testSimpleTimeIntegration( int argc, char *argv[] )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::string> integrators;

    if ( argc > 1 ) {

        for ( int i = 1; i < argc; i++ )
            integrators.emplace_back( argv[i] );

    } else {
        // List of integrators
        // We need to look at the errors for the first order -- whether they are acceptable
        integrators = { "ExplicitEuler", "RK2",  "RK4",  "RK12", "RK23", "RK34", "RK45", "CN",
                        "BDF1",          "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    }
    // Run the tests
    for ( auto tmp : integrators )
        runBasicIntegratorTests( tmp, ut );

    ut.report( 3 );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
