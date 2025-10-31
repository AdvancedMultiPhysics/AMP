#include "AMP/IO/PIO.h"
#include "AMP/operators/FunctionOperator.h"
#include "AMP/operators/Operator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#ifdef AMP_USE_TRILINOS_NOX
    #include "AMP/solvers/trilinos/nox/TrilinosNOXSolverParameters.h"
#endif

static double val        = 0.25;
static double init_guess = 0.325;


auto getSolverParameters( std::shared_ptr<AMP::Database> input_db )
{
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
#ifdef AMP_USE_MPI
    auto solverComm = globalComm.dup(); // Create a unique solver comm to test proper cleanup
#else
    auto solverComm = globalComm; // a dup fails for no MPI
#endif

    // Create the solver
    std::shared_ptr<AMP::Solver::SolverStrategyParameters> parameters;
    auto name = input_db->getString( "name" );
    if ( name == "TrilinosNOXSolver" ) {
#ifdef AMP_USE_TRILINOS_NOX
        parameters = std::make_shared<AMP::Solver::TrilinosNOXSolverParameters>( input_db );
#else
        AMP_ERROR( "AMP built without support for Trilinos NOX" );
#endif
    } else {
        parameters = std::make_shared<AMP::Solver::SolverStrategyParameters>( input_db );
    }

    parameters->d_comm      = solverComm;
    parameters->d_db        = input_db;
    parameters->d_global_db = input_db;

    return parameters;
}

static void nullRhsTest( AMP::UnitTest *ut, std::shared_ptr<AMP::Database> input_db )
{

    AMP_ASSERT( input_db );
    auto name = input_db->getString( "name" );
    AMP::pout << "Running nullRhs test with non-zero initial guess and solver " << name
              << std::endl;

    // Create the solution and function variables
    const size_t N = 1;
    auto u         = AMP::LinearAlgebra::createSimpleVector<double>( N, "x" );

    // Create the operator
    auto op =
        std::make_shared<AMP::Operator::FunctionOperator>( []( double x ) { return x * x - val; } );

    auto parameters         = getSolverParameters( input_db );
    parameters->d_pOperator = op;

    auto nonlinearSolver = AMP::Solver::SolverFactory::create( parameters );

    // Call solve with a simple vector
    u->setToScalar( init_guess );
    nonlinearSolver->setZeroInitialGuess( false );
    nonlinearSolver->apply( nullptr, u );

    u->addScalar( *u, -std::sqrt( val ) );
    auto error = static_cast<double>( u->L2Norm() );
    auto msg1{ " solve with simple vector, null rhs, and non-zero initial guess" };
    if ( fabs( error ) < 1e-14 )
        ut->passes( name + msg1 + " passed" );
    else
        ut->failure( name + msg1 + " failed" );

    // AMP::pout << "Running nullRhs test with zero initial guess and solver " << name << std::endl;
    // u->setToScalar( init_guess );
    // nonlinearSolver->setZeroInitialGuess( true );
    // nonlinearSolver->apply( nullptr, u );

    // u->addScalar( *u, -std::sqrt( val ) );
    // error = static_cast<double>( u->L2Norm() );
    // auto msg2{ " solve with simple vector, null rhs, and zero initial guess" };
    // if ( fabs( error ) < 1e-14 )
    //     ut->passes( name + msg2 + " passed" );
    // else
    //     ut->failure( name + msg2 + " failed" );
}

static void nonNullRhsTest( AMP::UnitTest *ut, std::shared_ptr<AMP::Database> input_db )
{

    AMP_ASSERT( input_db );
    auto name = input_db->getString( "name" );
    AMP::pout << "Running test with non null rhs and non-zero initial guess with solver " << name
              << std::endl;

    // Create the solution and function variables
    const size_t N = 1;
    auto u         = AMP::LinearAlgebra::createSimpleVector<double>( N, "x" );
    auto f         = AMP::LinearAlgebra::createSimpleVector<double>( N, "f" );
    f->setToScalar( val );

    // Create the operator
    auto op = std::make_shared<AMP::Operator::FunctionOperator>( []( double x ) { return x * x; } );

    auto parameters         = getSolverParameters( input_db );
    parameters->d_pOperator = op;

    auto nonlinearSolver = AMP::Solver::SolverFactory::create( parameters );

    // Call solve with a simple vector
    u->setToScalar( init_guess );
    nonlinearSolver->setZeroInitialGuess( false );
    nonlinearSolver->apply( f, u );

    u->addScalar( *u, -std::sqrt( val ) );
    auto error = static_cast<double>( u->L2Norm() );
    std::string msg1{ " solve with simple vector, non null rhs, non-zero initial guess" };
    if ( fabs( error ) < 1e-14 )
        ut->passes( name + msg1 + "  passed" );
    else
        ut->failure( name + msg1 + " failed" );

    // AMP::pout << "Running test with non null rhs and zero initial guess with solver " << name
    //           << std::endl;
    // u->setToScalar( init_guess );
    // nonlinearSolver->reset( {} );
    // nonlinearSolver->setZeroInitialGuess( true );
    // nonlinearSolver->apply( f, u );

    // u->addScalar( *u, -std::sqrt( val ) );
    // error = static_cast<double>( u->L2Norm() );
    // std::string msg2{ " solve with simple vector, non null rhs, zero initial guess" };
    // if ( fabs( error ) < 1e-14 )
    //     ut->passes( name + msg2 + "  passed" );
    // else
    //     ut->failure( name + msg2 + " failed" );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::shared_ptr<AMP::Database>> solverDBs;

    auto nka_db = AMP::Database::create( "name",
                                         "NKASolver",
                                         "print_info_level",
                                         2,
                                         "max_vectors",
                                         3,
                                         "absolute_tolerance",
                                         1.0e-14,
                                         "relative_tolerance",
                                         1.0e-18,
                                         "angle_tolerance",
                                         0.2,
                                         "uses_preconditioner",
                                         false );

    solverDBs.emplace_back( std::move( nka_db ) );

#ifdef AMP_USE_PETSC

    auto snes_db = AMP::Database::create( "name",
                                          "SNESSolver",
                                          "print_info_level",
                                          3,
                                          "max_iterations",
                                          20,
                                          "absolute_tolerance",
                                          1.0e-14,
                                          "relative_tolerance",
                                          1.0e-18,
                                          "stepTolerance",
                                          1.0e-16,
                                          "usesJacobian",
                                          false,
                                          "uses_preconditioner",
                                          false );
    solverDBs.emplace_back( std::move( snes_db ) );
#endif

#ifdef AMP_USE_TRILINOS_NOX
    auto nox_db = AMP::Database::create( "name",
                                         "TrilinosNOXSolver",
                                         "print_info_level",
                                         2,
                                         "solver",
                                         "JFNK",
                                         "max_iterations",
                                         20,
                                         "absolute_tolerance",
                                         1.0e-14,
                                         "relative_tolerance",
                                         1.0e-18,
                                         "uses_preconditioner",
                                         false );

    auto linear_db = AMP::Database::create( "print_info_level",
                                            2,
                                            "linearSolverType",
                                            "Belos",
                                            "linearSolver",
                                            "Pseudo Block GMRES",
                                            "max_iterations",
                                            100,
                                            "absolute_tolerance",
                                            1.0e-14,
                                            "relative_tolerance",
                                            1.0e-01,
                                            "zero_initial_guess",
                                            false );

    nox_db->putDatabase( "LinearSolver", std::move( linear_db ) );

    solverDBs.emplace_back( std::move( nox_db ) );
#endif

    for ( auto &db : solverDBs ) {
        nullRhsTest( &ut, db );
        nonNullRhsTest( &ut, db );
    }
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
