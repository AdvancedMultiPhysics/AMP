// This tests checks the creation of nonlinear solvers
// Note: the comm used should NOT be comm_world as there are cleanup issues for other comms when
// using the monitor
// option
#include "AMP/IO/PIO.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/IdentityOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <string>

static void
myTest( AMP::UnitTest *ut, std::shared_ptr<AMP::Database> input_db, const std::string &banner )
{
    size_t N_error0 = ut->NumFailLocal();
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
#ifdef AMP_USE_MPI
    auto solverComm = globalComm.dup(); // Create a unique solver comm to test proper cleanup
#else
    auto solverComm = globalComm; // a dup fails for no MPI
#endif
    // Create the solution and function variables
    auto db = AMP::Database::create(
        "name", "IdentityOperator", "InputVariable", "x", "OutputVariable", "x", "localSize", 20 );
    input_db->putDatabase( "IdentityOperator", std::move( db ) );
    // Create the operator
    auto op =
        AMP::Operator::OperatorBuilder::createOperator( nullptr, "IdentityOperator", input_db );

    auto u = std::dynamic_pointer_cast<AMP::Operator::IdentityOperator>( op )->getInputVector();
    auto f = u->clone();


    // Create the solver
    auto linearSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, solverComm, u, op );

    ut->passes( "Solver created" );

    // Call solve with a simple vector
    u->setRandomValues();
    f->setRandomValues();
    linearSolver->setZeroInitialGuess( true );
    linearSolver->apply( f, u );
    auto iter   = linearSolver->getIterations();
    auto solver = linearSolver->type();
    u->subtract( *u, *f );
    auto error = static_cast<double>( u->L2Norm() );
    if ( iter == 1 && fabs( error ) < 1.0e-14 )
        ut->passes( solver +
                    " solves identity operator in one iteration withj zero initial guess" );
    else
        ut->failure( solver +
                     " fails to solve identity operator in one iteration with zero initial guess" );

    // Call solve with a simple vector
    u->setRandomValues();
    f->setRandomValues();
    linearSolver->setZeroInitialGuess( false );
    linearSolver->apply( f, u );
    iter = linearSolver->getIterations();
    u->subtract( *u, *f );
    error = static_cast<double>( u->L2Norm() );
    if ( iter == 1 && fabs( error ) < 1.0e-14 )
        ut->passes( solver +
                    " solves identity operator in one iteration with non-zero initial guess" );
    else
        ut->failure(
            solver +
            " fails to solve identity operator in one iteration with non-zero initial guess" );

        // Call solve with a multivector (there can be bugs when solve is called with a single
        // vector and then a multivector)
#if 0
    auto mu = AMP::LinearAlgebra::MultiVector::create( "multivector", solverComm );
    auto mf = AMP::LinearAlgebra::MultiVector::create( "multivector", solverComm );
    mu->addVector( u );
    mf->addVector( f );
    mu->setRandomValues();
    mf->zero();
    linearSolver->apply( mf, mu );
#endif
    if ( N_error0 == ut->NumFailLocal() )
        ut->passes( banner );
    else
        ut->failure( banner );
}

static void myTest( AMP::UnitTest *ut, const std::string &inputName )
{
    std::string input_file = inputName;
    AMP::pout << "Running with input " << input_file << std::endl;
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );
    std::string banner( "Identity solve with input file: " + input_file );
    myTest( ut, input_db, banner );
}

static void myTest( AMP::UnitTest *ut )
{
    std::vector<std::pair<std::string, std::string>> solvers{ { "CG", "NoPC" },
                                                              { "GMRES", "NoPC" },
                                                              { "FGMRES", "NoPC" },
                                                              { "BiCGSTAB", "NoPC" },
                                                              { "TFQMR", "NoPC" } };

    for ( auto &[primary, nested] : solvers ) {
        std::shared_ptr<AMP::Database> db = std::make_shared<AMP::Database>( "SolverDatabase" );
        auto use_nested                   = ( nested == "NoPC" ) ? false : true;
        db->putDatabase(
            "LinearSolver",
            AMP::Solver::Test::SolverParameters::getParameters( primary, use_nested ) );
        if ( use_nested ) {
            db->putDatabase(
                "Preconditioner",
                AMP::Solver::Test::SolverParameters::getParameters( nested, use_nested ) );
        }

        std::string banner;
        if ( use_nested )
            banner = "Running " + primary + " with PC " + nested + " on identity";
        else
            banner = "Running " + primary + " on identity";
        AMP::pout << banner << std::endl;
        myTest( ut, db, banner );
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::string> inputNames;

    if ( argc > 1 ) {
        inputNames.push_back( argv[1] );
        myTest( &ut, inputNames[0] );
    } else {
        myTest( &ut );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
