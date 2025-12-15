// This tests checks the creation of nonlinear solvers
// Note: the comm used should NOT be comm_world as there are cleanup issues for other comms when
// using the monitor
// option
#include "AMP/IO/PIO.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/IdentityOperator.h"
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


static void myTest( AMP::UnitTest *ut, const std::string &inputName )
{
    std::string input_file = inputName;
    std::string log_file   = "output_" + inputName;

    AMP::pout << "Running with input " << input_file << std::endl;

    size_t N_error0 = ut->NumFailLocal();

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
#ifdef AMP_USE_MPI
    auto solverComm = globalComm.dup(); // Create a unique solver comm to test proper cleanup
#else
    auto solverComm = globalComm; // a dup fails for no MPI
#endif
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Create the solution and function variables
    auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "x" );
    auto u   = AMP::LinearAlgebra::createSimpleVector<double>( 10, var, solverComm );
    auto f   = u->clone();

    // Create the operator
    auto op = std::make_shared<AMP::Operator::IdentityOperator>();
    op->setInputVariable( var );
    op->setOutputVariable( var );

    // Get the databases for the nonlinear and linear solvers
    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );

    const auto solverName =
        input_db->getDatabase( "NonlinearSolver" )->getScalar<std::string>( "name" );

    // Create the solver
    auto nonlinearSolver =
        AMP::Solver::Test::buildSolver( "NonlinearSolver", input_db, solverComm, u, op );

    ut->passes( solverName + " created" );

    // Call solve with a simple vector
    u->setRandomValues();
    f->setRandomValues();
    nonlinearSolver->apply( f, u );
    ut->passes( solverName + " solve called with simple vector" );
    auto x = u->clone();
    x->subtract( *u, *f );
    auto error = static_cast<double>( x->L2Norm() ) / static_cast<double>( f->L2Norm() );
    if ( fabs( error ) < 1e-8 )
        ut->passes( "Solve with simple vector passed" );
    else
        ut->failure( "Solve with simple vector failed" );

    // Call solve with a multivector (there can be bugs when solve is called with a single vector
    // and then a
    // multivector)
    auto mu = AMP::LinearAlgebra::MultiVector::create( "multivector", solverComm );
    auto mf = AMP::LinearAlgebra::MultiVector::create( "multivector", solverComm );
    mu->addVector( u );
    mf->addVector( f );
    mu->setRandomValues();
    mf->zero();
    nonlinearSolver->apply( mf, mu );

    if ( N_error0 == ut->NumFailLocal() )
        ut->passes( inputName );
    else
        ut->failure( inputName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::string> inputNames;

    if ( argc > 1 ) {
        inputNames.push_back( argv[1] );
    } else {
#ifdef AMP_USE_PETSC
        inputNames.emplace_back( "input_PetscSNESSolver" );
#endif
#ifdef AMP_USE_TRILINOS_NOX
        inputNames.emplace_back( "input_TrilinosNOXSolver" );
#endif
    }

    for ( auto &inputName : inputNames )
        myTest( &ut, inputName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
