// This tests checks the creation of a TrilinosNOXSolver
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include <iostream>
#include <string>

#include "AMP/ampmesh/Mesh.h"
#include "AMP/operators/IdentityOperator.h"
#include "AMP/operators/NullOperator.h"
#include "AMP/solvers/trilinos/nox/TrilinosNOXSolver.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/NullVector.h"
#include "AMP/vectors/SimpleVector.h"


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    AMP::AMP_MPI solverComm =
        globalComm.dup(); // Create a unique solver comm to test proper cleanup


    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Create the solution and function variables
    AMP::LinearAlgebra::Variable::shared_ptr var( new AMP::LinearAlgebra::Variable( "x" ) );
    AMP::LinearAlgebra::Vector::shared_ptr u =
        AMP::LinearAlgebra::SimpleVector<double>::create( 25, var, solverComm );
    AMP::LinearAlgebra::Vector::shared_ptr f     = u->cloneVector();
    AMP::LinearAlgebra::Vector::shared_ptr icVec = u->cloneVector();

    // Create the operator
    std::shared_ptr<AMP::Operator::IdentityOperator> op( new AMP::Operator::IdentityOperator() );
    op->setInputVariable( var );
    op->setOutputVariable( var );

    // Get the databases for the nonlinear and linear solvers
    std::shared_ptr<AMP::Database> nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
    // std::shared_ptr<AMP::Database> linearSolver_db =
    // nonlinearSolver_db->getDatabase("LinearSolver");

    // initialize the nonlinear solver parameters
    std::shared_ptr<AMP::Solver::TrilinosNOXSolverParameters> nonlinearSolverParams(
        new AMP::Solver::TrilinosNOXSolverParameters( nonlinearSolver_db ) );
    nonlinearSolverParams->d_comm            = solverComm;
    nonlinearSolverParams->d_pInitialGuess   = icVec;
    nonlinearSolverParams->d_pOperator       = op;
    nonlinearSolverParams->d_pLinearOperator = op;

    // Create the nonlinear solver
    std::shared_ptr<AMP::Solver::TrilinosNOXSolver> nonlinearSolver(
        new AMP::Solver::TrilinosNOXSolver( nonlinearSolverParams ) );
    ut->passes( "TrilinosNOXSolver created" );

    // Call solve with a simple vector
    u->setRandomValues();
    f->setRandomValues();
    nonlinearSolver->solve( f, u );
    ut->passes( "TrilinosNOXSolver solve called with simple vector" );
    AMP::LinearAlgebra::Vector::shared_ptr x = u->cloneVector();
    x->subtract( u, f );
    double error = x->L2Norm() / std::max( f->L2Norm(), 1.0 );
    if ( fabs( error ) < 1e-8 )
        ut->passes( "Solve with simple vector passed" );
    else
        ut->failure( "Solve with simple vector failed" );


    // Call solve with a multivector (there can be bugs when solve is called with a single vector
    // and then a
    // multivector)
    std::shared_ptr<AMP::LinearAlgebra::MultiVector> mu =
        AMP::LinearAlgebra::MultiVector::create( "multivector", solverComm );
    std::shared_ptr<AMP::LinearAlgebra::MultiVector> mf =
        AMP::LinearAlgebra::MultiVector::create( "multivector", solverComm );
    mu->addVector( u );
    mf->addVector( f );
    mu->setRandomValues();
    mf->zero();
    nonlinearSolver->solve( mf, mu );
    ut->passes( "TrilinosNOXSolver solve called with multivector" );
}


int testTrilinosNOXSolver( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    myTest( &ut, "testTrilinosNOXSolver" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}