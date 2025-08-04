#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>

std::shared_ptr<AMP::Solver::SolverStrategy>
buildSolver( std::shared_ptr<AMP::Database> input_db,
             const std::string &solver_name,
             const AMP::AMP_MPI &comm,
             std::shared_ptr<AMP::Operator::LinearOperator> &op )
{


    AMP_INSIST( input_db->keyExists( solver_name ), "Key " + solver_name + " is missing!" );

    const auto &db = input_db->getDatabase( solver_name );

    std::shared_ptr<AMP::Solver::SolverStrategyParameters> parameters;
    if ( db->keyExists( "name" ) ) {

        auto name = db->getString( "name" );

        if ( ( name == "GMRESSolver" ) || ( name == "CGSolver" ) || ( name == "BiCGSTABSolver" ) ) {

            // check if we need to construct a preconditioner
            auto uses_preconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );
            std::shared_ptr<AMP::Solver::SolverStrategy> pcSolver;

            if ( uses_preconditioner ) {

                auto pc_name = db->getWithDefault<std::string>( "pc_name", "Preconditioner" );

                pcSolver = AMP::Solver::Test::buildSolver( pc_name, input_db, comm, nullptr, op );

                AMP_INSIST( pcSolver, "null preconditioner" );
            }

            parameters         = std::make_shared<AMP::Solver::SolverStrategyParameters>( db );
            parameters->d_comm = comm;
            parameters->d_pNestedSolver = pcSolver;

        } else {
            parameters = std::make_shared<AMP::Solver::SolverStrategyParameters>( db );
        }

        AMP_INSIST( parameters != nullptr, "null parameter object" );
        parameters->d_pOperator = op;

    } else {
        AMP_ERROR( "Key name does not exist in solver database" );
    }

    auto solver = AMP::Solver::SolverFactory::create( parameters );

    return solver;
}

void userLinearOperatorTest( AMP::UnitTest *const ut, const std::string &inputFileName )
{
    // Test create
    const std::string input_file = inputFileName;
    const std::string log_file   = "output_" + inputFileName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // read the input file into a database
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // extract the Mesh database and create the mesh parameters
    const auto meshDB = input_db->getDatabase( "Mesh" );
    auto params       = std::make_shared<AMP::Mesh::MeshParameters>( meshDB );
    params->setComm( globalComm );

    // create the mesh
    const auto mesh = AMP::Mesh::MeshFactory::create( params );

    // create a linear diffusion operator
    auto linearOperator =
        AMP::Operator::OperatorBuilder::createOperator( mesh, "DiffusionBVPOperator", input_db );
    auto diffOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( linearOperator );

    // extract the internal matrix
    const auto &userMat = diffOp->getMatrix();

    AMP_INSIST( userMat->numGlobalColumns() == userMat->numGlobalRows(), "matrix is not square" );

    // extract the right vector
    const auto userVector = userMat->getInputVector();

    // concludes creation of a native linear operator
    // ************************************************************************************************

    // extract information about the local size and mpi comm
    const auto localSize = userVector->getLocalSize();
    const auto ampComm   = userVector->getComm();

    // construct a dof manager
    const auto userDM     = userVector->getDOFManager();
    const auto dofManager = std::make_shared<AMP::Discretization::DOFManager>(
        localSize, ampComm, userDM->getRemoteDOFs() );

    const auto copyVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "copyVariable" );

    // create a vector based on the dofs and variable
    auto ampVector = AMP::LinearAlgebra::createVector( dofManager, copyVariable );
    AMP_INSIST( ampVector != nullptr, "ampVector is null" );

    // copy values from one vector to another
    std::copy( userVector->begin(), userVector->end(), ampVector->begin() );
    ampVector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    // concludes demonstrating how to initialize an AMP vector from a user vector
    // ************************************************************************************************

    // create a lambda that returns non zero column ids given a global row id
    auto getColumnIDS = [userMat]( size_t row ) { return userMat->getColumnIDs( row ); };

    // create a matrix based on the dimensions of the copied vector
    auto ampMat = AMP::LinearAlgebra::createMatrix( ampVector, ampVector, "auto", getColumnIDS );

    // construct a LinearOperator and set its matrix
    const auto linearOpDB = std::make_shared<AMP::Database>( "linearOperatorDB" );
    linearOpDB->putScalar<int>( "print_info_level", 0 );
    auto linearOpParameters = std::make_shared<AMP::Operator::OperatorParameters>( linearOpDB );
    auto linearOp           = std::make_shared<AMP::Operator::LinearOperator>( linearOpParameters );
    linearOp->setMatrix( ampMat );

    ampMat->axpy( 1.0, userMat );
    ampMat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    // concludes demonstrating how to initialize an AMP linear operator from a user matrix
    // ************************************************************************************************

    auto f  = userVector->clone();
    auto u  = userVector->clone();
    auto ri = userVector->clone();
    auto rf = userVector->clone();

    linearOp->setVariables( u->getVariable(), f->getVariable() );

    f->zero();
    u->setRandomValues();

    // ************************************************************************************************
    // make sure the database on theinput file exists for the linear solver
    AMP_INSIST( input_db->keyExists( "LinearSolver" ), "Key ''LinearSolver'' is missing!" );

    auto linearSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, globalComm, nullptr, linearOp );

    // Use a random initial guess
    linearSolver->setZeroInitialGuess( false );

    // Compute the initial residual
    linearOp->residual( f, u, ri );

    // Solve the problem.
    linearSolver->apply( f, u );

    // Compute the final residual
    linearOp->residual( f, u, rf );

    // Check the L2 norm of the residuals.
    const double finalResidualNorm   = static_cast<double>( rf->L2Norm() );
    const double initialResidualNorm = static_cast<double>( ri->L2Norm() );
    AMP::pout << "Initial Residual Norm: " << initialResidualNorm << std::endl;
    AMP::pout << "Final Residual Norm  : " << finalResidualNorm << std::endl;

    if ( finalResidualNorm / initialResidualNorm > 1.0e-7 ) {
        ut->failure( "solver could NOT solve a linear thermal problem" );
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files;

    if ( argc > 1 ) {

        files.emplace_back( argv[1] );

    } else {

#ifdef AMP_USE_TRILINOS
        files.emplace_back( "input_testSolversForUserMatrix-ML" );
#endif
#ifdef AMP_USE_HYPRE
        files.emplace_back( "input_testSolversForUserMatrix-BoomerAMG" );
#endif
#ifdef AMP_USE_TRILINOS_MUELU
        files.emplace_back( "input_testSolversForUserMatrix-MueLu" );
#endif
    }

    for ( const auto &file : files )
        userLinearOperatorTest( &ut, file );

    ut.report();

    const int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
