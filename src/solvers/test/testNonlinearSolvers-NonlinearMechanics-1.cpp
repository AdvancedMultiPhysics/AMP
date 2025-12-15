#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <iostream>
#include <string>

#include "AMP/solvers/testHelpers/testSolverHelpers.h"

void myTest( AMP::UnitTest *ut, const std::string &fileName )
{
    PROFILE2( fileName );

    std::string input_file = fileName;
    std::string log_file   = "output_" + fileName;

    AMP::pout << "Running with input " << input_file << std::endl;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // create the Mesh
    const auto mesh = createMesh( input_db );

    AMP_INSIST( input_db->keyExists( "NumberOfLoadingSteps" ),
                "Key ''NumberOfLoadingSteps'' is missing!" );
    auto NumberOfLoadingSteps = input_db->getScalar<int>( "NumberOfLoadingSteps" );

    auto nonlinBvpOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "nonlinearMechanicsBVPOperator", input_db ) );

    // For RHS (Point Forces)
    auto dirichletLoadVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );

    auto var = nonlinBvpOperator->getOutputVariable();

    dirichletLoadVecOp->setVariable( var );

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto mechNlSolVec       = AMP::LinearAlgebra::createVector( dofMap, var, true );
    auto mechNlRhsVec       = mechNlSolVec->clone();
    auto mechNlResVec       = mechNlSolVec->clone();
    auto mechNlScaledRhsVec = mechNlSolVec->clone();

    // Initial guess for NL solver must satisfy the displacement boundary conditions
    mechNlSolVec->setToScalar( 0.0 );
    nonlinBvpOperator->modifyInitialSolutionVector( mechNlSolVec );
    nonlinBvpOperator->apply( mechNlSolVec, mechNlResVec );

    // Point forces
    mechNlRhsVec->setToScalar( 0.0 );
    dirichletLoadVecOp->apply( nullVec, mechNlRhsVec );

    // Create the solver
    auto nonlinearSolver = AMP::Solver::Test::buildSolver(
        "NonlinearSolver", input_db, globalComm, mechNlSolVec, nonlinBvpOperator );

    nonlinearSolver->setZeroInitialGuess( false );

    for ( int step = 0; step < NumberOfLoadingSteps; step++ ) {
        AMP::pout << "########################################" << std::endl;
        AMP::pout << "The current loading step is " << ( step + 1 ) << std::endl;

        double scaleValue = ( (double) step + 1.0 ) / NumberOfLoadingSteps;
        mechNlScaledRhsVec->scale( scaleValue, *mechNlRhsVec );
        mechNlScaledRhsVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        AMP::pout << "L2 Norm of RHS at loading step " << ( step + 1 ) << " is "
                  << mechNlScaledRhsVec->L2Norm() << std::endl;

        nonlinBvpOperator->residual( mechNlScaledRhsVec, mechNlSolVec, mechNlResVec );
        double initialResidualNorm = static_cast<double>( mechNlResVec->L2Norm() );
        AMP::pout << "Initial Residual Norm for loading step " << ( step + 1 ) << " is "
                  << initialResidualNorm << std::endl;

        AMP::pout << "Starting Nonlinear Solve..." << std::endl;
        nonlinearSolver->apply( mechNlScaledRhsVec, mechNlSolVec );

        nonlinBvpOperator->residual( mechNlScaledRhsVec, mechNlSolVec, mechNlResVec );
        double finalResidualNorm = static_cast<double>( mechNlResVec->L2Norm() );
        AMP::pout << "Final Residual Norm for loading step " << ( step + 1 ) << " is "
                  << finalResidualNorm << std::endl;

        if ( finalResidualNorm > ( 1.0e-10 * initialResidualNorm ) ) {
            ut->failure( "Nonlinear solve for current loading step" );
        } else {
            ut->passes( "Nonlinear solve for current loading step" );
        }

        auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
        auto tmpParams =
            std::make_shared<AMP::Operator::MechanicsNonlinearFEOperatorParameters>( tmp_db );
        nonlinBvpOperator->getVolumeOperator()->reset( tmpParams );
        nonlinearSolver->setZeroInitialGuess( false );
    }

    ut->passes( fileName );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    PROFILE_ENABLE( 8 );

    std::vector<std::string> inputNames;

    if ( argc > 1 ) {
        inputNames.push_back( argv[1] );
    } else {
#ifdef AMP_USE_TRILINOS_ML
    #ifdef AMP_USE_PETSC
        inputNames.emplace_back( "input_testPetscSNESSolver-JFNK-ML-NonlinearMechanics-1-normal" );
    #endif
        inputNames.emplace_back( "input_NKASolver-ML-NonlinearMechanics-1-normal" );
#endif
#ifdef AMP_USE_HYPRE
    #ifdef AMP_USE_PETSC
        inputNames.emplace_back(
            "input_testPetscSNESSolver-JFNK-BoomerAMG-NonlinearMechanics-1-normal" );
    #endif
        inputNames.emplace_back( "input_NKASolver-BoomerAMG-NonlinearMechanics-1-normal" );
#endif
    }

    for ( auto &inputName : inputNames )
        myTest( &ut, inputName );

    ut.report();

    PROFILE_SAVE( "testNonlinearSolvers-NonlinearMechanics-1" );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
