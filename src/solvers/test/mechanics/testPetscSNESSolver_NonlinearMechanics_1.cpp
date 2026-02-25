#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <iostream>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &inputName )
{
    std::string input_file = inputName;
    std::string log_file   = "output_" + inputName;
    AMP::pout << "Running " << input_file << std::endl;
    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Create the meshes from the input database
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    AMP_INSIST( input_db->keyExists( "NumberOfLoadingSteps" ),
                "Key ''NumberOfLoadingSteps'' is missing!" );
    int NumberOfLoadingSteps = input_db->getScalar<int>( "NumberOfLoadingSteps" );

    auto nonlinBvpOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "nonlinearMechanicsBVPOperator", input_db ) );

    auto displacementVariable = nonlinBvpOperator->getOutputVariable();

    // For RHS (Point Forces)
    auto dirichletLoadVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    dirichletLoadVecOp->setVariable( displacementVariable );

    // For Initial-Guess
    auto dirichletDispInVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Displacement_Boundary", input_db ) );
    dirichletDispInVecOp->setVariable( displacementVariable );

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto mechNlSolVec = AMP::LinearAlgebra::createVector( dofMap, displacementVariable, true );
    auto mechNlRhsVec = mechNlSolVec->clone();
    auto mechNlResVec = mechNlSolVec->clone();
    auto mechNlScaledRhsVec = mechNlSolVec->clone();

    // Initial guess for NL solver must satisfy the displacement boundary conditions
    mechNlSolVec->setToScalar( 0.0 );
    dirichletDispInVecOp->apply( nullVec, mechNlSolVec );
    mechNlSolVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    nonlinBvpOperator->apply( mechNlSolVec, mechNlResVec );
    auto linBvpOperator = std::make_shared<AMP::Operator::LinearBVPOperator>(
        nonlinBvpOperator->getParameters( "Jacobian", nullptr ) );

    mechNlRhsVec->setToScalar( 0.0 );
    dirichletLoadVecOp->apply( nullVec, mechNlRhsVec );

    std::cout << "Initial Solution Norm: " << mechNlSolVec->L2Norm() << std::endl;

    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
    auto linearSolver_db    = nonlinearSolver_db->getDatabase( "LinearSolver" );

    // initialize the linear solver
    auto linearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
    linearSolverParams->d_pOperator = linBvpOperator;
    linearSolverParams->d_comm      = globalComm;
    auto linearSolver = std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );

    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nonlinBvpOperator;
    nonlinearSolverParams->d_pNestedSolver = linearSolver;
    nonlinearSolverParams->d_pInitialGuess = mechNlSolVec;

    auto nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

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

        double finalSolNorm = static_cast<double>( mechNlSolVec->L2Norm() );

        AMP::pout << "Final Solution Norm: " << finalSolNorm << std::endl;

        auto mechUvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride( 0, 3 ) );
        auto mechVvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride( 1, 3 ) );
        auto mechWvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride( 2, 3 ) );

        double finalMaxU = static_cast<double>( mechUvec->maxNorm() );
        double finalMaxV = static_cast<double>( mechVvec->maxNorm() );
        double finalMaxW = static_cast<double>( mechWvec->maxNorm() );

        AMP::pout << "Maximum U displacement: " << finalMaxU << std::endl;
        AMP::pout << "Maximum V displacement: " << finalMaxV << std::endl;
        AMP::pout << "Maximum W displacement: " << finalMaxW << std::endl;

        auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
        auto tmpParams =
            std::make_shared<AMP::Operator::MechanicsNonlinearFEOperatorParameters>( tmp_db );
        nonlinBvpOperator->getVolumeOperator()->reset( tmpParams );
        nonlinearSolver->setZeroInitialGuess( false );

        mesh->displaceMesh( mechNlSolVec );
    }

    double finalSolNorm = static_cast<double>( mechNlSolVec->L2Norm() );
    AMP::pout << "Final Solution Norm: " << finalSolNorm << std::endl;

    ut->passes( inputName );
}

int testPetscSNESSolver_NonlinearMechanics_1( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> inputNames;
    inputNames.emplace_back( "input_testPetscSNESSolver-NonlinearMechanics-PlateWithHole-1" );
    inputNames.emplace_back( "input_testPetscSNESSolver-LU-NonlinearMechanics-1-normal" );
    inputNames.emplace_back( "input_testPetscSNESSolver-ML-NonlinearMechanics-1-normal" );
    inputNames.emplace_back( "input_testPetscSNESSolver-LU-NonlinearMechanics-1-reduced" );
    inputNames.emplace_back( "input_testPetscSNESSolver-ML-NonlinearMechanics-1-reduced" );

    for ( auto &inputName : inputNames )
        myTest( &ut, inputName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
