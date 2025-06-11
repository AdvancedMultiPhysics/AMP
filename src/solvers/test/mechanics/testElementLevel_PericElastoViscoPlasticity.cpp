#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/libmesh/initializeLibMesh.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearElement.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/operators/mechanics/ThermalStrainMaterialModel.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include "libmesh/mesh_communication.h"

#include <iostream>
#include <memory>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file  = "input_" + exeName;
    std::string output_file = "output_" + exeName + ".txt";
    std::string log_file    = "log_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    [[maybe_unused]] auto libmeshInit =
        std::make_shared<AMP::Mesh::initializeLibMesh>( globalComm );
    {

        // Read the input file
        auto input_db = AMP::Database::parseInputFile( input_file );
        input_db->print( AMP::plog );

        auto mesh_file = input_db->getString( "mesh_file" );
        auto mesh      = AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD );

        auto NodalVectorDOF = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, 3 );

        AMP_INSIST( input_db->keyExists( "NumberOfLoadingSteps" ),
                    "Key ''NumberOfLoadingSteps'' is missing!" );
        int NumberOfLoadingSteps = input_db->getScalar<int>( "NumberOfLoadingSteps" );

        // Create a nonlinear BVP operator for mechanics
        AMP_INSIST( input_db->keyExists( "NonlinearMechanicsOperator" ), "key missing!" );
        auto nonlinearMechanicsDatabase = input_db->getDatabase( "NonlinearMechanicsOperator" );
        auto dirichletVectorCorrectionDatabaseName =
            nonlinearMechanicsDatabase->getString( "BoundaryOperator" );
        auto dirichletVectorCorrectionDatabase =
            input_db->getDatabase( dirichletVectorCorrectionDatabaseName );
        auto nonlinearMechanicsBVPoperator =
            std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
                AMP::Operator::OperatorBuilder::createOperator(
                    mesh, "NonlinearMechanicsOperator", input_db ) );
        auto nonlinearMechanicsVolumeOperator =
            std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
                nonlinearMechanicsBVPoperator->getVolumeOperator() );
        std::shared_ptr<AMP::Operator::ElementPhysicsModel> mechanicsMaterialModel =
            nonlinearMechanicsVolumeOperator->getMaterialModel();

        // Create a Linear BVP operator for mechanics
        AMP_INSIST( input_db->keyExists( "LinearMechanicsOperator" ), "key missing!" );
        auto linearMechanicsBVPoperator =
            std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
                AMP::Operator::OperatorBuilder::createOperator(
                    mesh, "LinearMechanicsOperator", input_db, mechanicsMaterialModel ) );

        // Create the variables
        auto mechanicsNonlinearVolumeOperator =
            std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
                nonlinearMechanicsBVPoperator->getVolumeOperator() );

        auto multivariable = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(
            mechanicsNonlinearVolumeOperator->getInputVariable() );
        auto dispVar = multivariable->getVariable( AMP::Operator::Mechanics::DISPLACEMENT );

        /* auto mechanicsLinearVolumeOperator =
            std::dynamic_pointer_cast<AMP::Operator::MechanicsLinearFEOperator>(
                linearMechanicsBVPoperator->getVolumeOperator());  */
        auto mechanicsNonlinearMaterialModel =
            std::dynamic_pointer_cast<AMP::Operator::MechanicsMaterialModel>(
                mechanicsNonlinearVolumeOperator->getMaterialModel() );

        // For RHS (Point Forces)
        std::shared_ptr<AMP::Operator::ElementPhysicsModel> dummyModel;
        auto dirichletLoadVecOp =
            std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
                AMP::Operator::OperatorBuilder::createOperator(
                    mesh, "Load_Boundary", input_db, dummyModel ) );
        dirichletLoadVecOp->setVariable( dispVar );

        // Create the vectors
        AMP::LinearAlgebra::Vector::shared_ptr nullVec;
        auto solVec       = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
        auto rhsVec       = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
        auto resVec       = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
        auto scaledRhsVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );

        // Initial guess
        solVec->zero();
        nonlinearMechanicsBVPoperator->modifyInitialSolutionVector( solVec );

        // RHS
        rhsVec->zero();
        dirichletLoadVecOp->apply( nullVec, rhsVec );
        nonlinearMechanicsBVPoperator->modifyRHSvector( rhsVec );

        // We need to reset the linear operator before the solve since TrilinosML does
        // the factorization of the matrix during construction and so the matrix must
        // be correct before constructing the TrilinosML object.
        nonlinearMechanicsBVPoperator->apply( solVec, resVec );
        linearMechanicsBVPoperator->reset(
            nonlinearMechanicsBVPoperator->getParameters( "Jacobian", solVec ) );

        double epsilon =
            1.0e-13 *
            static_cast<double>(
                ( ( linearMechanicsBVPoperator->getMatrix() )->extractDiagonal() )->L1Norm() );

        auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
        auto linearSolver_db    = nonlinearSolver_db->getDatabase( "LinearSolver" );

        // ---- first initialize the preconditioner
        auto pcSolver_db = linearSolver_db->getDatabase( "Preconditioner" );
        auto pcSolverParams =
            std::make_shared<AMP::Solver::TrilinosMLSolverParameters>( pcSolver_db );
        pcSolverParams->d_pOperator = linearMechanicsBVPoperator;
        auto pcSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( pcSolverParams );

        // HACK to prevent a double delete on Petsc Vec
        std::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver;

        // initialize the linear solver
        auto linearSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
        linearSolverParams->d_pOperator     = linearMechanicsBVPoperator;
        linearSolverParams->d_comm          = globalComm;
        linearSolverParams->d_pNestedSolver = pcSolver;
        auto linearSolver = std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );

        // initialize the nonlinear solver
        auto nonlinearSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
        // change the next line to get the correct communicator out
        nonlinearSolverParams->d_comm          = globalComm;
        nonlinearSolverParams->d_pOperator     = nonlinearMechanicsBVPoperator;
        nonlinearSolverParams->d_pNestedSolver = linearSolver;
        nonlinearSolverParams->d_pInitialGuess = solVec;
        nonlinearSolver.reset( new AMP::Solver::PetscSNESSolver( nonlinearSolverParams ) );

        nonlinearSolver->setZeroInitialGuess( false );

        double epsilon_dot = 0.1;
        double delta_time  = 0.01;

        for ( int step = 0; step < NumberOfLoadingSteps; step++ ) {
            AMP::pout << "########################################" << std::endl;
            AMP::pout << "The current loading step is " << ( step + 1 ) << std::endl;

            nonlinearMechanicsBVPoperator->modifyInitialSolutionVector( solVec );

            double current_time = delta_time * ( (double) step + 1.0 );
            mechanicsNonlinearMaterialModel->updateTime( current_time );

            double scaleValue = ( (double) step + 1.0 ) / NumberOfLoadingSteps;
            scaledRhsVec->scale( scaleValue, *rhsVec );
            AMP::pout << "L2 Norm of RHS at loading step " << ( step + 1 ) << " is "
                      << scaledRhsVec->L2Norm() << std::endl;

            nonlinearMechanicsBVPoperator->residual( scaledRhsVec, solVec, resVec );
            double initialResidualNorm = static_cast<double>( resVec->L2Norm() );
            AMP::pout << "Initial Residual Norm for loading step " << ( step + 1 ) << " is "
                      << initialResidualNorm << std::endl;

            nonlinearSolver->apply( scaledRhsVec, solVec );

            nonlinearMechanicsBVPoperator->residual( scaledRhsVec, solVec, resVec );
            double finalResidualNorm = static_cast<double>( resVec->L2Norm() );
            AMP::pout << "Final Residual Norm for loading step " << ( step + 1 ) << " is "
                      << finalResidualNorm << std::endl;

            if ( finalResidualNorm > ( 1.0e-10 * initialResidualNorm ) ) {
                ut->failure( "Nonlinear solve for current loading step" );
            } else {
                ut->passes( "Nonlinear solve for current loading step" );
            }

            double finalSolNorm = static_cast<double>( solVec->L2Norm() );
            AMP::pout << "Final Solution Norm: " << finalSolNorm << std::endl;

            AMP_ASSERT( solVec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
            auto mechUvec = solVec->select( AMP::LinearAlgebra::VS_Stride( 0, 3 ) );
            auto mechVvec = solVec->select( AMP::LinearAlgebra::VS_Stride( 1, 3 ) );
            auto mechWvec = solVec->select( AMP::LinearAlgebra::VS_Stride( 2, 3 ) );
            AMP_ASSERT( solVec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

            AMP::pout << "Maximum U displacement: " << mechUvec->maxNorm() << std::endl;
            AMP::pout << "Maximum V displacement: " << mechVvec->maxNorm() << std::endl;
            AMP::pout << "Maximum W displacement: " << mechWvec->maxNorm() << std::endl;

            auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
            auto tmpParams =
                std::make_shared<AMP::Operator::MechanicsNonlinearFEOperatorParameters>( tmp_db );
            nonlinearMechanicsBVPoperator->getVolumeOperator()->reset( tmpParams );
            nonlinearSolver->setZeroInitialGuess( false );

            current_time = delta_time * ( (double) step + 2.0 );

            dirichletVectorCorrectionDatabase->putScalar( "value_6_0",
                                                          ( epsilon_dot * current_time ) );
            auto bndParams = std::make_shared<AMP::Operator::DirichletVectorCorrectionParameters>(
                dirichletVectorCorrectionDatabase );
            nonlinearMechanicsBVPoperator->getBoundaryOperator()->reset( bndParams );

            std::string number1 = std::to_string( step );
            std::string fname   = exeName + "_Stress_Strain_" + number1 + ".txt";

            AMP_ASSERT( solVec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
            std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
                nonlinearMechanicsBVPoperator->getVolumeOperator() )
                ->printStressAndStrain( solVec, fname );
        }

        AMP::pout << "epsilon = " << epsilon << std::endl;

        AMP::pout << solVec << std::endl;

        mechanicsNonlinearVolumeOperator->printStressAndStrain( solVec, output_file );
    }

    ut->passes( exeName );
}

int testElementLevel_PericElastoViscoPlasticity( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testElementLevel-PericElastoViscoPlasticity-1" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
