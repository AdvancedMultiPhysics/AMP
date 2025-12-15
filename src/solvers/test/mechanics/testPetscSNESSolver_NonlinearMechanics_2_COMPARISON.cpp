#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
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
#include "AMP/operators/mechanics/ThermalVonMisesMatModel.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <iostream>
#include <memory>
#include <string>


static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testPetscSNESSolver-NonlinearMechanics-2_COMPARISON-3" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create the DOFManagers
    auto NodalScalarDOF =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );
    auto NodalVectorDOF =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 3 );

    AMP_INSIST( input_db->keyExists( "NumberOfLoadingSteps" ),
                "Key ''NumberOfLoadingSteps'' is missing!" );
    int NumberOfLoadingSteps = input_db->getScalar<int>( "NumberOfLoadingSteps" );

    auto nonlinBvpOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "nonlinearMechanicsBVPOperator", input_db ) );
    auto nonlinearMechanicsVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinBvpOperator->getVolumeOperator() );
    auto elementPhysicsModel = nonlinearMechanicsVolumeOperator->getMaterialModel();

    auto multivariable = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(
        nonlinBvpOperator->getInputVariable() );
    auto displacementVariable =
        multivariable->getVariable( AMP::Operator::Mechanics::DISPLACEMENT );
    auto temperatureVariable = multivariable->getVariable( AMP::Operator::Mechanics::TEMPERATURE );

    auto initTempVec  = AMP::LinearAlgebra::createVector( NodalScalarDOF, temperatureVariable );
    auto finalTempVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, temperatureVariable );

    double Temp_0 = 400.0;
    double Temp_1 = 2000.0;
    initTempVec->setToScalar( Temp_0 );
    initTempVec->abs( *initTempVec );
    auto initTempConst = input_db->getWithDefault<double>( "INIT_TEMP_CONST", 1.0 );
    initTempVec->scale( initTempConst );
    initTempVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    bool setFinalTempEqualsInitialTemp =
        input_db->getWithDefault<bool>( "SET_FINAL_TEMP_EQUALS_INIT_TEMP", false );

    if ( setFinalTempEqualsInitialTemp ) {
        finalTempVec->copyVector( initTempVec );
    } else {
        double Temp_n = Temp_0 + ( ( Temp_1 - Temp_0 ) / ( (double) ( NumberOfLoadingSteps ) ) );
        AMP::pout << "Temp_n = " << Temp_n << std::endl;
        finalTempVec->setToScalar( Temp_n );
        auto finalTempConst = input_db->getWithDefault<double>( "FINAL_TEMP_CONST", 1.0 );
        finalTempVec->scale( finalTempConst );
    }
    finalTempVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
        nonlinBvpOperator->getVolumeOperator() )
        ->setReferenceTemperature( initTempVec );
    std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
        nonlinBvpOperator->getVolumeOperator() )
        ->setVector( AMP::Operator::Mechanics::TEMPERATURE, finalTempVec );


    // For RHS (Point Forces)
    auto dirichletLoadVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    dirichletLoadVecOp->setVariable( displacementVariable );

    // For Initial-Guess
    auto dirichletDispInVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Displacement_Boundary", input_db ) );
    dirichletDispInVecOp->setVariable( displacementVariable );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto mechNlSolVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
    auto mechNlRhsVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
    auto mechNlResVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
    auto mechNlScaledRhsVec =
        AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );

    // Initial guess for NL solver must satisfy the displacement boundary conditions
    mechNlSolVec->setToScalar( 0.0 );
    dirichletDispInVecOp->apply( nullVec, mechNlSolVec );
    mechNlSolVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    mechNlRhsVec->setToScalar( 0.0 );
    dirichletLoadVecOp->apply( nullVec, mechNlRhsVec );

    std::cout << "Initial Solution Norm: " << mechNlSolVec->L2Norm() << std::endl;

    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );

    auto linearSolver_db = nonlinearSolver_db->getDatabase( "LinearSolver" );

    auto linBvpOperator = std::make_shared<AMP::Operator::LinearBVPOperator>(
        nonlinBvpOperator->getParameters( "Jacobian", nullptr ) );

    // ---- first initialize the preconditioner
    auto pcSolver_db    = linearSolver_db->getDatabase( "Preconditioner" );
    auto pcSolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( pcSolver_db );
    pcSolverParams->d_pOperator = linBvpOperator;
    auto pcSolver               = std::make_shared<AMP::Solver::TrilinosMLSolver>( pcSolverParams );

    // HACK to prevent a double delete on Petsc Vec
    std::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver;

    // initialize the linear solver
    auto linearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
    linearSolverParams->d_pOperator     = linBvpOperator;
    linearSolverParams->d_comm          = globalComm;
    linearSolverParams->d_pNestedSolver = pcSolver;
    auto linearSolver = std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nonlinBvpOperator;
    nonlinearSolverParams->d_pNestedSolver = linearSolver;
    nonlinearSolverParams->d_pInitialGuess = mechNlSolVec;
    nonlinearSolver.reset( new AMP::Solver::PetscSNESSolver( nonlinearSolverParams ) );
    nonlinearSolver->setZeroInitialGuess( false );

    for ( int step = 0; step < NumberOfLoadingSteps; step++ ) {
        AMP::pout << "########################################" << std::endl;
        AMP::pout << "The current loading step is " << ( step + 1 ) << std::endl;

        if ( step > 0 ) {
            double Temp_n =
                Temp_0 + ( ( (double) ( step + 1 ) ) *
                           ( ( Temp_1 - Temp_0 ) / ( (double) ( NumberOfLoadingSteps ) ) ) );
            AMP::pout << "Temp_n = " << Temp_n << std::endl;
            finalTempVec->setToScalar( Temp_n );
            ( std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
                  nonlinBvpOperator->getVolumeOperator() ) )
                ->setVector( AMP::Operator::Mechanics::TEMPERATURE, finalTempVec );
        }

        double scaleValue = ( (double) step + 1.0 ) / NumberOfLoadingSteps;
        mechNlScaledRhsVec->scale( scaleValue, *mechNlRhsVec );
        mechNlScaledRhsVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        AMP::pout << "L2 Norm at loading step " << ( step + 1 ) << " is "
                  << mechNlScaledRhsVec->L2Norm() << std::endl;

        nonlinBvpOperator->residual( mechNlScaledRhsVec, mechNlSolVec, mechNlResVec );
        double initialResidualNorm = static_cast<double>( mechNlResVec->L2Norm() );
        AMP::pout << "Initial Residual Norm for loading step " << ( step + 1 ) << " is "
                  << initialResidualNorm << std::endl;

        if ( initialResidualNorm < 1.0e-2 ) {
            ut->passes( "Nonlinear solve for current loading step" );
        } else {
            AMP::pout << "Starting Nonlinear Solve..." << std::endl;
            nonlinearSolver->apply( mechNlScaledRhsVec, mechNlSolVec );

            nonlinBvpOperator->residual( mechNlScaledRhsVec, mechNlSolVec, mechNlResVec );
            double finalResidualNorm = static_cast<double>( mechNlResVec->L2Norm() );
            AMP::pout << "Final Residual Norm for loading step " << ( step + 1 ) << " is "
                      << finalResidualNorm << std::endl;
            AMP::pout << "Maxx value in the final sol for step " << ( step + 1 ) << ": "
                      << mechNlSolVec->max() << std::endl;

            if ( finalResidualNorm > ( 1.0e-8 * initialResidualNorm ) ) {
                ut->failure( "Nonlinear solve for current loading step" );
            } else {
                ut->passes( "Nonlinear solve for current loading step" );
            }
        }

        auto mechUvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride( 0, 3 ) );
        auto mechVvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride( 1, 3 ) );
        auto mechWvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride( 2, 3 ) );

        AMP::pout << "Maximum U displacement: " << mechUvec->maxNorm() << std::endl;
        AMP::pout << "Maximum V displacement: " << mechVvec->maxNorm() << std::endl;
        AMP::pout << "Maximum W displacement: " << mechWvec->maxNorm() << std::endl;

        auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
        auto tmpParams =
            std::make_shared<AMP::Operator::MechanicsNonlinearFEOperatorParameters>( tmp_db );
        ( nonlinBvpOperator->getVolumeOperator() )->reset( tmpParams );
        nonlinearSolver->setZeroInitialGuess( false );
    }

    double finalSolNorm = static_cast<double>( mechNlSolVec->L2Norm() );
    AMP::pout << "Final Solution Norm: " << finalSolNorm << std::endl;
    AMP::pout << "Maxx value in the final sol: " << mechNlSolVec->max() << std::endl;

    ut->passes( exeName );
}

int testPetscSNESSolver_NonlinearMechanics_2_COMPARISON( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
