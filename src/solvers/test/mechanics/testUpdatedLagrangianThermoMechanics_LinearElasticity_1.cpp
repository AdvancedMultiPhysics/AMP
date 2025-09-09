#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/operators/mechanics/ThermalStrainMaterialModel.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <iostream>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "log_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    AMP_INSIST( input_db->keyExists( "NumberOfLoadingSteps" ),
                "Key ''NumberOfLoadingSteps'' is missing!" );
    int NumberOfLoadingSteps = input_db->getScalar<int>( "NumberOfLoadingSteps" );

    // Create a nonlinear BVP operator for mechanics
    AMP_INSIST( input_db->keyExists( "NonlinearMechanicsOperator" ), "key missing!" );
    auto nonlinearMechanicsBVPoperator =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "NonlinearMechanicsOperator", input_db ) );

    // Create the variables
    auto mechanicsNonlinearVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinearMechanicsBVPoperator->getVolumeOperator() );
    auto dispVar   = mechanicsNonlinearVolumeOperator->getOutputVariable();
    auto inputVars = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(
        mechanicsNonlinearVolumeOperator->getInputVariable() );
    auto tempVar = inputVars->getVariable( AMP::Operator::Mechanics::TEMPERATURE );

    // For RHS (Point Forces)
    auto dirichletLoadVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    dirichletLoadVecOp->setVariable( dispVar );

    auto dispDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    auto tempDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto solVec       = AMP::LinearAlgebra::createVector( dispDofMap, dispVar, true );
    auto rhsVec       = solVec->clone();
    auto resVec       = solVec->clone();
    auto scaledRhsVec = solVec->clone();
    auto refTempVec   = AMP::LinearAlgebra::createVector( tempDofMap, tempVar, true );
    auto curTempVec   = refTempVec->clone();

    // Initial guess
    solVec->zero();
    nonlinearMechanicsBVPoperator->modifyInitialSolutionVector( solVec );

    // RHS
    rhsVec->zero();
    dirichletLoadVecOp->apply( nullVec, rhsVec );
    nonlinearMechanicsBVPoperator->modifyRHSvector( rhsVec );

    // Set the temperatures
    refTempVec->setToScalar( 300.0 );
    curTempVec->setToScalar( 700.0 );

    // Set the reference temperature
    std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
        nonlinearMechanicsBVPoperator->getVolumeOperator() )
        ->setReferenceTemperature( refTempVec );
    std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
        nonlinearMechanicsBVPoperator->getVolumeOperator() )
        ->setVector( AMP::Operator::Mechanics::TEMPERATURE, curTempVec );

    // We need to reset the linear operator before the solve since TrilinosML does
    // the factorization of the matrix during construction and so the matrix must
    // be correct before constructing the TrilinosML object.
    nonlinearMechanicsBVPoperator->apply( solVec, resVec );

    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
    // HACK to prevent a double delete on Petsc Vec
    std::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver;

    // initialize the nonlinear solver
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nonlinearMechanicsBVPoperator;
    nonlinearSolverParams->d_pInitialGuess = solVec;
    nonlinearSolver.reset( new AMP::Solver::PetscSNESSolver( nonlinearSolverParams ) );

    nonlinearSolver->setZeroInitialGuess( false );

    int loadingSubSteps = 1;
    double scaleValue;

    for ( int step = 0; step < NumberOfLoadingSteps; step++ ) {
        AMP::pout << "########################################" << std::endl;
        AMP::pout << "The current loading step is " << ( step + 1 ) << std::endl;

        if ( step > 3 ) {
            loadingSubSteps = 1;
        }
        for ( int subStep = 0; subStep < loadingSubSteps; subStep++ ) {
            if ( step <= 3 ) {
                scaleValue = ( (double) step + 1.0 ) / NumberOfLoadingSteps;
            } else {
                scaleValue =
                    ( ( (double) step ) / NumberOfLoadingSteps ) +
                    ( ( (double) subStep + 1.0 ) / ( NumberOfLoadingSteps * loadingSubSteps ) );
            }

            AMP::pout << "########################################" << std::endl;
            AMP::pout << "The current loading sub step is " << ( subStep + 1 )
                      << " and scaleValue = " << scaleValue << std::endl;

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

            AMP::pout << "Final Solution Norm: " << solVec->L2Norm() << std::endl;

            auto mechUvec = solVec->select( AMP::LinearAlgebra::VS_Stride( 0, 3 ) );
            auto mechVvec = solVec->select( AMP::LinearAlgebra::VS_Stride( 1, 3 ) );
            auto mechWvec = solVec->select( AMP::LinearAlgebra::VS_Stride( 2, 3 ) );

            double finalMaxU = static_cast<double>( mechUvec->maxNorm() );
            double finalMaxV = static_cast<double>( mechVvec->maxNorm() );
            double finalMaxW = static_cast<double>( mechWvec->maxNorm() );

            AMP::pout << "Maximum U displacement: " << finalMaxU << std::endl;
            AMP::pout << "Maximum V displacement: " << finalMaxV << std::endl;
            AMP::pout << "Maximum W displacement: " << finalMaxW << std::endl;

            auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
            auto tmpParams =
                std::make_shared<AMP::Operator::MechanicsNonlinearFEOperatorParameters>( tmp_db );
            nonlinearMechanicsBVPoperator->getVolumeOperator()->reset( tmpParams );
            nonlinearSolver->setZeroInitialGuess( false );
        }
    }

    ut->passes( exeName );
}

int testUpdatedLagrangianThermoMechanics_LinearElasticity_1( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testUpdatedLagrangianThermoMechanics-LinearElasticity-1" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
