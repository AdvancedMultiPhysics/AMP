#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
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
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
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

static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file  = "input_" + exeName;
    std::string output_file = "output_" + exeName + ".txt";
    std::string log_file    = "log_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    AMP_INSIST( input_db->keyExists( "NumberOfLoadingSteps" ),
                "Key ''NumberOfLoadingSteps'' is missing!" );
    int NumberOfLoadingSteps = input_db->getScalar<int>( "NumberOfLoadingSteps" );

    bool ExtractData = input_db->getWithDefault<bool>( "ExtractStressStrainData", false );
    FILE *fout123;
    std::string ss_file = exeName + "_UniaxialTmperatureDisplacement.txt";
    fout123             = fopen( ss_file.c_str(), "w" );

    // Create a nonlinear BVP operator for mechanics
    AMP_INSIST( input_db->keyExists( "NonlinearMechanicsOperator" ), "key missing!" );
    auto nonlinearMechanicsBVPoperator =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "NonlinearMechanicsOperator", input_db ) );
    auto nonlinearMechanicsVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinearMechanicsBVPoperator->getVolumeOperator() );
    auto mechanicsMaterialModel = nonlinearMechanicsVolumeOperator->getMaterialModel();

    // Create the variables
    auto mechanicsNonlinearVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinearMechanicsBVPoperator->getVolumeOperator() );

    auto multivariable = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(
        mechanicsNonlinearVolumeOperator->getInputVariable() );
    auto dispVar = multivariable->getVariable( AMP::Operator::Mechanics::DISPLACEMENT );
    auto tempVar = multivariable->getVariable( AMP::Operator::Mechanics::TEMPERATURE );
    auto burnVar = multivariable->getVariable( AMP::Operator::Mechanics::BURNUP );

    // For RHS (Point Forces)
    auto dirichletLoadVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    dirichletLoadVecOp->setVariable( dispVar );

    // Create the DOFManagers
    auto NodalVectorDOF =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 3 );
    auto NodalScalarDOF =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto solVec       = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
    auto rhsVec       = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
    auto resVec       = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
    auto scaledRhsVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
    auto tempVecRef   = AMP::LinearAlgebra::createVector( NodalScalarDOF, tempVar );
    auto tempVec      = AMP::LinearAlgebra::createVector( NodalScalarDOF, tempVar );
    auto burnVec      = AMP::LinearAlgebra::createVector( NodalScalarDOF, burnVar );

    // Initial guess
    solVec->zero();
    nonlinearMechanicsBVPoperator->modifyInitialSolutionVector( solVec );

    // RHS
    rhsVec->zero();
    dirichletLoadVecOp->apply( nullVec, rhsVec );
    nonlinearMechanicsBVPoperator->modifyRHSvector( rhsVec );

    // Adding the Temperature and Burnup
    tempVecRef->setToScalar( 301.0 );
    tempVec->setToScalar( 301.0 );
    burnVec->setToScalar( 10.0 );

    mechanicsNonlinearVolumeOperator->setReferenceTemperature( tempVecRef );
    mechanicsNonlinearVolumeOperator->setVector( AMP::Operator::Mechanics::TEMPERATURE, tempVec );
    mechanicsNonlinearVolumeOperator->setVector( AMP::Operator::Mechanics::BURNUP, burnVec );

    auto mechanicsNonlinearMaterialModel =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsMaterialModel>(
            mechanicsNonlinearVolumeOperator->getMaterialModel() );

    // Create the solver
    auto nonlinearSolver = AMP::Solver::Test::buildSolver(
        "NonlinearSolver", input_db, globalComm, solVec, nonlinearMechanicsBVPoperator );

    nonlinearSolver->setZeroInitialGuess( false );

    double scaleValue = 1.0;
    scaledRhsVec->scale( scaleValue, *rhsVec );
    AMP::pout << "L2 Norm of RHS at loading step 1 is " << scaledRhsVec->L2Norm() << std::endl;

    double currTime = 1000.0;
    mechanicsNonlinearMaterialModel->updateTime( currTime );

    if ( ExtractData ) {
        fprintf( fout123, "%f %f %f %f\n", 301.0, 0.0, 0.0, 0.0 );
    }

    for ( int step = 0; step < NumberOfLoadingSteps; step++ ) {
        currTime = ( (double) ( step + 2 ) ) * 1000.0;

        AMP::pout << "########################################" << std::endl;
        AMP::pout << "The current loading step is " << ( step + 1 ) << std::endl;

        double finalTemperature = 500.0;
        tempVec->setToScalar( finalTemperature );

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

        mechanicsNonlinearMaterialModel->updateTime( currTime );

        nonlinearSolver->setZeroInitialGuess( false );

        std::string number1 = std::to_string( step );
        std::string fname   = exeName + "_Stress_Strain_" + number1 + ".txt";

        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinearMechanicsBVPoperator->getVolumeOperator() )
            ->printStressAndStrain( solVec, fname );

        // double prev_stress, prev_strain, slope;
        if ( ExtractData ) {
            fprintf( fout123, "%f %f %f %f\n", finalTemperature, finalMaxU, finalMaxV, finalMaxW );
        }
    }

    mechanicsNonlinearVolumeOperator->printStressAndStrain( solVec, output_file );

    ut->passes( exeName );
    fclose( fout123 );
}

int testFixedBeam_CreepLoading( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testFixedBeam-CreepLoading-1" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
