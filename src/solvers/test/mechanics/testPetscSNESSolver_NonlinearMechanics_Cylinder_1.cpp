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
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/PressureBoundaryOperator.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/operators/trilinos/ml/TrilinosMatrixShellOperator.h"
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


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
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
    auto mechanicsMaterialModel = nonlinearMechanicsVolumeOperator->getMaterialModel();

    auto linBvpOperator = std::make_shared<AMP::Operator::LinearBVPOperator>(
        nonlinBvpOperator->getParameters( "Jacobian", nullptr ) );

    auto multivariable = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(
        nonlinBvpOperator->getVolumeOperator()->getInputVariable() );
    auto displacementVariable =
        multivariable->getVariable( AMP::Operator::Mechanics::DISPLACEMENT );
    auto residualVariable = nonlinBvpOperator->getOutputVariable();

    // For RHS (Point Forces)
    auto dirichletLoadVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    dirichletLoadVecOp->setVariable( residualVariable );

    // Pressure RHS
    auto pressureLoadVecOp = std::dynamic_pointer_cast<AMP::Operator::PressureBoundaryOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Pressure_Boundary", input_db ) );
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    auto mechNlSolVec    = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
    auto mechNlRhsVec    = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
    auto mechPressureVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
    auto mechNlResVec    = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
    auto mechNlScaledRhsVec =
        AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );

    // Initial guess for NL solver must satisfy the displacement boundary conditions
    mechNlSolVec->setToScalar( 0.0 );
    mechPressureVec->setToScalar( 0.0 );

    nonlinBvpOperator->apply( mechNlSolVec, mechNlResVec );
    linBvpOperator->reset( nonlinBvpOperator->getParameters( "Jacobian", mechNlSolVec ) );

    // Point forces
    mechNlRhsVec->setToScalar( 0.0 );
    dirichletLoadVecOp->apply( nullVec, mechNlRhsVec );

    // Applying the pressure load
    pressureLoadVecOp->addRHScorrection( mechPressureVec );
    mechNlRhsVec->add( *mechNlRhsVec, *mechPressureVec );

    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );

    // initialize the nonlinear solver
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nonlinBvpOperator;
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

        if ( finalResidualNorm > ( 1.0e-10 * initialResidualNorm ) ) {
            ut->failure( "Nonlinear solve for current loading step" );
        } else {
            ut->passes( "Nonlinear solve for current loading step" );
        }

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
        ( nonlinBvpOperator->getVolumeOperator() )->reset( tmpParams );
        nonlinearSolver->setZeroInitialGuess( false );
    }

    double finalSolNorm = static_cast<double>( mechNlSolVec->L2Norm() );
    AMP::pout << "Final Solution Norm: " << finalSolNorm << std::endl;

    ut->passes( exeName );
}

int testPetscSNESSolver_NonlinearMechanics_Cylinder_1( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testPetscSNESSolver-NonlinearMechanics-Cylinder-1" );
    // exeNames.push_back("testPetscSNESSolver-NonlinearMechanics-PlateWithHole-1");

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
