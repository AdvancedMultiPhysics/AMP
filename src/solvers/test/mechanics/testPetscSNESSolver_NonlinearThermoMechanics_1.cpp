#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
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
    AMP::pout << "Running test for input " << input_file << std::endl;

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    // create a nonlinear BVP operator for nonlinear mechanics
    AMP_INSIST( input_db->keyExists( "testNonlinearMechanicsOperator" ), "key missing!" );

    auto nonlinearMechanicsOperator =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "testNonlinearMechanicsOperator", input_db ) );
    auto nonlinearMechanicsVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinearMechanicsOperator->getVolumeOperator() );
    auto mechanicsMaterialModel = nonlinearMechanicsVolumeOperator->getMaterialModel();

    // create a nonlinear BVP operator for nonlinear thermal diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearThermalOperator" ), "key missing!" );

    auto nonlinearThermalOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testNonlinearThermalOperator", input_db ) );
    auto nonlinearThermalVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator->getVolumeOperator() );
    auto thermalTransportModel = nonlinearThermalVolumeOperator->getTransportModel();

    // create a column operator object for nonlinear thermomechanics
    auto nonlinearThermoMechanicsOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    nonlinearThermoMechanicsOperator->append( nonlinearMechanicsOperator );
    nonlinearThermoMechanicsOperator->append( nonlinearThermalOperator );

    // initialize the input multi-variable
    auto mechanicsVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinearMechanicsOperator->getVolumeOperator() );

    // initialize the output multi-variable
    auto displacementVar = nonlinearMechanicsOperator->getOutputVariable();
    auto temperatureVar  = nonlinearThermalOperator->getOutputVariable();

    auto vectorDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    auto scalarDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );

    // create solution, rhs, and residual vectors
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto displacementVec = AMP::LinearAlgebra::createVector( vectorDofMap, displacementVar, true );
    auto temperatureVec  = AMP::LinearAlgebra::createVector( scalarDofMap, temperatureVar, true );
    auto solVec          = AMP::LinearAlgebra::MultiVector::create( "multiVector", globalComm );
    auto multiVec        = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( solVec );
    multiVec->addVector( displacementVec );
    multiVec->addVector( temperatureVec );

    auto rhsVec = solVec->clone();
    auto resVec = solVec->clone();

    auto referenceTemperatureVec = temperatureVec->clone();
    referenceTemperatureVec->setToScalar( 300.0 );
    mechanicsVolumeOperator->setReferenceTemperature( referenceTemperatureVec );

    // Initial-Guess for mechanics
    auto dirichletDispInVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MechanicsInitialGuess", input_db ) );
    dirichletDispInVecOp->setVariable( displacementVar );

    // Initial-Guess for thermal
    auto dirichletThermalInVecOp =
        std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "ThermalInitialGuess", input_db ) );
    dirichletThermalInVecOp->setVariable( temperatureVar );

    // Random initial guess
    solVec->setToScalar( 0.0 );
    const double initialTemperature = 301.0;
    temperatureVec->addScalar( *temperatureVec, initialTemperature );

    // Initial guess for mechanics must satisfy the displacement boundary conditions
    dirichletDispInVecOp->apply( nullVec, solVec );
    // Initial guess for thermal must satisfy the thermal Dirichlet boundary conditions
    dirichletThermalInVecOp->apply( nullVec, solVec );

    // We need to reset the linear operator before the solve since TrilinosML does
    // the factorization of the matrix during construction and so the matrix must
    // be correct before constructing the TrilinosML object.
    // The thermal operator does not expect an apply to be called before calling
    // getJacobianParams and so it need not be called. So, any of the following
    // apply calls will work:
    nonlinearThermoMechanicsOperator->apply( solVec, resVec );

    resVec->setToScalar( 0.0 );
    rhsVec->setToScalar( 0.0 );

    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );

    // initialize the nonlinear solver
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );

    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nonlinearThermoMechanicsOperator;
    nonlinearSolverParams->d_pInitialGuess = solVec;
    auto nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

    nonlinearThermoMechanicsOperator->residual( rhsVec, solVec, resVec );
    double initialResidualNorm = static_cast<double>( resVec->L2Norm() );

    AMP::pout << "Initial Residual Norm: " << initialResidualNorm << std::endl;

    nonlinearSolver->setZeroInitialGuess( false );

    nonlinearSolver->apply( rhsVec, solVec );

    nonlinearThermoMechanicsOperator->residual( rhsVec, solVec, resVec );

    double finalResidualNorm = static_cast<double>( resVec->L2Norm() );
    std::cout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    auto mechUvec = displacementVec->select( AMP::LinearAlgebra::VS_Stride( 0, 3 ) );
    auto mechVvec = displacementVec->select( AMP::LinearAlgebra::VS_Stride( 1, 3 ) );
    auto mechWvec = displacementVec->select( AMP::LinearAlgebra::VS_Stride( 2, 3 ) );

    double finalMaxU = static_cast<double>( mechUvec->maxNorm() );
    double finalMaxV = static_cast<double>( mechVvec->maxNorm() );
    double finalMaxW = static_cast<double>( mechWvec->maxNorm() );

    AMP::pout << "Maximum U displacement: " << finalMaxU << std::endl;
    AMP::pout << "Maximum V displacement: " << finalMaxV << std::endl;
    AMP::pout << "Maximum W displacement: " << finalMaxW << std::endl;

    if ( finalResidualNorm > initialResidualNorm * 1.0e-10 + 1.0e-05 ) {
        ut->failure( "Error" );
    } else {
        ut->passes( "PetscSNES Solver successfully solves a nonlinear thermo-mechanics equation "
                    "with JFNK, FGMRES for "
                    "Krylov, block diagonal preconditioning with ML solvers" );
    }
    ut->passes( exeName );
}


int testPetscSNESSolver_NonlinearThermoMechanics_1( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testPetscSNESSolver-NonlinearThermoMechanics-1" );
    exeNames.emplace_back( "testPetscSNESSolver-NonlinearThermoMechanics-1a" );
    exeNames.emplace_back( "testPetscSNESSolver-NonlinearThermoMechanics-1b" );
    exeNames.emplace_back( "testPetscSNESSolver-NonlinearThermoMechanics-1c" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
