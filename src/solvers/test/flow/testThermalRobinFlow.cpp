#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/CoupledOperator.h"
#include "AMP/operators/CoupledOperatorParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"
#include "AMP/operators/boundary/libmesh/RobinMatrixCorrection.h"
#include "AMP/operators/boundary/libmesh/RobinVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/libmesh/MassLinearElement.h"
#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/operators/map/MapOperatorParameters.h"
#include "AMP/operators/map/libmesh/Map1Dto3D.h"
#include "AMP/operators/map/libmesh/Map3Dto1D.h"
#include "AMP/operators/subchannel/FlowFrapconJacobian.h"
#include "AMP/operators/subchannel/FlowFrapconOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/libmesh/Flow1DSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void flowTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    auto input_db          = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP::logAllNodes( log_file );
    AMP::AMP_MPI globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );

    // Create the Mesh
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode          = 1;
    int DOFsPerElement       = 8;
    int nodalGhostWidth      = 1;
    int gaussPointGhostWidth = 1;
    bool split               = true;
    auto nodalDofMap         = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, gaussPointGhostWidth, DOFsPerElement, split );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    auto intguess = input_db->getWithDefault<double>( "InitialGuess", 400 );

    // CREATE THE NONLINEAR THERMAL OPERATOR 1
    AMP_INSIST( input_db->keyExists( "NonlinearThermalOperator" ), "key missing!" );
    auto thermalNonlinearOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "NonlinearThermalOperator", input_db ) );

    // initialize the input variable
    auto thermalVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            thermalNonlinearOperator->getVolumeOperator() );

    auto globalSolVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, thermalVolumeOperator->getOutputVariable() );
    auto globalRhsVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, thermalVolumeOperator->getOutputVariable() );
    auto globalResVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, thermalVolumeOperator->getOutputVariable() );

    globalSolVec->setToScalar( intguess );

    // CREATE THE LINEAR THERMAL OPERATOR

    auto thermalLinearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearThermalOperator", input_db ) );

    // CREATE THE NEUTRONICS SOURCE
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db  = input_db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams = std::make_shared<AMP::Operator::OperatorParameters>( neutronicsOp_db );
    auto neutronicsOperator = std::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );

    auto SpecificPowerVar = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, SpecificPowerVar );

    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    // Integrate Nuclear Rhs over Desnity * GeomType::Cell
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "VolumeIntegralOperator", input_db ) );

    // Create the power (heat source) vector.
    auto PowerInWattsVar = sourceOperator->getOutputVariable();
    auto PowerInWattsVec = AMP::LinearAlgebra::createVector( nodalDofMap, PowerInWattsVar );
    PowerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, PowerInWattsVec );

    AMP_INSIST( input_db->keyExists( "NonlinearSolver" ), "Key ''NonlinearSolver'' is missing!" );

    ut->passes( "set up to the iterations passes." );


    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
    auto linearSolver_db    = nonlinearSolver_db->getDatabase( "LinearSolver" );

    // initialize the nonlinear solver
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );

    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = thermalNonlinearOperator;
    nonlinearSolverParams->d_pInitialGuess = globalSolVec;
    auto nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

    // initialize the column preconditioner which is a diagonal block preconditioner
    auto columnPreconditioner_db = linearSolver_db->getDatabase( "Preconditioner" );

    auto thermalPreconditioner_db =
        columnPreconditioner_db->getDatabase( "pelletThermalPreconditioner" );
    auto thermalPreconditionerParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( thermalPreconditioner_db );
    thermalPreconditionerParams->d_pOperator = thermalLinearOperator;
    auto thermalPreconditioner =
        std::make_shared<AMP::Solver::TrilinosMLSolver>( thermalPreconditionerParams );

    // register the preconditioner with the Jacobian free Krylov solver
    auto linearSolver = nonlinearSolver->getKrylovSolver();
    linearSolver->setNestedSolver( thermalPreconditioner );

    nonlinearSolver->setZeroInitialGuess( false );


    globalRhsVec->zero();

    globalRhsVec->copyVector( PowerInWattsVec );
    std::cout << "PowerInWattsVec norm  inside loop = " << globalRhsVec->L2Norm() << "\n";
    double expectedVal   = input_db->getScalar<double>( "RhsNorm" );
    double globalRhsNorm = static_cast<double>( globalRhsVec->L2Norm() );
    if ( !AMP::Utilities::approx_equal( expectedVal, globalRhsNorm, 1e-5 ) ) {
        ut->failure( "the PowerInWattsVec norm has changed." );
    }

    // robinBoundaryOp->reset(correctionParameters);

    thermalNonlinearOperator->modifyRHSvector( globalRhsVec );
    thermalNonlinearOperator->modifyInitialSolutionVector( globalSolVec );

    thermalNonlinearOperator->residual( globalRhsVec, globalSolVec, globalResVec );
    AMP::pout << "Initial Residual Norm for Step is: " << globalResVec->L2Norm() << std::endl;
    expectedVal          = input_db->getScalar<double>( "InitialResidual" );
    double globalResNorm = static_cast<double>( globalResVec->L2Norm() );
    if ( !AMP::Utilities::approx_equal( expectedVal, globalResNorm, 1e-5 ) ) {
        ut->failure( "the Initial Residual Norm has changed." );
    }

    std::cout << " RHS Vec L2 Norm " << globalRhsVec->L2Norm() << std::endl;
    nonlinearSolver->apply( globalRhsVec, globalSolVec );

    std::cout << "Final Solution Norm: " << globalSolVec->L2Norm() << std::endl;
    expectedVal        = input_db->getScalar<double>( "FinalSolution" );
    auto globalSolNorm = static_cast<double>( globalSolVec->L2Norm() );
    if ( !AMP::Utilities::approx_equal( expectedVal, globalSolNorm, 1e-5 ) ) {
        ut->failure( "the Final Solution Norm has changed." );
    }

    thermalNonlinearOperator->residual( globalRhsVec, globalSolVec, globalResVec );
    globalResNorm = static_cast<double>( globalResVec->L2Norm() );
    AMP::pout << "Final   Residual Norm for Step is: " << globalResNorm << std::endl;
    expectedVal = 1. - 10;
    if ( !AMP::Utilities::approx_equal( expectedVal, globalResNorm, 10.0 ) ) {
        ut->failure( "the Final Residual Norm has changed." );
    }

    if ( globalResNorm < 10e-6 ) {
        ut->passes( "Seggregated solve of Composite Operator using control loop of "
                    "Thermal+Robin->Map->Flow->Map ." );
    } else {
        ut->failure( "Seggregated solve of Composite Operator using control loop of "
                     "Thermal+Robin->Map->Flow->Map ." );
    }

    input_db.reset();

    ut->passes( exeName );
}

int testThermalRobinFlow( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    flowTest( &ut, "testThermalRobinFlow-2" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
