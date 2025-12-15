#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/PressureBoundaryOperator.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <fstream>
#include <iostream>
#include <string>


static void linearElasticTest( AMP::UnitTest *ut, const std::string &exeName, int )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName + ".txt";

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    auto bvpOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MechanicsBVPOperator", input_db ) );

    auto dispVar = bvpOperator->getOutputVariable();

    auto dirichletVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    // This has an in-place apply. So, it has an empty input variable and
    // the output variable is the same as what it is operating on.
    dirichletVecOp->setVariable( dispVar );

    // Pressure RHS
    auto pressureLoadVecOp = std::dynamic_pointer_cast<AMP::Operator::PressureBoundaryOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Pressure_Boundary", input_db ) );
    // This has an in-place apply. So, it has an empty input variable and
    // the output variable is the same as what it is operating on.

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto mechSolVec      = AMP::LinearAlgebra::createVector( dofMap, dispVar, true );
    auto mechRhsVec      = mechSolVec->clone();
    auto mechResVec      = mechSolVec->clone();
    auto mechPressureVec = mechSolVec->clone();

    mechSolVec->setToScalar( 0.0 );
    mechRhsVec->setToScalar( 0.0 );
    mechResVec->setToScalar( 0.0 );
    mechPressureVec->setToScalar( 0.0 );

    dirichletVecOp->apply( nullVec, mechRhsVec );

    AMP::pout << "RHS Norm after Dirichlet Apply: " << mechRhsVec->L2Norm() << std::endl;

    AMP::pout << "Pressure Norm before Apply: " << mechPressureVec->L2Norm() << std::endl;

    // Applying the pressure load
    pressureLoadVecOp->addRHScorrection( mechPressureVec );
    AMP::pout << "Pressure Norm after Apply: " << mechPressureVec->L2Norm() << std::endl;

    mechRhsVec->add( *mechRhsVec, *mechPressureVec );
    mechRhsVec->makeConsistent();

    AMP::pout << "Total RHS Norm: " << mechRhsVec->L2Norm() << std::endl;
    AMP::pout << "Initial Solution Norm: " << mechSolVec->L2Norm() << std::endl;

    bvpOperator->residual( mechRhsVec, mechSolVec, mechResVec );

    double initResidualNorm = static_cast<double>( mechResVec->L2Norm() );
    AMP::pout << "Initial Residual Norm: " << initResidualNorm << std::endl;

    auto linearSolver_db = input_db->getDatabase( "LinearSolver" );

    // ---- first initialize the preconditioner
    auto pcSolver_db    = linearSolver_db->getDatabase( "Preconditioner" );
    auto pcSolverParams = std::make_shared<AMP::Solver::TrilinosMLSolverParameters>( pcSolver_db );
    pcSolverParams->d_pOperator = bvpOperator;
    auto pcSolver               = std::make_shared<AMP::Solver::TrilinosMLSolver>( pcSolverParams );

    // initialize the linear solver
    auto linearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
    linearSolverParams->d_pOperator     = bvpOperator;
    linearSolverParams->d_comm          = globalComm;
    linearSolverParams->d_pNestedSolver = pcSolver;
    auto linearSolver = std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );

    linearSolver->setZeroInitialGuess( false );

    linearSolver->apply( mechRhsVec, mechSolVec );

    AMP::pout << "Final Solution Norm: " << mechSolVec->L2Norm() << std::endl;

    std::string fname = exeName + "_StressAndStrain.txt";

    std::dynamic_pointer_cast<AMP::Operator::MechanicsLinearFEOperator>(
        bvpOperator->getVolumeOperator() )
        ->printStressAndStrain( mechSolVec, fname );

    auto mechUvec = mechSolVec->select( AMP::LinearAlgebra::VS_Stride( 0, 3 ) );
    auto mechVvec = mechSolVec->select( AMP::LinearAlgebra::VS_Stride( 1, 3 ) );
    auto mechWvec = mechSolVec->select( AMP::LinearAlgebra::VS_Stride( 2, 3 ) );

    AMP::pout << "Maximum U displacement: " << mechUvec->maxNorm() << std::endl;
    AMP::pout << "Maximum V displacement: " << mechVvec->maxNorm() << std::endl;
    AMP::pout << "Maximum W displacement: " << mechWvec->maxNorm() << std::endl;

    bvpOperator->residual( mechRhsVec, mechSolVec, mechResVec );

    double finalResidualNorm = static_cast<double>( mechResVec->L2Norm() );

    AMP::pout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    if ( finalResidualNorm > ( 1e-10 * initResidualNorm ) ) {
        ut->failure( exeName );
    } else {
        ut->passes( exeName );
    }

    double epsilon =
        1.0e-13 * static_cast<double>( bvpOperator->getMatrix()->extractDiagonal()->L1Norm() );
    AMP::pout << "epsilon = " << epsilon << std::endl;

    mesh->displaceMesh( mechSolVec );
}

int testLinearMechanics_PressureBoundary( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;

    if ( argc == 1 ) {
        exeNames.emplace_back( "testLinearMechanics-PressureBoundary-1" );
    } else {
        for ( int i = 1; i < argc; ++i ) {
            auto inpName =
                AMP::Utilities::stringf( "testLinearMechanics-PressureBoundary-%s", argv[i] );
            exeNames.emplace_back( inpName );
        } // end for i
    }

    for ( size_t i = 0; i < exeNames.size(); ++i ) {
        linearElasticTest( &ut, exeNames[i], i );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
