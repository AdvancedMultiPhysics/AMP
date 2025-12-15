#include "AMP/IO/PIO.h"
#include "AMP/IO/WriteSolutionToFile.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <fstream>
#include <iostream>
#include <string>


static void linearElasticTest( AMP::UnitTest *ut, int reduced, std::string mesh_file )
{
    std::string exeName;
    std::string input_file;

    if ( reduced ) {
        exeName    = "testLinearElasticity-reduced-" + mesh_file;
        input_file = "input_testLinearElasticity-reduced-mesh2elem";
    } else {
        exeName    = "testLinearElasticity-normal-" + mesh_file;
        input_file = "input_testLinearElasticity-normal-mesh2elem";
    }

    std::string log_file = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );

    [[maybe_unused]] auto libmeshInit =
        std::make_shared<AMP::Mesh::initializeLibMesh>( globalComm );

    if ( globalComm.getSize() == 1 ) {

        auto input_db = AMP::Database::parseInputFile( input_file );
        input_db->print( AMP::plog );

        auto mesh = AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD );

        auto bvpOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "MechanicsBVPOperator", input_db ) );

        auto dirichletVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
            AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );

        auto var = bvpOperator->getOutputVariable();

        // This has an in-place apply. So, it has an empty input variable and
        // the output variable is the same as what it is operating on.
        dirichletVecOp->setVariable( var );

        auto dofMap = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

        AMP::LinearAlgebra::Vector::shared_ptr nullVec;
        auto mechSolVec = AMP::LinearAlgebra::createVector( dofMap, var, true );
        auto mechRhsVec = mechSolVec->clone();
        auto mechResVec = mechSolVec->clone();

        mechSolVec->setToScalar( 0.5 );
        mechRhsVec->setToScalar( 0.0 );
        mechResVec->setToScalar( 0.0 );

        dirichletVecOp->apply( nullVec, mechRhsVec );

        double rhsNorm = static_cast<double>( mechRhsVec->L2Norm() );

        AMP::pout << "RHS Norm: " << rhsNorm << std::endl;

        double initSolNorm = static_cast<double>( mechSolVec->L2Norm() );

        AMP::pout << "Initial Solution Norm: " << initSolNorm << std::endl;

        bvpOperator->residual( mechRhsVec, mechSolVec, mechResVec );

        double initResidualNorm = static_cast<double>( mechResVec->L2Norm() );

        AMP::pout << "Initial Residual Norm: " << initResidualNorm << std::endl;

        auto mlSolver_db = input_db->getDatabase( "LinearSolver" );

        auto mlSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( mlSolver_db );

        mlSolverParams->d_pOperator = bvpOperator;

        // create the ML solver interface
        auto mlSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams );

        mlSolver->setZeroInitialGuess( false );

        mlSolver->apply( mechRhsVec, mechSolVec );

        AMP::pout << "Final Solution Norm: " << mechSolVec->L2Norm() << std::endl;

        bvpOperator->residual( mechRhsVec, mechSolVec, mechResVec );

        double finalResidualNorm = static_cast<double>( mechResVec->L2Norm() );

        AMP::pout << "Final Residual Norm: " << finalResidualNorm << std::endl;

        printSolution( mesh, mechSolVec, exeName );

        if ( finalResidualNorm > ( 1e-10 * initResidualNorm ) ) {
            ut->failure( exeName );
        } else {
            ut->passes( exeName );
        }
    } else {
        AMP::pout << "WARNING: This is a single processor test!" << std::endl;
        ut->passes( exeName );
    }
}

int testLinearElasticity_2_element( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    auto libmeshInit = std::make_shared<AMP::Mesh::initializeLibMesh>( AMP_COMM_WORLD );

    AMP::UnitTest ut;

    std::vector<std::string> mesh_files;

    if ( argc == 1 ) {
        for ( int i = 1; i <= 6; i++ ) {
            auto name = AMP::Utilities::stringf( "mesh2elem-%d", i );
            mesh_files.emplace_back( name );
        } // end for i
    } else {
        for ( int i = 1; i < argc; i++ ) {
            auto name = AMP::Utilities::stringf( "mesh2elem-%d", atoi( argv[i] ) );
            mesh_files.emplace_back( name );
        } // end for i
    }

    for ( auto &mesh_file : mesh_files ) {
        for ( int reduced = 0; reduced < 2; reduced++ ) {
            try {
                linearElasticTest( &ut, reduced, mesh_file );
            } catch ( std::exception &err ) {
                AMP::pout << "ERROR: " << err.what() << std::endl;
                ut.failure( "ERROR" );
            } catch ( ... ) {
                AMP::pout << "ERROR: "
                          << "An unknown exception was thrown." << std::endl;
                ut.failure( "ERROR" );
            }
        } // end for reduced
    }     // end for i

    ut.report();
    int num_failed = ut.NumFailGlobal();

    libmeshInit.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
