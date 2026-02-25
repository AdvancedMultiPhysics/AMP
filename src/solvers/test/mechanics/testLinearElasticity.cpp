#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <fstream>
#include <iostream>
#include <string>


static void linearElasticTest( AMP::UnitTest *ut,
                               const std::string &input_file,
                               const std::string &input_mesh = "" )
{
    AMP_INSIST( AMP::AMP_MPI( AMP_COMM_WORLD ).getSize() == 1, "This is a single processor test!" );

    std::cout << "Running " << input_file << " " << input_mesh << std::endl;
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    auto mesh_file = input_mesh;
    if ( input_mesh.empty() )
        mesh_file = input_db->getString( "mesh_file" );

    auto mesh = AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD );

    auto bvpOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MechanicsBVPOperator", input_db ) );

    auto var = bvpOperator->getOutputVariable();

    std::shared_ptr<AMP::Operator::DirichletVectorCorrection> dirichletVecOp;
    if ( input_db->keyExists( "Load_Boundary" ) ) {
        dirichletVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
            AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
        dirichletVecOp->setVariable( var );
    }

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    auto mechSolVec = AMP::LinearAlgebra::createVector( dofMap, var, true );
    auto mechRhsVec = mechSolVec->clone();
    auto mechResVec = mechSolVec->clone();
    mechSolVec->setToScalar( 0.5 );
    mechRhsVec->setToScalar( 0.0 );
    mechResVec->setToScalar( 0.0 );

    if ( dirichletVecOp )
        dirichletVecOp->apply( nullptr, mechRhsVec );
    else
        bvpOperator->modifyRHSvector( mechRhsVec );

    AMP::pout << "RHS Norm: " << mechRhsVec->L2Norm() << std::endl;
    AMP::pout << "Initial Solution Norm: " << mechSolVec->L2Norm() << std::endl;

    bvpOperator->residual( mechRhsVec, mechSolVec, mechResVec );

    double initResidualNorm = static_cast<double>( mechResVec->L2Norm() );

    AMP::pout << "Initial Residual Norm: " << initResidualNorm << std::endl;

    auto mlSolver_db = input_db->getDatabase( "LinearSolver" );

    auto mlSolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( mlSolver_db );

    mlSolverParams->d_pOperator = bvpOperator;

    // create the ML solver interface
    auto mlSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams );

    mlSolver->setZeroInitialGuess( false );

    mlSolver->apply( mechRhsVec, mechSolVec );

    AMP::pout << "Final Solution Norm: " << mechSolVec->L2Norm() << std::endl;

    bvpOperator->residual( mechRhsVec, mechSolVec, mechResVec );

    double finalResidualNorm = static_cast<double>( mechResVec->L2Norm() );

    AMP::pout << "Final Residual Norm: " << finalResidualNorm << std::endl << std::endl;

    auto testname = input_file;
    if ( !input_mesh.empty() )
        testname += "-" + input_mesh;
    if ( finalResidualNorm > ( 1e-10 * initResidualNorm ) ) {
        ut->failure( testname );
    } else {
        ut->passes( testname );
    }
}

int testLinearElasticity( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );

    AMP::UnitTest ut;

    if ( argc == 2 ) {
        linearElasticTest( &ut, argv[1] );
    } else if ( argc == 3 ) {
        linearElasticTest( &ut, argv[1], argv[2] );
    } else {
        linearElasticTest( &ut, "input_testLinearElasticity-patch-1-normal" );
        linearElasticTest( &ut, "input_testLinearElasticity-patch-1-reduced" );
        linearElasticTest( &ut, "input_testLinearElasticity-patch-2-normal" );
        linearElasticTest( &ut, "input_testLinearElasticity-patch-2-reduced" );
        for ( int i = 1; i <= 6; i++ ) {
            auto mesh = AMP::Utilities::stringf( "mesh2elem-%d", i );
            linearElasticTest( &ut, "input_testLinearElasticity-reduced-mesh2elem", mesh );
            linearElasticTest( &ut, "input_testLinearElasticity-normal-mesh2elem", mesh );
        }
    }

    ut.report();
    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
