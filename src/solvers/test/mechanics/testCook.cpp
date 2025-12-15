#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/libmesh/initializeLibMesh.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"
#include <memory>

// libMesh files
DISABLE_WARNINGS
#include "libmesh/boundary_info.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_communication.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS

#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>


static void linearElasticTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName + ".txt";

    AMP::logOnlyNodeZero( log_file );

    auto globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );
    auto input_db   = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    auto mesh_file = input_db->getString( "mesh_file" );
    auto mesh = AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD, "cook" );
    AMP_ASSERT( mesh );

    auto bvpOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MechanicsBVPOperator", input_db ) );

    auto dirichletVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    // This has an in-place apply. So, it has an empty input variable and
    // the output variable is the same as what it is operating on.
    dirichletVecOp->setVariable( bvpOperator->getOutputVariable() );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    auto DOF_vector = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );
    auto mechSolVec =
        AMP::LinearAlgebra::createVector( DOF_vector, bvpOperator->getOutputVariable(), true );
    auto mechRhsVec =
        AMP::LinearAlgebra::createVector( DOF_vector, bvpOperator->getOutputVariable(), true );
    auto mechResVec =
        AMP::LinearAlgebra::createVector( DOF_vector, bvpOperator->getOutputVariable(), true );

    mechSolVec->setToScalar( 0.5 );
    mechRhsVec->setToScalar( 0.0 );
    mechResVec->setToScalar( 0.0 );

    dirichletVecOp->apply( nullVec, mechRhsVec );


    AMP::pout << "RHS Norm: " << mechRhsVec->L2Norm() << std::endl;
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

    // Petsc either has a bug in the preonly interface for KSP that assumes that
    // a non-zero initial guess is intended for a petsc preconditioner -we are using
    // a non-petsc preconditioner or else it's a user error on our part
    linearSolver->setZeroInitialGuess( true );

    linearSolver->apply( mechRhsVec, mechSolVec );

    AMP::pout << "Final Solution Norm: " << mechSolVec->L2Norm() << std::endl;

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
}

int testCook( int argc, char *argv[] )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;

    if ( argc == 1 ) {
        exeNames.emplace_back( "testCook-normal-mesh0" );
        exeNames.emplace_back( "testCook-normal-mesh1" );
        exeNames.emplace_back( "testCook-normal-mesh2" );
        exeNames.emplace_back( "testCook-reduced-mesh0" );
        exeNames.emplace_back( "testCook-reduced-mesh1" );
        exeNames.emplace_back( "testCook-reduced-mesh2" );
    } else {
        for ( int i = 1; i < argc; i++ )
            exeNames.emplace_back( argv[i] );
    }

    for ( auto &exeName : exeNames )
        linearElasticTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
