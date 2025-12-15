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
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "libmesh/mesh_communication.h"

#include <string>


void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testAMGuniform" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logAllNodes( log_file );

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );


    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    auto mesh_file = input_db->getString( "mesh_file" );
    auto mesh      = AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD );


    auto bvpOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearBVPOperator", input_db ) );

    /* auto mat     = bvpOperator->getMatrix();
    size_t matSz = mat->numGlobalRows();
    for ( size_t i = 0; i < matSz; ++i ) {
        std::vector<size_t> cols;
        std::vector<double> vals;
        mat->getRowByGlobalID( i, cols, vals );
        for ( size_t j = 0; j < cols.size(); ++j ) {
            std::cout << "A[" << i << "][" << ( cols[j] ) << "] = " << std::setprecision( 15 )
                      << ( vals[j] ) << std::endl;
        }
        std::cout << std::endl;
    } */

    int DOFsPerNode = 1;
    // int DOFsPerElement = 8;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto solVec = AMP::LinearAlgebra::createVector( nodalDofMap, bvpOperator->getOutputVariable() );
    auto rhsVec = solVec->clone();

    solVec->setRandomValues();
    bvpOperator->apply( solVec, rhsVec );
    solVec->zero();

    auto mlSolver_db    = input_db->getDatabase( "LinearSolver" );
    auto mlSolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( mlSolver_db );
    mlSolverParams->d_pOperator = bvpOperator;
    auto mlSolver               = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams );

    mlSolver->setZeroInitialGuess( true );

    mlSolver->apply( rhsVec, solVec );

    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    auto libmeshInit =
        std::make_shared<AMP::Mesh::initializeLibMesh>( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    myTest( &ut );

    ut.report();
    int num_failed = ut.NumFailGlobal();

    libmeshInit.reset();

    AMP::AMPManager::shutdown();
    return num_failed;
}
