#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/map/dtk/DTKMapOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

double testFunction1( const std::vector<double> &coords )
{
    return coords[0] + coords[1] + coords[2] + 2.0;
}

static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testDTKMapOperator" );
    std::string log_file = "output_" + exeName;
    std::string msgPrefix;
    AMP::logOnlyNodeZero( log_file );

    // load the source mesh
    AMP::pout << "Loading the source mesh" << std::endl;

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    std::string input_file = "input_" + exeName;
    auto input_db          = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    auto sourceMeshDatabase = input_db->getDatabase( "SourceMesh" );
    auto sourceMeshParams   = std::make_shared<AMP::Mesh::MeshParameters>( sourceMeshDatabase );
    sourceMeshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto sourceMesh                = AMP::Mesh::MeshFactory::create( sourceMeshParams );
    size_t numVerticesOnSourceMesh = sourceMesh->numGlobalElements( AMP::Mesh::GeomType::Vertex );
    size_t numElementsOnSourceMesh = sourceMesh->numGlobalElements( AMP::Mesh::GeomType::Cell );
    AMP::pout << "source mesh contains " << numVerticesOnSourceMesh << " vertices\n";
    AMP::pout << "source mesh contains " << numElementsOnSourceMesh << " elements\n";

    // build source vector
    AMP::pout << "Building the source vector" << std::endl;
    bool const split      = true;
    int const ghostWidth  = 1;
    int const dofsPerNode = 1;
    auto sourceDofManager = AMP::Discretization::simpleDOFManager::create(
        sourceMesh, AMP::Mesh::GeomType::Vertex, ghostWidth, dofsPerNode );
    auto variable     = std::make_shared<AMP::LinearAlgebra::Variable>( "dummy" );
    auto sourceVector = AMP::LinearAlgebra::createVector( sourceDofManager, variable, split );
    // and fill it
    std::vector<std::size_t> dofIndices;
    double value;
    AMP::pout << "Filling source vector" << std::endl;
    auto sourceMeshIterator = sourceMesh->getIterator( AMP::Mesh::GeomType::Vertex );
    for ( sourceMeshIterator = sourceMeshIterator.begin();
          sourceMeshIterator != sourceMeshIterator.end();
          ++sourceMeshIterator ) {
        sourceDofManager->getDOFs( sourceMeshIterator->globalID(), dofIndices );
        AMP_ASSERT( dofIndices.size() == 1 );
        value = testFunction1( sourceMeshIterator->coord() );
        sourceVector->setLocalValueByGlobalID( dofIndices[0], value );
    }
    // load the target mesh
    AMP::pout << "Loading the target mesh" << std::endl;
    auto targetMeshDatabase = input_db->getDatabase( "TargetMesh" );
    auto targetMeshParams   = std::make_shared<AMP::Mesh::MeshParameters>( targetMeshDatabase );
    targetMeshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto targetMesh                = AMP::Mesh::MeshFactory::create( targetMeshParams );
    size_t numVerticesOnTargetMesh = targetMesh->numGlobalElements( AMP::Mesh::GeomType::Vertex );
    size_t numElementsOnTargetMesh = targetMesh->numGlobalElements( AMP::Mesh::GeomType::Cell );
    AMP::pout << "target mesh contains " << numVerticesOnTargetMesh << " vertices\n";
    AMP::pout << "target mesh contains " << numElementsOnTargetMesh << " elements\n";

    AMP::pout << "Building the target vector" << std::endl;
    auto targetDofManager = AMP::Discretization::simpleDOFManager::create(
        targetMesh, AMP::Mesh::GeomType::Vertex, ghostWidth, dofsPerNode );
    auto targetVector = AMP::LinearAlgebra::createVector( targetDofManager, variable, split );

    // create dtk map operator.
    std::shared_ptr<AMP::Database> null_db;
    auto dtk_op_params = std::make_shared<AMP::Operator::DTKMapOperatorParameters>( null_db );
    dtk_op_params->d_domain_mesh = sourceMesh;
    dtk_op_params->d_range_mesh  = targetMesh;
    dtk_op_params->d_domain_dofs = sourceDofManager;
    dtk_op_params->d_range_dofs  = targetDofManager;
    dtk_op_params->d_globalComm  = AMP::AMP_MPI( AMP_COMM_WORLD );
    auto dtk_operator            = std::make_shared<AMP::Operator::DTKMapOperator>( dtk_op_params );

    // apply the map.
    AMP::pout << "Apply dtk operator" << std::endl;
    AMP::LinearAlgebra::Vector::shared_ptr null_vector;
    dtk_operator->apply( sourceVector, targetVector );

    // checking the answer
    AMP::pout << "Check answer" << std::endl;
    AMP::pout << "source vector l2 norm = " << sourceVector->L2Norm() << std::endl;
    AMP::pout << "target vector l2 norm = " << targetVector->L2Norm() << std::endl;

    double const atol       = 1.0e-14;
    double const rtol       = 1.0e-14;
    double const tol        = atol + rtol * targetVector->L2Norm();
    auto targetMeshIterator = targetMesh->getIterator( AMP::Mesh::GeomType::Vertex );
    for ( targetMeshIterator = targetMeshIterator.begin();
          targetMeshIterator != targetMeshIterator.end();
          ++targetMeshIterator ) {
        targetDofManager->getDOFs( targetMeshIterator->globalID(), dofIndices );
        AMP_ASSERT( dofIndices.size() == 1 );
        value = testFunction1( targetMeshIterator->coord() );
        targetVector->addLocalValueByGlobalID( dofIndices[0], -value );
    }
    AMP::pout << "error l2 norm = " << targetVector->L2Norm() << std::endl;
    AMP_ASSERT( targetVector->L2Norm() < tol );

    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
