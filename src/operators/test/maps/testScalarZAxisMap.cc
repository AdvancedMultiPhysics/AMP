#include "utils/Database.h"
#include "ampmesh/Mesh.h"
#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "operators/map/AsyncMapColumnOperator.h"
#include "operators/map/ScalarZAxisMap.h"
#include "utils/AMPManager.h"
#include "utils/AMP_MPI.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/PIO.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "vectors/Variable.h"
#include "vectors/VectorBuilder.h"


void setBoundary( int id,
                  AMP::LinearAlgebra::Vector::shared_ptr &v1,
                  AMP::Mesh::Mesh::shared_ptr mesh )
{
    if ( mesh.get() == nullptr )
        return;

    AMP::Discretization::DOFManager::shared_ptr d1 = v1->getDOFManager();

    AMP::Mesh::MeshIterator curBnd =
        mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, id, 0 );
    AMP::Mesh::MeshIterator endBnd = curBnd.end();

    std::vector<size_t> ids;
    while ( curBnd != endBnd ) {
        d1->getDOFs( curBnd->globalID(), ids );
        std::vector<double> x = curBnd->coord();
        v1->setLocalValuesByGlobalID( ids.size(), &ids[0], &x[2] );
        ++curBnd;
    }
}


void runTest( const std::string &fname, AMP::UnitTest *ut )
{
    // Read the input file
    AMP::shared_ptr<AMP::InputDatabase> input_db( new AMP::InputDatabase( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile( fname, input_db );
    input_db->printClassData( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    AMP::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase( "Mesh" );
    AMP::shared_ptr<AMP::Mesh::MeshParameters> params( new AMP::Mesh::MeshParameters( mesh_db ) );
    params->setComm( globalComm );

    // Create the meshes from the input database
    AMP::shared_ptr<AMP::Mesh::Mesh> mesh = AMP::Mesh::Mesh::buildMesh( params );

    // Get the database for the node to node maps
    AMP::shared_ptr<AMP::Database> map_db = input_db->getDatabase( "MeshToMeshMaps" );

    // Create a simple DOFManager and the vectors
    int DOFsPerNode     = map_db->getInteger( "DOFsPerObject" );
    std::string varName = map_db->getString( "VariableName" );
    AMP::LinearAlgebra::Variable::shared_ptr nodalVariable(
        new AMP::LinearAlgebra::Variable( varName ) );
    AMP::Discretization::DOFManagerParameters::shared_ptr DOFparams(
        new AMP::Discretization::DOFManagerParameters( mesh ) );
    AMP::Discretization::DOFManager::shared_ptr DOFs =
        AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, DOFsPerNode );

    // Test the creation/destruction of ScalarZAxisMap (no apply call)
    try {
        AMP::shared_ptr<AMP::Operator::AsyncMapColumnOperator> gapmaps;
        gapmaps = AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::ScalarZAxisMap>(
            mesh, map_db );
        gapmaps.reset();
        ut->passes( "Created / Destroyed ScalarZAxisMap" );
    } catch ( ... ) {
        ut->failure( "Created / Destroyed ScalarZAxisMap" );
    }

    // Perform a complete test of ScalarZAxisMap
    AMP::shared_ptr<AMP::Operator::AsyncMapColumnOperator> gapmaps;
    gapmaps =
        AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::ScalarZAxisMap>( mesh, map_db );

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr dummy;
    AMP::LinearAlgebra::Vector::shared_ptr v1 =
        AMP::LinearAlgebra::createVector( DOFs, nodalVariable );
    AMP::LinearAlgebra::Vector::shared_ptr v2 =
        AMP::LinearAlgebra::createVector( DOFs, nodalVariable );
    gapmaps->setVector( v2 );

    // Initialize the vectors
    v1->setToScalar( 0.0 );
    v2->setToScalar( 0.0 );
    size_t N_maps                  = (size_t) map_db->getInteger( "N_maps" );
    std::vector<std::string> mesh1 = map_db->getStringArray( "Mesh1" );
    std::vector<std::string> mesh2 = map_db->getStringArray( "Mesh2" );
    std::vector<int> surface1      = map_db->getIntegerArray( "Surface1" );
    std::vector<int> surface2      = map_db->getIntegerArray( "Surface2" );
    AMP_ASSERT( mesh1.size() == N_maps || mesh1.size() == 1 );
    AMP_ASSERT( mesh2.size() == N_maps || mesh2.size() == 1 );
    AMP_ASSERT( surface1.size() == N_maps || surface1.size() == 1 );
    AMP_ASSERT( surface2.size() == N_maps || surface2.size() == 1 );
    for ( size_t i = 0; i < N_maps; i++ ) {
        std::string meshname1, meshname2;
        if ( mesh1.size() == N_maps ) {
            meshname1 = mesh1[i];
            meshname2 = mesh2[i];
        } else {
            meshname1 = mesh1[0];
            meshname2 = mesh2[0];
        }
        int surface_id1, surface_id2;
        if ( surface1.size() == N_maps ) {
            surface_id1 = surface1[i];
            surface_id2 = surface2[i];
        } else {
            surface_id1 = surface1[0];
            surface_id2 = surface2[0];
        }
        AMP::Mesh::Mesh::shared_ptr curMesh = mesh->Subset( meshname1 );
        setBoundary( surface_id1, v1, curMesh );
        curMesh = mesh->Subset( meshname2 );
        setBoundary( surface_id2, v1, curMesh );
    }

    // Apply the maps
    globalComm.barrier();
    gapmaps->apply( v1, v2 );
    v1->subtract( v1, v2 );
    if ( v1->maxNorm() < 1.e-12 )
        ut->passes( "Node to node map test" );
    else
        ut->failure( "Node to node map test" );

    gapmaps.reset();
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    AMP::AMP_MPI globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // int  numNodes = globalComm.getSize();
    runTest( "inputScalarZAxisMap-1", &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
