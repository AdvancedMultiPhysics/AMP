#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/map/AsyncMapColumnOperator.h"
#include "AMP/operators/map/NodeToNodeMap.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"


static void setBoundary( int id,
                         AMP::LinearAlgebra::Vector::shared_ptr &v1,
                         std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    if ( mesh.get() == nullptr )
        return;

    auto d1     = v1->getDOFManager();
    auto curBnd = mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, id, 0 );
    auto endBnd = curBnd.end();

    std::vector<size_t> ids;
    while ( curBnd != endBnd ) {
        d1->getDOFs( curBnd->globalID(), ids );
        auto x = curBnd->coord();
        v1->setValuesByGlobalID( ids.size(), &ids[0], &x[0] );
        ++curBnd;
    }
}


static void runTest( const std::string &fname, AMP::UnitTest *ut )
{

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( fname );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto mesh_db = input_db->getDatabase( "Mesh" );
    auto params  = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create a surface mesh
    auto surfaceMesh = mesh->Subset( mesh->getSurfaceIterator( AMP::Mesh::GeomType::Face, 1 ) );

    // Create a surface vector with the surface ids
    auto idDOFs = AMP::Discretization::simpleDOFManager::create(
        surfaceMesh, AMP::Mesh::GeomType::Face, 1, 1 );
    auto id_var = std::make_shared<AMP::LinearAlgebra::Variable>( "id" );
    auto id_vec = AMP::LinearAlgebra::createVector( idDOFs, id_var );
    std::vector<size_t> dofs;
    for ( auto &id : surfaceMesh->getBoundaryIDs() ) {
        auto val = double( id );
        for ( auto elem : surfaceMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, id, 0 ) ) {
            idDOFs->getDOFs( elem.globalID(), dofs );
            AMP_ASSERT( dofs.size() == 1 );
            id_vec->setValuesByGlobalID( 1, &dofs[0], &val );
        }
    }

    // Get the database for the node to node maps
    auto map_db = input_db->getDatabase( "NodeToNodeMaps" );

    // Create a simple DOFManager and the vectors
    int DOFsPerNode     = map_db->getScalar<int>( "DOFsPerObject" );
    std::string varName = map_db->getString( "VariableName" );
    auto nodalVariable  = std::make_shared<AMP::LinearAlgebra::Variable>( varName );
    auto DOFparams      = std::make_shared<AMP::Discretization::DOFManagerParameters>( mesh );
    auto DOFs           = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, DOFsPerNode );

    // Test the creation/destruction of NodeToNodeMap (no apply call)
    try {
        auto n2nmaps = AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::NodeToNodeMap>(
            mesh, map_db );
        n2nmaps.reset();
        ut->passes( "Created / Destroyed NodeToNodeMap (" + fname + ")" );
    } catch ( ... ) {
        ut->failure( "Created / Destroyed NodeToNodeMap (" + fname + ")" );
    }

    // Build the maps
    auto n2nmaps =
        AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::NodeToNodeMap>( mesh, map_db );

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr dummy;
    auto v1 = AMP::LinearAlgebra::createVector( DOFs, nodalVariable );
    auto v2 = AMP::LinearAlgebra::createVector( DOFs, nodalVariable );
    n2nmaps->setVector( v2 );

    // Initialize the vectors
    v1->setToScalar( 0.0 );
    v2->setToScalar( 0.0 );
    size_t N_maps = (size_t) map_db->getScalar<int>( "N_maps" );
    auto mesh1    = map_db->getVector<std::string>( "Mesh1" );
    auto mesh2    = map_db->getVector<std::string>( "Mesh2" );
    auto surface1 = map_db->getVector<int>( "Surface1" );
    auto surface2 = map_db->getVector<int>( "Surface2" );
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
        auto curMesh = mesh->Subset( meshname1 );
        setBoundary( surface_id1, v1, curMesh );
        curMesh = mesh->Subset( meshname2 );
        setBoundary( surface_id2, v1, curMesh );
    }

    // Apply the maps
    globalComm.barrier();
    std::cout << v1->maxNorm() << "  " << v2->maxNorm() << std::endl;
    double start_time = AMP::AMP_MPI::time();
    n2nmaps->apply( v1, v2 );
    globalComm.barrier();
    double end_time = AMP::AMP_MPI::time();

    // Check the results
    std::cout << v1->maxNorm() << "  " << v2->maxNorm() << std::endl;
    v1->subtract( *v1, *v2 );
    std::cout << v1->maxNorm() << std::endl;
    if ( v1->maxNorm() < 1.e-12 ) {
        ut->passes( "Node to node map test (" + fname + ")" );
    } else {
        ut->failure( "Node to node map test (" + fname + ")" );
    }

    std::cout << "Time for apply call: " << end_time - start_time << std::endl;
}


int testNodeToNodeMap( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    if ( argc <= 1 ) {
        ut.failure( "No input files specified" );
    } else {
        for ( int i = 1; i < argc; i++ ) {
            AMP::pout << "Running test with input file: " << argv[i] << std::endl;
            runTest( argv[i], &ut );
        }
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
