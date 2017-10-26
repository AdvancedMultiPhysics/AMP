#include "ampmesh/Mesh.h"
#include "ampmesh/structured/BoxMesh.h"
#include "ampmesh/testHelpers/meshTests.h"

#include "utils/AMPManager.h"
#include "utils/AMP_MPI.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/MemoryDatabase.h"
#include "utils/UnitTest.h"

#include "ProfilerApp.h"

#include "meshGenerators.h"


// Function to test the creation/destruction of a mesh with the mesh generators
// Note: this only runs the mesh tests, not the vector or matrix tests
void testMeshGenerators( AMP::UnitTest *ut )
{
    PROFILE_START( "testMeshGenerators" );
    AMP::shared_ptr<AMP::unit_test::MeshGenerator> generator;
    // AMP mesh generators
    generator = AMP::make_shared<AMP::unit_test::AMPCubeGenerator<4>>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, generator->getMesh() );
    generator = AMP::make_shared<AMP::unit_test::AMPCylinderGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, generator->getMesh() );
    generator = AMP::make_shared<AMP::unit_test::AMPTubeGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, generator->getMesh() );
    generator = AMP::make_shared<AMP::unit_test::AMPMultiMeshGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
#ifdef USE_EXT_LIBMESH
    // libmesh generators
    // Test the libmesh cube generator
    generator = AMP::make_shared<AMP::unit_test::LibMeshCubeGenerator<5>>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    // Test the libmesh reader generator
    generator = AMP::make_shared<AMP::unit_test::ExodusReaderGenerator<>>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    generator = AMP::make_shared<AMP::unit_test::ExodusReaderGenerator<2>>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    // Test the ThreeElementLGenerator generator
    generator = AMP::make_shared<AMP::unit_test::libMeshThreeElementGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    // Test the multimesh generator
    generator = AMP::make_shared<AMP::unit_test::MultiMeshGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, generator->getMesh() );
#endif
    PROFILE_STOP( "testMeshGenerators" );
}


// Function to test the creation/destruction of a native AMP mesh
void testAMPMesh( AMP::UnitTest *ut )
{
    PROFILE_START( "testAMPMesh" );
    // Create the AMP mesh
    AMP::shared_ptr<AMP::unit_test::MeshGenerator> generator;
    generator = AMP::make_shared<AMP::unit_test::AMPCubeGenerator<5>>();
    generator->build_mesh();
    auto mesh = generator->getMesh();

    // Check the basic dimensions
    std::vector<size_t> size( 3, 5 );
    size_t N_elements_global = size[0] * size[1] * size[2];
    size_t N_faces_global    = ( size[0] + 1 ) * size[1] * size[2] +
                            size[0] * ( size[1] + 1 ) * size[2] +
                            size[0] * size[1] * ( size[2] + 1 );
    size_t N_edges_global = size[0] * ( size[1] + 1 ) * ( size[2] + 1 ) +
                            ( size[0] + 1 ) * size[1] * ( size[2] + 1 ) +
                            ( size[0] + 1 ) * ( size[1] + 1 ) * size[2];
    size_t N_nodes_global = ( size[0] + 1 ) * ( size[1] + 1 ) * ( size[2] + 1 );
    if ( mesh->numGlobalElements( AMP::Mesh::GeomType::Vertex ) == N_nodes_global )
        ut->passes( "Simple structured mesh has expected number of nodes" );
    else
        ut->failure( "Simple structured mesh has expected number of nodes" );
    if ( mesh->numGlobalElements( AMP::Mesh::GeomType::Edge ) == N_edges_global )
        ut->passes( "Simple structured mesh has expected number of edges" );
    else
        ut->failure( "Simple structured mesh has expected number of edges" );
    if ( mesh->numGlobalElements( AMP::Mesh::GeomType::Face ) == N_faces_global )
        ut->passes( "Simple structured mesh has expected number of faces" );
    else
        ut->failure( "Simple structured mesh has expected number of faces" );
    if ( mesh->numGlobalElements( AMP::Mesh::GeomType::Volume ) == N_elements_global )
        ut->passes( "Simple structured mesh has expected number of elements" );
    else
        ut->failure( "Simple structured mesh has expected number of elements" );

    // Check the volumes
    std::vector<double> range( 6, 0.0 );
    range[1]      = 1.0;
    range[3]      = 1.0;
    range[5]      = 1.0;
    double dx     = range[1] / size[0];
    auto iterator = mesh->getIterator( AMP::Mesh::GeomType::Edge );
    bool passes   = true;
    for ( size_t i = 0; i < iterator.size(); i++ ) {
        if ( !AMP::Utilities::approx_equal( iterator->volume(), dx, 1e-12 ) )
            passes = false;
    }
    if ( passes )
        ut->passes( "Simple structured mesh has correct edge legth" );
    else
        ut->failure( "Simple structured mesh has correct edge legth" );
    iterator = mesh->getIterator( AMP::Mesh::GeomType::Face );
    passes   = true;
    for ( size_t i = 0; i < iterator.size(); i++ ) {
        if ( !AMP::Utilities::approx_equal( iterator->volume(), dx * dx, 1e-12 ) )
            passes = false;
    }
    if ( passes )
        ut->passes( "Simple structured mesh has correct face area" );
    else
        ut->failure( "Simple structured mesh has correct face area" );
    iterator = mesh->getIterator( AMP::Mesh::GeomType::Volume );
    passes   = true;
    for ( size_t i = 0; i < iterator.size(); i++ ) {
        if ( !AMP::Utilities::approx_equal( iterator->volume(), dx * dx * dx, 1e-12 ) )
            passes = false;
    }
    if ( passes )
        ut->passes( "Simple structured mesh has correct element volume" );
    else
        ut->failure( "Simple structured mesh has correct element volume" );

    // Run the mesh tests
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    AMP::Mesh::meshTests::getParents( ut, mesh );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshMatrixTestLoop( ut, mesh );
    PROFILE_STOP( "testAMPMesh" );
}


// Function to test the creation/destruction of a STKmesh mesh
void testSTKMesh( AMP::UnitTest *ut )
{
#ifdef USE_TRILINOS_STKCLASSIC
    PROFILE_START( "testSTKMesh" );
    // Create a generic MeshParameters object
    auto database = AMP::make_shared<AMP::MemoryDatabase>( "Mesh" );
    database->putInteger( "dim", 3 );
    database->putString( "MeshType", "STKMesh" );
    database->putString( "MeshName", "pellet_lo_res.e" );
    database->putString( "FileName", "pellet_lo_res.e" );
    auto params = AMP::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create a STKMesh mesh
    auto mesh = AMP::Mesh::Mesh::buildMesh( params );
    AMP_ASSERT( mesh != nullptr );

    // Run the mesh tests
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshMatrixTestLoop( ut, mesh );
    PROFILE_STOP( "testSTKMesh" );
#else
    ut->expected_failure( "testSTKMesh disabled (compiled without STKClassic)" );
#endif
}


// Function to test the creation/destruction of a libmesh mesh

void testlibMesh( AMP::UnitTest *ut )
{
#ifdef USE_EXT_LIBMESH
    PROFILE_START( "testlibMesh" );
    // Create a generic MeshParameters object
    auto database = AMP::make_shared<AMP::MemoryDatabase>( "Mesh" );
    database->putInteger( "dim", 3 );
    database->putString( "MeshType", "libMesh" );
    database->putString( "MeshName", "pellet_lo_res.e" );
    database->putString( "FileName", "pellet_lo_res.e" );
    auto params = AMP::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create a libMesh mesh
    auto mesh = AMP::Mesh::Mesh::buildMesh( params );
    AMP_ASSERT( mesh != nullptr );

    // Run the mesh tests
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshMatrixTestLoop( ut, mesh );
    PROFILE_STOP( "testlibMesh" );
#else
    ut->expected_failure( "testlibMesh disabled (compiled without libmesh)" );
#endif
}



// Function to test the creation/destruction of a moab mesh
void testMoabMesh( AMP::UnitTest *ut )
{
#ifdef USE_EXT_MOAB
    PROFILE_START( "testMoabMesh" );
    // Create a generic MeshParameters object
    auto database = AMP::make_shared<AMP::MemoryDatabase>( "Mesh" );
    database->putInteger( "dim", 3 );
    database->putString( "MeshType", "MOAB" );
    database->putString( "MeshName", "pellet_lo_res.e" );
    database->putString( "FileName", "pellet_lo_res.e" );
    auto params = AMP::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create a MOAB mesh
    auto mesh = AMP::Mesh::Mesh::buildMesh( params );
    AMP_ASSERT( mesh != nullptr );

    // Run the mesh tests
    ut->expected_failure( "Mesh tests not working on a MOAB mesh yet" );
    // MeshTestLoop( ut, mesh );
    // MeshVectorTestLoop( ut, mesh );
    // MeshMatrixTestLoop( ut, mesh );
    PROFILE_STOP( "testMoabMesh" );
#else
    ut->expected_failure( "testMoabMesh disabled (compiled without MOAB)" );
#endif
}


void testInputMesh( AMP::UnitTest *ut, std::string filename )
{
    PROFILE_START( "testInputMesh" );
    // Read the input file
    auto input_db = AMP::make_shared<AMP::InputDatabase>( "input_db" );
    AMP::InputManager::getManager()->parseInputFile( filename, input_db );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = AMP::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::Mesh::buildMesh( params );
    AMP_ASSERT( mesh != nullptr );

    // Run the mesh tests
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, mesh, true );
    AMP::Mesh::meshTests::MeshMatrixTestLoop( ut, mesh, true );
    PROFILE_STOP( "testInputMesh" );
}


void testSubsetMesh( AMP::UnitTest *ut )
{
    PROFILE_START( "testSubsetMesh" );
#ifdef USE_EXT_LIBMESH
    // Subset a mesh for a surface without ghost cells and test
    AMP::shared_ptr<AMP::unit_test::MeshGenerator> generator;
    generator = AMP::make_shared<
        AMP::unit_test::SurfaceSubsetGenerator<AMP::unit_test::ExodusReaderGenerator<>, 0>>();
    generator->build_mesh();
    AMP::Mesh::Mesh::shared_ptr mesh = generator->getMesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    // MeshVectorTestLoop( ut, mesh );
    // MeshMatrixTestLoop( ut, mesh );

    // Subset a mesh for a surface with ghost cells and test
    generator = AMP::make_shared<
        AMP::unit_test::SurfaceSubsetGenerator<AMP::unit_test::ExodusReaderGenerator<3>, 1>>();
    generator->build_mesh();
    mesh = generator->getMesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
// MeshVectorTestLoop( ut, mesh );
// MeshMatrixTestLoop( ut, mesh );
#else
    NULL_USE( ut );
#endif
    PROFILE_STOP( "testSubsetMesh" );
}


// Run the default tests/mesh generators
void testDefaults( AMP::UnitTest &ut )
{
    // Run the ID test
    AMP::Mesh::meshTests::testID( &ut );

    // Run tests on a native AMP mesh
    testAMPMesh( &ut );

    // Run tests on a STKmesh mesh (currently disabled)
    // testSTKMesh( &ut );

    // Run tests on a libmesh mesh
    testlibMesh( &ut );

    // Run tests on a moab mesh
    testMoabMesh( &ut );

    // Run tests on the input file
    #ifdef USE_LIBMESH
        testInputMesh( &ut, "input_Mesh" );
    #endif

    // Run the basic tests on all mesh generators
    testMeshGenerators( &ut );

    // Run the tests on the subset meshes
    testSubsetMesh( &ut );
}


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;
    PROFILE_ENABLE( 2 );
    PROFILE_START( "Run tests" );

    if ( argc == 1 ) {
        // Run the default tests
        testDefaults( ut );
    } else {
        // Test each given input file
        for ( int i = 1; i < argc; i++ )
            testInputMesh( &ut, argv[i] );
    }

    // Save the timing results
    PROFILE_STOP( "Run tests" );
    PROFILE_SAVE( "test_Mesh" );

    // Print the results and return
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
