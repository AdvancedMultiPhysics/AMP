#include "AMP/AMP_TPLs.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/mesh/testHelpers/meshTests.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#ifdef AMP_USE_LIBMESH
    #include "AMP/mesh/libmesh/initializeLibMesh.h"
#endif

#include "ProfilerApp.h"


// Function to test the creation/destruction of a mesh with the mesh generators
// Note: this only runs the mesh tests, not the vector or matrix tests
void testMeshGenerators( AMP::UnitTest &ut )
{
    PROFILE( "testMeshGenerators" );
    std::shared_ptr<AMP::unit_test::MeshGenerator> generator;
    // AMP mesh generators
    generator = std::make_shared<AMP::unit_test::AMPCubeGenerator>( 4 );
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshGeometryTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, generator->getMesh() );
    generator = std::make_shared<AMP::unit_test::AMPCylinderGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshGeometryTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, generator->getMesh() );
    generator = std::make_shared<AMP::unit_test::AMPTubeGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshGeometryTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, generator->getMesh() );
    generator = std::make_shared<AMP::unit_test::AMPMultiMeshGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    AMP::Mesh::meshTests::MeshGeometryTestLoop( ut, generator->getMesh() );
#ifdef AMP_USE_LIBMESH
    // libmesh generators
    // Test the libmesh cube generator
    generator = std::make_shared<AMP::unit_test::LibMeshCubeGenerator>( 5 );
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    // Test the libmesh reader generator
    if ( AMP::IO::exists( "pellet_1x.e" ) ) {
        generator = std::make_shared<AMP::unit_test::ExodusReaderGenerator>( "pellet_1x.e" );
        generator->build_mesh();
        AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    }
    // Test the ThreeElementLGenerator generator
    generator = std::make_shared<AMP::unit_test::libMeshThreeElementGenerator>();
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
    // Test the multimesh generator
    if ( AMP::IO::exists( "pellet_1x.e" ) ) {
        generator = std::make_shared<AMP::unit_test::MultiMeshGenerator>();
        generator->build_mesh();
        AMP::Mesh::meshTests::MeshTestLoop( ut, generator->getMesh() );
        AMP::Mesh::meshTests::MeshVectorTestLoop( ut, generator->getMesh() );
    }
#endif
}


// Function to test the creation/destruction of a native AMP mesh
void testAMPMesh( AMP::UnitTest &ut )
{
    PROFILE( "testAMPMesh" );
    // Create the AMP mesh
    auto generator = std::make_shared<AMP::unit_test::AMPCubeGenerator>( 5 );
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
        ut.passes( "Simple structured mesh has expected number of nodes" );
    else
        ut.failure( "Simple structured mesh has expected number of nodes" );
    if ( mesh->numGlobalElements( AMP::Mesh::GeomType::Edge ) == N_edges_global )
        ut.passes( "Simple structured mesh has expected number of edges" );
    else
        ut.failure( "Simple structured mesh has expected number of edges" );
    if ( mesh->numGlobalElements( AMP::Mesh::GeomType::Face ) == N_faces_global )
        ut.passes( "Simple structured mesh has expected number of faces" );
    else
        ut.failure( "Simple structured mesh has expected number of faces" );
    if ( mesh->numGlobalElements( AMP::Mesh::GeomType::Cell ) == N_elements_global )
        ut.passes( "Simple structured mesh has expected number of elements" );
    else
        ut.failure( "Simple structured mesh has expected number of elements" );

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
        ut.passes( "Simple structured mesh has correct edge legth" );
    else
        ut.failure( "Simple structured mesh has correct edge legth" );
    iterator = mesh->getIterator( AMP::Mesh::GeomType::Face );
    passes   = true;
    for ( size_t i = 0; i < iterator.size(); i++ ) {
        if ( !AMP::Utilities::approx_equal( iterator->volume(), dx * dx, 1e-12 ) )
            passes = false;
    }
    if ( passes )
        ut.passes( "Simple structured mesh has correct face area" );
    else
        ut.failure( "Simple structured mesh has correct face area" );
    iterator = mesh->getIterator( AMP::Mesh::GeomType::Cell );
    passes   = true;
    for ( size_t i = 0; i < iterator.size(); i++ ) {
        if ( !AMP::Utilities::approx_equal( iterator->volume(), dx * dx * dx, 1e-12 ) )
            passes = false;
    }
    if ( passes )
        ut.passes( "Simple structured mesh has correct element volume" );
    else
        ut.failure( "Simple structured mesh has correct element volume" );

    // Run the mesh tests
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    AMP::Mesh::meshTests::getParents( ut, mesh );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshMatrixTestLoop( ut, mesh );
}


// Function to test the creation/destruction of a libmesh mesh

void testlibMesh( AMP::UnitTest &ut )
{
#if defined( AMP_USE_LIBMESH )
    if ( !AMP::IO::exists( "pellet_1x.e" ) ) {
        ut.expected_failure( "testlibMesh disabled (missing pellet_1x.e)" );
        return;
    }
    PROFILE( "testlibMesh" );
    // Create a generic MeshParameters object
    auto database = std::make_shared<AMP::Database>( "Mesh" );
    database->putScalar( "dim", 3 );
    database->putScalar( "MeshType", "libMesh" );
    database->putScalar( "MeshName", "pellet" );
    database->putScalar( "FileName", "pellet_1x.e" );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create a libMesh mesh
    auto mesh = AMP::Mesh::MeshFactory::create( params );
    AMP_ASSERT( mesh );

    // Run the mesh tests
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshMatrixTestLoop( ut, mesh );
#else
    ut.expected_failure( "testlibMesh disabled (compiled without libmesh)" );
#endif
}


void testInputMesh( [[maybe_unused]] AMP::UnitTest &ut,
                    [[maybe_unused]] const std::string &filename )
{
    PROFILE( "testInputMesh" );
    // Read the input file
    auto input_db = AMP::Database::parseInputFile( filename );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );
    AMP_ASSERT( mesh );

    // Run the mesh tests
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshGeometryTestLoop( ut, mesh );
    AMP::Mesh::meshTests::MeshVectorTestLoop( ut, mesh, true );
    AMP::Mesh::meshTests::MeshMatrixTestLoop( ut, mesh, true );
}


void testSubsetMesh( [[maybe_unused]] AMP::UnitTest &ut )
{
    PROFILE( "testSubsetMesh" );
    // Subset a mesh for a surface with ghost cells and test
    auto generator = std::make_shared<AMP::unit_test::SurfaceSubsetGenerator>(
        std::make_shared<AMP::unit_test::AMPCylinderGenerator>(), 1 );
    generator->build_mesh();
    auto mesh = generator->getMesh();
    AMP::Mesh::meshTests::MeshTestLoop( ut, mesh );
    // MeshVectorTestLoop( ut, mesh );
    // MeshMatrixTestLoop( ut, mesh );
}


// Test initializing libMesh
void testIntializeLibmesh( [[maybe_unused]] AMP::UnitTest &ut )
{
#if defined( AMP_USE_LIBMESH )
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    AMP::AMP_MPI splitComm = globalComm.split( globalComm.getRank() % 2 );
    AMP_ASSERT( !AMP::Mesh::initializeLibMesh::isInitialized() );
    AMP_ASSERT( AMP::Mesh::initializeLibMesh::canBeInitialized( globalComm ) );
    AMP_ASSERT( AMP::Mesh::initializeLibMesh::canBeInitialized( splitComm ) );
    auto libmesh = std::make_shared<AMP::Mesh::initializeLibMesh>( splitComm );
    AMP_ASSERT( AMP::Mesh::initializeLibMesh::isInitialized() );
    if ( globalComm.getSize() > 1 )
        AMP_ASSERT( !AMP::Mesh::initializeLibMesh::canBeInitialized( globalComm ) );
    AMP_ASSERT( AMP::Mesh::initializeLibMesh::canBeInitialized( splitComm ) );
    libmesh.reset();
    AMP_ASSERT( AMP::Mesh::initializeLibMesh::canBeInitialized( globalComm ) );
    ut.passes( "Initialize libMesh" );
#endif
}


// Run the default tests/mesh generators
void testDefaults( AMP::UnitTest &ut )
{
    // Run the ID test
    AMP::Mesh::meshTests::testID( ut );

    // Run tests on a native AMP mesh
    AMP::Mesh::meshTests::testBoxMeshIndicies( ut, 1 );
    AMP::Mesh::meshTests::testBoxMeshIndicies( ut, 2 );
    AMP::Mesh::meshTests::testBoxMeshIndicies( ut, 3 );
    testAMPMesh( ut );

#if defined( AMP_USE_LIBMESH )
    // Run tests on a libmesh mesh
    testIntializeLibmesh( ut );
    testlibMesh( ut );

    // Run tests on the input file
    if ( AMP::IO::exists( "pellet_1x.e" ) )
        testInputMesh( ut, "input_libMesh" );
#endif

    // Run the basic tests on all mesh generators
    testMeshGenerators( ut );

    // Run the tests on the subset meshes
    testSubsetMesh( ut );
}


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;
    PROFILE_ENABLE( 2 );
    PROFILE( "Run tests" );

    if ( argc == 1 ) {
        // Run the default tests
        testDefaults( ut );
    } else {
        // Test each given input file
        for ( int i = 1; i < argc; i++ )
            testInputMesh( ut, argv[i] );
    }

    // Save the timing results
    PROFILE_SAVE( "test_Mesh" );

    // Print the results and return
    int num_failed = ut.NumFailGlobal();
    ut.report( 1 );
    ut.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
