#include "AMP/IO/FileSystem.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/mesh/testHelpers/meshTests.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"

#include "ProfilerApp.h"


template<class GENERATOR, class... Args>
void runTest( AMP::UnitTest &ut, Args... ts )
{
    auto generator = std::make_shared<GENERATOR>( ts... );
    generator->build_mesh();
    AMP::Mesh::meshTests::MeshPerformance( ut, generator->getMesh() );
}


void testMeshGenerators( AMP::UnitTest &ut )
{
    PROFILE( "testMeshGenerators" );
    // AMP mesh generators
    runTest<AMP::unit_test::AMPCubeGenerator>( ut, 4 );
    runTest<AMP::unit_test::AMPCylinderGenerator>( ut );
    runTest<AMP::unit_test::AMPMultiMeshGenerator>( ut );
    runTest<AMP::unit_test::AMPCubeGenerator>( ut, 4 );
// libMesh generators
#ifdef AMP_USE_LIBMESH
    runTest<AMP::unit_test::LibMeshCubeGenerator>( ut, 5 );
    runTest<AMP::unit_test::libMeshThreeElementGenerator>( ut );
    if ( AMP::IO::exists( "pellet_1x.e" ) )
        runTest<AMP::unit_test::ExodusReaderGenerator>( ut, "pellet_1x.e" );
#endif
}


void testInputMesh( AMP::UnitTest &ut, const std::string &filename )
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

    // Run the mesh tests
    AMP::Mesh::meshTests::MeshPerformance( ut, mesh );
}


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;
    PROFILE_ENABLE();
    PROFILE( "Run tests" );

    if ( argc == 1 ) {
        // Run the default tests
        testMeshGenerators( ut );
    } else {
        // Test each given input file
        for ( int i = 1; i < argc; i++ )
            testInputMesh( ut, argv[i] );
    }

    // Save the timing results
    PROFILE_SAVE( "test_MeshPerformance" );

    // Print the results and return
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
