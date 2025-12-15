#include "AMP/AMP_TPLs.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/discretization/testHelpers/discretizationTests.h"
#include "AMP/discretization/testHelpers/discretizationTestsLoop.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"


using namespace AMP::unit_test;


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    // startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    // Run the simpleDOFManager tests
    testLogicalDOFMap( ut );
    testSimpleDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), ut );
    testSimpleDOFManager( std::make_shared<AMPMultiMeshGenerator>(), ut );
#ifdef AMP_USE_LIBMESH
    testSimpleDOFManager( std::make_shared<LibMeshCubeGenerator>( 5 ), ut );
    if ( AMP::IO::exists( "pellet_1x.e" ) ) {
        testSimpleDOFManager( std::make_shared<ExodusReaderGenerator>( "pellet_1x.e" ), ut );
        testSimpleDOFManager( std::make_shared<MultiMeshGenerator>(), ut );
    }
#endif

    // Run the multiDOFManager tests
    testMultiDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), ut );
#ifdef AMP_USE_LIBMESH
    testMultiDOFManager( std::make_shared<LibMeshCubeGenerator>( 5 ), ut );
    if ( AMP::IO::exists( "pellet_1x.e" ) )
        testMultiDOFManager( std::make_shared<MultiMeshGenerator>(), ut );
#endif

    // Run the subsetDOFManager tests
    testSubsetDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), false, ut );
    testSubsetDOFManager( std::make_shared<AMPMultiMeshGenerator>(), false, ut );
    testSubsetDOFManager( std::make_shared<AMPMultiMeshGenerator>(), true, ut );
#if defined( AMP_USE_LIBMESH )
    if ( AMP::IO::exists( "pellet_1x.e" ) ) {
        testSubsetDOFManager( std::make_shared<ExodusReaderGenerator>( "pellet_1x.e" ), false, ut );
        testSubsetDOFManager( std::make_shared<MultiMeshGenerator>(), false, ut );
        testSubsetDOFManager( std::make_shared<MultiMeshGenerator>(), true, ut );
    }
#endif

    // Run the tests for the structureMeshDOFManager
    testStructureDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), 1, 0, 0, 1, ut );
    testStructureDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), 0, 1, 0, 1, ut );
    testStructureDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), 0, 0, 1, 1, ut );
    testStructureDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), 1, 1, 1, 1, ut );
    testStructureDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), 1, 1, 1, 0, ut );
    testStructureDOFManager( std::make_shared<AMPCubeGenerator>( 10 ), 1, 1, 1, 2, ut );

    // Print the results and return
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
