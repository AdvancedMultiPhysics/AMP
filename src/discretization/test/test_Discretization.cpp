#include "test_Discretization.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    // startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    // Run the simpleDOFManager tests
    testSimpleDOFManager<AMPCubeGenerator<10>>( &ut );
    testSimpleDOFManager<AMPMultiMeshGenerator>( &ut );
#ifdef AMP_USE_LIBMESH
    testSimpleDOFManager<LibMeshCubeGenerator<5>>( &ut );
    testSimpleDOFManager<ExodusReaderGenerator<1>>( &ut );
    testSimpleDOFManager<ExodusReaderGenerator<3>>( &ut );
    testSimpleDOFManager<MultiMeshGenerator>( &ut );
#endif

    // Run the multiDOFManager tests
    testMultiDOFManager<AMPCubeGenerator<10>>( &ut );
#ifdef AMP_USE_LIBMESH
    testMultiDOFManager<LibMeshCubeGenerator<5>>( &ut );
    testMultiDOFManager<MultiMeshGenerator>( &ut );
#endif

    // Run the subsetDOFManager tests
    testSubsetDOFManager<AMPCubeGenerator<10>, false>( &ut );
    testSubsetDOFManager<AMPMultiMeshGenerator, false>( &ut );
    testSubsetDOFManager<AMPMultiMeshGenerator, true>( &ut );
#ifdef AMP_USE_LIBMESH
    testSubsetDOFManager<ExodusReaderGenerator<3>, false>( &ut );
    testSubsetDOFManager<MultiMeshGenerator, false>( &ut );
    testSubsetDOFManager<MultiMeshGenerator, true>( &ut );
#endif

    // Run the tests for the structureMeshDOFManager
    testStructureDOFManager<AMPCubeGenerator<10>, 1, 0, 0, 1>( &ut );
    testStructureDOFManager<AMPCubeGenerator<10>, 0, 1, 0, 1>( &ut );
    testStructureDOFManager<AMPCubeGenerator<10>, 0, 0, 1, 1>( &ut );
    testStructureDOFManager<AMPCubeGenerator<10>, 1, 1, 1, 1>( &ut );
    testStructureDOFManager<AMPCubeGenerator<10>, 1, 1, 1, 0>( &ut );
    testStructureDOFManager<AMPCubeGenerator<10>, 1, 1, 1, 2>( &ut );

    // Print the results and return
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
