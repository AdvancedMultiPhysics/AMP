#include "AMP/geometry/Geometry.h"
#include "AMP/geometry/testHelpers/geometryTests.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"

#include "ProfilerApp.h"


void testInputGeometries( AMP::UnitTest &ut, const std::string &filename )
{
    PROFILE( "testInputGeometries" );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( filename );

    // Loop through the databases
    for ( auto &key : input_db->getAllKeys() ) {
        // Load the geometry database
        auto db = input_db->getDatabase( key );
        // Create the geometry
        auto geom = AMP::Geometry::Geometry::buildGeometry( db );
        // Run the tests
        testGeometry( *geom, ut );
    }
}


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    PROFILE_ENABLE( 2 );

    // Test each given input file
    for ( int i = 1; i < argc; i++ )
        testInputGeometries( ut, argv[i] );

    // Save the timing results
    PROFILE_SAVE( "test_Geometry" );

    // Print the results and return
    int num_failed = ut.NumFailGlobal();
    ut.report();
    ut.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
