#include "AMP/IO/PIO.h"
#include "AMP/mesh/SAMRAI/SAMRAIHierarchyAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRBuilder.h"
#include "AMP/mesh/SAMRAI/SAMRLevel.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/UtilityMacros.h"

#include "SAMRAI/hier/Patch.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/PatchLevel.h"

#include <string>


int main( int argc, char *argv[] )
{

    // Startup
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Process command line arguments and dump to log file.
    std::string input_file;
    std::string log_file;

    AMP::Mesh::SAMRBuilder::processCommandLine( argc, argv, input_file, log_file );

    AMP::logAllNodes( log_file );

    /*
     * Create block to force pointer deallocation.  If this is not done
     * then there will be memory leaks reported.
     */
    {

        // Create input database and parse all data in input file.
        auto input_db = AMP::Database::parseInputFile( input_file );
        input_db->setDefaultAddKeyBehavior( AMP::Database::Check::Overwrite, true );
        input_db->print( AMP::plog );

        // Create the patch hierarchy
        std::shared_ptr<SAMRAI::mesh::StandardTagAndInitStrategy> test_object;

        std::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> gridding_algorithm;

        auto samrHierarchy =
            AMP::Mesh::SAMRBuilder::buildHierarchy( input_db, test_object, gridding_algorithm );
        auto hierarchy_adaptor =
            std::dynamic_pointer_cast<AMP::Mesh::SAMRAIHierarchyAdaptor>( samrHierarchy );
        AMP_ASSERT( hierarchy_adaptor );
        auto hierarchy = std::dynamic_pointer_cast<SAMRAI::hier::PatchHierarchy>(
            hierarchy_adaptor->getSAMRAIHierarchy() );
        AMP_ASSERT( hierarchy );
        hierarchy->recursivePrint( AMP::pout, "", 2 );
        auto dim   = samrHierarchy->getDim();
        auto level = samrHierarchy->getPatchLevel( 0 );
        auto shift = level->getPeriodicShift();
        bool pass  = true;
        for ( int d = 0; d < dim; ++d )
            pass = pass && ( shift[d] != 0 );
        if ( pass )
            ut.passes( "All directions periodic" );
        else
            ut.failure( "All directions not periodic" );
    }
    ut.report();

    // Shutdown
    AMP::AMPManager::shutdown();

    return 0;
}
