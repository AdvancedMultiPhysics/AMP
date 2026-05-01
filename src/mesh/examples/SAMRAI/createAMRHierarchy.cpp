// Test creation of AMR hierarchies
#include "AMP/IO/PIO.h"
#include "AMP/mesh/SAMRAI/SAMRAIHierarchyAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRAILevelAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRAIPatchAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRBuilder.h"
#include "AMP/mesh/SAMRAI/SAMRHierarchy.h"
#include "AMP/utils/AMPManager.h"

#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/hier/Patch.h"
#include "SAMRAI/geom/CartesianPatchGeometry.h"

#include <string>
#include <vector>


// Demonstrate traversing an AMR hierarchy using the hierarchy,
// level, and patch adaptors that SAMRSolvers provides. Extract the
// individual SAMRAI patch data structure and print out info.
// This is the preferred approach when detailed hierarchy or
// level information is not required.
void visitHierarchy( std::shared_ptr<AMP::Mesh::SAMRHierarchy> samrHierarchy )
{
    // iterate over all refinement levels
    AMP::plog << "Local Hierarchy Information" << std::endl;
    for ( int ln = 0; ln < samrHierarchy->getNumberOfLevels(); ++ln ) {

        AMP::plog << "==========================================" << std::endl;
        AMP::plog << "Local Refinement Level: " << ln << std::endl;
        const auto &samrLevel = samrHierarchy->getPatchLevel( ln );
        // iterate over all patches on a refinement level
        int pn = 0;
        for ( const auto &samrPatch : *samrLevel ) {

            // extract a SAMRAI patch from the general SAMRPatch object
            auto samrPatchAdaptor =
                std::dynamic_pointer_cast<AMP::Mesh::SAMRAIPatchAdaptor>( samrPatch );
            AMP_ASSERT( samrPatchAdaptor );
            auto patch = samrPatchAdaptor->getSAMRAIPatch();
            AMP_ASSERT( patch );
            AMP::plog << "----------" << std::endl;
            AMP::plog << "Info for Patch " << pn << std::endl;
            AMP::plog << "----------" << std::endl;
            AMP::plog << "GlobalId info: " << patch->getGlobalId() << std::endl;
            AMP::plog << "Local Id info: " << patch->getLocalId() << std::endl;
            AMP::plog << "Patch refinement level: " << patch->getPatchLevelNumber() << std::endl;
            AMP::plog << "Dimension: " << patch->getDim() << std::endl;
            AMP::plog << "Number of data objects: " << patch->numPatchData() << std::endl;
            AMP::plog << "--------------" << std::endl;
            AMP::plog << "Box Info for Patch " << pn << std::endl;
            AMP::plog << "--------------" << std::endl;
            AMP::plog << patch->getBox() << std::endl;
            ;
            AMP::plog << "-------------------" << std::endl;
            AMP::plog << "Geometry Info for Patch " << pn << std::endl;
            AMP::plog << "-------------------" << std::endl;
            // for now assume its a Cartesian geometry
            auto geometry = std::dynamic_pointer_cast<SAMRAI::geom::CartesianPatchGeometry>(
                patch->getPatchGeometry() );
            AMP_ASSERT( geometry );
            geometry->printClassData( AMP::plog );
            ++pn;
        }
    }
}

int main( int argc, char *argv[] )
{

    // startup initializes MPI and all packages
    AMP::AMPManager::startup( argc, argv );

    // This extra code block is used to scope some temporaries that are
    // created, it forces the destruction before the manager is shutdown.
    {

        std::string inputFile;
        std::string logFile;

        if ( argc > 1 ) {
            AMP::Mesh::SAMRBuilder::processCommandLine( argc, argv, inputFile, logFile );
        } else {
            inputFile = "input_HierarchyExample_16b4l";
            logFile   = "log_HierarchyExample_16b4l";
        }

        AMP::logAllNodes( logFile );

        // read the entire input file into a single AMP database
        // a database can recursively contain databases or scalars (bool, int, float, double,
        // string) or vectors of scalars
        auto inputDatabase = AMP::Database::parseInputFile( inputFile );

        // print the entire database into a log stream (optional)
        AMP::plog << inputFile << " Database log " << std::endl;
        inputDatabase->print( AMP::plog );
        AMP::plog << "===========================================" << std::endl;

        // the maximum ghost cell width for data that will be created on the AMR hierarchy
        const int max_gcw = 1;

        // here we choose to duplicate the global AMP communicator and create
        // a separate communicator for SAMRAI. We could choose to do a split
        // of an AMP communicator over a subset of ranks also.
        auto globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );
        auto dupComm    = globalComm.dup();

        // create an AMR hierarchy. At present only SAMRAI hierarchies can
        // be created. A more detailed interface that also takes in StandardTagAndInitStrategy
        // and GriddingAlgorithmStrategy exists in SAMRBuiulder but it is outside the scope
        // of this example
        auto samrHierarchy =
            AMP::Mesh::SAMRBuilder::buildHierarchy( inputDatabase, max_gcw, dupComm );

        // check we were able to to successfully create a hierarchy
        AMP_ASSERT( samrHierarchy );

        visitHierarchy( samrHierarchy );
    }

    // shutdown cleanly shuts down individual packages
    AMP::AMPManager::shutdown();
    return 0;
}
