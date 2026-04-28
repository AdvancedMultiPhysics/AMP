#include "AMP/IO/PIO.h"
#include "AMP/mesh/SAMRAI/SAMRAIHierarchyAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRAILevelAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRBuilder.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/threadpool/ThreadPool.h"

#include "ProfilerApp.h"

#include <string>


template<typename Derived, int dimension, int NC0, int NC1>
struct Expected {
    static constexpr int dim                                                = dimension;
    static constexpr int finest_level                                       = 1;
    static constexpr int number_of_levels                                   = 2;
    static constexpr std::array<int, number_of_levels> number_of_patches    = { 1, 1 };
    static constexpr std::array<unsigned, number_of_levels> number_of_cells = { NC0, NC1 };

    constexpr int ratio_to_level_zero( int l, int d )
    {
        return static_cast<Derived &>( *this ).ratio_to_level_zero_[this->dim * l + d];
    }
};

struct inp_1_hier_1 : Expected<inp_1_hier_1, 3, 64, 64> {
    static constexpr char name[] = "input 1, Hierarchy 1";
    static constexpr std::array<int, number_of_levels * dim> ratio_to_level_zero_ = { 1, 1, 1,
                                                                                      2, 2, 2 };
};

struct inp_1_hier_2 : Expected<inp_1_hier_2, 3, 512, 128> {
    static constexpr char name[] = "input 1, Hierarchy 2";
    static constexpr std::array<int, number_of_levels * dim> ratio_to_level_zero_ = { 1, 1, 1,
                                                                                      2, 1, 1 };
};

struct inp_2_hier_1 : Expected<inp_2_hier_1, 2, 16, 16> {
    static constexpr char name[] = "input 2, Hierarchy 1";
    static constexpr std::array<int, number_of_levels * dim> ratio_to_level_zero_ = { 1, 1, 2, 2 };
};

struct inp_2_hier_2 : Expected<inp_2_hier_2, 3, 512, 128> {
    static constexpr char name[] = "input 2, Hierarchy 2";
    static constexpr std::array<int, number_of_levels * dim> ratio_to_level_zero_ = { 1, 1, 1,
                                                                                      2, 1, 1 };
};

template<typename T, typename U>
auto expected_actual( T expected, U actual )
{
    return ", expected: " + std::to_string( expected ) + " actual: " + std::to_string( actual );
}

template<typename Expected>
void testHierarchy( const std::shared_ptr<AMP::Mesh::SAMRAIHierarchyAdaptor> &h, AMP::UnitTest &ut )
{
    bool result = Expected{}.finest_level == h->getFinestLevelNumber();
    auto name =
        std::string( "testHierarchy, getFinestLevelNumber" ) + std::string( Expected{}.name );
    result ? ut.passes( name ) : ut.failure( name );

    result = Expected{}.number_of_levels == h->getNumberOfLevels();
    name   = std::string( "testHierarchy, getNumberOfLevels" ) + std::string( Expected{}.name );
    result ? ut.passes( name ) : ut.failure( name );

    for ( int ln = 0; ln < h->getNumberOfLevels(); ++ln ) {
        auto level         = h->getPatchLevel( ln );
        auto level_adaptor = std::dynamic_pointer_cast<AMP::Mesh::SAMRAILevelAdaptor>( level );

        result = Expected{}.number_of_patches[ln] == level_adaptor->getLocalNumberOfPatches();
        name   = std::string( "testHierarchy, getLocalNumberOfPatches, level " ) +
               std::to_string( ln ) + std::string( " , " ) + std::string( Expected{}.name );
        result ? ut.passes( name ) : ut.failure( name );

        result = Expected{}.number_of_cells[ln] == level_adaptor->getGlobalNumberOfCells();
        name   = std::string( "testHierarchy, getGlobalNumberOfCells, level " ) +
               std::to_string( ln ) + std::string( " , " ) + std::string( Expected{}.name );
        result ? ut.passes( name ) : ut.failure( name );

        auto ratio = level_adaptor->getRatioToLevelZero();
        for ( int d = 0; d < Expected{}.dim; ++d ) {
            result = Expected{}.ratio_to_level_zero( ln, d ) == ratio[d];
            name   = std::string( "testHierarchy, getRatioToLevelZero, level " ) +
                   std::to_string( ln ) + std::string( " , " ) + std::string( Expected{}.name );
            result ? ut.passes( name ) : ut.failure( name );
        }
    }
}

// The test below creates two separate hierarchies on two comms
// that live on different sets of processors
int main( int argc, char *argv[] )
{
    // Startup
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // This extra code block is used to scope some temporaries that are
    // created, it forces the destruction before the manager is shutdown.
    {

        // Process command line arguments and dump to log file.
        std::string inputFile;
        std::string logFile;

        AMP::Mesh::SAMRBuilder::processCommandLine( argc, argv, inputFile, logFile );

        auto world_comm   = AMP::AMP_MPI( AMP_COMM_WORLD );
        int world_rank    = world_comm.getRank();
        int world_size    = world_comm.getSize();
        int color         = world_rank / ( world_size / 2 );
        auto sub_comm     = world_comm.split( color, world_rank );
        int sub_comm_size = sub_comm.getSize();

        auto ampInputDatabase = AMP::Database::parseInputFile( inputFile );
        ampInputDatabase->print( AMP::plog );

        // Create the patch hierarchy
        auto hierarchy_str = ( color ) ? "SecondHierarchy" : "FirstHierarchy";
        auto hierarchyDB   = ampInputDatabase->getDatabase( hierarchy_str );
        auto hierarchy     = AMP::Mesh::SAMRBuilder::buildHierarchy( hierarchyDB, 0, sub_comm );
        auto hierarchy_adaptor =
            std::dynamic_pointer_cast<AMP::Mesh::SAMRAIHierarchyAdaptor>( hierarchy );
        AMP_ASSERT( hierarchy_adaptor );

        if ( color == 0 && sub_comm_size == 1 ) {
            // input file 1; Hierarchy 1
            testHierarchy<inp_1_hier_1>( hierarchy_adaptor, ut );
        } else if ( color == 1 && sub_comm_size == 1 ) {
            // input file 1; Hierarchy 2
            testHierarchy<inp_1_hier_2>( hierarchy_adaptor, ut );
        } else if ( color == 0 && sub_comm_size == 2 && world_rank != 1 ) {
            // input file 2; Hierarchy 1
            testHierarchy<inp_2_hier_1>( hierarchy_adaptor, ut );
        } else if ( color == 1 && sub_comm_size == 2 && world_rank != 3 ) {
            // input file 2; Hierarchy 2
            testHierarchy<inp_2_hier_2>( hierarchy_adaptor, ut );
        }

    } // End code block

    // That's all, folks!
    //    mpi.Barrier();
    int N_errors = ut.NumFailGlobal();
    ut.report();

    // Shutdown
    AMP::AMPManager::shutdown();
    return N_errors;
}
