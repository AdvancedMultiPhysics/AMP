#include "AMP/mesh/SAMRAI/SAMRBuilder.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/mesh/SAMRAI/BogusTagAndInitStrategy.h"
#include "AMP/mesh/SAMRAI/SAMRAIHierarchyAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRAILoadBalanceFactory.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"

#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/hier/Box.h"
#include "SAMRAI/hier/BoxContainer.h"
#include "SAMRAI/hier/IntVector.h"
#include "SAMRAI/hier/Patch.h"
#include "SAMRAI/hier/PatchGeometry.h"
#include "SAMRAI/hier/PatchHierarchy.h"
#include "SAMRAI/hier/PatchLevel.h"
#include "SAMRAI/mesh/BergerRigoutsos.h"
#include "SAMRAI/mesh/CascadePartitioner.h"
#include "SAMRAI/mesh/GriddingAlgorithm.h"
#include "SAMRAI/mesh/StandardTagAndInitialize.h"
#include "SAMRAI/mesh/TileClustering.h"
#include "SAMRAI/tbox/Database.h"
#include "SAMRAI/tbox/Dimension.h"
#include "SAMRAI/tbox/SAMRAI_MPI.h"
#include "SAMRAI/tbox/Utilities.h"

// System include
#include <string>


namespace AMP::Mesh {


std::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> SAMRBuilder::buildGriddingAlgorithm(
    const std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy,
    const std::shared_ptr<AMP::Database> amp_db,
    std::shared_ptr<SAMRAI::mesh::StandardTagAndInitStrategy> &test_object )
{
    auto db = amp_db->cloneToSAMRAI();
    AMP_ASSERT( db );
    const SAMRAI::tbox::Dimension dim( static_cast<unsigned short>( db->getInteger( "dim" ) ) );

    const auto print_info_level = db->getIntegerWithDefault( "print_info_level", 0 );

    // Get naming suffix
    std::string suffix = "";
    if ( db->keyExists( "suffix" ) )
        suffix = "_" + db->getString( "suffix" );

    std::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> gridding_algorithm;
    // Create an application object (if necessary)
    if ( test_object.get() == nullptr )
        test_object = std::make_shared<BogusTagAndInitStrategy>();

    // Classes for error detection, box_generation, and load balancing are needed to build the
    // mesh::GriddingAlgorithm.
    auto error_detector = std::make_shared<SAMRAI::mesh::StandardTagAndInitialize>(
        "StandardTagAndInitialize",
        test_object.get(),
        db->getDatabase( "StandardTagAndInitialize" ) );

    // Load the box generator
    std::string box_generator_name;
    if ( db->keyExists( "BoxGeneratorStrategy" ) )
        box_generator_name = db->getString( "BoxGeneratorStrategy" );
    else if ( db->keyExists( "BergerRigoutsos" ) )
        box_generator_name = "BergerRigoutsos";
    else if ( db->keyExists( "TileClustering" ) )
        box_generator_name = "TileClustering";
    else
        box_generator_name = "BergerRigoutsos";
    std::shared_ptr<SAMRAI::tbox::Database> box_generator_db;
    if ( db->keyExists( box_generator_name ) )
        box_generator_db = db->getDatabase( box_generator_name );
    std::shared_ptr<SAMRAI::mesh::BoxGeneratorStrategy> box_generator;
    if ( box_generator_name == "BergerRigoutsos" ) {
        if ( print_info_level > 0 )
            AMP::pout << "BergerRigoutsos clustering being used for box generation " << std::endl;
        if ( box_generator_db )
            box_generator =
                std::make_shared<SAMRAI::mesh::BergerRigoutsos>( dim, box_generator_db );
        else
            box_generator = std::make_shared<SAMRAI::mesh::BergerRigoutsos>( dim );
    } else if ( box_generator_name == "TileClustering" ) {
        if ( print_info_level > 0 )
            AMP::pout << "Tile clustering being used for box generation " << std::endl;
        box_generator = std::make_shared<SAMRAI::mesh::TileClustering>( dim, box_generator_db );
    } else {
        AMP_ERROR( "Unknown box generator (" + box_generator_name + ")" );
    }

    // Load the load balancer
    std::string load_balancer_name;
    if ( db->keyExists( "LoadBalanceStrategy" ) )
        load_balancer_name = db->getString( "LoadBalanceStrategy" );
    else if ( db->keyExists( "CascadePartitioner" ) )
        load_balancer_name = "CascadePartitioner";
    else if ( db->keyExists( "TreeLoadBalancer" ) )
        load_balancer_name = "TreeLoadBalancer";
    else if ( db->keyExists( "ChopAndPackLoadBalancer" ) )
        load_balancer_name = "ChopAndPackLoadBalancer";
    else if ( db->keyExists( "DisjointCascadeBalancer" ) )
        load_balancer_name = "DisjointCascadeBalancer";
    else if ( db->keyExists( "SimpleLoadBalancer" ) )
        load_balancer_name = "SimpleLoadBalancer";
    else
        AMP_ERROR( "Unable to determine the load balancer" );
    std::shared_ptr<SAMRAI::tbox::Database> load_balancer_db;
    if ( db->keyExists( load_balancer_name ) )
        load_balancer_db = db->getDatabase( load_balancer_name );
    if ( print_info_level > 0 )
        AMP::pout << load_balancer_name << " being used for load balance" << std::endl;
    std::shared_ptr<SAMRAI::mesh::LoadBalanceStrategy> load_balancer =
        SAMRAILoadBalanceFactory::create( dim, load_balancer_name, load_balancer_db );
    AMP_ASSERT( load_balancer );
    if ( load_balancer_db )
        load_balancer_db->printClassData( AMP::plog );

    // Create the gridding algorithm
    if ( db->keyExists( "GriddingAlgorithm" ) ) {
        gridding_algorithm = std::make_shared<SAMRAI::mesh::GriddingAlgorithm>(
            hierarchy,
            "GriddingAlgorithm" + suffix,
            db->getDatabase( "GriddingAlgorithm" ),
            error_detector,
            box_generator,
            load_balancer );

    } else
        AMP_ERROR( "Unable to determine the gridding algorithm" );

    return gridding_algorithm;
}

std::shared_ptr<AMP::Mesh::SAMRAIHierarchyAdaptor> SAMRBuilder::buildHierarchy(
    std::shared_ptr<AMP::Database> db, int max_gcw, const AMP::AMP_MPI &mpi )
{
    AMP_ASSERT( db );
    bool restrict_gcw_domain = db->getWithDefault<bool>( "restrict_gcw_domain", true );
    std::shared_ptr<SAMRAI::mesh::StandardTagAndInitStrategy> test_object       = nullptr;
    std::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> gridding_algorithm = nullptr;
    return buildHierarchy( db, test_object, gridding_algorithm, max_gcw, restrict_gcw_domain, mpi );
}

std::shared_ptr<AMP::Mesh::SAMRAIHierarchyAdaptor> SAMRBuilder::buildHierarchy(
    std::shared_ptr<AMP::Database> amp_db,
    std::shared_ptr<SAMRAI::mesh::StandardTagAndInitStrategy> &test_object,
    std::shared_ptr<SAMRAI::mesh::GriddingAlgorithmStrategy> &gridding_algorithm,
    int max_gcw_in,
    bool restrict_gcw_domain,
    const AMP::AMP_MPI &mpi )
{
    AMP_ASSERT( amp_db );
    auto db = amp_db->cloneToSAMRAI();
    AMP_ASSERT( db );
    const SAMRAI::tbox::Dimension dim( static_cast<unsigned short>( db->getInteger( "dim" ) ) );

    // Get naming suffix
    std::string suffix = "";
    if ( db->keyExists( "suffix" ) )
        suffix = "_" + db->getString( "suffix" );
    // Create geometry object.
    auto grid_geometry = std::make_shared<SAMRAI::geom::CartesianGridGeometry>(
        dim, "CartesianGeometry" + suffix, db->getDatabase( "CartesianGeometry" ) );

    // Create patch hierarchy
    auto samrai_mpi = static_cast<SAMRAI::tbox::SAMRAI_MPI>( mpi );

    auto hierarchy = std::make_shared<SAMRAI::hier::PatchHierarchy>(
        "PatchHierarchy" + suffix, grid_geometry, db->getDatabase( "PatchHierarchy" ), samrai_mpi );

    // Set a minimum value for the ghost width to be 2 or the size of the domain
    SAMRAI::hier::IntVector max_gcw   = SAMRAI::hier::IntVector( dim, max_gcw_in );
    SAMRAI::hier::BoxContainer domain = grid_geometry->getPhysicalDomain();
    if ( restrict_gcw_domain ) {
        for ( auto it = domain.begin(); it != domain.end(); ++it )
            max_gcw.min( it->numberCells() );
    }
    hierarchy->getPatchDescriptor()->setMinGhostWidth( max_gcw );

    // Create the gridding_algorithm (if necessary)
    if ( gridding_algorithm.get() == nullptr ) {
        gridding_algorithm = buildGriddingAlgorithm( hierarchy, amp_db, test_object );
    } else if ( test_object.get() == nullptr ) {
        // Return the StandardTagAndInitStrategy if the gridding algorthm is provided and the
        // StandardTagAndInitStrategy is not
        test_object = std::dynamic_pointer_cast<SAMRAI::mesh::StandardTagAndInitStrategy>(
            gridding_algorithm->getTagAndInitializeStrategy() );
    }

    bool manually_create_levels = false;
    std::shared_ptr<SAMRAI::tbox::Database> box_db;
    if ( db->keyExists( "ManualBoxes" ) ) {
        box_db                 = db->getDatabase( "ManualBoxes" );
        manually_create_levels = box_db->getBoolWithDefault( "manually_create_levels", false );
    }
    if ( manually_create_levels ) {
        const int rank = mpi.getRank();
        const int size = mpi.getSize();
        auto ratio     = SAMRAI::hier::IntVector::getOne( hierarchy->getDim() );
        auto ratio_db  = db->getDatabase( "PatchHierarchy" )->getDatabase( "ratio_to_coarser" );
        for ( int ln = 0; ln < db->getDatabase( "PatchHierarchy" )->getInteger( "max_levels" );
              ln++ ) {
            auto level_name = Utilities::stringf( "level_%i", ln );
            if ( ln > 0 ) {
                auto r = ratio_db->getIntegerVector( level_name );
                for ( size_t d = 0; d < r.size(); d++ )
                    ratio( d ) *= r[d];
            }
            // Load the database
            auto box_data  = box_db->getDatabase( level_name )->getDatabaseBoxVector( "boxes" );
            auto rank_data = box_db->getDatabase( level_name )->getIntegerVector( "rank" );
            AMP_ASSERT( box_data.size() == rank_data.size() );
            // Create the boxes
            SAMRAI::hier::BoxContainer boxes;
            for ( size_t i = 0; i < box_data.size(); i++ ) {
                AMP_ASSERT( rank_data[i] < size );
                if ( rank_data[i] == rank ) {
                    SAMRAI::hier::Box box( box_data[i] );
                    box.setBlockId( SAMRAI::hier::BlockId( 0 ) );
                    box.setId( SAMRAI::hier::BoxId( SAMRAI::hier::LocalId( i ), rank ) );
                    boxes.insert( box );
                }
            }
            auto geom = grid_geometry->makeRefinedGridGeometry( level_name, ratio );
            SAMRAI::hier::BoxLevel boxlevel(
                boxes, ratio, grid_geometry, samrai_mpi, SAMRAI::hier::BoxLevel::DISTRIBUTED );
            boxlevel.finalize();
            boxlevel.cacheGlobalReducedData();
            boxlevel.setParallelState( SAMRAI::hier::BoxLevel::GLOBALIZED );
            SAMRAI::hier::BoxLevelConnectorUtils blcu;
            blcu.addPeriodicImages( boxlevel,
                                    hierarchy->getGridGeometry()->getDomainSearchTree(),
                                    SAMRAI::hier::IntVector::max(
                                        SAMRAI::hier::IntVector::getOne( dim ),
                                        hierarchy->getRequiredConnectorWidth( 0, 0, true ) ) );
            hierarchy->makeNewPatchLevel( ln, boxlevel );
        }
    } else {
        // Create the coarse level
        double t0 = 0.0; // dummy time
        gridding_algorithm->makeCoarsestLevel( t0 );
        // Create the fine levels
        for ( int ln = 0; hierarchy->levelCanBeRefined( ln ); ln++ ) {
            gridding_algorithm->makeFinerLevel( 0, true, 0, t0 );
        }
    }
    // Return a wrapper to the created hierarchy
    return std::make_shared<AMP::Mesh::SAMRAIHierarchyAdaptor>( hierarchy );
}


static std::string replace( const std::string &str, const std::string &src, const std::string &dst )
{
    auto str2 = str;
    size_t i  = str2.find( src );
    while ( i != std::string::npos ) {
        str2.replace( i, sizeof( src ), dst );
        i = str2.find( src );
    }
    return str2;
}
void SAMRBuilder::processCommandLine( int argc,
                                      char *argv[],
                                      std::string &input_file,
                                      std::string &log_file )
{
    if ( argc == 2 ) {
        input_file = argv[1];
        log_file   = AMP::IO::filename( argv[0] ) + "." + AMP::IO::filename( argv[1] );
        int size   = AMP::AMP_MPI( AMP_COMM_WORLD ).getSize();
        if ( size > 1 )
            log_file += ".p" + std::to_string( size );
        log_file = replace( log_file, "input_", "" );
        log_file = replace( log_file, "input.", "" );
        log_file += ".log";
    } else if ( argc == 3 ) {
        input_file = argv[1];
        log_file   = argv[2];
    } else {
        AMP::pout << "USAGE:  " << argv[0] << " <input file> <log file> " << std::endl;
        exit( -1 );
    }
}


} // namespace AMP::Mesh
