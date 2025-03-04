#include "AMP/IO/PIO.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/map/dtk/DTKAMPMeshEntityIterator.h"
#include "AMP/operators/map/dtk/DTKAMPMeshEntityLocalMap.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>


bool selectAll( DataTransferKit::Entity entity ) { return true; }

static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testDTKEntityLocalMap" );
    std::string log_file = "output_" + exeName;
    std::string msgPrefix;
    AMP::logOnlyNodeZero( log_file );

    AMP::pout << "Loading the  mesh" << std::endl;

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    std::string input_file = "input_" + exeName;
    auto input_db          = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto meshDatabase = input_db->getDatabase( "Mesh" );

    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( meshDatabase );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    // get the volume iterator
    auto vol_iterator = mesh->getIterator( AMP::Mesh::GeomType::Cell );

    // get the vertex iterator
    auto vert_iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex );

    // map the volume ids to dtk ids
    auto vol_id_map =
        std::make_shared<std::map<AMP::Mesh::MeshElementID, DataTransferKit::EntityId>>();
    {
        int counter = 0;
        for ( vol_iterator = vol_iterator.begin(); vol_iterator != vol_iterator.end();
              ++vol_iterator, ++counter ) {
            vol_id_map->emplace( vol_iterator->globalID(), counter );
        }
        int comm_rank = globalComm.getRank();
        int comm_size = globalComm.getSize();
        std::vector<std::size_t> offsets( comm_size, 0 );
        globalComm.allGather( vol_id_map->size(), offsets.data() );
        for ( int n = 1; n < comm_size; ++n ) {
            offsets[n] += offsets[n - 1];
        }
        if ( comm_rank > 0 ) {
            for ( auto &i : *vol_id_map )
                i.second += offsets[comm_rank - 1];
        }
    }

    // map the vertex ids to dtk ids
    auto vert_id_map =
        std::make_shared<std::map<AMP::Mesh::MeshElementID, DataTransferKit::EntityId>>();
    {
        int counter = 0;
        for ( vert_iterator = vert_iterator.begin(); vert_iterator != vert_iterator.end();
              ++vert_iterator, ++counter ) {
            vert_id_map->emplace( vert_iterator->globalID(), counter );
        }
        int comm_rank = globalComm.getRank();
        int comm_size = globalComm.getSize();
        std::vector<std::size_t> offsets( comm_size, 0 );
        globalComm.allGather( vert_id_map->size(), offsets.data() );
        for ( int n = 1; n < comm_size; ++n ) {
            offsets[n] += offsets[n - 1];
        }
        if ( comm_rank > 0 ) {
            for ( auto &i : *vert_id_map )
                i.second += offsets[comm_rank - 1];
        }
    }

    // make the rank map.
    auto rank_map     = std::make_shared<std::unordered_map<int, int>>();
    auto global_ranks = mesh->getComm().globalRanks();
    int size          = mesh->getComm().getSize();
    for ( int n = 0; n < size; ++n ) {
        rank_map->emplace( global_ranks[n], n );
    }

    auto dtk_iterator =
        AMP::Operator::AMPMeshEntityIterator( rank_map, vol_id_map, vol_iterator, selectAll );

    // Create and test a local map.
    auto dtk_local_map = std::make_shared<AMP::Operator::AMPMeshEntityLocalMap>();

    int num_points = 10;

    double epsilon = 1.0e-12;

    // Test the local map with elements
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dist( -1, 1 );
    for ( dtk_iterator = dtk_iterator.begin(); dtk_iterator != dtk_iterator.end();
          ++dtk_iterator ) {
        // Get the bounding box.
        Teuchos::Tuple<double, 6> element_box;
        dtk_iterator->boundingBox( element_box );

        // Check the measure.
        double measure = ( element_box[3] - element_box[0] ) * ( element_box[4] - element_box[1] ) *
                         ( element_box[5] - element_box[2] );
        AMP_ASSERT( std::abs( measure - dtk_local_map->measure( *dtk_iterator ) ) < epsilon );

        // Check the centroid.
        std::vector<double> gold_centroid( 3 );
        gold_centroid[0] = ( element_box[3] - element_box[0] ) / 2.0 + element_box[0];
        gold_centroid[1] = ( element_box[4] - element_box[1] ) / 2.0 + element_box[1];
        gold_centroid[2] = ( element_box[5] - element_box[2] ) / 2.0 + element_box[2];
        Teuchos::Array<double> centroid( 3 );
        dtk_local_map->centroid( *dtk_iterator, centroid() );
        AMP_ASSERT( std::abs( gold_centroid[0] - centroid[0] ) < epsilon );
        AMP_ASSERT( std::abs( gold_centroid[1] - centroid[1] ) < epsilon );
        AMP_ASSERT( std::abs( gold_centroid[2] - centroid[2] ) < epsilon );

        Teuchos::Array<double> point( 3 );
        Teuchos::Array<double> ref_point( 3 );
        Teuchos::Array<double> phys_point( 3 );
        for ( int n = 0; n < num_points; ++n ) {
            // Create a random point in the neighborhood of the box.
            point[0] = centroid[0] + ( element_box[3] - element_box[0] ) * dist( gen );
            point[1] = centroid[1] + ( element_box[4] - element_box[1] ) * dist( gen );
            point[2] = centroid[2] + ( element_box[5] - element_box[2] ) * dist( gen );

            // Determine if it is in the box.
            bool gold_inclusion =
                ( point[0] >= element_box[0] ) && ( point[0] <= element_box[3] ) &&
                ( point[1] >= element_box[1] ) && ( point[1] <= element_box[4] ) &&
                ( point[2] >= element_box[2] ) && ( point[2] <= element_box[5] );

            // Check safety of mapping to the reference frame.
            bool is_safe = dtk_local_map->isSafeToMapToReferenceFrame( *dtk_iterator, point() );
            AMP_ASSERT( is_safe == gold_inclusion );

            // Map to the reference frame.
            bool map_success =
                dtk_local_map->mapToReferenceFrame( *dtk_iterator, point(), ref_point() );
            AMP_ASSERT( map_success );

            // Check point inclusion.
            bool point_inclusion = dtk_local_map->checkPointInclusion( *dtk_iterator, ref_point() );
            AMP_ASSERT( point_inclusion == gold_inclusion );

            // Check mapping to the physical frame.
            dtk_local_map->mapToPhysicalFrame( *dtk_iterator, ref_point(), phys_point() );
            AMP_ASSERT( std::abs( point[0] - phys_point[0] ) < epsilon );
            AMP_ASSERT( std::abs( point[1] - phys_point[1] ) < epsilon );
            AMP_ASSERT( std::abs( point[2] - phys_point[2] ) < epsilon );
        }
    }

    // Test the local map with nodes.
    auto node_iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex );
    for ( node_iterator = node_iterator.begin(); node_iterator != node_iterator.end();
          ++node_iterator ) {
        auto dtk_node = AMP::Operator::AMPMeshEntity( *node_iterator, *rank_map, *vert_id_map );
        Teuchos::Array<double> centroid( 3 );
        dtk_local_map->centroid( dtk_node, centroid() );
        std::vector<double> coords = node_iterator->coord();
        AMP_ASSERT( coords[0] == centroid[0] );
        AMP_ASSERT( coords[1] == centroid[1] );
        AMP_ASSERT( coords[2] == centroid[2] );
    }

    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
