#include "AMP/IO/PIO.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"

#include <algorithm>
#include <functional>
#include <unordered_map>


union id_mask {
    uint64_t id_i[2];
    char id_c[sizeof( uint64_t[2] )];
};

size_t hash_id( const AMP::Mesh::MeshElementID &id )
{
    id_mask m;

    // get the first 64 bits from the mesh id
    m.id_i[0] = id.meshID().getData();

    // construct the next 64 from the element id.
    unsigned int tmp = 0x00000000;
    if ( id.is_local() )
        tmp = 0x80000000;
    tmp += ( 0x007FFFFF & id.owner_rank() ) << 8;
    auto type = (char) id.type();
    tmp += ( (unsigned char) type );
    m.id_i[1] = ( ( (uint64_t) tmp ) << 32 ) + ( (uint64_t) id.local_id() );

    // hash the id
    std::hash<std::string> hasher;
    return hasher( std::string( m.id_c, sizeof( uint64_t[2] ) ) );
}

void makeCommRankMap( const AMP::AMP_MPI &comm, std::unordered_map<int, int> &rank_map )
{
    auto global_ranks = comm.globalRanks();
    int size          = comm.getSize();
    for ( int n = 0; n < size; ++n ) {
        rank_map.emplace( global_ranks[n], n );
    }
}

int mapElementOwnerRank( const AMP::Mesh::Mesh &mesh,
                         const std::unordered_map<int, int> &rank_map,
                         const AMP::Mesh::MeshElement &element )
{
    return rank_map.find( element.globalOwnerRank( mesh ) )->second;
}

void testMultiMeshOwnerRank( AMP::UnitTest &ut )
{
    std::string const exeName   = "testMultiMeshOwnerRank";
    std::string const inputFile = "input_" + exeName;
    std::string const logFile   = "output_" + exeName;

    AMP::logAllNodes( logFile );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Parse the input File
    auto inputDatabase = AMP::Database::parseInputFile( inputFile );

    // Read the mesh database
    auto meshDatabase = inputDatabase->getDatabase( "Mesh" );

    // Build the mesh
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( meshDatabase );
    meshParams->setComm( globalComm );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    // Subset the mesh boundary on surface 0
    auto arrayMesh = mesh->Subset( "Mesh1" );
    std::shared_ptr<AMP::Mesh::Mesh> arrayBoundaryMesh;
    if ( arrayMesh ) {
        auto it           = arrayMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 0 );
        arrayBoundaryMesh = arrayMesh->Subset( it );
    }

    bool failure = false;
    if ( arrayBoundaryMesh ) {
        // Create the rank mapping.
        std::unordered_map<int, int> rank_map;
        makeCommRankMap( arrayBoundaryMesh->getComm(), rank_map );

        // Iterate through the vertices on the boundaries of array and see if the
        // ranks are correct.
        int bnd_comm_rank            = arrayBoundaryMesh->getComm().getRank();
        bool owner_rank_is_comm_rank = true;
        auto it       = arrayBoundaryMesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
        auto it_begin = it.begin();
        auto it_end   = it.end();
        for ( it = it_begin; it != it_end; ++it ) {
            // If the owner rank of the vertex is the same as the comm rank of
            // the boundary mesh that we got the vertex from then it should be
            // locally owned.
            owner_rank_is_comm_rank =
                ( bnd_comm_rank == mapElementOwnerRank( *arrayBoundaryMesh, rank_map, *it ) );

            // If the vertex thinks it is locally owned but its owner rank
            // and mesh comm rank dont match then this is a failure.
            failure = ( owner_rank_is_comm_rank != it->globalID().is_local() );

            // Exit the for loop on failure.
            if ( failure )
                break;
        }
    }

    // Return pass/fail.
    if ( !failure ) {
        ut.passes( "Owner ranks are correct" );
    } else {
        ut.failure( "Owner ranks failed" );
    }

    // Do a reduction to make sure we only get one instance of locally owned elements.
    std::vector<size_t> local_ids( 0 );
    if ( arrayBoundaryMesh ) {
        auto it         = arrayMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Cell, 0 );
        auto volBndMesh = arrayMesh->Subset( it );
        it              = volBndMesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
        auto it_begin   = it.begin();
        auto it_end     = it.end();
        for ( it = it_begin; it != it_end; ++it ) {
            local_ids.push_back( hash_id( it->globalID() ) );
        }
    }
    auto global_ids = globalComm.allGather( local_ids );
    failure         = false;
    for ( auto i : global_ids ) {
        auto count = std::count( global_ids.begin(), global_ids.end(), i );
        if ( count > 1 ) {
            failure = true;
            break;
        }
    }

    // Return pass/fail.
    if ( !failure ) {
        ut.passes( "Global IDs are correct" );
    } else {
        ut.failure( "Repeated global ids" );
    }
}

// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    // Run the MultiMesh subest test
    testMultiMeshOwnerRank( ut );

    // Print the results and return
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
