#include "AMP/ampmesh/testHelpers/meshTests.h"
#include "AMP/ampmesh/Mesh.h"
#include "AMP/ampmesh/MeshElement.h"
#include "AMP/ampmesh/MeshElementVectorIterator.h"
#include "AMP/ampmesh/MeshIterator.h"
#include "AMP/ampmesh/MultiGeometry.h"
#include "AMP/ampmesh/MultiMesh.h"
#include "AMP/ampmesh/SubsetMesh.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#ifdef USE_AMP_VECTORS
#include "AMP/vectors/Vector.h"
#endif

#include <set>
#include <vector>

namespace AMP {
namespace Mesh {


// Some global variables
AMP::Mesh::Mesh::shared_ptr meshTests::globalMeshForMeshVectorFactory =
    AMP::Mesh::Mesh::shared_ptr();


// Helper function to create a map from the base mesh communicator rank to the main mesh
// communicator
std::map<AMP::Mesh::MeshID, std::vector<int>>
meshTests::createRankMap( AMP::Mesh::Mesh::shared_ptr mesh )
{
    std::map<AMP::Mesh::MeshID, std::vector<int>> proc_map;
    auto meshIDs = mesh->getBaseMeshIDs();
    std::vector<std::pair<int, int>> tmp( mesh->getComm().getSize() );
    for ( auto &meshID : meshIDs ) {
        auto mesh2 = mesh->Subset( meshID );
        int N_send = 0;
        std::pair<int, int> map;
        if ( mesh2 ) {
            map    = std::pair<int, int>( mesh2->getComm().getRank(), mesh->getComm().getRank() );
            N_send = 1;
        }
        int N = mesh->getComm().allGather( &map, N_send, &tmp[0] );
        std::vector<int> rank( N );
        for ( int j = 0; j < N; j++ )
            rank[tmp[j].first] = tmp[j].second;
        proc_map.insert( std::pair<AMP::Mesh::MeshID, std::vector<int>>( meshID, rank ) );
    }
    return proc_map;
}


// This test checks a single mesh element iterator
void meshTests::ElementIteratorTest( AMP::UnitTest &ut,
                                     AMP::Mesh::Mesh::shared_ptr mesh,
                                     const AMP::Mesh::MeshIterator &iterator,
                                     const size_t N_local,
                                     const size_t N_ghost,
                                     const AMP::Mesh::GeomType type )
{
    auto name = mesh->getName();
    // For each mesh, get a mapping of it's processor id's to the comm of the mesh
    auto proc_map = createRankMap( mesh );
    // Check that we can get the begin and end iterator
    auto begin_it = iterator.begin();
    auto end_it   = iterator.end();
    if ( N_local + N_ghost == 0 ) {
        if ( begin_it == end_it )
            ut.passes( name + "-trival iterator begin and end returned" );
        else
            ut.failure( name + "-trival iterator begin and end returned" );
        return;
    }
    if ( begin_it != end_it )
        ut.passes( name + "-iterator begin and end returned" );
    else
        ut.failure( name + "-iterator begin and end returned" );

    // Check that the iterator iterates through the proper number of elements
    if ( iterator.size() == N_local + N_ghost )
        ut.passes( name + "-regular iterator size()" );
    else
        ut.failure( name + "-regular iterator size()" );
    size_t number_of_local_elements = 0;
    size_t number_of_ghost_elements = 0;
    std::set<AMP::Mesh::MeshElementID> id_set;
    bool pass_position = true;
    {
        auto it = iterator.begin();
        for ( size_t i = 0; i < iterator.size(); i++, ++it ) {
            if ( it.position() != i )
                pass_position = false;
            AMP::Mesh::MeshElementID id = it->globalID();
            id_set.insert( id );
            if ( id.is_local() )
                number_of_local_elements++;
            else
                number_of_ghost_elements++;
        }
    }
    if ( pass_position )
        ut.passes( name + "-iterator.position()" );
    else
        ut.failure( name + "-iterator.position()" );
    if ( number_of_local_elements == N_local && number_of_ghost_elements == N_ghost )
        ut.passes( name + "-regular iterator count" );
    else
        ut.failure( name + "-regular iterator count" );
    if ( id_set.size() == N_local + N_ghost )
        ut.passes( name + "-regular iterator uniqueness" );
    else
        ut.failure( name + "-regular iterator uniqueness" );

    // Check that we can increment and decrement properly
    if ( iterator.size() >= 2 ) {
        bool pass = true;
        auto it1  = iterator.begin();
        auto it2  = iterator.begin();
        auto it3  = iterator.begin();
        auto it4  = iterator.begin();
        auto it5  = iterator.begin();
        it1++;
        ++it1;
        it2 = it2 + 2;
        it3 += 2;
        it4 += 0;
        it5 += iterator.size();
        if ( it1 != it2 || it1 != it3 || it4 != iterator.begin() || it5 != iterator.end() )
            pass = false;
        if ( static_cast<int>( it1.type() ) >= 2 ) {
            it1--;
            --it1;
            it2 = it2 - 2;
            it3 -= 2;
            if ( it1 != iterator.begin() || it2 != iterator.begin() || it3 != iterator.begin() )
                pass = false;
        }
        if ( pass )
            ut.passes( name + "-regular iterator increments/decrements" );
        else
            ut.failure( name + "-regular iterator increments/decrements" );
    }

    // Run element tests
    bool id_pass       = true;
    bool type_pass     = true;
    bool volume_pass   = true;
    bool coord_pass    = true;
    bool centroid_pass = true;
    bool elements_pass = true;
    bool block_pass    = true;
    int neighbor_pass  = 1;
    int myRank         = mesh->getComm().getRank();
    int maxRank        = mesh->getComm().getSize() - 1;
    int myGlobalRank   = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    auto blockIds      = mesh->getBlockIDs();
    std::vector<AMP::Mesh::MeshElementID> ids;
    ids.reserve( iterator.size() );
    for ( const auto &element : iterator ) {
        // Get the current id
        auto id = element.globalID();
        ids.push_back( id );
        if ( id != element.globalID() )
            id_pass = false;
        // Get the owner rank
        auto meshID         = id.meshID();
        const auto &map     = proc_map.find( meshID )->second;
        int ownerRank       = map[id.owner_rank()];
        int globalOwnerRank = element.globalOwnerRank();
        // Perform some simple checks
        if ( element.elementType() != type )
            type_pass = false;
        if ( type == AMP::Mesh::GeomType::Vertex ) {
            auto coord = element.coord();
            if ( coord.size() != mesh->getDim() )
                coord_pass = false;
        } else {
            if ( element.volume() <= 0.0 )
                volume_pass = false;
        }
        if ( id.type() == AMP::Mesh::GeomType::Volume ) {
            bool in_a_block = false;
            for ( int blockId : blockIds ) {
                if ( element.isInBlock( blockId ) )
                    in_a_block = true;
            }
            if ( !in_a_block )
                block_pass = false;
        }
        auto centroid = element.centroid();
        if ( centroid.size() != mesh->getDim() )
            centroid_pass = false;
        if ( type == AMP::Mesh::GeomType::Vertex ) {
            auto coord = element.coord();
            for ( size_t i = 0; i < centroid.size(); i++ ) {
                if ( centroid[i] != coord[i] )
                    centroid_pass = false;
            }
        }
        if ( id.is_local() ) {
            for ( int t2 = 0; t2 <= (int) type; t2++ ) {
                auto type2  = static_cast<AMP::Mesh::GeomType>( t2 );
                auto pieces = element.getElements( type2 );
                ids.resize( pieces.size() );
                for ( size_t j = 0; j < pieces.size(); j++ )
                    ids[j] = pieces[j].globalID();
                AMP::Utilities::unique( ids );
                if ( pieces.empty() || pieces.size() != ids.size() ) {
                    pieces        = element.getElements( type2 );
                    elements_pass = false;
                }
            }
            auto neighbors = element.getNeighbors();
            if ( neighbors.empty() ) {
                if ( element.elementType() == AMP::Mesh::GeomType::Vertex ||
                     static_cast<int>( element.elementType() ) == mesh->getDim() )
                    neighbor_pass = 0; // All nodes / elements should have neighbors
                else if ( neighbor_pass == 1 )
                    neighbor_pass = 2; // Neighbors of other element types are not always supported
            } else {
                for ( auto neighbor : neighbors ) {
                    if ( neighbor ) {
                        // Verify that the neighbors does not include self
                        if ( *neighbor == element )
                            neighbor_pass = 0;
                    } else {
                        // Neighbor is empty
                        neighbor_pass = 0;
                    }
                }
            }
            if ( ownerRank != myRank )
                id_pass = false;
            if ( globalOwnerRank != myGlobalRank )
                id_pass = false;
        } else {
            if ( ownerRank > maxRank || ownerRank < 0 || ownerRank == myRank )
                id_pass = false;
            if ( globalOwnerRank < 0 || globalOwnerRank == myGlobalRank )
                id_pass = false;
        }
    }
    if ( id_pass && type_pass && volume_pass && coord_pass && elements_pass && neighbor_pass == 1 &&
         block_pass ) {
        ut.passes( name + "-elements passed" );
    } else {
        if ( !id_pass )
            ut.failure( name + "-elements failed id test" );
        if ( !type_pass )
            ut.failure( name + "-elements failed type test" );
        if ( !volume_pass )
            ut.failure( name + "-elements failed volume test" );
        if ( !coord_pass )
            ut.failure( name + "-elements failed coord test" );
        if ( !centroid_pass )
            ut.failure( name + "-elements failed centroid test" );
        if ( !elements_pass )
            ut.failure( name + "-elements failed getElements test" );
        if ( neighbor_pass == 0 )
            ut.failure( name + "-elements failed getNeighbors test" );
        else if ( neighbor_pass == 2 )
            ut.expected_failure( name + "-elements failed getNeighbors test" );
        if ( !block_pass )
            ut.failure( name + "-Element is not in a block" );
    }
    // Check that we can get the element from the global id for all elements
    bool getElem_pass = true;
    for ( const auto &element : iterator ) {
        auto id1  = element.globalID();
        auto elem = mesh->getElement( id1 );
        auto id2  = elem.globalID();
        if ( id1 != id2 )
            getElem_pass = false;
    }
    if ( getElem_pass )
        ut.passes( name + "-Got elements from element ids" );
    else
        ut.failure( name + "-Got elements from element ids" );
    // Check nearest
    try {
        bool pass = true;
        for ( const auto &element : iterator ) {
            auto centroid = element.centroid();
            pass          = pass && element.nearest( centroid ) == centroid;
        }
        if ( pass )
            ut.passes( name + "-elements nearest passed" );
        else
            ut.passes( name + "-elements nearest failed" );
    } catch ( const StackTrace::abort_error &e ) {
        auto pos = e.message.find( "nearest is not implimented" );
        if ( pos != std::string::npos )
            ut.expected_failure( name + "-elements " + e.message );
        else
            ut.failure( name + "-elements nearest exception: " + e.message );
    } catch ( ... ) {
        ut.failure( name + "-nearest distance unknown exception" );
    }
    // Check distance
    try {
        ut.expected_failure( name + "-elements distance test not implimented yet" );
        /*bool pass  = true;
        for ( const auto &element : iterator ) {
            auto centroid = element.centroid();
            pass = pass && element.nearest( centroid ) == centroid;
        }
        if ( pass )
            ut.passes( name + "-elements distance passed" );
        else
            ut.passes( name + "-elements distance failed" );*/
    } catch ( const StackTrace::abort_error &e ) {
        auto pos = e.message.find( "distance is not implimented" );
        if ( pos != std::string::npos )
            ut.passes( name + "-elements " + e.message );
        else
            ut.failure( name + "-elements distance exception: " + e.message );
    } catch ( ... ) {
        ut.failure( name + "-elements distance unknown exception" );
    }
}


// Check the different mesh element iterators
void meshTests::MeshIteratorTest( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    char message[1000];
    // Loop through different ghost widths
    for ( int gcw = 0; gcw <= 1; gcw++ ) {
        // Loop through the different geometric entities
        for ( int i = 0; i <= (int) mesh->getGeomType(); i++ ) {
            auto type = (AMP::Mesh::GeomType) i;
            // Try to create the iterator
            size_t N_local = 0;
            size_t N_ghost = 0;
            AMP::Mesh::MeshIterator iterator;
            bool iterator_created = true;
            try {
                N_local  = mesh->numLocalElements( type );
                N_ghost  = mesh->numGhostElements( type, gcw );
                iterator = mesh->getIterator( type, gcw );
                sprintf( message, "Element iterator created (gcw=%i)", gcw );
                ut.passes( message );
            } catch ( ... ) {
                iterator_created = false;
                if ( i == 0 ) {
                    sprintf( message, "Node iterator failed (gcw=%i)", gcw );
                    ut.failure( message );
                } else if ( type == mesh->getGeomType() ) {
                    sprintf( message, "Geometric element iterator failed (gcw=%i)", gcw );
                    ut.failure( message );
                } else {
                    sprintf( message, "Intermediate element iterator failed (gcw=%i)", gcw );
                    ut.expected_failure( message );
                }
            }
            if ( !iterator_created )
                continue;
            // Test the regular iterator over local elements
            ElementIteratorTest( ut, mesh, iterator, N_local, N_ghost, type );
            // Add tests with gcw != 0
            // Add const iterator tests
        }
    }
}


// Test operator operations for iterator
void meshTests::MeshIteratorOperationTest( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    // Create some iterators to work with
    auto A        = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 1 );
    auto B        = mesh->getIterator( mesh->getGeomType(), 0 );
    auto elements = std::make_shared<std::vector<AMP::Mesh::MeshElement>>( A.size() );
    auto tmp      = A.begin();
    for ( size_t i = 0; i < A.size(); i++, ++tmp )
        ( *elements )[i] = *tmp;

    // Check operator== and operator!=
    auto C = AMP::Mesh::MultiVectorIterator( elements );
    if ( A == A && B == B && C == C )
        ut.passes( "Iterator == with same iterator" );
    else
        ut.failure( "Iterator == with same iterator" );
    if ( !( A != A ) && !( B != B ) && !( C != C ) )
        ut.passes( "Iterator != with same iterator" );
    else
        ut.failure( "Iterator != with same iterator" );
    if ( !( A == B ) )
        ut.passes( "Iterator == with same type, different iterator" );
    else
        ut.failure( "Iterator == with same type, different iterator" );
    if ( A != B )
        ut.passes( "Iterator != with same type, different iterator" );
    else
        ut.failure( "Iterator != with same type, different iterator" );
    if ( A == C && C == A && !( B == C ) && !( C == B ) )
        ut.passes( "Iterator == with different type" );
    else
        ut.failure( "Iterator == with different type" );
    if ( !( A != C ) && !( C != A ) && B != C && C != B )
        ut.passes( "Iterator != with different type" );
    else
        ut.failure( "Iterator != with different type" );
}


// Test set operations for the iterators
void meshTests::MeshIteratorSetOPTest( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    auto A = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 1 );
    auto B = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto C = AMP::Mesh::MeshIterator();
    AMP::Mesh::MeshIterator R1, R2, R3;
    // Check SetOP::Union
    R1 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Union, A, B );
    R2 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Union, B, C );
    R3 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Union, B.end(), C.end() );
    if ( R1.size() == A.size() && R2.size() == B.size() && R2 == R3 )
        ut.passes( "SetOP::Union iterator create" );
    else
        ut.failure( "SetOP::Union iterator create" );
    // Check SetOP::Intersection
    R1 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Intersection, A, B );
    R2 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Intersection, B, C );
    R3 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Intersection, B.end(), C.end() );
    if ( R1.size() == B.size() && R2.size() == 0 && R2 == R3 )
        ut.passes( "SetOP::Intersection iterator create" );
    else
        ut.failure( "SetOP::Intersection iterator create" );
    // Check SetOP::Complement
    R1 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Complement, A, B );
    R2 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Complement, B, C );
    R3 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Complement, B.end(), C.end() );
    if ( R1.size() == mesh->numGhostElements( AMP::Mesh::GeomType::Vertex, 1 ) &&
         R2.size() == B.size() && R2 == R3 )
        ut.passes( "SetOP::Complement iterator create" );
    else
        ut.failure( "SetOP::Complement iterator create" );
}


// Test the number of elements in the mesh
void meshTests::MeshCountTest( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    AMP::AMP_MPI comm = mesh->getComm();
    for ( int i = 0; i <= (int) mesh->getGeomType(); i++ ) {
        auto type             = (AMP::Mesh::GeomType) i;
        const size_t N_local  = mesh->numLocalElements( type );
        const size_t N_global = mesh->numGlobalElements( type );
        const size_t N_ghost0 = mesh->numGhostElements( type, 0 );
        const size_t N_ghost1 = comm.sumReduce( mesh->numGhostElements( type, 1 ) );
        const size_t N_sum    = comm.sumReduce( N_local );
        if ( N_global > 0 )
            ut.passes( "Non-trivial mesh created" );
        else
            ut.failure( "Non-trivial mesh created" );
        if ( N_sum == N_global )
            ut.passes( "Sum of local mesh counts matches global count" );
        else
            ut.failure( "Sum of local mesh counts matches global count: " + mesh->getName() );
        if ( N_ghost0 == 0 )
            ut.passes( "gcw=0 has no ghost elements" );
        else
            ut.failure( "gcw=0 has no ghost elements: " + mesh->getName() );
        auto ids          = mesh->getBaseMeshIDs();
        bool is_base_mesh = ids.size() == 1 && ids[0] == mesh->meshID();
        if ( N_local != N_global && is_base_mesh ) {
            if ( N_ghost1 > 0 )
                ut.passes( "gcw=1 has ghost elements" );
            else
                ut.failure( "gcw=1 has ghost elements: " + mesh->getName() );
        }
    }
}


// Test some basic Mesh properties
void meshTests::MeshBasicTest( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    // test that we can get the mesh ID
    auto meshID = mesh->meshID();
    if ( meshID > 0 && meshID != AMP::Mesh::MeshID() )
        ut.passes( "got meshID" );
    else
        ut.failure( "got meshID" );
    // Test that we can subset the mesh for it's self using the meshID
    auto mesh2 = mesh->Subset( meshID );
    if ( mesh2.get() == mesh.get() )
        ut.passes( "subset on meshID for self" );
    else
        ut.failure( "subset on meshID for self" );
    // test that we can get and set the mesh name
    auto meshName = mesh->getName();
    mesh->setName( "testing mesh name" );
    bool setName = mesh->getName().compare( "testing mesh name" ) == 0;
    mesh->setName( meshName );
    if ( meshName.compare( "NULL" ) != 0 )
        ut.passes( "non-null mesh name" );
    else
        ut.failure( "non-null mesh name" );
    if ( setName )
        ut.passes( "get/set mesh name" );
    else
        ut.failure( "get/set mesh name" );
    // Test that we can subset the mesh by the mesh name
    mesh2 = mesh->Subset( meshName );
    if ( mesh2.get() == mesh.get() )
        ut.passes( "subset on mesh name for self" );
    else
        ut.failure( "subset on mesh name for self" );
    mesh2 = mesh->Subset( "Garbage name" );
    if ( mesh2.get() == nullptr )
        ut.passes( "subset on mesh name for garbage" );
    else
        ut.failure( "subset on mesh name for garbage" );
    // Check that the bounding box matches on all processors
    auto box1 = mesh->getBoundingBox();
    auto box2 = box1;
    mesh->getComm().bcast( &box2[0], (int) box1.size(), 0 );
    bool box_match = true;
    for ( size_t i = 0; i < box1.size(); i++ ) {
        if ( box1[i] != box2[i] )
            box_match = false;
    }
    if ( box_match )
        ut.passes( "mesh->getBoundingBox returns global bounding box" );
    else
        ut.failure( "mesh->getBoundingBox returns global bounding box" );
    // Check that the sum of all local elements == number of global elements
    bool pass = true;
    for ( int d = 0; d <= static_cast<int>( mesh->getGeomType() ); d++ ) {
        auto geom        = static_cast<AMP::Mesh::GeomType>( d );
        size_t N_local   = mesh->numLocalElements( geom );
        size_t N_global  = mesh->numGlobalElements( geom );
        size_t N_global2 = mesh->getComm().sumReduce( N_local );
        pass             = pass && N_global == N_global2;
    }
    if ( pass )
        ut.passes( "sum(numLocalElements) matches numGlobalElements" );
    else
        ut.failure( "sum(numLocalElements) matches numGlobalElements: " + mesh->getName() );
}


// This tests checks that all ghost elements are owned by "owner processor"
void meshTests::VerifyGhostIsOwned( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    for ( int type = 0; type <= (int) mesh->getGeomType(); type++ ) {
        int gcw = mesh->getMaxGhostWidth();
        // Build a list of the owned and ghost elements
        std::vector<AMP::Mesh::MeshElementID> owned, ghost;
        owned.reserve( mesh->numLocalElements( (AMP::Mesh::GeomType) type ) );
        ghost.reserve( mesh->numGhostElements( (AMP::Mesh::GeomType) type, gcw ) );
        auto iterator = mesh->getIterator( (AMP::Mesh::GeomType) type, gcw );
        for ( size_t i = 0; i < iterator.size(); i++ ) {
            auto id = iterator->globalID();
            if ( id.is_local() )
                owned.push_back( id );
            else
                ghost.push_back( id );
            ++iterator;
        }
        // Broadcast the list of ghost ids to everybody
        size_t N_ghost_global = mesh->getComm().sumReduce( ghost.size() );
        if ( N_ghost_global == 0 )
            continue;
        std::vector<AMP::Mesh::MeshElementID> ghost_global( N_ghost_global );
        AMP::Mesh::MeshElementID *send_data = nullptr;
        if ( !ghost.empty() ) {
            send_data = &ghost[0];
        }
        auto recv_data = &ghost_global[0];
        mesh->getComm().allGather( send_data, (int) ghost.size(), recv_data );
        // Check that each ghost appears in the owner's rank's list
        AMP::Utilities::quicksort( owned );        // Sort for search
        AMP::Utilities::quicksort( ghost_global ); // Sort for speed
        std::vector<int> found( ghost_global.size(), 0 );
        auto my_mesh    = mesh;
        auto my_rank    = my_mesh->getComm().getRank();
        auto my_mesh_id = my_mesh->meshID();
        for ( size_t i = 0; i < N_ghost_global; i++ ) {
            // Get the current mesh
            if ( ghost_global[i].meshID() != my_mesh_id ) {
                my_mesh_id = ghost_global[i].meshID();
                my_mesh    = mesh->Subset( my_mesh_id );
                if ( my_mesh.get() == nullptr )
                    continue;
                my_rank = my_mesh->getComm().getRank();
            }
            // Check if we are the owning rank
            if ( my_mesh.get() == nullptr )
                continue;
            if ( (int) ghost_global[i].owner_rank() != my_rank )
                continue;
            // Check if we have the element
            size_t index = AMP::Utilities::findfirst( owned, ghost_global[i] );
            if ( index == owned.size() ) {
                index--;
            }
            if ( owned[index] == ghost_global[i] )
                found[i] = 1;
        }
        mesh->getComm().maxReduce( &found[0], (int) found.size() );
        bool all_found = true;
        for ( int i : found ) {
            if ( i == 0 )
                all_found = false;
        }
        if ( all_found )
            ut.passes( "All ghosts are owned by somebody" );
        else
            ut.failure( "All ghosts are owned by somebody" );
    }
}


// This tests loops over all boundary ids
void meshTests::VerifyBoundaryIDNodeIterator( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    const auto bids = mesh->getBoundaryIDs();
    for ( int bid : bids ) {
        for ( int gcw = 0; gcw <= 0; gcw++ ) {
            // Get the iterator over the current boundary id
            auto curNode = mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, bid, gcw );
            auto endNode = curNode.end();
            // Get the set of all nodes in the iterator
            bool testPassed = true;
            std::set<AMP::Mesh::MeshElementID> node_ids;
            while ( curNode != endNode ) {
                node_ids.insert( curNode->globalID() );
                if ( !curNode->isOnBoundary( bid ) )
                    testPassed = false;
                ++curNode;
            }
            size_t total_size = mesh->getComm().sumReduce( node_ids.size() );
            if ( total_size == 0 )
                testPassed = false;
            // Verify that all nodes were found
            size_t numFound = 0;
            auto curMNode   = mesh->getIterator( AMP::Mesh::GeomType::Vertex, gcw );
            auto endMNode   = curMNode.end();
            while ( curMNode != endMNode ) {
                if ( curMNode->isOnBoundary( bid ) ) {
                    numFound++;
                    if ( node_ids.find( curMNode->globalID() ) == node_ids.end() )
                        testPassed = false;
                }
                ++curMNode;
            }
            if ( numFound != node_ids.size() )
                testPassed = false;
            if ( testPassed )
                ut.passes( "Found all boundary nodes" );
            else
                ut.failure( "Found all boundary nodes" );
        }
    }
}


// This tests loops over the boundary
void meshTests::VerifyBoundaryIterator( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    bool is_surfaceMesh = static_cast<int>( mesh->getGeomType() ) < mesh->getDim();
    for ( int gcw = 0; gcw <= 0; gcw++ ) {
        for ( int type2 = 0; type2 <= (int) mesh->getGeomType(); type2++ ) {
            auto type = (AMP::Mesh::GeomType) type2;
            // Get the iterator over the current boundary id
            auto iterator      = mesh->getSurfaceIterator( type, gcw );
            size_t global_size = mesh->getComm().sumReduce( iterator.size() );
            bool passes        = global_size > 0;
            if ( std::dynamic_pointer_cast<AMP::Mesh::SubsetMesh>( mesh ).get() == nullptr ) {
                if ( mesh->numGlobalElements( type ) >= 100 )
                    passes = passes && ( global_size < mesh->numGlobalElements( type ) );
            }
            if ( passes )
                ut.passes( "Non-trivial surface iterator created" );
            else if ( is_surfaceMesh )
                ut.expected_failure( "Non-trivial surface iterator created: " + mesh->getName() );
            else
                ut.failure( "Non-trivial surface iterator created: " + mesh->getName() );
        }
    }
}


// This tests checks the block ids
void meshTests::testBlockIDs( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    const std::vector<int> blockIDs = mesh->getBlockIDs();
    if ( !blockIDs.empty() )
        ut.passes( "Block ids found" );
    else if ( (int) mesh->getGeomType() != mesh->getDim() )
        ut.expected_failure( "Block ids need work for surface meshes" );
    else
        ut.failure( "Block ids found" );
}


// This tests basic id info
void meshTests::testID( AMP::UnitTest &ut )
{
    unsigned int num_failed0 = ut.NumFailLocal();
    // Create some IDs for testing
    AMP::Mesh::MeshElementID id0;
    AMP::Mesh::MeshElementID id1( false, AMP::Mesh::GeomType::Vertex, 2, 1, 103 );
    AMP::Mesh::MeshElementID id2( true, AMP::Mesh::GeomType::Vertex, 2, 1, 103 );
    AMP::Mesh::MeshElementID id3( true, AMP::Mesh::GeomType::Volume, 2, 1, 103 );
    AMP::Mesh::MeshElementID id4( true, AMP::Mesh::GeomType::Vertex, 3, 1, 103 );
    AMP::Mesh::MeshElementID id5( true, AMP::Mesh::GeomType::Vertex, 2, 4, 103 );
    AMP::Mesh::MeshElementID id6( true, AMP::Mesh::GeomType::Vertex, 2, 1, 105 );
    // Test the default values
    if ( id0.meshID() != 0xFFFFFFFFFFFFFFFF || id0.is_local() ||
         id0.type() != AMP::Mesh::GeomType::null || id0.owner_rank() != 0 ||
         id0.local_id() != 0xFFFFFFFF )
        ut.failure( "MeshElementID test defaults" );
    // Test == and != operators
    if ( !( id1 == id1 ) || !( id1 == id2 ) )
        ut.failure( "MeshElementID test ==" );
    if ( ( id1 != id1 ) || ( id1 != id2 ) )
        ut.failure( "MeshElementID test !=" );
    if ( id1 == id3 || id1 == id4 || id1 == id5 || id1 == id6 )
        ut.failure( "MeshElementID test == (2)" );
    // Test that the basic properties were assigned correctly
    if ( id1.is_local() || !id2.is_local() )
        ut.failure( "MeshElementID test is_local" );
    if ( id1.type() != AMP::Mesh::GeomType::Vertex || id1.local_id() != 2 ||
         id1.owner_rank() != 1 || id1.meshID() != 103 )
        ut.failure( "MeshElementID test values" );
    id1.set_is_local( true );
    id2.set_is_local( false );
    if ( !id1.is_local() || id2.is_local() )
        ut.failure( "MeshElementID test is_local (2)" );
    // test greater than and less than operators
    if ( !( id1 <= id2 ) || !( id1 >= id2 ) || id1 > id2 || id1 < id2 )
        ut.failure( "MeshElementID test <,> (1)" );
    if ( id1 > id3 || id1 >= id3 || id3 < id1 || id3 <= id1 )
        ut.failure( "MeshElementID test <,> (2)" );
    if ( id1 > id4 || id1 >= id4 || id4 < id1 || id4 <= id1 )
        ut.failure( "MeshElementID test <,> (3)" );
    if ( id1 > id5 || id1 >= id5 || id5 < id1 || id5 <= id1 )
        ut.failure( "MeshElementID test <,> (4)" );
    if ( id1 > id6 || id1 >= id6 || id6 < id1 || id6 <= id1 )
        ut.failure( "MeshElementID test <,> (5)" );
    // The elements should sort by meshID, processor id, type, then local id
    std::vector<AMP::Mesh::MeshElementID> list( 6 );
    list[0] = id0;
    list[1] = id3;
    list[2] = id4;
    list[3] = id6;
    list[4] = id5;
    list[5] = id1;
    AMP::Utilities::quicksort( list );
    if ( list[0] != id1 || list[1] != id4 || list[2] != id3 || list[3] != id5 || list[4] != id6 ||
         list[5] != id0 )
        ut.failure( "MeshElementID test sort" );
    if ( num_failed0 == ut.NumFailLocal() )
        ut.passes( "MeshElementID tests" );
    else
        ut.failure( "MeshElementID tests" );
}


// Test if we correctly identify the node neighbors
void meshTests::getNodeNeighbors( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    std::map<AMP::Mesh::MeshElementID, std::vector<AMP::Mesh::MeshElementID>> neighbor_list;
    // Get a list of all neighors for each local node
    auto nodeIterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    std::vector<AMP::Mesh::MeshElementID> neighbors( 100 );
    for ( size_t i = 0; i < nodeIterator.size(); i++ ) {
        auto elements = nodeIterator->getNeighbors();
        // Store the neighbor list
        neighbors.resize( 0 );
        for ( auto &element : elements ) {
            if ( element )
                neighbors.push_back( element->globalID() );
        }
        // Sort the neighbor list for easy searching
        AMP::Utilities::quicksort( neighbors );
        auto entry = std::make_pair( nodeIterator->globalID(), neighbors );
        neighbor_list.insert( entry );
        ++nodeIterator;
    }
    // First check if the neighbor lists are unique and don't contain self
    {
        bool contains_self      = false;
        bool contains_duplicate = false;
        for ( auto it = neighbor_list.begin(); it != neighbor_list.end(); ++it ) {
            auto neighbors = it->second;
            for ( size_t i = 0; i < neighbors.size(); i++ ) {
                if ( neighbors[i] == it->first )
                    contains_self = true;
                for ( size_t j = 0; j < i; j++ ) {
                    if ( neighbors[j] == neighbors[i] )
                        contains_duplicate = true;
                }
            }
        }
        if ( !contains_self )
            ut.passes( "Neighbor nodes does not contain self" );
        else
            ut.failure( "Neighbor nodes does not contain self" );
        if ( !contains_duplicate )
            ut.passes( "Neighbor nodes does not contain duplicates" );
        else
            ut.failure( "Neighbor nodes does not contain duplicates" );
    }
    // If there are ghost nodes, then some of them must be neighbors
    if ( mesh->numGhostElements( AMP::Mesh::GeomType::Vertex, 1 ) > 0 ) {
        bool ghost_neighbors = false;
        for ( auto it = neighbor_list.begin(); it != neighbor_list.end(); ++it ) {
            auto neighbors = it->second;
            for ( auto &neighbor : neighbors ) {
                if ( !neighbor.is_local() )
                    ghost_neighbors = true;
            }
        }
        if ( ghost_neighbors )
            ut.passes( "Found ghost neighbor nodes" );
        else
            ut.failure( "Found ghost neighbor nodes" );
    }
    // Loop through all elements (including ghosts), for each owned node, check that all other nodes
    // are neighbors
    bool passed          = true;
    auto elementIterator = mesh->getIterator( mesh->getGeomType(), 1 );
    for ( size_t i = 0; i < elementIterator.size(); i++ ) {
        auto nodes = elementIterator->getElements( AMP::Mesh::GeomType::Vertex );
        for ( size_t j = 0; j < nodes.size(); j++ ) {
            if ( !nodes[j].globalID().is_local() )
                continue; // Node is not owned, move on
            auto iterator = neighbor_list.find( nodes[j].globalID() );
            if ( iterator == neighbor_list.end() ) {
                passed = false;
                break;
            }
            const auto &neighbors = iterator->second;
            if ( neighbors.empty() ) {
                passed = false;
                break;
            }
            for ( size_t k = 0; k < nodes.size(); k++ ) {
                if ( k == j )
                    continue;
                size_t index = AMP::Utilities::findfirst( neighbors, nodes[k].globalID() );
                if ( index == neighbors.size() )
                    passed = false;
            }
        }
        ++elementIterator;
    }
    if ( passed )
        ut.passes( "Node neighbors found all neighbors" );
    else
        ut.failure( "Node neighbors found all neighbors" );
    // Transfer all neighbor lists to all processors and check that every neighbor node
    // also has the current node as a neighbor (not finished)
}


// Test the displacement of the mesh
void meshTests::DisplaceMeshScalar( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    // Test the scalar displacement
    auto box1 = mesh->getBoundingBox();
    mesh->displaceMesh( std::vector<double>( mesh->getDim(), 1 ) );
    auto box2     = mesh->getBoundingBox();
    double volume = 1.0;
    for ( int i = 0; i < mesh->getDim(); i++ )
        volume *= box1[2 * i + 1] - box1[2 * i + 0];
    if ( volume > 0.0 )
        ut.passes( "non-zero bounding box" );
    else
        ut.failure( "non-zero bounding box" );
    bool passes = true;
    for ( size_t i = 0; i < box1.size(); i++ ) {
        if ( fabs( box2[i] - box1[i] - 1.0 ) > 1e-12 )
            passes = false;
    }
    if ( passes )
        ut.passes( "scalar displacement test" );
    else
        ut.failure( "scalar displacement test: " + mesh->getName() );
    mesh->displaceMesh( std::vector<double>( mesh->getDim(), -1 ) );
}
void meshTests::DisplaceMeshVector( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
// Test displacement vector
#ifdef USE_AMP_VECTORS
    // Get the volume of each element
    size_t numElements = mesh->numLocalElements( mesh->getGeomType() );
    std::vector<double> orig_vol( numElements, 0.0 );
    auto cur_elem = mesh->getIterator( mesh->getGeomType(), 0 );
    for ( size_t i = 0; i < numElements; i++ ) {
        orig_vol[i] = cur_elem->volume();
        ++cur_elem;
    }
    // Get the position of the nodes
    auto posVec1 = mesh->getPositionVector( "pos_before" );
    // Displace the mesh
    auto dispVec = posVec1->cloneVector( "displ" );
    dispVec->copyVector( posVec1 );
    dispVec->scale( 1e-3 );
    mesh->displaceMesh( dispVec );
    // Get the new positions
    auto posVec2 = mesh->getPositionVector( "pos_after" );
    auto diff    = dispVec->cloneVector( "diff" );
    diff->subtract( *posVec2, *posVec1 );
    if ( diff->equals( *dispVec ) )
        ut.passes( "displacement successfully applied" );
    else
        ut.failure( "displacement failed: " + mesh->getName() );
    // Get the new volumes (only valid for meshes with matching geometric and physical dimensions
    if ( static_cast<int>( mesh->getGeomType() ) == mesh->getDim() ) {
        bool volume_passed = true;
        cur_elem           = mesh->getIterator( mesh->getGeomType(), 0 );
        double vol_ratio   = 1.0;
        for ( int i = 0; i < mesh->getDim(); i++ )
            vol_ratio *= 1.001;
        for ( size_t i = 0; i < numElements; i++ ) {
            double ratio = cur_elem->volume() / orig_vol[i];
            // The new volume should be (1+10^-3)^dim the original volume
            if ( !AMP::Utilities::approx_equal( ratio, vol_ratio, 1e-9 ) )
                volume_passed = false;
            ++cur_elem;
        }
        if ( volume_passed )
            ut.passes( "displacement changed volumes" );
        else
            ut.failure( "displacement changed volumes: " + mesh->getName() );
    }
    for ( auto &tmp : *dispVec )
        tmp = -tmp;
    mesh->displaceMesh( dispVec );
#endif
}


// Test getting parent elements for each mesh element
void meshTests::getParents( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    bool pass = true;
    int gcw   = mesh->getMaxGhostWidth();
    for ( int type1 = 0; type1 <= (int) mesh->getGeomType(); type1++ ) {
        auto it = mesh->getIterator( (AMP::Mesh::GeomType) type1, gcw );
        for ( size_t k = 0; k < it.size(); k++ ) {
            for ( int type2 = 0; type2 < type1; type2++ ) {
                auto elements = it->getElements( (AMP::Mesh::GeomType) type2 );
                for ( auto &element : elements ) {
                    if ( !element.globalID().is_local() )
                        continue;
                    auto parents = mesh->getElementParents( element, (AMP::Mesh::GeomType) type1 );
                    // Check that the current parent was found (find all parents)
                    bool found = false;
                    for ( auto &parent : parents ) {
                        if ( parent == *it )
                            found = true;
                    }
                    if ( !found )
                        pass = false;
                    // Check that all parents do have the current element as a child (no extra
                    // parents found)
                    for ( auto &parent : parents ) {
                        auto children = parent.getElements( (AMP::Mesh::GeomType) type2 );
                        found         = false;
                        for ( auto &m : children ) {
                            if ( m == element )
                                found = true;
                        }
                        if ( !found )
                            pass = false;
                    }
                }
            }
            ++it;
        }
    }
    if ( pass )
        ut.passes( "getParents passed" );
    else
        ut.failure( "getParents passed" );
}


// VerifyElementForNode
void meshTests::VerifyElementForNode( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        // Mesh is a multimesh and test is not valid if multimesh contains meshes with
        //   different geometric types
        for ( auto mesh2 : multimesh->getMeshes() )
            meshTests::VerifyElementForNode( ut, mesh2 );
    } else if ( mesh->meshClass().find( "libmeshMesh" ) != std::string::npos ) {
        // libmeshMesh does not currently support getElementParents
        ut.expected_failure( "VerifyElementForNode" + mesh->getName() );
        return;
    } else {
        auto element_has_node = []( const AMP::Mesh::MeshElement &elem,
                                    const AMP::Mesh::MeshElement &n ) {
            std::vector<MeshElementID> ids;
            elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
            auto id0 = n.globalID();
            for ( auto id : ids ) {
                if ( id == id0 )
                    return true;
            }
            return false;
        };
        auto type = mesh->getGeomType();
        bool pass = true;
        for ( auto node : mesh->getIterator( AMP::Mesh::GeomType::Vertex ) ) {
            for ( auto elem : mesh->getElementParents( node, type ) )
                pass = pass && element_has_node( elem, node );
        }
        if ( pass )
            ut.passes( "All elements found are correct" );
        else
            ut.failure( "Found an incorrect element" );
    }
}


// VerifyNodeElemMapIteratorTest
void meshTests::VerifyNodeElemMapIteratorTest( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        // Mesh is a multimesh and test is not valid if multimesh contains meshes with
        //   different geometric types
        for ( auto mesh2 : multimesh->getMeshes() )
            meshTests::VerifyNodeElemMapIteratorTest( ut, mesh2 );
    } else if ( mesh->meshClass().find( "libmeshMesh" ) != std::string::npos ) {
        // libmeshMesh does not currently support getElementParents
        ut.expected_failure( "VerifyElementForNode" + mesh->getName() );
        return;
    } else {
        auto verify_node = []( const AMP::Mesh::MeshElement &node,
                               AMP::Mesh::Mesh::shared_ptr mesh ) {
            std::set<AMP::Mesh::MeshElementID> elems_from_node, elems_from_mesh;
            auto elements = mesh->getElementParents( node, mesh->getGeomType() );
            for ( const auto &elem : elements )
                elems_from_node.insert( elem.globalID() );
            std::vector<MeshElementID> ids;
            for ( const auto &elem : mesh->getIterator( mesh->getGeomType(), 1 ) ) {
                elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
                for ( const auto &id : ids )
                    if ( id == node.globalID() )
                        elems_from_mesh.insert( elem.globalID() );
            }
            return elems_from_node == elems_from_mesh;
        };
        int i    = 0;
        int SKIP = mesh->numLocalElements( AMP::Mesh::GeomType::Vertex ) / 20;
        for ( const auto &node : mesh->getIterator( AMP::Mesh::GeomType::Vertex ) ) {
            if ( i % SKIP == 0 ) {
                if ( !verify_node( node, mesh ) ) {
                    ut.failure( "Verify Node<->Element map iterator" );
                    return;
                }
            }
            i++;
        }
        ut.passes( "Verify Node<->Element map iterator" );
    }
}


// VerifyBoundaryIteratorTest
void meshTests::VerifyBoundaryIteratorTest( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        // Mesh is a multimesh and test is not valid if multimesh contains meshes with
        //   different geometric types
        for ( auto mesh2 : multimesh->getMeshes() )
            meshTests::VerifyBoundaryIteratorTest( ut, mesh2 );
    } else if ( mesh->meshClass().find( "libmeshMesh" ) != std::string::npos ) {
        // libmeshMesh does not currently support getElementParents
        ut.expected_failure( "VerifyElementForNode" + mesh->getName() );
        return;
    } else {
        auto isBoundaryElement = []( const AMP::Mesh::MeshElement &elem ) {
            auto neighbors = elem.getNeighbors();
            for ( const auto &neighbor : neighbors )
                if ( !neighbor )
                    return true;
            return false;
        };
        auto isNodeOnBoundary = [isBoundaryElement]( const AMP::Mesh::MeshElement &node,
                                                     AMP::Mesh::Mesh::shared_ptr mesh ) {
            auto elements = mesh->getElementParents( node, mesh->getGeomType() );
            for ( const auto &elem : elements ) {
                if ( !isBoundaryElement( elem ) )
                    return false;
            }
            return true;
        };
        for ( auto bid : mesh->getBoundaryIDs() ) {
            for ( const auto &node :
                  mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, bid ) ) {
                if ( !isNodeOnBoundary( node, mesh ) ) {
                    ut.expected_failure( "Verify Boundary iterator" );
                    return;
                }
            }
        }
        ut.passes( "Verify Boundary iterator" );
    }
}


// Test that cloning a mesh does not modify the existing mesh
void meshTests::cloneMesh( AMP::UnitTest &ut, AMP::Mesh::Mesh::const_shared_ptr mesh )
{
    // Run the tests on each individual mesh (if we are dealing with a multimesh)
    auto multimesh = std::dynamic_pointer_cast<const AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        for ( auto mesh2 : multimesh->getMeshes() )
            cloneMesh( ut, mesh2 );
    }
    // Clone is not supported for certain meshes
    if ( mesh->meshClass().find( "SubsetMesh" ) != std::string::npos )
        return;
    // Get the coordinates of the mesh points
    std::vector<AMP::Mesh::Point> coord;
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex ) )
        coord.push_back( elem.coord() );
    // Clone the mesh
    auto mesh2 = mesh->clone();
    AMP_ASSERT( mesh->meshID() != mesh2->meshID() );
    if ( mesh->getGeometry() && mesh2->getGeometry() )
        AMP_ASSERT( mesh->getGeometry().get() != mesh2->getGeometry().get() );
    // Displace the mesh
    AMP::Mesh::Point p0( mesh->getDim(), { 0.0 } );
    if ( mesh2->isMeshMovable() >= AMP::Mesh::Mesh::Movable::Displace ) {
        std::vector<double> x0 = { 1.0, 2.0, 3.0 };
        x0.resize( mesh2->getDim() );
        p0 = AMP::Mesh::Point( mesh2->getDim(), x0.data() );
        mesh2->displaceMesh( x0 );
    }
    // Get the original coordinates and make sure they match
    bool pass = true;
    auto it   = mesh2->getIterator( AMP::Mesh::GeomType::Vertex );
    for ( size_t i = 0; i < coord.size(); i++, ++it ) {
        auto p   = it->coord();
        auto err = ( p - p0 - coord[i] ).abs();
        pass     = pass && err < 1e-6;
    }
    if ( pass )
        ut.passes( "cloneMesh" );
    else
        ut.failure( "cloneMesh " + mesh->getName() );
}


// Test the performance of some common mesh operations
static inline double runAndTime( std::function<void( AMP::Mesh::Mesh::shared_ptr )> fun,
                                 AMP::Mesh::Mesh::shared_ptr mesh,
                                 int N = 1 )
{
    auto start = AMP::Utilities::time();
    for ( int i = 0; i < N; i++ )
        fun( mesh );
    auto stop = AMP::Utilities::time();
    return ( stop - start ) / N;
}
static inline void getIterator( AMP::Mesh::Mesh::shared_ptr mesh )
{
    auto it = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    NULL_USE( it );
}
static inline void incIterator( AMP::Mesh::Mesh::shared_ptr mesh )
{
    size_t N = 0;
    auto it  = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto end = it.end();
    for ( ; it != end; N++, ++it ) {}
    AMP_ASSERT( N == it.size() );
}
static inline void rangeLoop( AMP::Mesh::Mesh::shared_ptr mesh )
{
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) )
        NULL_USE( elem );
}
static inline void globalID( AMP::Mesh::Mesh::shared_ptr mesh )
{
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) ) {
        auto id = elem.globalID();
        NULL_USE( id );
    }
}
static inline void coord1( AMP::Mesh::Mesh::shared_ptr mesh )
{
    bool pass = true;
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) ) {
        auto x = elem.coord();
        pass   = pass && x.size() > 0;
    }
    AMP_ASSERT( pass );
}
static inline void coord2( AMP::Mesh::Mesh::shared_ptr mesh )
{
    bool pass = true;
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) ) {
        auto x = elem.coord( 0 );
        pass   = pass && x == x;
    }
    AMP_ASSERT( pass );
}
static inline void centroid( AMP::Mesh::Mesh::shared_ptr mesh )
{
    bool pass = true;
    for ( const auto &elem : mesh->getIterator( mesh->getGeomType(), 0 ) ) {
        auto x = elem.centroid();
        pass   = pass && x == x;
    }
    AMP_INSIST( pass, "NaN centroid: " + mesh->getName() );
}
static inline void volume( AMP::Mesh::Mesh::shared_ptr mesh )
{
    for ( const auto &elem : mesh->getIterator( mesh->getGeomType(), 0 ) ) {
        auto V = elem.volume();
        if ( V <= 0 || V != V ) {
            auto msg = "Failed volume check: " + mesh->getName() + "\n";
            msg += elem.print( 6 );
            AMP_ERROR( msg );
        }
    }
}
static inline void getElementIDs( AMP::Mesh::Mesh::shared_ptr mesh )
{
    auto type = mesh->getGeomType();
    if ( type > AMP::Mesh::GeomType::Vertex ) {
        bool pass = true;
        std::vector<AMP::Mesh::MeshElementID> ids;
        for ( const auto &elem : mesh->getIterator( type, 0 ) ) {
            elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
            pass = pass && !ids.empty();
        }
        AMP_ASSERT( pass );
    }
}
static inline void getElements1( AMP::Mesh::Mesh::shared_ptr mesh )
{
    auto type = mesh->getGeomType();
    if ( type > AMP::Mesh::GeomType::Vertex ) {
        bool pass = true;
        std::vector<AMP::Mesh::MeshElement> x;
        for ( const auto &elem : mesh->getIterator( type, 0 ) ) {
            elem.getElements( AMP::Mesh::GeomType::Vertex, x );
            pass = pass && !x.empty();
        }
        AMP_ASSERT( pass );
    }
}
static inline void getElements2( AMP::Mesh::Mesh::shared_ptr mesh )
{
    auto type = mesh->getGeomType();
    if ( type > AMP::Mesh::GeomType::Vertex ) {
        bool pass = true;
        for ( const auto &elem : mesh->getIterator( type, 0 ) ) {
            auto x = elem.getElements( AMP::Mesh::GeomType::Vertex );
            pass   = pass && x.size() > 0;
        }
        AMP_ASSERT( pass );
    }
}
void meshTests::MeshPerformance( AMP::UnitTest &ut, AMP::Mesh::Mesh::shared_ptr mesh )
{
    if ( AMP::AMP_MPI( AMP_COMM_WORLD ).getRank() != 0 )
        return;
    printf( "%s performance:\n", mesh->getName().c_str() );
    const size_t N_nodes = mesh->numLocalElements( AMP::Mesh::GeomType::Vertex );
    const size_t N_elem  = mesh->numLocalElements( mesh->getGeomType() );
    // Get the test timing
    auto t1  = runAndTime( getIterator, mesh, 1000 );
    auto t2  = runAndTime( incIterator, mesh, 10 );
    auto t3  = runAndTime( rangeLoop, mesh, 10 );
    auto t4  = runAndTime( globalID, mesh, 10 );
    auto t5  = runAndTime( coord1, mesh, 10 );
    auto t6  = runAndTime( coord2, mesh, 10 );
    auto t7  = runAndTime( centroid, mesh, 10 );
    auto t8  = runAndTime( getElementIDs, mesh, 10 );
    auto t9  = runAndTime( getElements1, mesh, 10 );
    auto t10 = runAndTime( getElements2, mesh, 10 );
    auto t11 = runAndTime( volume, mesh, 10 );
    // Print the results
    auto to_ns = []( double time, size_t N ) {
        return static_cast<int>( 1e9 * std::max( time, 0.0 ) / N );
    };
    printf( "   getIterator: %i ns\n", static_cast<int>( 1e9 * t1 ) );
    printf( "   ++iterator: %i ns\n", to_ns( t2, N_nodes ) );
    printf( "   rangeLoop: %i ns\n", to_ns( t3, N_nodes ) );
    printf( "   globalID: %i ns\n", to_ns( t4 - t3, N_nodes ) );
    printf( "   coord (1): %i ns\n", to_ns( t5 - t3, N_nodes ) );
    printf( "   coord (2): %i ns\n", to_ns( t6 - t3, N_nodes ) );
    printf( "   centroid: %i ns\n", to_ns( t7 - t3, N_elem ) );
    printf( "   getElementIDs: %i ns\n", to_ns( t8 - t3, N_elem ) );
    printf( "   getElements (1): %i ns\n", to_ns( t9 - t3, N_elem ) );
    printf( "   getElements (2): %i ns\n", to_ns( t10 - t3, N_elem ) );
    printf( "   volume: %i ns\n", to_ns( t11 - t3, N_elem ) );
    // Repeat the tests for all base meshes if we are dealing with a multimesh
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        auto ids = multimesh->getLocalBaseMeshIDs();
        for ( auto id : ids ) {
            auto mesh2 = multimesh->Subset( id );
            MeshPerformance( ut, mesh2 );
        }
    }
}


} // namespace Mesh
} // namespace AMP
