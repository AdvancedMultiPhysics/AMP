#include "AMP/mesh/testHelpers/meshTests.h"
#include "AMP/geometry/MultiGeometry.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/MeshIterator.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/SubsetMesh.h"
#include "AMP/mesh/structured/PureLogicalMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/Vector.h"

#include "ProfilerApp.h"

#include <set>
#include <vector>


using AMP::Utilities::stringf;


namespace AMP::Mesh {


// Some global variables
std::shared_ptr<AMP::Mesh::Mesh> meshTests::globalMeshForMeshVectorFactory =
    std::shared_ptr<AMP::Mesh::Mesh>();


// Get the local rank for each mesh id
static std::map<AMP::Mesh::MeshID, int> createRankMap( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    std::map<AMP::Mesh::MeshID, int> map;
    for ( auto &meshID : mesh->getBaseMeshIDs() ) {
        auto mesh2 = mesh->Subset( meshID );
        if ( mesh2 )
            map[meshID] = mesh2->getComm().getRank();
        else
            map[meshID] = -1;
    }
    return map;
}


// This test checks a single mesh element iterator
std::pair<size_t, size_t> meshTests::ElementIteratorTest( AMP::UnitTest &ut,
                                                          std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                                          const AMP::Mesh::MeshIterator &iterator,
                                                          const AMP::Mesh::GeomType type,
                                                          const std::vector<int> &blockIds,
                                                          const std::string &name )
{
    PROFILE( "ElementIteratorTest" );
    // For each mesh, get a mapping of it's processor id's to the comm of the mesh
    auto rank_map = createRankMap( mesh );
    // Check that we can get the begin and end iterator
    auto begin_it = iterator.begin();
    auto end_it   = iterator.end();
    if ( iterator.empty() ) {
        ut.pass_fail( begin_it == end_it, name + " begin and end returned" );
        return { 0, 0 };
    }
    ut.pass_fail( begin_it != end_it, name + " begin and end returned" );

    // Check that the iterator iterates through the proper number of elements
    size_t N_local     = 0;
    size_t N_ghost     = 0;
    bool pass_position = true;
    {
        std::vector<AMP::Mesh::MeshElementID> id_set( iterator.size() );
        auto it = iterator.begin();
        for ( size_t i = 0; i < iterator.size(); i++, ++it ) {
            if ( it.position() != i )
                pass_position = false;
            AMP::Mesh::MeshElementID id = it->globalID();
            id_set[i]                   = id;
            if ( id.is_local() )
                N_local++;
            else
                N_ghost++;
        }
        AMP::Utilities::unique( id_set );
        ut.pass_fail( iterator.size() == N_local + N_ghost, name + " size()" );
        ut.pass_fail( pass_position, name + " position()" );
        ut.pass_fail( id_set.size() == N_local + N_ghost, name + " uniqueness" );
    }

    // Check that we can increment and decrement properly
    if ( iterator.size() >= 4 ) {
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
        ut.pass_fail( pass, name + " increments/decrements" );
    }

    // Run element tests
    bool id_pass       = true;
    bool type_pass     = true;
    bool volume_pass   = true;
    bool coord_pass    = true;
    bool centroid_pass = true;
    bool elements_pass = true;
    bool block_pass    = true;
    bool skip_nearest  = false;
    for ( const auto &element : iterator ) {
        if ( element.elementClass() == "libmeshMeshElement" )
            skip_nearest = true;
    }
    int N_skip        = skip_nearest ? 10 : 3;
    int neighbor_pass = 1;
    int commSize      = mesh->getComm().getSize();
    int myGlobalRank  = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    int globalSize    = AMP::AMP_MPI( AMP_COMM_WORLD ).getSize();
    std::vector<AMP::Mesh::MeshElementID> ids;
    ids.reserve( iterator.size() );
    size_t testElementIndex = 0;
    for ( const auto &element : iterator ) {
        // Get the current id
        auto id = element.globalID();
        ids.push_back( id );
        id_pass = id_pass && !element.isNull();
        id_pass = id_pass && element.globalID() != MeshElementID();
        id_pass = id_pass && element.getTypeID() != AMP::getTypeID<MeshElement>();
        // Check the owner rank
        int ownerRank       = id.owner_rank();
        int globalOwnerRank = element.globalOwnerRank( *mesh );
        id_pass             = id_pass && ownerRank >= 0 && ownerRank < commSize;
        id_pass             = id_pass && globalOwnerRank >= 0 && globalOwnerRank < globalSize;
        if ( id.is_local() ) {
            int rank = rank_map[id.meshID()];
            id_pass  = id_pass && ownerRank == rank;
            id_pass  = id_pass && globalOwnerRank == myGlobalRank;
        }
        // Perform some simple checks
        if ( element.elementType() != type )
            type_pass = false;
        if ( type == AMP::Mesh::GeomType::Vertex ) {
            auto coord = element.coord();
            if ( coord.size() != mesh->getDim() )
                coord_pass = false;
        } else {
            try {
                if ( element.volume() <= 0.0 )
                    volume_pass = false;
            } catch ( ... ) {
                volume_pass = false;
            }
        }
        if ( id.type() == AMP::Mesh::GeomType::Cell ) {
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
            bool skip = ( testElementIndex++ ) % N_skip != 0;
            if ( !skip ) {
                // Test getElements
                for ( int t2 = 0; t2 <= (int) type; t2++ ) {
                    MeshElementID ids[16];
                    auto type2    = static_cast<AMP::Mesh::GeomType>( t2 );
                    auto pieces   = element.getElements( type2 );
                    size_t N      = element.getElementsID( type2, ids );
                    elements_pass = elements_pass && !pieces.empty() && pieces.size() == N;
                    for ( size_t j = 0; j < pieces.size(); j++ )
                        elements_pass = elements_pass && ids[j] == pieces[j]->globalID();
                }
                // Test getNeighbors
                // Note: some neighbors may be null (e.g. surfaces)
                auto neighbors     = element.getNeighbors();
                size_t N_neighbors = 0;
                for ( auto &neighbor : neighbors ) {
                    if ( neighbor ) {
                        N_neighbors++;
                        // Verify that the neighbors does not include self
                        if ( *neighbor == element )
                            neighbor_pass = 0;
                    }
                }
                if ( N_neighbors == 0 ) {
                    if ( element.elementType() == AMP::Mesh::GeomType::Vertex ||
                         static_cast<int>( element.elementType() ) == mesh->getDim() )
                        neighbor_pass = 0; // All nodes / elements should have some neighbors
                    else if ( neighbor_pass == 1 )
                        neighbor_pass =
                            2; // Neighbors of other element types are not always supported
                }
            }
        }
    }
    if ( id_pass && type_pass && volume_pass && coord_pass && elements_pass && neighbor_pass == 1 &&
         block_pass ) {
        ut.passes( name + " elements passed" );
    } else {
        if ( !id_pass )
            ut.failure( name + " elements failed id test" );
        if ( !type_pass )
            ut.failure( name + " elements failed type test" );
        if ( !volume_pass )
            ut.failure( name + " elements failed volume test" );
        if ( !coord_pass )
            ut.failure( name + " elements failed coord test" );
        if ( !centroid_pass )
            ut.failure( name + " elements failed centroid test" );
        if ( !elements_pass )
            ut.failure( name + " elements failed getElements test" );
        if ( neighbor_pass == 0 )
            ut.failure( name + " elements failed getNeighbors test" );
        else if ( !block_pass )
            ut.failure( name + " Element is not in a block" );
    }
    // Check that we can get the element from the global id for all elements
    bool getElem_pass = true;
    for ( const auto &element : iterator ) {
        auto id1  = element.globalID();
        auto elem = mesh->getElement( id1 );
        auto id2  = elem->globalID();
        if ( id1 != id2 )
            getElem_pass = false;
    }
    ut.pass_fail( getElem_pass, name + " Got elements from element ids" );
    // Check nearest
    if ( !skip_nearest ) {
        try {
            bool pass = true;
            int i     = 0;
            for ( const auto &element : iterator ) {
                if ( ( i++ ) % N_skip != 0 )
                    continue;
                auto x = element.centroid();
                auto y = element.nearest( x );
                auto d = ( x - y ).norm();
                pass   = pass && d < 1e-12;
            }
            ut.pass_fail( pass, name + " elements nearest passed" );
        } catch ( const StackTrace::abort_error &e ) {
            std::string msg = e.message;
            auto pos        = msg.find( "nearest is not implemented" );
            if ( pos != std::string::npos ) {
                // Nearest is not supported for all types
            } else {
                ut.failure( name + " elements nearest exception: " + msg );
            }
        } catch ( ... ) {
            ut.failure( name + " nearest distance unknown exception" );
        }
    }
    // Check distance (not finished)
    /*bool pass  = true;
    for ( const auto &element : iterator ) {
        auto centroid = element.centroid();
        pass = pass && element.nearest( centroid ) == centroid;
    }
    if ( pass )
        ut.passes( name + "-elements distance passed" );
    else
        ut.passes( name + "-elements distance failed" );*/
    return { N_local, N_ghost };
}


// Check the different mesh element iterators
void meshTests::MeshIteratorTest( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "MeshIteratorTest" );
    auto bndIDs = mesh->getBoundaryIDs();
    auto blocks = mesh->getBlockIDs();
    // Get an iterator over a boundary
    auto getBndIterator = [mesh, &ut]( AMP::Mesh::GeomType type, int id, int gcw ) {
        AMP::Mesh::MeshIterator it;
        try {
            it = mesh->getBoundaryIDIterator( type, id, gcw );
        } catch ( ... ) {
            if ( type == AMP::Mesh::GeomType::Vertex && gcw == 0 )
                ut.failure( "Failed to get boundary iterator: " + mesh->getName() );
        }
        return it;
    };
    // Get an iterator over a block
    auto getBlockIterator = [mesh]( AMP::Mesh::GeomType type, int id, int gcw ) {
        AMP::Mesh::MeshIterator it;
        try {
            it = mesh->getBlockIDIterator( type, id, gcw );
        } catch ( ... ) {
        }
        return it;
    };
    // Loop through different ghost widths
    for ( int gcw = 0; gcw <= 1; gcw++ ) {
        // Loop through the different geometric entities
        for ( int i = 0; i <= (int) mesh->getGeomType(); i++ ) {
            auto type = (AMP::Mesh::GeomType) i;
            // Test the regular iterator
            auto name     = stringf( "%s - getIterator(%i,%i)", mesh->getName().data(), i, gcw );
            auto iterator = mesh->getIterator( type, gcw );
            auto [Nl, Ng] = ElementIteratorTest( ut, mesh, iterator, type, blocks, name );
            bool pass =
                Nl == mesh->numLocalElements( type ) && Ng == mesh->numGhostElements( type, gcw );
            ut.pass_fail( pass, name + " count" );
            // Test the surface iterator
            name     = stringf( "%s - getSurfaceIterator(%i,%i)", mesh->getName().data(), i, gcw );
            iterator = mesh->getSurfaceIterator( type, gcw );
            ElementIteratorTest( ut, mesh, iterator, type, blocks, name );
            // Test the boundary iterators
            for ( auto id : bndIDs ) {
                name = stringf(
                    "%s - getBoundaryIDIterator(%i,%i,%i)", mesh->getName().data(), i, id, gcw );
                iterator = getBndIterator( type, id, gcw );
                ElementIteratorTest( ut, mesh, iterator, type, blocks, name );
            }
            // Test the block iterators
            for ( auto id : blocks ) {
                name = stringf(
                    "%s - getBlockIDIterator(%i,%i,%i)", mesh->getName().data(), i, id, gcw );
                iterator = getBlockIterator( type, id, gcw );
                ElementIteratorTest( ut, mesh, iterator, type, blocks, name );
            }
        }
    }
    // Test all meshes within a multimesh
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        for ( auto mesh2 : multimesh->getMeshes() )
            MeshIteratorTest( ut, mesh2 );
    }
}


// Test operator operations for iterator
void meshTests::MeshIteratorOperationTest( AMP::UnitTest &ut,
                                           std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "MeshIteratorOperationTest" );
    // Create some iterators to work with
    auto A = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 1 );
    auto B = mesh->getIterator( mesh->getGeomType(), 0 );
    auto elements =
        std::make_shared<std::vector<std::unique_ptr<AMP::Mesh::MeshElement>>>( A.size() );
    auto tmp = A.begin();
    for ( size_t i = 0; i < A.size(); i++, ++tmp )
        ( *elements )[i] = tmp->clone();

    // Check operator== and operator!=
    auto C = AMP::Mesh::MeshElementVectorIterator( elements );
    ut.pass_fail( A == A && B == B && C == C, "Iterator == with same iterator" );
    ut.pass_fail( !( A != A ) && !( B != B ) && !( C != C ), "Iterator != with same iterator" );
    ut.pass_fail( !( A == B ), "Iterator == with same type, different iterator" );
    ut.pass_fail( A != B, "Iterator != with same type, different iterator" );
    ut.pass_fail( A == C && C == A && !( B == C ) && !( C == B ),
                  "Iterator == with different type" );
    ut.pass_fail( !( A != C ) && !( C != A ) && B != C && C != B,
                  "Iterator != with different type" );
}


// Test set operations for the iterators
void meshTests::MeshIteratorSetOPTest( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "MeshIteratorSetOPTest" );
    auto A = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 1 );
    auto B = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto C = AMP::Mesh::MeshIterator();
    AMP::Mesh::MeshIterator R1, R2, R3;
    // Check SetOP::Union
    R1        = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Union, A, B );
    R2        = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Union, B, C );
    R3        = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Union, B.end(), C.end() );
    bool pass = R1.size() == A.size() && R2.size() == B.size() && R2 == R3;
    ut.pass_fail( pass, "SetOP::Union iterator create" );
    // Check SetOP::Intersection
    R1   = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Intersection, A, B );
    R2   = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Intersection, B, C );
    R3   = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Intersection, B.end(), C.end() );
    pass = R1.size() == B.size() && R2.size() == 0 && R2 == R3;
    ut.pass_fail( pass, "SetOP::Intersection iterator create" );
    // Check SetOP::Complement
    R1   = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Complement, A, B );
    R2   = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Complement, B, C );
    R3   = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Complement, B.end(), C.end() );
    pass = R1.size() == mesh->numGhostElements( AMP::Mesh::GeomType::Vertex, 1 ) &&
           R2.size() == B.size() && R2 == R3;
    ut.pass_fail( pass, "SetOP::Complement iterator create" );
}


// Test the number of elements in the mesh
void meshTests::MeshCountTest( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "MeshCountTest" );
    AMP::AMP_MPI comm = mesh->getComm();
    for ( int i = 0; i <= (int) mesh->getGeomType(); i++ ) {
        auto type             = (AMP::Mesh::GeomType) i;
        const size_t N_local  = mesh->numLocalElements( type );
        const size_t N_global = mesh->numGlobalElements( type );
        const size_t N_ghost0 = mesh->numGhostElements( type, 0 );
        const size_t N_ghost1 = mesh->numGhostElements( type, 1 );
        const size_t N_sum    = comm.sumReduce( N_local );
        ut.pass_fail( N_global > 0, "Non-trivial mesh created" );
        ut.pass_fail( N_sum == N_global, "Sum of local mesh counts matches global count" );
        ut.pass_fail( N_ghost0 == 0, "gcw=0 has no ghost elements" );
        auto ids          = mesh->getBaseMeshIDs();
        bool is_base_mesh = ids.size() == 1 && ids[0] == mesh->meshID();
        if ( N_local != N_global && is_base_mesh ) {
            ut.pass_fail( N_ghost1 > 0, "gcw=1 has ghost elements: " + mesh->getName() );
            if ( N_ghost1 + N_local >= N_global )
                ut.expected_failure( "gcw=1 has all elements: " + mesh->getName() );
        }
    }
}


// Test some basic Mesh properties
void meshTests::MeshBasicTest( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "MeshBasicTest" );
    // test that we can get the mesh ID
    auto meshID = mesh->meshID();
    ut.pass_fail( meshID > 0 && meshID != AMP::Mesh::MeshID(), "got meshID" );
    // Test that we can subset the mesh for it's self using the meshID
    auto mesh2 = mesh->Subset( meshID );
    ut.pass_fail( mesh2.get() == mesh.get(), "subset on meshID for self" );
    // test that we can get and set the mesh name
    auto meshName = mesh->getName();
    mesh->setName( "testing mesh name" );
    bool setName = mesh->getName().compare( "testing mesh name" ) == 0;
    mesh->setName( meshName );
    ut.pass_fail( meshName.compare( "NULL" ) != 0, "non-null mesh name" );
    ut.pass_fail( setName, "get/set mesh name" );
    // Test that we can subset the mesh by the mesh name
    mesh2 = mesh->Subset( meshName );
    ut.pass_fail( mesh2.get() == mesh.get(), "subset on mesh name for self" );
    mesh2 = mesh->Subset( "Garbage name" );
    ut.pass_fail( mesh2.get() == nullptr, "subset on mesh name for garbage" );
    // Check that the bounding box matches on all processors
    auto box1 = mesh->getBoundingBox();
    auto box2 = box1;
    mesh->getComm().bcast( &box2[0], (int) box1.size(), 0 );
    bool box_match = true;
    for ( size_t i = 0; i < box1.size(); i++ ) {
        if ( box1[i] != box2[i] )
            box_match = false;
    }
    ut.pass_fail( box_match, "mesh->getBoundingBox returns global bounding box" );
    // Check that the sum of all local elements == number of global elements
    bool pass = true;
    for ( int d = 0; d <= static_cast<int>( mesh->getGeomType() ); d++ ) {
        auto geom        = static_cast<AMP::Mesh::GeomType>( d );
        size_t N_local   = mesh->numLocalElements( geom );
        size_t N_global  = mesh->numGlobalElements( geom );
        size_t N_global2 = mesh->getComm().sumReduce( N_local );
        pass             = pass && N_global == N_global2;
    }
    ut.pass_fail( pass, "sum(numLocalElements) matches numGlobalElements: " + mesh->getName() );
}


// This tests checks that all ghost elements are owned by "owner processor"
void meshTests::VerifyGhostIsOwned( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "VerifyGhostIsOwned" );
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
        auto ghost_global = mesh->getComm().allGather( ghost );
        if ( ghost_global.empty() )
            continue; // Check that each ghost appears in the owner's rank's list
        AMP::Utilities::quicksort( owned );        // Sort for search
        AMP::Utilities::quicksort( ghost_global ); // Sort for speed
        std::vector<int> found( ghost_global.size(), 0 );
        auto my_mesh    = mesh;
        auto my_rank    = my_mesh->getComm().getRank();
        auto my_mesh_id = my_mesh->meshID();
        for ( size_t i = 0; i < ghost_global.size(); i++ ) {
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
        for ( size_t i = 0; i < found.size(); i++ ) {
            if ( found[i] == 0 )
                all_found = false;
        }
        ut.pass_fail( all_found, "All ghosts are owned by somebody" );
    }
}


// This tests loops over all boundary ids
void meshTests::VerifyBoundaryIDNodeIterator( AMP::UnitTest &ut,
                                              std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "VerifyBoundaryIDNodeIterator" );
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
            ut.pass_fail( testPassed, "Found all boundary nodes" );
        }
    }
}


// This tests loops over the boundary
void meshTests::VerifyBoundaryIterator( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "VerifyBoundaryIterator" );
    // Test all meshes within a multimesh
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        for ( auto mesh2 : multimesh->getMeshes() )
            meshTests::VerifyBoundaryIterator( ut, mesh2 );
    }
    // Surface meshes may not have their own surfaces
    if ( static_cast<int>( mesh->getGeomType() ) < mesh->getDim() )
        return;
    // Check that we can create surface iterators
    bool passCreated  = true;
    bool passInterior = true;
    bool passUnique   = true;
    for ( int type2 = 0; type2 <= (int) mesh->getGeomType(); type2++ ) {
        auto type = (AMP::Mesh::GeomType) type2;
        // Get the iterator over the current boundary id
        auto iterator      = mesh->getSurfaceIterator( type, 0 );
        size_t global_size = mesh->getComm().sumReduce( iterator.size() );
        passCreated        = passCreated && global_size > 0;
        passInterior       = passInterior && global_size < mesh->numGlobalElements( type );
        // Check for duplicate elements
        std::vector<AMP::Mesh::MeshElementID> ids;
        ids.reserve( iterator.size() );
        for ( const auto &elem : iterator )
            ids.push_back( elem.globalID() );
        AMP::Utilities::unique( ids );
        passUnique = passUnique && ids.size() == iterator.size();
    }
    // Verify that all nodes were found
    bool passFound = true;
    for ( int gcw = 0; gcw <= 1; gcw++ ) {
        auto iterator   = mesh->getSurfaceIterator( AMP::Mesh::GeomType::Vertex, gcw );
        size_t numFound = 0;
        auto nodes      = mesh->getIterator( AMP::Mesh::GeomType::Vertex, gcw );
        for ( auto &node : nodes ) {
            if ( node.isOnSurface() )
                numFound++;
        }
        if ( numFound != iterator.size() ) {
            auto msg = stringf( "Surface iterator found all points <%i,%i,%i>: %s",
                                (int) numFound,
                                (int) iterator.size(),
                                gcw,
                                mesh->getName().data() );
            ut.failure( msg );
            passFound = false;
        }
    }
    if ( !passCreated ) {
        ut.failure( "Non-trivial surface iterator created: " + mesh->getName() );
        return;
    }
    if ( !passInterior )
        ut.expected_failure( "Surface iterator created (no interior points): " + mesh->getName() );
    if ( !passUnique )
        ut.failure( "Surface iterator has duplicate points: " + mesh->getName() );
    if ( passCreated && passInterior && passUnique && passFound )
        ut.passes( ": " + mesh->getName() );
    // Check that the nodes of surface elements < physical dim are on the surface
    std::vector<AMP::Mesh::MeshElementID> nodes;
    for ( auto &elem : mesh->getSurfaceIterator( AMP::Mesh::GeomType::Vertex, 1 ) )
        nodes.push_back( elem.globalID() );
    AMP::Utilities::unique( nodes );
    AMP_ASSERT( !nodes.empty() );
    AMP::Mesh::MeshElementID tmp[32];
    for ( int type2 = 1; type2 <= (int) mesh->getGeomType() && type2 < mesh->getDim(); type2++ ) {
        bool pass = true;
        auto type = (AMP::Mesh::GeomType) type2;
        for ( auto &elem : mesh->getSurfaceIterator( type, 0 ) ) {
            int N = elem.getElementsID( AMP::Mesh::GeomType::Vertex, tmp );
            AMP_ASSERT( N < 32 );
            for ( int j = 0; j < N; j++ ) {
                auto id  = tmp[j];
                size_t i = std::min( AMP::Utilities::findfirst( nodes, id ), nodes.size() - 1 );
                if ( nodes[i] != id )
                    pass = false;
            }
        }
        auto msg =
            stringf( "Surface elements have all nodes on surface <%i>: ", type2 ) + mesh->getName();
        if ( pass )
            ut.passes( msg );
        else
            ut.expected_failure( msg );
    }
}


// This tests checks the block ids
void meshTests::testBlockIDs( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
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
    int rank                 = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    unsigned int num_failed0 = ut.NumFailLocal();
    // Print some basic meshID hashes
    if ( rank == 0 ) {
        std::cout << "MeshID( 0, 0 ).hash() = " << MeshID( 0, 0 ).getHash() << std::endl;
        std::cout << "MeshID( 0, 1 ).hash() = " << MeshID( 0, 1 ).getHash() << std::endl;
        std::cout << "MeshID( 1, 0 ).hash() = " << MeshID( 1, 0 ).getHash() << std::endl;
        std::cout << "MeshID( 1, 1 ).hash() = " << MeshID( 1, 1 ).getHash() << std::endl;
    }
    // Create some IDs for testing
    AMP::Mesh::MeshElementID id0;
    AMP::Mesh::MeshElementID id1( false, AMP::Mesh::GeomType::Vertex, 2, 1, 103 );
    AMP::Mesh::MeshElementID id2( true, AMP::Mesh::GeomType::Vertex, 2, 1, 103 );
    AMP::Mesh::MeshElementID id3( true, AMP::Mesh::GeomType::Cell, 2, 1, 103 );
    AMP::Mesh::MeshElementID id4( true, AMP::Mesh::GeomType::Vertex, 3, 1, 103 );
    AMP::Mesh::MeshElementID id5( true, AMP::Mesh::GeomType::Vertex, 2, 4, 103 );
    AMP::Mesh::MeshElementID id6( true, AMP::Mesh::GeomType::Vertex, 2, 1, 105 );
    // Test the default values
    if ( id0.meshID() != AMP::Mesh::MeshID( 0xFFFFFFFFFFFFFFFF ) || id0.is_local() ||
         id0.type() != AMP::Mesh::GeomType::Nullity || id0.owner_rank() != 0 ||
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
         id1.owner_rank() != 1 || id1.meshID() != AMP::Mesh::MeshID( 103 ) )
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
    // Test basic MeshElement info
    MeshElement elem;
    bool passElem = elem.isNull() && elem.getTypeID() == AMP::getTypeID<MeshElement>() &&
                    elem.elementClass() == "MeshElement";
    ut.pass_fail( passElem, "MeshElement()" );
}


// Function to test the index conversions for BoxMesh
static inline bool inBox( const BoxMesh::MeshElementIndex &index, const BoxMesh::Box &box )
{
    auto [i, j, k] = index.index();
    return i >= box.first[0] && i <= box.last[0] && j >= box.first[1] && j <= box.last[1] &&
           k >= box.first[2] && k <= box.last[2];
}
static inline bool inBox( const MeshIterator &it, const BoxMesh::Box &box, const BoxMesh &mesh )
{
    bool pass = true;
    for ( const auto &elem : it ) {
        auto index = mesh.convert( elem.globalID() );
        pass       = pass && inBox( index, box );
    }
    return pass;
}
void meshTests::testBoxMeshIndicies( AMP::UnitTest &ut, int ndim )
{
    PROFILE( "testBoxMeshIndicies" );
    bool pass = true;
    auto msg  = stringf( "testBoxMeshIndicies<%i>", ndim );

    // Create the mesh
    std::vector<size_t> size = { 15, 12, 8 };
    size.resize( ndim );
    std::vector<bool> per( size.size(), true );
    auto db =
        AMP::Database::create( "MeshName", "domain", "Size", size, "Periodic", per, "GCW", 2 );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( std::move( db ) );
    params->setComm( AMP_COMM_WORLD );
    auto mesh = std::make_shared<AMP::Mesh::PureLogicalMesh>( params );
    auto type = mesh->getGeomType();

    // Check box domain
    auto globalSize = mesh->size();
    auto globalBox  = mesh->getGlobalBox( 2 );
    AMP_ASSERT( globalSize == size );
    for ( int d = 0; d < ndim; d++ ) {
        AMP_ASSERT( globalBox.first[d] == -2 );
        AMP_ASSERT( globalBox.last[d] == (int) size[d] + 1 );
    }

    // Get a list of element ids
    auto local = mesh->getIterator( type, 0 );
    auto ghost = mesh->getIterator( type, 1 );
    std::vector<AMP::Mesh::MeshElementID> ghostIDs;
    for ( auto &element : ghost )
        ghostIDs.push_back( element.globalID() );
    AMP::Utilities::unique( ghostIDs );
    if ( !inBox( local, mesh->getLocalBox( 0 ), *mesh ) ) {
        ut.failure( msg + " - local iterator not in box" );
        pass = false;
    }

    // Check the conversion between ids
    for ( auto id : ghostIDs ) {
        auto index = mesh->convert( id );
        auto id2   = mesh->convert( index );
        if ( id != id2 ) {
            std::cout << "Failed mesh id conversion:" << std::endl;
            std::cout << "  id: " << id << std::endl;
            std::cout << "  index: " << index << std::endl;
            std::cout << "  id2: " << id2 << std::endl;
            ut.failure( msg );
            return;
        }
    }

    // For each local element, check the neighbors
    bool pass2 = true;
    std::vector<std::unique_ptr<MeshElement>> neighbors;
    std::vector<AMP::Mesh::MeshElementID> notFound;
    size_t N_neighbors = 2 * ndim;
    for ( auto &element : local ) {
        neighbors.clear();
        element.getNeighbors( neighbors );
        pass2 = pass2 && neighbors.size() == N_neighbors;
        for ( auto &neighbor : neighbors ) {
            auto id = neighbor->globalID();
            int i   = AMP::Utilities::findfirst( ghostIDs, id );
            i       = std::min<int>( i, ghostIDs.size() - 1 );
            if ( ghostIDs[i] != id )
                notFound.push_back( id );
        }
    }
    pass2 = pass2 && notFound.empty();
    pass  = pass && pass2;

    // Print any failed elements
    AMP::AMP_MPI comm( AMP_COMM_WORLD );
    for ( int r = 0; r < comm.getSize(); r++ ) {
        comm.barrier();
        if ( r == comm.getRank() && !notFound.empty() ) {
            auto box = mesh->getLocalBox( 0 );
            printf( "Rank %i\n", r );
            printf( "N_local: %i\n", (int) mesh->numLocalElements( type ) );
            printf( "N_ghost: %i\n", (int) mesh->numGhostElements( type, 1 ) );
            printf( "N_total: %i\n", (int) ghostIDs.size() );
            printf( "Box: (%i,%i,%i)-(%i,%i,%i)\n",
                    box.first[0],
                    box.first[1],
                    box.first[2],
                    box.last[0],
                    box.last[1],
                    box.last[2] );
            printf( "Not found:\n" );
            for ( auto id : notFound )
                std::cout << "   " << mesh->convert( id ) << std::endl;
        }
    }
    comm.barrier();
    if ( pass )
        ut.passes( msg );
    else
        ut.failure( msg );
}


// Test if we correctly identify the node neighbors
void meshTests::getNodeNeighbors( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "getNodeNeighbors" );
    std::map<AMP::Mesh::MeshElementID, std::vector<AMP::Mesh::MeshElementID>> neighbor_list;
    // Get a list of all neighors for each local node
    auto nodeIterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    std::vector<AMP::Mesh::MeshElementID> neighbors( 100 );
    for ( const auto &node : nodeIterator ) {
        auto elements = node.getNeighbors();
        // Store the neighbor list
        neighbors.resize( 0 );
        for ( auto &element : elements ) {
            if ( element )
                neighbors.push_back( element->globalID() );
        }
        // Sort the neighbor list for easy searching
        AMP::Utilities::quicksort( neighbors );
        auto entry = std::make_pair( node.globalID(), neighbors );
        neighbor_list.insert( entry );
    }
    // First check if the neighbor lists are unique and don't contain self
    {
        bool contains_self      = false;
        bool contains_duplicate = false;
        for ( auto &it : neighbor_list ) {
            auto neighbors = it.second;
            for ( size_t i = 0; i < neighbors.size(); i++ ) {
                if ( neighbors[i] == it.first )
                    contains_self = true;
                for ( size_t j = 0; j < i; j++ ) {
                    if ( neighbors[j] == neighbors[i] )
                        contains_duplicate = true;
                }
            }
        }
        ut.pass_fail( !contains_self, "Neighbor nodes does not contain self" );
        ut.pass_fail( !contains_duplicate, "Neighbor nodes does not contain duplicates" );
    }
    // If there are ghost nodes, then some of them must be neighbors
    if ( mesh->numGhostElements( AMP::Mesh::GeomType::Vertex, 1 ) > 0 ) {
        bool ghost_neighbors = false;
        for ( auto &it : neighbor_list ) {
            auto neighbors = it.second;
            for ( auto &neighbor : neighbors ) {
                if ( !neighbor.is_local() )
                    ghost_neighbors = true;
            }
        }
        ut.pass_fail( ghost_neighbors, "Found ghost neighbor nodes" );
    }
    // Loop through all elements (including ghosts), for each owned node, check that all other nodes
    // are neighbors
    bool passed          = true;
    auto elementIterator = mesh->getIterator( mesh->getGeomType(), 1 );
    for ( size_t i = 0; i < elementIterator.size(); i++ ) {
        auto nodes = elementIterator->getElements( AMP::Mesh::GeomType::Vertex );
        for ( size_t j = 0; j < nodes.size(); j++ ) {
            auto id = nodes[j]->globalID();
            if ( !id.is_local() )
                continue; // Node is not owned, move on
            auto iterator = neighbor_list.find( id );
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
                size_t index = AMP::Utilities::findfirst( neighbors, nodes[k]->globalID() );
                if ( index == neighbors.size() )
                    passed = false;
            }
        }
        ++elementIterator;
    }
    ut.pass_fail( passed, "Node neighbors found all neighbors" );
    // Transfer all neighbor lists to all processors and check that every neighbor node
    // also has the current node as a neighbor (not finished)
}


// Test the displacement of the mesh
void meshTests::DisplaceMeshScalar( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "DisplaceMeshScalar" );
    // Test the scalar displacement
    auto box1 = mesh->getBoundingBox();
    mesh->displaceMesh( std::vector<double>( mesh->getDim(), 1 ) );
    auto box2     = mesh->getBoundingBox();
    double volume = 1.0;
    for ( int i = 0; i < mesh->getDim(); i++ )
        volume *= box1[2 * i + 1] - box1[2 * i + 0];
    ut.pass_fail( volume > 0.0, "non-zero bounding box" );
    bool passes = true;
    for ( size_t i = 0; i < box1.size(); i++ ) {
        if ( fabs( box2[i] - box1[i] - 1.0 ) > 1e-12 )
            passes = false;
    }
    ut.pass_fail( passes, "scalar displacement test: " + mesh->getName() );
    mesh->displaceMesh( std::vector<double>( mesh->getDim(), -1 ) );
}
void meshTests::DisplaceMeshVector( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "DisplaceMeshVector" );
    // Test displacement vector
    // Get the volume of each element
    size_t numElements = mesh->numLocalElements( mesh->getGeomType() );
    std::vector<double> orig_vol( numElements, 0.0 );
    auto cur_elem = mesh->getIterator( mesh->getGeomType(), 0 );
    try {
        for ( size_t i = 0; i < numElements; i++ ) {
            orig_vol[i] = cur_elem->volume();
            ++cur_elem;
        }
    } catch ( ... ) {
        ut.failure( "volume error in " + mesh->getName() );
        return;
    }
    // Get the position of the nodes
    auto posVec1 = mesh->getPositionVector( "pos_before" );
    // Displace the mesh
    auto dispVec = posVec1->clone( "displ" );
    dispVec->copyVector( posVec1 );
    dispVec->scale( 1e-3 );
    mesh->displaceMesh( dispVec );
    // Get the new positions
    auto posVec2 = mesh->getPositionVector( "pos_after" );
    auto diff    = dispVec->clone( "diff" );
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
        ut.pass_fail( volume_passed, "displacement changed volumes: " + mesh->getName() );
    }
    for ( auto &tmp : *dispVec )
        tmp = -tmp;
    mesh->displaceMesh( dispVec );
}


// Test getting parent elements for each mesh element
void meshTests::getParents( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "getParents" );
    bool pass = true;
    int gcw   = mesh->getMaxGhostWidth();
    for ( int type1 = 0; type1 <= (int) mesh->getGeomType(); type1++ ) {
        auto it = mesh->getIterator( (AMP::Mesh::GeomType) type1, gcw );
        for ( size_t k = 0; k < it.size(); k++ ) {
            for ( int type2 = 0; type2 < type1; type2++ ) {
                auto elements = it->getElements( (AMP::Mesh::GeomType) type2 );
                for ( auto &element : elements ) {
                    if ( !element->globalID().is_local() )
                        continue;
                    auto parents = mesh->getElementParents( *element, (AMP::Mesh::GeomType) type1 );
                    // Check that the current parent was found (find all parents)
                    bool found = false;
                    for ( auto &parent : parents ) {
                        if ( *parent == *it )
                            found = true;
                    }
                    if ( !found )
                        pass = false;
                    // Check that all parents do have the current element as a child (no extra
                    // parents found)
                    for ( auto &parent : parents ) {
                        auto children = parent->getElements( (AMP::Mesh::GeomType) type2 );
                        found         = false;
                        for ( auto &m : children ) {
                            if ( *m == *element )
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
    ut.pass_fail( pass, "getParents passed" );
}


// VerifyElementForNode
void meshTests::VerifyElementForNode( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "VerifyElementForNode" );
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        // Mesh is a multimesh and test is not valid if multimesh contains meshes with
        //   different geometric types
        for ( auto mesh2 : multimesh->getMeshes() )
            meshTests::VerifyElementForNode( ut, mesh2 );
    } else if ( mesh->meshClass().find( "libmeshMesh" ) != std::string::npos ) {
        // libmeshMesh does not currently support getElementParents
        ut.expected_failure( "VerifyElementForNode not supported for libMesh" );
        return;
    } else {
        auto element_has_node = []( const AMP::Mesh::MeshElement &elem,
                                    const AMP::Mesh::MeshElement &n ) {
            MeshElementID ids[32];
            int N = elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
            AMP_ASSERT( N < 32 );
            auto id0 = n.globalID();
            for ( int i = 0; i < N; i++ ) {
                if ( ids[i] == id0 )
                    return true;
            }
            return false;
        };
        auto type = mesh->getGeomType();
        bool pass = true;
        for ( auto &node : mesh->getIterator( AMP::Mesh::GeomType::Vertex ) ) {
            for ( auto &elem : mesh->getElementParents( node, type ) )
                pass = pass && element_has_node( *elem, node );
        }
        if ( pass )
            ut.passes( "All elements found are correct" );
        else
            ut.failure( "Found an incorrect element" );
    }
}


// VerifyNodeElemMapIteratorTest
void meshTests::VerifyNodeElemMapIteratorTest( AMP::UnitTest &ut,
                                               std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "VerifyNodeElemMapIteratorTest" );
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        // Mesh is a multimesh and test is not valid if multimesh contains meshes with
        //   different geometric types
        for ( auto mesh2 : multimesh->getMeshes() )
            meshTests::VerifyNodeElemMapIteratorTest( ut, mesh2 );
    } else if ( mesh->meshClass().find( "libmeshMesh" ) != std::string::npos ) {
        // libmeshMesh does not currently support getElementParents
        ut.expected_failure( "Verify Node<->Element not supported for libMesh" );
        return;
    } else {
        auto verify_node = []( const AMP::Mesh::MeshElement &node,
                               std::shared_ptr<AMP::Mesh::Mesh> mesh ) {
            std::set<AMP::Mesh::MeshElementID> elems_from_node, elems_from_mesh;
            auto elements = mesh->getElementParents( node, mesh->getGeomType() );
            for ( const auto &elem : elements )
                elems_from_node.insert( elem->globalID() );
            MeshElementID ids[32];
            for ( const auto &elem : mesh->getIterator( mesh->getGeomType(), 1 ) ) {
                int N = elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
                AMP_ASSERT( N < 32 );
                for ( int i = 0; i < N; i++ )
                    if ( ids[i] == node.globalID() )
                        elems_from_mesh.insert( elem.globalID() );
            }
            return elems_from_node == elems_from_mesh;
        };
        int i    = 0;
        int SKIP = mesh->numLocalElements( AMP::Mesh::GeomType::Vertex ) / 20;
        for ( const auto &node : mesh->getIterator( AMP::Mesh::GeomType::Vertex ) ) {
            if ( i % SKIP == 0 ) {
                if ( !verify_node( node, mesh ) ) {
                    if ( mesh->meshClass().find( "TriangleMesh" ) != std::string::npos )
                        ut.expected_failure( "Verify Node<->Element map iterator (TriangleMesh)" );
                    else if ( mesh->meshClass().find( "SubsetMesh" ) != std::string::npos )
                        ut.expected_failure( "Verify Node<->Element map iterator (SubsetMesh)" );
                    else
                        ut.failure( "Verify Node<->Element map iterator (" + mesh->meshClass() +
                                    ")" );
                    return;
                }
            }
            i++;
        }
        ut.passes( "Verify Node<->Element map iterator" );
    }
}


// VerifyBoundaryIteratorTest
void meshTests::VerifyBoundaryIteratorTest( AMP::UnitTest &ut,
                                            std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "VerifyBoundaryIteratorTest" );
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        // Mesh is a multimesh and test is not valid if multimesh contains meshes with
        //   different geometric types
        for ( auto mesh2 : multimesh->getMeshes() )
            meshTests::VerifyBoundaryIteratorTest( ut, mesh2 );
    } else if ( mesh->meshClass().find( "libmeshMesh" ) != std::string::npos ) {
        // libmeshMesh does not currently support getElementParents
        ut.expected_failure( "Verify Boundary iterator not supported for libMesh" );
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
                                                     std::shared_ptr<AMP::Mesh::Mesh> mesh ) {
            auto elements = mesh->getElementParents( node, mesh->getGeomType() );
            for ( const auto &elem : elements ) {
                if ( !isBoundaryElement( *elem ) )
                    return false;
            }
            return true;
        };
        for ( auto bid : mesh->getBoundaryIDs() ) {
            for ( const auto &node :
                  mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, bid ) ) {
                if ( !isNodeOnBoundary( node, mesh ) ) {
                    ut.expected_failure( "Verify Boundary iterator: " + mesh->getName() );
                    return;
                }
            }
        }
        ut.passes( "Verify Boundary iterator: " + mesh->getName() );
    }
}


// Test that cloning a mesh does not modify the existing mesh
void meshTests::cloneMesh( AMP::UnitTest &ut, std::shared_ptr<const AMP::Mesh::Mesh> mesh )
{
    PROFILE( "cloneMesh" );
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
    ut.pass_fail( pass, "cloneMesh " + mesh->getName() );
}


// Test the performance of some common mesh operations
static inline double runAndTime( std::function<void( std::shared_ptr<AMP::Mesh::Mesh> )> fun,
                                 std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                 int N = 1 )
{
    auto start = AMP::Utilities::time();
    for ( int i = 0; i < N; i++ )
        fun( mesh );
    auto stop = AMP::Utilities::time();
    return ( stop - start ) / N;
}
static inline void getIterator( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    [[maybe_unused]] auto it = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
}
static inline void incIterator( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    size_t N = 0;
    auto it  = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto end = it.end();
    for ( ; it != end; N++, ++it ) {}
    AMP_ASSERT( N == it.size() );
}
static inline void rangeLoop( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) ) {
        [[maybe_unused]] auto id = elem.globalID();
    }
}
static inline void globalID( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) ) {
        [[maybe_unused]] auto id = elem.globalID();
    }
}
static inline void coord1( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    bool pass = true;
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) ) {
        auto x = elem.coord();
        pass   = pass && x.size() > 0;
    }
    AMP_ASSERT( pass );
}
static inline void coord2( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    bool pass = true;
    for ( const auto &elem : mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) ) {
        auto x = elem.coord( 0 );
        pass   = pass && x == x;
    }
    AMP_ASSERT( pass );
}
static inline void centroid( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    bool pass = true;
    for ( const auto &elem : mesh->getIterator( mesh->getGeomType(), 0 ) ) {
        auto x = elem.centroid();
        pass   = pass && x == x;
    }
    AMP_INSIST( pass, "NaN centroid: " + mesh->getName() );
}
static inline void volume( std::shared_ptr<AMP::Mesh::Mesh> mesh )
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
static inline void getElementIDs( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    auto type = mesh->getGeomType();
    if ( type > AMP::Mesh::GeomType::Vertex ) {
        bool pass = true;
        AMP::Mesh::MeshElementID ids[32];
        for ( const auto &elem : mesh->getIterator( type, 0 ) ) {
            auto N = elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
            pass   = pass && N > 0;
        }
        AMP_ASSERT( pass );
    }
}
static inline void getElements1( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    auto type = mesh->getGeomType();
    if ( type > AMP::Mesh::GeomType::Vertex ) {
        bool pass = true;
        std::vector<std::unique_ptr<AMP::Mesh::MeshElement>> x;
        for ( const auto &elem : mesh->getIterator( type, 0 ) ) {
            elem.getElements( AMP::Mesh::GeomType::Vertex, x );
            pass = pass && !x.empty();
        }
        AMP_ASSERT( pass );
    }
}
static inline void getElements2( std::shared_ptr<AMP::Mesh::Mesh> mesh )
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
void meshTests::MeshPerformance( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    if ( AMP::AMP_MPI( AMP_COMM_WORLD ).getRank() != 0 )
        return;
    try {
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
    } catch ( ... ) {
        ut.failure( "Caught exception testing performance: " + mesh->getName() );
    }
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


} // namespace AMP::Mesh
