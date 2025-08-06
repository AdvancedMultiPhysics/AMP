#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"
#include "AMP/mesh/testHelpers/meshTests.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"


using AMP::Utilities::stringf;


namespace AMP::Mesh {


static void testElementFromPoint( AMP::UnitTest &ut,
                                  std::shared_ptr<const AMP::Mesh::BoxMesh> mesh )
{
    if ( mesh->meshClass() == "PureLogicalMesh" )
        return;
    if ( mesh->isMeshMovable() == Mesh::Movable::Deform )
        return;
    PROFILE( "testElementFromPoint" );
    bool pass   = true;
    auto vertex = AMP::Mesh::GeomType::Vertex;
    for ( const auto &node : mesh->getIterator( vertex, 0 ) ) {
        auto point = node.centroid();
        auto index = mesh->getElementFromPhysical( point, vertex );
        auto node2 = mesh->getElement( index );
        pass       = pass && node == node2;
    }
    auto volume = mesh->getGeomType();
    for ( const auto &elem : mesh->getIterator( volume, 0 ) ) {
        auto point = elem.centroid();
        auto index = mesh->getElementFromPhysical( point, volume );
        auto elem2 = mesh->getElement( index );
        pass       = pass && elem == elem2;
    }
    auto logicalDim = static_cast<int>( mesh->getGeomType() );
    bool test       = false;
    auto name       = mesh->getName();
    for ( int s = 0; s < 2 * logicalDim; s++ )
        test = test || mesh->getSurfaceID( s ) == -2;
    if ( pass )
        ut.passes( name + " - getElementFromPhysical" );
    else if ( test )
        ut.expected_failure( name + " - getElementFromPhysical (boundary ids of -2)" );
    else
        ut.failure( name + " - getElementFromPhysical" );
}


static void testSurface( AMP::UnitTest &ut, std::shared_ptr<const AMP::Mesh::BoxMesh> mesh )
{
    PROFILE( "testSurface" );
    bool allPass    = true;
    auto logicalDim = static_cast<int>( mesh->getGeomType() );
    const auto name = mesh->getName();
    for ( int t = 0; t <= logicalDim; t++ ) {
        auto type     = static_cast<AMP::Mesh::GeomType>( t );
        auto typeName = to_string( type );
        // Get a list of element ids on each surface id
        std::map<int, std::vector<BoxMesh::MeshElementIndex>> map;
        for ( auto id : mesh->getBoundaryIDs() ) {
            auto it    = mesh->getBoundaryIDIterator( type, id, 0 );
            auto &list = map[id];
            list.reserve( it.size() );
            for ( auto &elem : it ) {
                auto elem2 = dynamic_cast<const structuredMeshElement *>( elem.getRawElement() );
                list.push_back( elem2->getIndex() );
            }
            map[id]  = mesh->getComm().allGather( list );
            size_t N = map[id].size();
            AMP::Utilities::unique( map[id] );
            if ( N != map[id].size() )
                ut.failure( stringf(
                    "%s - getSurface (getBoundaryIDIterator,%s)", name.data(), typeName.data() ) );
        }
        // Check that each surface is in the list
        for ( int s = 0; s < 2 * logicalDim; s++ ) {
            auto id    = mesh->getSurfaceID( s );
            auto boxes = mesh->getSurface( s, type );
            auto it    = mesh->createIterator( boxes );
            auto list  = map[id];
            bool pass  = true;
            for ( auto &elem : it ) {
                auto elem2 = dynamic_cast<const structuredMeshElement *>( elem.getRawElement() );
                auto index = elem2->getIndex();
                if ( !std::binary_search( list.begin(), list.end(), index ) )
                    pass = false;
            }
            if ( !pass ) {
                auto it2 = mesh->getBoundaryIDIterator( type, id, 0 );
                ut.failure( stringf( "%s - getSurface (%s,%i)", name.data(), typeName.data(), s ) );
                allPass = false;
            }
        }
    }
    if ( allPass )
        ut.passes( name + " - getSurface" );
}


void meshTests::testBoxMesh( AMP::UnitTest &ut, std::shared_ptr<const AMP::Mesh::Mesh> mesh )
{
    // Get the box mesh
    auto multimesh = std::dynamic_pointer_cast<const MultiMesh>( mesh );
    if ( multimesh ) {
        for ( auto mesh2 : multimesh->getMeshes() )
            testBoxMesh( ut, mesh2 );
        return;
    }
    auto boxmesh = std::dynamic_pointer_cast<const BoxMesh>( mesh );
    if ( !boxmesh )
        return;
    // Run the tests
    testElementFromPoint( ut, boxmesh );
    testSurface( ut, boxmesh );
}


} // namespace AMP::Mesh
