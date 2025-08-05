#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/testHelpers/meshTests.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"


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
    // Get a list of element ids on each surface id
    std::map<int, std::vector<MeshElementID>> map;
    for ( auto id : mesh->getBoundaryIDs() ) {
        auto it    = mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, id, 0 );
        auto &list = map[id];
        list.reserve( it.size() );
        for ( auto &elem : it )
            list.push_back( elem.globalID() );
        map[id] = mesh->getComm().allGather( list );
        AMP::Utilities::unique( map[id] );
    }
    // Check that each surface is in the list
    auto logicalDim = static_cast<int>( mesh->getGeomType() );
    bool pass       = true;
    for ( int s = 0; s < 2 * logicalDim; s++ ) {
        auto id   = mesh->getSurfaceID( s );
        auto it   = mesh->createIterator( mesh->getSurface( s, AMP::Mesh::GeomType::Vertex ) );
        auto list = map[id];
        for ( auto &elem : it )
            pass = pass && std::binary_search( list.begin(), list.end(), elem.globalID() );
    }
    if ( pass )
        ut.passes( mesh->getName() + " - getSurface" );
    else
        ut.failure( mesh->getName() + " - getSurface" );
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
