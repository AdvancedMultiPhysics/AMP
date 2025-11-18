#include "AMP/AMP_TPLs.h"
#include "AMP/mesh/testHelpers/meshTests.h"

#include "ProfilerApp.h"


namespace AMP::Mesh {


void meshTests::MeshTestLoop( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "MeshTestLoop" );
    // Run some basic sanity checks
    MeshBasicTest( ut, mesh );
    // Test the number of elements
    MeshCountTest( ut, mesh );
    // Test the iterators
    MeshIteratorTest( ut, mesh );
    MeshIteratorOperationTest( ut, mesh );
    VerifyBoundaryIDNodeIterator( ut, mesh );
    VerifyBoundaryIterator( ut, mesh );
    testBlockIDs( ut, mesh );
    MeshIteratorSetOPTest( ut, mesh );
    VerifyGhostIsOwned( ut, mesh );
    // Test the node neighbors
    getNodeNeighbors( ut, mesh );
    // Test displacement
    if ( mesh->isMeshMovable() >= AMP::Mesh::Mesh::Movable::Displace )
        DisplaceMeshScalar( ut, mesh );
    if ( mesh->isMeshMovable() >= AMP::Mesh::Mesh::Movable::Deform )
        DisplaceMeshVector( ut, mesh );
    // Test cloneMesh
    cloneMesh( ut, mesh );
    // Test the elements
    VerifyNodeElemMapIteratorTest( ut, mesh );
    VerifyBoundaryIteratorTest( ut, mesh );
    VerifyElementForNode( ut, mesh );
    // Test BoxMesh
    testBoxMesh( ut, mesh );
    // Test performance
    MeshPerformance( ut, mesh );
}


void meshTests::MeshGeometryTestLoop( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    // Return if we do not have a geometry to work with
    if ( !mesh->getGeometry() )
        return;
    PROFILE( "MeshGeometryTestLoop" );
    // Run the mesh-geometry based tests
    TestBasicGeometry( ut, mesh );
    TestInside( ut, mesh );
    TestPhysicalLogical( ut, mesh );
    TestNormalGeometry( ut, mesh );
}


void meshTests::MeshVectorTestLoop( AMP::UnitTest &ut,
                                    std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                    bool fast )
{
    // Run the vector tests
    PROFILE( "MeshVectorTestLoop" );
    VerifyGetVectorTest<1, true>( ut, mesh );
    if ( !fast ) {
        VerifyGetVectorTest<3, true>( ut, mesh );
        VerifyGetVectorTest<1, false>( ut, mesh );
        VerifyGetVectorTest<3, false>( ut, mesh );
    }
}


void meshTests::MeshMatrixTestLoop( AMP::UnitTest &ut,
                                    std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                    bool fast )
{
    // Run the matrix tests
    bool run_tests = true;
    if ( run_tests ) {
        PROFILE( "MeshMatrixTestLoop" );
        VerifyGetMatrixTrivialTest<1, true>( ut, mesh );
        GhostWriteTest<1, true>( ut, mesh );
        if ( !fast ) {
            VerifyGetMatrixTrivialTest<3, true>( ut, mesh );
            VerifyGetMatrixTrivialTest<1, false>( ut, mesh );
            VerifyGetMatrixTrivialTest<3, false>( ut, mesh );
            GhostWriteTest<3, true>( ut, mesh );
            GhostWriteTest<1, false>( ut, mesh );
            GhostWriteTest<3, false>( ut, mesh );
        }
    }
}


} // namespace AMP::Mesh
