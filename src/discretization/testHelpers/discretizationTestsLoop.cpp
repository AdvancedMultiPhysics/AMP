#include "AMP/discretization/testHelpers/discretizationTestsLoop.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/structuredFaceDOFManager.h"
#include "AMP/discretization/subsetDOFManager.h"
#include "AMP/discretization/testHelpers/discretizationTests.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MeshVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"


#define PASS_FAIL( test, MSG ) \
    do {                       \
        if ( test )            \
            ut.passes( MSG );  \
        else                   \
            ut.failure( MSG ); \
    } while ( 0 )


namespace AMP::unit_test {


constexpr auto Vertex = AMP::Mesh::GeomType::Vertex;


// Function to test subsetting a DOF manager
void testSubsetDOFManager( std::shared_ptr<MeshGenerator> generator, bool split, AMP::UnitTest &ut )
{
    // Get the mesh
    generator->build_mesh();
    auto name = generator->name();
    auto mesh = generator->getMesh();

    // Create a simple DOF manager
    auto DOF = AMP::Discretization::simpleDOFManager::create( mesh, Vertex, 1, 1, split );
    testGetDOFIterator( ut, mesh->getIterator( Vertex, 1 ), DOF );

    // Run basic tests that should be valid for all DOFManagers
    testBasics( DOF, ut );
    testSubsetComm( DOF, ut );
    testSubsetMesh( mesh, DOF, true, 1, 1, ut );

    // Subset for iterators
    auto iterator  = mesh->getIterator( Vertex, 1 );
    auto subsetDOF = DOF->subset( iterator, mesh->getComm() );
    testGetDOFIterator( ut, mesh->getIterator( Vertex, 1 ), subsetDOF );
    PASS_FAIL( *DOF == *subsetDOF, "Subset DOF on full mesh iterator: " + name );
    iterator  = mesh->getSurfaceIterator( Vertex, 1 );
    subsetDOF = DOF->subset( iterator, mesh->getComm() );
    testGetDOFIterator( ut, mesh->getSurfaceIterator( Vertex, 1 ), subsetDOF );
    iterator = mesh->getSurfaceIterator( Vertex, 0 );
    PASS_FAIL( subsetDOF->numGlobalDOF() < DOF->numGlobalDOF() &&
                   subsetDOF->numLocalDOF() == iterator.size(),
               "Subset DOF on surface mesh iterator: " + name );

    // Subset using VectorSelector
    auto submesh = mesh->Subset( mesh->getSurfaceIterator( AMP::Mesh::GeomType::Face, 1 ) );
    AMP::LinearAlgebra::MeshIteratorVariable meshSubsetVariable(
        "surfaceSubset", submesh->getIterator( Vertex, 1 ), submesh->getComm() );
    subsetDOF = meshSubsetVariable.getSubsetDOF( DOF );
    testGetDOFIterator( ut, submesh->getIterator( Vertex, 1 ), subsetDOF );

    // Test subsetting for a subset mesh
    if ( mesh->getGeomType() == AMP::Mesh::GeomType::Cell ) {
        auto surface_mesh =
            mesh->Subset( mesh->getSurfaceIterator( AMP::Mesh::GeomType::Face, 1 ) );
        subsetDOF = DOF->subset( surface_mesh );
        size_t N1 = surface_mesh->numGlobalElements( Vertex );
        size_t N2 = subsetDOF->numGlobalDOF();
        PASS_FAIL( N1 == N2, "Subset DOF on surface mesh: " + name );
    }
}


// Function to test the creation/destruction of a simpleDOFManager
void testSimpleDOFManager( std::shared_ptr<MeshGenerator> generator, AMP::UnitTest &ut )
{
    using AMP::Discretization::simpleDOFManager;

    // Get the mesh
    generator->build_mesh();
    auto name = generator->name();
    auto mesh = generator->getMesh();

    // Create some simple DOF managers
    auto DOF1 = simpleDOFManager::create( mesh, Vertex, 0, 1, false );
    auto DOF2 = simpleDOFManager::create( mesh, Vertex, 1, 1, false );
    auto DOF3 = simpleDOFManager::create( mesh, Vertex, 1, 3, false );
    auto DOF4 = simpleDOFManager::create( mesh, Vertex, 1, 1, true );

    // Run basic tests that should be valid for all DOFManagers
    testBasics( DOF1, ut );
    testBasics( DOF2, ut );
    testBasics( DOF3, ut );
    testBasics( DOF4, ut );
    testSubsetComm( DOF1, ut );
    testSubsetComm( DOF2, ut );
    testSubsetComm( DOF3, ut );
    testSubsetComm( DOF4, ut );
    testSubsetMesh( mesh, DOF1, true, 1, 0, ut );
    testSubsetMesh( mesh, DOF2, true, 1, 1, ut );
    testSubsetMesh( mesh, DOF3, true, 3, 1, ut );
    testSubsetMesh( mesh, DOF4, true, 1, 1, ut );

    // Check some simple properties
    PASS_FAIL( DOF1->numLocalDOF() > 0, "Non-empty DOFs: " + name );
    PASS_FAIL( DOF1->endDOF() - DOF1->beginDOF() == DOF1->numLocalDOF(),
               "Non-empty DOFs: " + name );
    PASS_FAIL( DOF2->numLocalDOF() == DOF1->numLocalDOF() &&
                   DOF3->numLocalDOF() == 3 * DOF1->numLocalDOF() &&
                   DOF4->numLocalDOF() == DOF1->numLocalDOF() &&
                   DOF2->beginDOF() == DOF1->beginDOF() &&
                   DOF3->beginDOF() == 3 * DOF1->beginDOF() &&
                   DOF4->beginDOF() == DOF1->beginDOF() && DOF2->endDOF() == DOF1->endDOF() &&
                   DOF3->endDOF() == 3 * DOF1->endDOF() && DOF4->endDOF() == DOF1->endDOF(),
               "DOFs agree: " + name );

    // Check that the iterator size matches the mesh
    auto meshIterator = mesh->getIterator( Vertex, 0 );
    auto DOF1Iterator = DOF1->getIterator();
    auto DOF2Iterator = DOF2->getIterator();
    auto DOF3Iterator = DOF3->getIterator();
    auto DOF4Iterator = DOF4->getIterator();
    PASS_FAIL( DOF1Iterator.size() == meshIterator.size(),
               "DOF1 has the correct size for the iterator: " + name );
    PASS_FAIL( DOF2Iterator.size() == meshIterator.size(),
               "DOF2 has the correct size for the iterator: " + name );
    PASS_FAIL( DOF3Iterator.size() == meshIterator.size(),
               "DOF3 has the correct size for the iterator: " + name );
    PASS_FAIL( DOF4Iterator.size() == meshIterator.size(),
               "DOF4 has the correct size for the iterator: " + name );

    // Check the iterator based constructor
    DOF1 = AMP::Discretization::simpleDOFManager::create( mesh, Vertex, 1, 1, false );
    DOF2 = AMP::Discretization::simpleDOFManager::create(
        mesh, mesh->getIterator( Vertex, 1 ), mesh->getIterator( Vertex, 0 ), 1 );
    PASS_FAIL( DOF1->numGlobalDOF() == DOF2->numGlobalDOF(),
               "iterator-based constructor created: " + name );

    // Check that we can get the DOFs
    testGetDOFIterator( ut, mesh->getIterator( Vertex, 0 ), DOF1 );
    testGetDOFIterator( ut, mesh->getIterator( Vertex, 1 ), DOF2 );
    testGetDOFIterator( ut, mesh->getIterator( Vertex, 1 ), DOF3 );
    testGetDOFIterator( ut, mesh->getIterator( Vertex, 1 ), DOF4 );
}


// Function to test the creation/destruction of a multiDOFManager
void testMultiDOFManager( std::shared_ptr<MeshGenerator> generator, AMP::UnitTest &ut )
{
    // Get the mesh
    generator->build_mesh();
    auto name = generator->name();
    auto mesh = generator->getMesh();

    // Create a simple DOF manager and check if it is a multiDOF manager
    auto DOFs = AMP::Discretization::simpleDOFManager::create( mesh, Vertex, 1, 1, true );
    if ( std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh ) ) {
        auto multiDOF = std::dynamic_pointer_cast<AMP::Discretization::multiDOFManager>( DOFs );
        if ( multiDOF ) {
            ut.passes( "Created multiDOFManager from simpleDOFManager: " + name );
            testMultiDOFMap( ut, multiDOF );
        } else {
            ut.failure( "Created multiDOFManager from simpleDOFManager: " + name );
        }
    }

    // Run basic tests that should be valid for all DOFManagers
    testBasics( DOFs, ut );
    testSubsetComm( DOFs, ut );
    testSubsetMesh( mesh, DOFs, true, 1, 1, ut );

    // Test a multivector that contains multiple vectors with the same DOFManager
    testMultiDOFVector( ut, DOFs );

    // Create a multiDOFManager with repeated mesh elements and make sure the iterator only iterates
    // once through each element
    std::vector<std::shared_ptr<AMP::Discretization::DOFManager>> managers( 2, DOFs );
    auto DOF2 = std::make_shared<AMP::Discretization::multiDOFManager>( DOFs->getComm(), managers );
    auto iterator1 = DOFs->getIterator();
    auto iterator2 = DOF2->getIterator();
    PASS_FAIL( iterator1.size() == iterator2.size(),
               "multiDOFManager iterates once through each element: " + name );

    // Run basic tests that should be valid for all DOFManagers
    testBasics( DOF2, ut );
    testSubsetComm( DOF2, ut );
    testSubsetMesh( mesh, DOF2, true, 2, 1, ut );
}


// Function to test the creation/destruction of a structuredFaceDOFManager
void testStructureDOFManager(
    std::shared_ptr<MeshGenerator> generator, int Nx, int Ny, int Nz, int GCW, AMP::UnitTest &ut )
{
    // Get the mesh
    generator->build_mesh();
    auto name = generator->name();
    auto mesh = generator->getMesh();

    // Create a simple DOF manager and check if it is a multiDOF manager
    int dofsPerFace[3] = { Nx, Ny, Nz };
    auto DOFs =
        std::make_shared<AMP::Discretization::structuredFaceDOFManager>( mesh, dofsPerFace, GCW );

    // Run basic tests that should be valid for all DOFManagers
    testBasics( DOFs, ut );
    testSubsetComm( DOFs, ut );
    testSubsetMesh( mesh, DOFs, false, 0, GCW, ut );

    // Check getRowDOFs
    if ( Nx > 0 && Ny > 0 && Nz > 0 ) {
        bool pass = true;
        auto it   = mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
        for ( size_t ii = 0; ii < it.size(); ++ii, ++it ) {
            auto faces = it->getElements( AMP::Mesh::GeomType::Face );
            for ( size_t i = 0; i < faces.size(); i++ ) {
                std::vector<size_t> rows = DOFs->getRowDOFs( faces[i].globalID() );
                std::vector<size_t> dofs;
                for ( size_t j = 0; j < faces.size(); j++ ) {
                    DOFs->getDOFs( faces[j].globalID(), dofs );
                    for ( size_t k = 0; k < dofs.size(); k++ ) {
                        size_t index = AMP::Utilities::findfirst( rows, dofs[k] );
                        if ( index == rows.size() ) {
                            index--;
                        }
                        if ( rows[index] != dofs[k] )
                            pass = false;
                    }
                }
            }
        }
        PASS_FAIL( pass, "getRowDOFs found all faces that share a volume: " + name );
    }
}


} // namespace AMP::unit_test
