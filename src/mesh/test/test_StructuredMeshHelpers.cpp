#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"


void runTest( AMP::UnitTest *ut )
{
    // Create a simple structured mesh
    size_t N_faces_tot = 227;
    AMP::unit_test::AMPCubeGenerator generator( 3, 4, 5 );
    generator.build_mesh();
    auto mesh = generator.getMesh();

    // Get iterators over each of the face types
    auto faces       = mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    auto x_faces     = AMP::Mesh::StructuredMeshHelper::getYZFaceIterator( mesh, 0 );
    auto y_faces     = AMP::Mesh::StructuredMeshHelper::getXZFaceIterator( mesh, 0 );
    auto z_faces     = AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( mesh, 0 );
    size_t N_faces   = mesh->getComm().sumReduce( faces.size() );
    size_t N_x_faces = mesh->getComm().sumReduce( x_faces.size() );
    size_t N_y_faces = mesh->getComm().sumReduce( y_faces.size() );
    size_t N_z_faces = mesh->getComm().sumReduce( z_faces.size() );
    if ( N_faces_tot == N_faces )
        ut->passes( "Total number of faces match" );
    else
        ut->failure( "Total number of faces match" );
    if ( N_x_faces + N_y_faces + N_z_faces == N_faces )
        ut->passes( "Number of faces match" );
    else
        ut->failure( "Number of faces match" );

    if ( mesh->getComm().getSize() > 1 ) {
        bool pass = false;
        for ( int d = 0; d < 3; d++ ) {
            auto face0 = AMP::Mesh::StructuredMeshHelper::getYZFaceIterator( mesh, 0 );
            auto face1 = AMP::Mesh::StructuredMeshHelper::getYZFaceIterator( mesh, 1 );
            if ( face1.size() > face0.size() )
                pass = true;
        }
        if ( pass )
            ut->passes( "Found ghost faces on all processors" );
        else
            ut->failure( "Found ghost faces on all processors" );
    }
}


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    runTest( &ut );
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
