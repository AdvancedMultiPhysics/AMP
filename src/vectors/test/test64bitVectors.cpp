/* WARNING:  THIS TEST REQUIRES A LOT OF MEMORY
 *
 * This test tries to create and use vectors that have more than 4e9 elements
 *   (the limit for a 32-bit integer.
 * It will adjust the number of DOFs per node to reach this limit across the meshes.
 * To run this test requires at least 64 GB of memory divided evenly across the processors used.
 * 128 GB or more is recommended.
 *
 */

#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"

// Create a vector with the desired number of unknowns and run some simple tests
static void simpleDOFManagerVectorTest( AMP::UnitTest *ut,
                                        std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                        size_t N_DOFs,
                                        bool split )
{
    // Calculate the number of DOFs per Node (we require that the # of DOFs is >= N_DOFs)
    double avgDOFsPerNode =
        ( (double) N_DOFs ) / ( (double) mesh->numGlobalElements( AMP::Mesh::GeomType::Vertex ) );
    auto DOFsPerNode = (int) ceil( avgDOFsPerNode );
    // Create a simple DOFManager
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, DOFsPerNode, split );
    // Create the vector
    double start_time  = AMP::AMP_MPI::time();
    auto nodalVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "test" );
    auto v1            = AMP::LinearAlgebra::createVector( DOFs, nodalVariable, split );
    mesh->getComm().barrier();
    double end_time = AMP::AMP_MPI::time();
    if ( mesh->getComm().getRank() == 0 ) {
        std::cout << std::endl << "Vector size: " << v1->getGlobalSize() << std::endl;
        std::cout << "Time for vector create: " << end_time - start_time << std::endl;
    }
    // Initialize the vector and set some random values
    v1->zero();
    AMP_ASSERT( v1->L2Norm() == 0.0 );
    size_t index = v1->getGlobalSize() - 4;
    auto val     = (double) index;
    if ( v1->containsGlobalElement( index ) )
        v1->setValuesByGlobalID( 1, &index, &val );
    // Time makeConsistentSet
    mesh->getComm().barrier();
    start_time = AMP::AMP_MPI::time();
    v1->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    mesh->getComm().barrier();
    end_time = AMP::AMP_MPI::time();
    if ( mesh->getComm().getRank() == 0 )
        std::cout << "Time for makeConsistent: " << end_time - start_time << std::endl;
    // Time L2Norm
    start_time   = AMP::AMP_MPI::time();
    double norm2 = static_cast<double>( v1->L2Norm() );
    AMP_ASSERT( norm2 == val );
    mesh->getComm().barrier();
    end_time = AMP::AMP_MPI::time();
    if ( mesh->getComm().getRank() == 0 )
        std::cout << "Time for L2 norm: " << end_time - start_time << std::endl;
    ut->passes( "Test ran to completion" );
}

static void runTest( AMP::UnitTest *ut, const std::string &input_file )
{
    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Run the test with > 2^24  DOFs
    simpleDOFManagerVectorTest( ut, mesh, 0x1000001, false );
    simpleDOFManagerVectorTest( ut, mesh, 0x1000001, true );

    // Run the test with > 2^27  DOFs
    // simpleDOFManagerVectorTest( ut, mesh, 0x8000001, false );

    // Run the test with > 2^30 DOFs
    // simpleDOFManagerVectorTest( ut, mesh, 0x40000001, false );

    // Run the test with > 2^31 DOFs
    // simpleDOFManagerVectorTest( ut, mesh, 0x80000001, true );

    // Run the test with > 2^32 DOFs
    // simpleDOFManagerVectorTest( ut, mesh, 0x100000001, true );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    runTest( &ut, "input-test64bitVectors" );
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
