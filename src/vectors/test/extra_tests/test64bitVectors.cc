/*   WARNING:  THIS TEST REQUIRES A LOT OF MEMORY
 *
 *  This test tries to create and use vectors that have more than 4e9 elements (the limit for a 32-bit integer.
 *  It will adjust the number of DOFs per node to reach this limit across the meshes.  To run this
 *  test requires at least 64 GB of memory divided evenly across the processors used.  128 GB or more
 *  is recommended.
 *
 */


#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/PIO.h"
#include "ampmesh/Mesh.h"
#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "discretization/NodalVariable.h"
#include "vectors/VectorBuilder.h"


// Create a vector with the desired number of unknowns and run some simple tests
void  simpleDOFManagerVectorTest ( AMP::UnitTest *ut, AMP::Mesh::Mesh::shared_ptr mesh, size_t N_DOFs )
{
    // Calculate the number of DOFs per Node (we require that the # of DOFs is >= N_DOFs)
    double avgDOFsPerNode = ((double) N_DOFs) / ((double) mesh->numGlobalElements(AMP::Mesh::Vertex));
    int DOFsPerNode = (int) ceil(avgDOFsPerNode);
    // Create a simple DOFManager
    std::string varName = "test";
    AMP::LinearAlgebra::Variable::shared_ptr nodalVariable( new AMP::Discretization::NodalVariable(DOFsPerNode,varName) );
    AMP::Discretization::DOFManagerParameters::shared_ptr DOFparams( new AMP::Discretization::DOFManagerParameters(mesh) );
    boost::shared_ptr<AMP::Discretization::simpleDOFManager> DOFs( new AMP::Discretization::simpleDOFManager(mesh,AMP::Mesh::Vertex,1,DOFsPerNode) );
    // Create the vector
    AMP::LinearAlgebra::Vector::shared_ptr v1 = createVector( DOFs, nodalVariable );
    std::cout << std::endl << "Vector size: " << v1->getGlobalSize() << std::endl;
    // Initialize the vector and set some random values
    v1->zero();
    AMP_ASSERT(v1->L2Norm()==0.0);
    size_t index = v1->getGlobalSize()-4;
    double val = (double) index;
    if ( v1->containsGlobalElement(index) )
        v1->setValueByGlobalID(index,val);
    // Time makeConsistentSet
    mesh->getComm().barrier();
    double start_time = AMP::AMP_MPI::time();
    v1->makeConsistent ( AMP::LinearAlgebra::Vector::CONSISTENT_SET );
    mesh->getComm().barrier();
    double end_time = AMP::AMP_MPI::time();
    std::cout << std::endl << "Time for makeConsistent: " << end_time-start_time << std::endl;
    AMP_ASSERT(v1->L2Norm()==val);
}



void  runTest ( AMP::UnitTest *ut, std::string input_file )
{
    // Read the input file
    boost::shared_ptr<AMP::InputDatabase>  input_db ( new AMP::InputDatabase ( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile ( input_file , input_db );

    // Get the Mesh database and create the mesh parameters
    boost::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
    boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
    params->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));

    // Create the meshes from the input database
    boost::shared_ptr<AMP::Mesh::Mesh> mesh = AMP::Mesh::Mesh::buildMesh(params);

    // Run the test with > 2^24  DOFs
    simpleDOFManagerVectorTest( ut, mesh, 0x1000001 );

    // Run the test with > 2^31 DOFs
    //simpleDOFManagerVectorTest( ut, mesh, 0x80000001 );

    // Run the test with > 2^32 DOFs
    //simpleDOFManagerVectorTest( ut, mesh, 0x100000001 );

}


int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    runTest ( &ut, "input-test64bitVectors" );
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}   

