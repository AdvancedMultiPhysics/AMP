#include <string>
#include <sstream>

#include "utils/Utilities.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/PIO.h"

#include "ampmesh/Mesh.h"
#include "ampmesh/SiloIO.h"

#ifdef USE_AMP_VECTORS
#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/VectorBuilder.h"
#include "vectors/Variable.h"
#include "vectors/Vector.h"
#include "vectors/VectorSelector.h"
#endif


void test_Silo( AMP::UnitTest *ut, std::string input_file ) {

    AMP::PIO::logOnlyNodeZero ( "outputMeshManagerTest1" );
    AMP::AMP_MPI globalComm(AMP_COMM_WORLD);
    globalComm.barrier();
    double t1 = AMP::AMP_MPI::time();

    // Read the input file
    boost::shared_ptr<AMP::InputDatabase>  input_db ( new AMP::InputDatabase ( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile ( input_file , input_db );
    input_db->printClassData (AMP::plog);

    // Get the Mesh database and create the mesh parameters
    boost::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
    boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
    params->setComm(globalComm);

    // Create the meshes from the input database
    boost::shared_ptr<AMP::Mesh::Mesh> mesh = AMP::Mesh::Mesh::buildMesh(params);
    globalComm.barrier();
    double t2 = AMP::AMP_MPI::time();

#ifdef USE_AMP_VECTORS
    // Create a simple DOFManager
    AMP::Discretization::DOFManagerParameters::shared_ptr DOFparams( new AMP::Discretization::DOFManagerParameters(mesh) );
    AMP::Discretization::DOFManager::shared_ptr DOF_scalar = AMP::Discretization::simpleDOFManager::create(mesh,AMP::Mesh::Vertex,1,1,true);
    AMP::Discretization::DOFManager::shared_ptr DOF_vector = AMP::Discretization::simpleDOFManager::create(mesh,AMP::Mesh::Vertex,1,3,true);

    // Create the vectors
    AMP::LinearAlgebra::Variable::shared_ptr rank_var( new AMP::LinearAlgebra::Variable("rank") );
    AMP::LinearAlgebra::Vector::shared_ptr rank_vec = AMP::LinearAlgebra::createVector( DOF_scalar, rank_var, true );
    AMP::LinearAlgebra::Variable::shared_ptr position_var( new AMP::LinearAlgebra::Variable("position") );
    AMP::LinearAlgebra::Vector::shared_ptr position = AMP::LinearAlgebra::createVector( DOF_vector, position_var, true );
    //AMP::LinearAlgebra::Variable::shared_ptr  gp_var ( new AMP::Mesh::SingleGaussPointVariable ( "gp_var" ) );
    //AMP::LinearAlgebra::Variable::shared_ptr  gp_var2 ( new AMP::LinearAlgebra::VectorVariable<AMP::Mesh::IntegrationPointVariable , 8> ( "gp_var2" ) );
    //gp_var->setUnits ( "newton-fathom / acre^2" );
    //AMP::LinearAlgebra::Vector::shared_ptr  gauss_pt = manager->createVector ( gp_var );
    //AMP::LinearAlgebra::Vector::shared_ptr  gauss_pt2 = manager->createVector ( gp_var2 );
    //AMP::LinearAlgebra::Vector::shared_ptr  displacement = manager->createPositionVector ( "displacement" );
    //displacement->getVariable()->setUnits ( "leagues" );
    //gauss_pt2->setToScalar ( 100 );
    globalComm.barrier();
#endif
    double t3 = AMP::AMP_MPI::time();

    // Create a subset mesh and view of a vector
    AMP::Mesh::Mesh::shared_ptr submesh = mesh->Subset( mesh->getSurfaceIterator(AMP::Mesh::Face) );
    #ifdef USE_AMP_VECTORS
        AMP::LinearAlgebra::VS_MeshIterator meshSelector( "positionSubset", submesh->getIterator(AMP::Mesh::Vertex,0), submesh->getComm() );
        AMP::LinearAlgebra::VS_Stride zSelector("thirds",2,3);
        AMP::LinearAlgebra::Vector::shared_ptr  vec_meshSubset = position->select( meshSelector, "mesh subset" );
        AMP_ASSERT(vec_meshSubset.get()!=NULL);
        AMP::LinearAlgebra::Vector::shared_ptr  z_surface = vec_meshSubset->select( zSelector, "z surface" );
        AMP_ASSERT(z_surface.get()!=NULL);
    #endif

    // Create the silo writer and register the data
    AMP::Mesh::SiloIO::shared_ptr  siloWriter( new AMP::Mesh::SiloIO);
    siloWriter->registerMesh( mesh );
    siloWriter->registerMesh( submesh );
#ifdef USE_AMP_VECTORS
    siloWriter->registerVector( rank_vec, mesh, AMP::Mesh::Vertex, "rank" );
    siloWriter->registerVector( position, mesh, AMP::Mesh::Vertex, "position" );
    siloWriter->registerVector( z_surface, submesh, AMP::Mesh::Vertex, "z_surface" );
    //siloWriter->registerVector( gauss_pt );
    //siloWriter->registerVector( gauss_pt2 );
#endif
    globalComm.barrier();
    double t4 = AMP::AMP_MPI::time();

    // Initialize the data
#ifdef USE_AMP_VECTORS
    rank_vec->setToScalar(globalComm.getRank());
    rank_vec->makeConsistent( AMP::LinearAlgebra::Vector::CONSISTENT_SET );
    std::vector<size_t> dofs;
    for (AMP::Mesh::MeshIterator it=DOF_vector->getIterator(); it!=it.end(); it++) {
        AMP::Mesh::MeshElementID id = it->globalID();
        DOF_vector->getDOFs( id, dofs );
        std::vector<double> pos = it->coord();
        position->setValuesByGlobalID( dofs.size(), &dofs[0], &pos[0] );
    }
    position->makeConsistent( AMP::LinearAlgebra::Vector::CONSISTENT_SET );
    globalComm.barrier();
#endif
    double t5 = AMP::AMP_MPI::time();

    // Write the file
    std::stringstream  fname;
    fname << "2pellet_clad_" << globalComm.getSize() << "proc";
    globalComm.barrier();
    siloWriter->writeFile( fname.str() , 0 );
    globalComm.barrier();
    double t6 = AMP::AMP_MPI::time();

    if ( globalComm.getRank() == 0 ) {
        std::cout << "Read in meshes: " << t2-t1 << std::endl;
        std::cout << "Allocate vectors: " << t3-t2 << std::endl;
        std::cout << "Register data: " << t4-t3 << std::endl;
        std::cout << "Initialize vectors: " << t5-t4 << std::endl;
        std::cout << "Write a file: " << t6-t5 << std::endl;
        std::cout << "Total time: " << t6-t1 << std::endl;
    }
}


int main ( int argc , char **argv )
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    #ifdef USE_SILO
        std::string filename = "input_SiloIO";
        if(argc == 2) filename = argv[1];
        test_Silo( &ut, filename );
    #else
        ut->expected_failure("AMP was not configured with silo");
    #endif

    ut.report();
    
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}

