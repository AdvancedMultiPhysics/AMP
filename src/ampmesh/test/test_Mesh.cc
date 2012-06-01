#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/AMP_MPI.h"
#include "utils/ProfilerApp.h"
#include "ampmesh/Mesh.h"
#include "ampmesh/MeshElement.h"
#include "utils/MemoryDatabase.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "meshTestLoop.h"
#include "meshTests.h"
#include "meshGenerators.h"

#include "ampmesh/structured/BoxMesh.h"
#ifdef USE_LIBMESH
    #include "ampmesh/libmesh/libMesh.h"
#endif
#ifdef USE_MOAB
    #include "ampmesh/moab/moabMesh.h"
#endif


// Function to test the creation/destruction of a mesh with the mesh generators
// Note: this only runs the mesh tests, not the vector or matrix tests
void testMeshGenerators( AMP::UnitTest *ut )
{
    boost::shared_ptr<AMP::unit_test::MeshGenerator> generator;
    // libmesh generators
    #ifdef USE_LIBMESH
        // Test the libmesh cube generator
        generator = boost::shared_ptr<AMP::unit_test::MeshGenerator>( new AMP::unit_test::LibMeshCubeGenerator<5> );
        generator->build_mesh();
        MeshTestLoop( ut, generator->getMesh() );
        // Test the libmesh reader generator
        generator = boost::shared_ptr<AMP::unit_test::MeshGenerator>( new AMP::unit_test::ExodusReaderGenerator<> );
        generator->build_mesh();
        MeshTestLoop( ut, generator->getMesh() );
        generator = boost::shared_ptr<AMP::unit_test::MeshGenerator>( new AMP::unit_test::ExodusReaderGenerator<2> );
        generator->build_mesh();
        MeshTestLoop( ut, generator->getMesh() );
        // Test the ThreeElementLGenerator generator
        generator = boost::shared_ptr<AMP::unit_test::MeshGenerator>( new AMP::unit_test::libMeshThreeElementGenerator );
        generator->build_mesh();
        MeshTestLoop( ut, generator->getMesh() );
    #endif
    // Test the multimesh generator
    generator = boost::shared_ptr<AMP::unit_test::MeshGenerator>( new AMP::unit_test::MultiMeshGenerator );
    generator->build_mesh();
    MeshTestLoop( ut, generator->getMesh() );
    MeshVectorTestLoop( ut, generator->getMesh() );
}


// Function to test the creation/destruction of a native AMP mesh
void testAMPMesh( AMP::UnitTest *ut )
{
    // Set the dimensions of the mesh
    std::vector<int> size(3,2);
    std::vector<double> range(6,0.0);
    range[1] = 1.0;
    range[3] = 1.0;
    range[5] = 1.0;
    // Create a generic MeshParameters object
    boost::shared_ptr<AMP::MemoryDatabase> database(new AMP::MemoryDatabase("Mesh"));
    database->putInteger("dim",3);
    database->putString("MeshName","mesh1");
    database->putString("Generator","cube");
    database->putIntegerArray("Size",size);
    database->putDoubleArray("Range",range);
    boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
    params->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));

    // Create an AMP mesh
    boost::shared_ptr<AMP::Mesh::BoxMesh> mesh(new AMP::Mesh::BoxMesh(params));    

    // Check the basic dimensions
    size_t N_elements_global = size[0]*size[1]*size[2];
    size_t N_faces_global = (size[0]+1)*size[1]*size[2] + size[0]*(size[1]+1)*size[2] + size[0]*size[1]*(size[2]+1);
    size_t N_edges_global = size[0]*(size[1]+1)*(size[2]+1) + (size[0]+1)*size[1]*(size[2]+1) + (size[0]+1)*(size[1]+1)*size[2];
    size_t N_nodes_global = (size[0]+1)*(size[1]+1)*(size[2]+1);
    if ( mesh->numGlobalElements(AMP::Mesh::Vertex) == N_nodes_global )
        ut->passes("Simple structured mesh has expected number of nodes");
    else
        ut->failure("Simple structured mesh has expected number of nodes");
    if ( mesh->numGlobalElements(AMP::Mesh::Edge) == N_edges_global )
        ut->passes("Simple structured mesh has expected number of edges");
    else
        ut->failure("Simple structured mesh has expected number of edges");
    if ( mesh->numGlobalElements(AMP::Mesh::Face) == N_faces_global )
        ut->passes("Simple structured mesh has expected number of faces");
    else
        ut->failure("Simple structured mesh has expected number of faces");
    if ( mesh->numGlobalElements(AMP::Mesh::Volume) == N_elements_global )
        ut->passes("Simple structured mesh has expected number of elements");
    else
        ut->failure("Simple structured mesh has expected number of elements");

    // Check the volumes
    double dx = range[1]/size[0];
    AMP::Mesh::MeshIterator iterator = mesh->getIterator(AMP::Mesh::Edge);
    bool passes = true;
    for (size_t i=0; i<iterator.size(); i++) {
        if ( !AMP::Utilities::approx_equal( iterator->volume(), dx, 1e-12 ) )
            passes = false;
    }
    if ( passes )
        ut->passes("Simple structured mesh has correct edge legth");
    else
        ut->failure("Simple structured mesh has correct edge legth");
    iterator = mesh->getIterator(AMP::Mesh::Face);
    passes = true;
    for (size_t i=0; i<iterator.size(); i++) {
        if ( !AMP::Utilities::approx_equal( iterator->volume(), dx*dx, 1e-12 ) )
            passes = false;
    }
    if ( passes )
        ut->passes("Simple structured mesh has correct face area");
    else
        ut->failure("Simple structured mesh has correct face area");
    iterator = mesh->getIterator(AMP::Mesh::Volume);
    passes = true;
    for (size_t i=0; i<iterator.size(); i++) {
        if ( !AMP::Utilities::approx_equal( iterator->volume(), dx*dx*dx, 1e-12 ) )
            passes = false;
    }
    if ( passes )
        ut->passes("Simple structured mesh has correct element volume");
    else
        ut->failure("Simple structured mesh has correct element volume");

    // Run the mesh tests
    MeshTestLoop( ut, mesh );
    MeshVectorTestLoop( ut, mesh );
    MeshMatrixTestLoop( ut, mesh );

}


// Function to test the creation/destruction of a libmesh mesh
#ifdef USE_LIBMESH
void testlibMesh( AMP::UnitTest *ut )
{
    // Create a generic MeshParameters object
    boost::shared_ptr<AMP::MemoryDatabase> database(new AMP::MemoryDatabase("Mesh"));
    database->putInteger("dim",3);
    database->putString("MeshName","mesh1");
    database->putString("FileName","pellet_lo_res.e");
    boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
    params->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));

    // Create an libMesh mesh
    boost::shared_ptr<AMP::Mesh::libMesh> mesh(new AMP::Mesh::libMesh(params));    

    // Run the mesh tests
    MeshTestLoop( ut, mesh );
    MeshVectorTestLoop( ut, mesh );
    MeshMatrixTestLoop( ut, mesh );

}
#endif


// Function to test the creation/destruction of a moab mesh
#ifdef USE_MOAB
void testMoabMesh( AMP::UnitTest *ut )
{
    // Create a generic MeshParameters object
    boost::shared_ptr<AMP::MemoryDatabase> database(new AMP::MemoryDatabase("Mesh"));
    database->putInteger("dim",3);
    database->putString("MeshName","mesh1");
    database->putString("FileName","pellet_lo_res.e");
    boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
    params->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));

    // Create an MOAB mesh
    try {
        boost::shared_ptr<AMP::Mesh::moabMesh> mesh(new AMP::Mesh::moabMesh(params));    
    } catch (...) {
        ut->expected_failure("MOAB meshes cannot be created yet");
    }

    // Run the mesh tests
    ut->expected_failure("Mesh tests not working on a MOAB mesh yet");
    //MeshTestLoop( ut, mesh );
    //MeshVectorTestLoop( ut, mesh );
    //MeshMatrixTestLoop( ut, mesh );

}
#endif


void testInputMesh( AMP::UnitTest *ut, std::string filename )
{
    // Read the input file
    boost::shared_ptr<AMP::InputDatabase>  input_db ( new AMP::InputDatabase ( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile ( filename , input_db );

    // Get the Mesh database and create the mesh parameters
    boost::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
    boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
    params->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));

    // Create the meshes from the input database
    boost::shared_ptr<AMP::Mesh::Mesh> mesh = AMP::Mesh::Mesh::buildMesh(params);

    // Run the mesh tests
    MeshTestLoop( ut, mesh );
    MeshVectorTestLoop( ut, mesh );
    MeshMatrixTestLoop( ut, mesh );

}


void testSubsetMesh( AMP::UnitTest *ut )
{
    #ifdef USE_LIBMESH
    // Subset a mesh for a surface without ghost cells and test
    boost::shared_ptr<AMP::unit_test::MeshGenerator>  generator( 
        new AMP::unit_test::SurfaceSubsetGenerator< AMP::unit_test::ExodusReaderGenerator<>,0> );
    generator->build_mesh();
    AMP::Mesh::Mesh::shared_ptr mesh = generator->getMesh();
    MeshTestLoop( ut, mesh );
    //MeshVectorTestLoop( ut, mesh );
    //MeshMatrixTestLoop( ut, mesh );

    // Subset a mesh for a surface with ghost cells and test
    generator = boost::shared_ptr<AMP::unit_test::MeshGenerator> ( 
        new AMP::unit_test::SurfaceSubsetGenerator< AMP::unit_test::ExodusReaderGenerator<3>,1> );
    generator->build_mesh();
    mesh = generator->getMesh();
    MeshTestLoop( ut, mesh );
    //MeshVectorTestLoop( ut, mesh );
    //MeshMatrixTestLoop( ut, mesh );
    #endif
}



// Main function
int main ( int argc , char ** argv )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup(argc,argv,startup_properties);
    AMP::UnitTest ut;
    PROFILE_ENABLE();
    PROFILE_START("Run tests");

    // Run the ID test
    testID( &ut );

    // Run tests on a native AMP mesh
    //testAMPMesh( &ut );

    // Run tests on a libmesh mesh
    #ifdef USE_LIBMESH
        testlibMesh( &ut );
    #endif

    // Run tests on a moab mesh
    #ifdef USE_MOAB
        testMoabMesh( &ut );
    #endif

    // Run tests on the input file
    #ifdef USE_LIBMESH
        testInputMesh( &ut, "input_Mesh" );
    #endif

    // Run the basic tests on all mesh generators
    #ifdef USE_LIBMESH
        testMeshGenerators( &ut );
    #endif

    // Run the tests on the subset meshes
    #ifdef USE_LIBMESH
        testSubsetMesh( &ut );
    #endif

    // Save the timing results
    PROFILE_STOP("Run tests");
    PROFILE_SAVE("test_Mesh");

    // Print the results and return
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;

}
