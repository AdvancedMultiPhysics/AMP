#include "AMP/IO/FileSystem.h"
#include "AMP/IO/Writer.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"

#include <sstream>
#include <string>


void createMesh( AMP::UnitTest *ut, const std::string &input_file )
{
    // Create a global communicator
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create the silo writer and write the mesh
    AMP::IO::recursiveMkdir( "output" );
    std::string output_file = "output/" + input_file;
    AMP::IO::Writer::WriterParameters writerParams;
    writerParams.decomposition = AMP::IO::Writer::DecompositionType::SINGLE;
    auto siloWriter            = AMP::IO::Writer::buildWriter( "Silo", writerParams );
    if ( siloWriter != nullptr ) {
        siloWriter->registerMesh( mesh, 1 );
        siloWriter->writeFile( output_file, 0 );
    }

    // Finished
    ut->passes( "example ran to completion" );
}


int main( int argc, char **argv )
{
    // Initialize AMP
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Run the example
    AMP_INSIST( argc == 2, "Usage is:  createmesh  inputFile" );
    std::string filename( argv[1] );
    createMesh( &ut, filename );

    // Report the results and exit
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
