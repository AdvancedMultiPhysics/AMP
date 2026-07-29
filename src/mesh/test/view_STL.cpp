#include "AMP/IO/PIO.h"
#include "AMP/IO/Writer.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"

#include <string>


int main( int argc, char **argv )
{
    // Startup
    AMP::AMPManager::startup( argc, argv );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Loop through the input arguments
    for ( int i = 1; i < argc; i++ ) {

        // Get the input filename
        std::string filename = argv[i];
        auto filename2       = filename.substr( 0, filename.rfind( '.' ) );
        if ( filename2.rfind( '/' ) != std::string::npos )
            filename2 = filename2.substr( filename2.rfind( '/' ) + 1 );
        if ( filename2.rfind( '\\' ) != std::string::npos )
            filename2 = filename2.substr( filename2.rfind( '\\' ) + 1 );
        std::cout << filename2 << ": " << filename << std::endl;

        // Create the input mesh database
        auto db = std::make_shared<AMP::Database>( "Mesh" );
        db->putScalar( "MeshType", "AMP" );
        db->putScalar( "MeshName", filename2 );
        db->putScalar( "FileName", filename );

        // Create the mesh
        auto params = std::make_shared<AMP::Mesh::MeshParameters>( db );
        params->setComm( globalComm );
        auto mesh = AMP::Mesh::MeshFactory::create( params );

        // Create the silo writer and register the data
        AMP::IO::Writer::WriterParameters writerParams;
        writerParams.decomposition = AMP::IO::Writer::DecompositionType::SINGLE;
        auto siloWriter            = AMP::IO::Writer::buildWriter( "Silo", writerParams );
        siloWriter->registerMesh( mesh, 1 );
        globalComm.barrier();

        // Write the output
        siloWriter->writeFile( filename2, 0 );
        globalComm.barrier();
    }

    // Shutdown
    AMP::AMPManager::shutdown();
    return 0;
}
