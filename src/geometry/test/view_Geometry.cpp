#include "AMP/IO/Writer.h"
#include "AMP/geometry/Geometry.h"
#include "AMP/geometry/LogicalGeometry.h"
#include "AMP/geometry/MultiGeometry.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/structured/StructuredGeometryMesh.h"
#include "AMP/utils/AMPManager.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>


std::shared_ptr<AMP::Mesh::Mesh> buildMesh( std::shared_ptr<AMP::Geometry::Geometry> geom )
{
    if ( std::dynamic_pointer_cast<AMP::Geometry::MultiGeometry>( geom ) ) {
        AMP_ERROR( "Not finished" );
    } else if ( std::dynamic_pointer_cast<AMP::Geometry::LogicalGeometry>( geom ) ) {
        std::cout << "Building mesh for " << geom->getName() << std::endl;
        auto logical = std::dynamic_pointer_cast<AMP::Geometry::LogicalGeometry>( geom );
        auto domain  = logical->box();
        auto length  = ( domain.second - domain.first ).abs();
        AMP::ArraySize size;
        if ( logical->getDim() == 1 )
            size = logical->getLogicalGridSize( std::vector<double>( 1, length / 100 ) );
        else if ( logical->getDim() == 2 )
            size = logical->getLogicalGridSize( std::vector<double>( 2, length / 50 ) );
        else if ( logical->getDim() == 3 )
            size = logical->getLogicalGridSize( std::vector<double>( 3, length / 20 ) );
        return std::make_shared<AMP::Mesh::StructuredGeometryMesh>( logical, size, AMP_COMM_WORLD );
    } else {
        AMP_ERROR( "Unknown geometry" );
    }
}


void viewGeometry( const std::string &input )
{
    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input );

    // Create the writer
    auto writer = AMP::IO::Writer::buildWriter( "auto", AMP_COMM_WORLD );

    // Loop through the databases, load the geometry, and build and register a mesh
    std::set<std::string> meshNames;
    for ( auto key : input_db->getAllKeys() ) {
        auto db   = input_db->getDatabase( key );
        auto geom = AMP::Geometry::Geometry::buildGeometry( db );
        auto mesh = buildMesh( geom );
        auto name = mesh->getName();
        std::replace( name.begin(), name.end(), '<', '_' );
        std::replace( name.begin(), name.end(), '>', '_' );
        int i = 2;
        while ( meshNames.find( name ) != meshNames.end() )
            name = mesh->getName() + "_" + std::to_string( i++ );
        meshNames.insert( name );
        mesh->setName( name );
        writer->registerMesh( mesh );
    }

    // Write the data
    writer->writeFile( input, 0 );
}


int main( int argc, char **argv )
{
    if ( argc != 2 ) {
        std::cerr << "Incorrect usage:\n";
        std::cerr << "   " << argv[0] << " <input>\n";
        return -1;
    }
    AMP::AMPManager::startup( argc, argv );

    viewGeometry( argv[1] );

    AMP::AMPManager::shutdown();
    return 0;
}
