#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/testHelpers/meshTests.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Array.hpp"
#include "AMP/utils/Database.h"


// Print the result of the comparisons
void printResults( const AMP::Array<AMP::Mesh::Mesh::CompareResult> &results )
{
    if ( AMP::AMP_MPI( AMP_COMM_WORLD ).getRank() != 0 )
        return;
    if ( results.size( 0 ) == 2 ) {
        auto r = results( 1 );
        std::cout << r.result() << ":\n";
        if ( r.equal ) {
            std::cout << "  Meshes are equal\n";
        } else {
            auto print2 = []( bool t, const std::string &msg ) {
                if ( t )
                    std::cout << msg << " match\n";
                else
                    std::cout << msg << " do not match\n";
            };
            print2( r.nodes, "  Nodes" );
            print2( r.block, "  Blocks" );
            print2( r.surface, "  Surface IDs" );
            print2( r.domain, "  Domains" );
            print2( r.geometry, "  Geometries" );
        }
    } else {
        AMP::Array<int> result2( results.size() );
        for ( size_t i = 0; i < results.length(); i++ )
            result2( i ) = results( i ).result();
        result2.print( std::cout, "compare" );
        std::cout << std::endl;
    }
}


// Compare meshes
bool compare( const std::string &filename )
{
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Create the meshes
    auto input_db = AMP::Database::parseInputFile( filename );
    std::vector<std::shared_ptr<AMP::Mesh::Mesh>> meshes;
    for ( int i = 1; i < 1000; i++ ) {
        auto key = "Mesh_" + std::to_string( i );
        if ( !input_db->keyExists( key ) )
            break;
        auto db     = input_db->getDatabase( key );
        auto params = std::make_shared<AMP::Mesh::MeshParameters>( db );
        params->setComm( globalComm );
        meshes.push_back( AMP::Mesh::MeshFactory::create( params ) );
    }

    // Compare the meshes
    AMP::Array<AMP::Mesh::Mesh::CompareResult> result( meshes.size(), meshes.size() );
    result.fill( 0 );
    for ( size_t i = 0; i < meshes.size(); i++ ) {
        for ( size_t j = 0; j < meshes.size(); j++ )
            result( i, j ) = AMP::Mesh::Mesh::compare( *meshes[i], *meshes[j] );
    }

    // Check we get the same result on all ranks
    int error = 0;
    auto tmp  = result;
    globalComm.bcast( tmp.data(), tmp.length(), 0 );
    if ( tmp != result ) {
        std::cerr << "Results do not match across ranks:\n";
        error = 1;
    }

    // Sanity check the comparison matrix
    for ( size_t i = 0; i < meshes.size(); i++ ) {
        AMP_ASSERT( result( i, i ).equal );
        for ( size_t j = 0; j < meshes.size(); j++ )
            AMP_ASSERT( result( i, j ) == result( j, i ) );
    }

    // Check the answer (if present)
    if ( input_db->keyExists( "ans" ) ) {
        auto ans  = input_db->getArray<int>( "ans" );
        bool test = ans == result;
        if ( !test ) {
            std::cerr << "Answer does not match\n";
            error = 1;
        }
    }

    // Print the results
    printResults( result );

    return globalComm.maxReduce( error );
}


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );

    if ( argc != 2 ) {
        std::cerr << "compare_Mesh input_file\n";
        AMP::AMPManager::shutdown();
        return 1;
    }

    int result = compare( argv[1] );

    AMP::AMPManager::shutdown();
    return result;
}
