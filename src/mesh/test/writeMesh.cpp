#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/utils/Database.hpp"

#include "ProfilerApp.h"

#include <iostream>
#include <string>
#include <vector>


void run( const std::string &exe, const std::vector<double> &args, const std::string &filename )
{
    auto CHECK_ARGS = [&args]( int N, const char *msg ) {
        if ( (int) args.size() != N ) {
            std::cerr << msg;
            exit( 1 );
        }
    };

    AMP::Mesh::MeshWriters::DatabasePtr db;
    if ( exe == "write2elementMesh" ) {
        CHECK_ARGS( 6, "Usage: <exe> write2elmenetMesh a ny nz Lx Ly Lz filename\n" );
        double a  = args[0];
        int ny    = args[1];
        int nz    = args[2];
        double Lx = args[3];
        double Ly = args[4];
        double Lz = args[5];
        db        = AMP::Mesh::MeshWriters::create2elementMesh( a, ny, nz, Lx, Ly, Lz );
    } else if ( exe == "write3pointConstrainedBox" ) {
        CHECK_ARGS( 6, "Usage: <exe> write3pointConstrainedBox nx ny nz Lx Ly Lz filename\n" );
        int nx    = args[0];
        int ny    = args[1];
        int nz    = args[2];
        double Lx = args[3];
        double Ly = args[4];
        double Lz = args[5];
        db        = AMP::Mesh::MeshWriters::createConstrainedMesh( nx, ny, nz, Lx, Ly, Lz );
    } else if ( exe == "write7elementMesh-1" ) {
        CHECK_ARGS( 0, "Usage: <exe> write7elementMesh-1 filename\n" );
        db = AMP::Mesh::MeshWriters::create7elementMesh( 2 );
    } else if ( exe == "write7elementMesh-2" ) {
        CHECK_ARGS( 0, "Usage: <exe> write7elementMesh-2 filename\n" );
        db = AMP::Mesh::MeshWriters::create7elementMesh( 8 );
    } else if ( exe == "writeBox" ) {
        CHECK_ARGS( 6, "Usage: <exe> writeBox nx ny nz Lx Ly Lz filename\n" );
        int nx    = args[0];
        int ny    = args[1];
        int nz    = args[2];
        double Lx = args[3];
        double Ly = args[4];
        double Lz = args[5];
        db        = AMP::Mesh::MeshWriters::createBox( nx, ny, nz, Lx, Ly, Lz );
    } else if ( exe == "writeDispValsForPatchTest-2" ) {
        CHECK_ARGS( 0, "Usage: <exe> writeDispValsForPatchTest-2 filename\n" );
        AMP::Mesh::MeshWriters::writeDispValsForPatchTest( filename );
    } else if ( exe == "writeDistortedElementMesh" ) {
        CHECK_ARGS( 0, "Usage: <exe> writeDistortedElementMesh filename\n" );
        db = AMP::Mesh::MeshWriters::createDistortedElement();
    } else if ( exe == "writePlateWithHole" ) {
        CHECK_ARGS( 8, "Usage: <exe> writePlateWithHole le me ne pe a b c r filename\n" );
        int le   = args[0];
        int me   = args[1];
        int ne   = args[2];
        int pe   = args[3];
        double a = args[4]; // X-dimension
        double b = args[5]; // Z-dimension
        double c = args[6]; // Y-dimension
        double r = args[7];
        db       = AMP::Mesh::MeshWriters::createPlateWithHole( le, me, ne, pe, a, b, c, r );
    } else if ( exe == "writeCookMesh" ) {
        CHECK_ARGS( 3, "Usage: <exe> writeCookMesh nx ny nz filename\n" );
        int nx = args[0];
        int ny = args[1];
        int nz = args[2];
        db     = AMP::Mesh::MeshWriters::createCookMesh( nx, ny, nz );
    } else if ( exe == "writeAMGMesh" ) {
        CHECK_ARGS( 6, "Usage: <exe> writeAMGMesh nx ny nz Lx Ly Lz filename\n" );
        int nx    = args[0];
        int ny    = args[1];
        int nz    = args[2];
        double Lx = args[3];
        double Ly = args[4];
        double Lz = args[5];
        db        = AMP::Mesh::MeshWriters::createAMGMesh( nx, ny, nz, Lx, Ly, Lz );
    } else if ( exe == "writeLUML" ) {
        CHECK_ARGS( 6, "Usage: <exe> writeLUML nx ny nz Lx Ly Lz filename\n" );
        int nx    = args[0];
        int ny    = args[1];
        int nz    = args[2];
        double Lx = args[3];
        double Ly = args[4];
        double Lz = args[5];
        db        = AMP::Mesh::MeshWriters::createLUML( nx, ny, nz, Lx, Ly, Lz );
    } else {
        std::cerr << "Unknown mesh: " << exe << std::endl;
        exit( 1 );
    }
    if ( db )
        AMP::Mesh::MeshWriters::writeTestMesh( *db, filename );
}


int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    PROFILE_ENABLE();
    if ( argc == 1 ) {
        // Generate all known meshes
        AMP::Mesh::MeshWriters::generateAll();
        // Generate some other meshes
        run( "write2elementMesh", { 0.5, 5, 10, 1.0, 2.0, 3.0 }, "outWrite2elementMesh" );
        run( "write3pointConstrainedBox", { 4, 7, 10, 1.0, 2.0, 3.0 }, "out3pointConstrainedBox" );
        run( "writeBox", { 4, 7, 10, 1.0, 2.0, 3.0 }, "outWriteBox" );
        run( "writeDispValsForPatchTest-2", {}, "outDispValsForPatchTest-2" );
        // Test Read/Write
        auto db = AMP::Mesh::MeshWriters::readTestMesh( "outWriteBox" );
        AMP::Mesh::MeshWriters::writeTestMesh( *db, "outWriteBox(copy)" );
    } else if ( std::string_view( argv[1] ) == "copy" ) {
        AMP_ASSERT( argc == 4 );
        auto db = AMP::Mesh::MeshWriters::readTestMesh( argv[2], false );
        AMP::Mesh::MeshWriters::writeTestMesh( *db, argv[3] );
    } else if ( std::string_view( argv[1] ) == "copyBinary" ) {
        AMP_ASSERT( argc == 4 );
        auto db = AMP::Mesh::MeshWriters::readBinaryTestMesh( argv[2] );
        AMP::Mesh::MeshWriters::writeTestMesh( *db, argv[3] );
    } else if ( std::string_view( argv[1] ) == "copyExodus" ) {
        // Copy an exodus file to the test mesh format
        AMP_ASSERT( argc == 4 );
        auto db0 = AMP::Database::create(
            "MeshName", "mesh", "MeshType", "libMesh", "FileName", argv[2], "dim", 3 );
        auto params = std::make_shared<AMP::Mesh::MeshParameters>( std::move( db0 ) );
        params->setComm( AMP_COMM_SELF );
        auto mesh = AMP::Mesh::MeshFactory::create( params );
        auto db   = AMP::Mesh::MeshWriters::createDatabase( *mesh );
        AMP::Mesh::MeshWriters::writeTestMesh( *db, argv[3] );
    } else if ( argc >= 3 ) {
        std::string exe( argv[1] );
        std::vector<double> args2( argc - 3 );
        for ( size_t i = 0; i < args2.size(); i++ )
            args2[i] = atof( argv[i + 2] );
        run( exe, args2, argv[argc - 1] );
    } else {
        std::cerr << "Invalid call\n";
        return -1;
    }
    PROFILE_SAVE( "writeMesh" );
    AMP::AMPManager::shutdown();
    return 0;
}
