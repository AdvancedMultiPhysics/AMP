#include "AMP/IO/Xdmf.h"
#include "AMP/utils/AMPManager.h"

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>


// Point mesh
template<std::size_t NDIM>
void write_points( size_t N, hid_t fid, const std::string &filename, AMP::Xdmf &xmf )
{
    std::string meshname = NDIM == 2 ? "Points_2D" : "Points_3D";
    meshname += "_" + std::to_string( AMP::AMP_MPI( AMP_COMM_WORLD ).getRank() );
    auto gid = AMP::IO::createGroup( fid, meshname );

    // Create the coordinate data
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dis;
    AMP::Array<double> x( NDIM, N );
    for ( size_t i = 0; i < N; i++ ) {
        for ( size_t d = 0; d < NDIM; d++ )
            x( d, i ) = dis( gen );
    }

    // Create the scalar data
    AMP::Array<double> distance( N );
    for ( size_t i = 0; i < N; i++ ) {
        double dist = 0;
        for ( size_t d = 0; d < NDIM; d++ )
            dist += x( d, i ) * x( d, i );
        distance( i ) = std::sqrt( dist );
    }

    // Write the data to HDF5
    AMP::IO::writeHDF5( gid, "XY", x );
    AMP::IO::writeHDF5( gid, "dist", distance );

    // Register the data with XDMF
    auto center = AMP::Xdmf::Center::Node;
    auto prefix = filename + ":/" + meshname + "/";
    auto mesh   = AMP::Xdmf::createPointMesh( meshname, NDIM, N, prefix + "XY" );
    mesh.addVariable(
        "distance", distance.size(), AMP::Xdmf::RankType::Scalar, center, prefix + "dist" );
    mesh.addVariable( "vector", x.size(), AMP::Xdmf::RankType::Vector, center, prefix + "XY" );
    xmf.addMesh( "points", mesh );
    AMP::IO::closeGroup( gid );
}


// Ray based mesh
void write_rays(
    size_t N_elem, size_t nodesPerElement, hid_t fid, const std::string &filename, AMP::Xdmf &xmf )
{
    std::string meshname = "rays";
    meshname += "_" + std::to_string( AMP::AMP_MPI( AMP_COMM_WORLD ).getRank() );
    auto gid = AMP::IO::createGroup( fid, meshname );

    // Create the coordinate data
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dis;
    AMP::Array<float> x( 3, nodesPerElement, N_elem );
    x.fill( 0 );
    for ( size_t i = 0; i < N_elem; i++ ) {
        x( 0, 0, i ) = dis( gen );
        x( 1, 0, i ) = dis( gen );
        x( 2, 0, i ) = 0;
        for ( size_t j = 1; j < nodesPerElement; j++ ) {
            double z     = static_cast<double>( j ) / ( nodesPerElement - 1 );
            x( 0, j, i ) = ( 1 - 0.5 * z ) * x( 0, 0, i );
            x( 1, j, i ) = ( 1 - 0.5 * z ) * x( 1, 0, i );
            x( 2, j, i ) = z;
        }
    }

    // Create the scalar data
    AMP::Array<double> distance( nodesPerElement, N_elem );
    for ( size_t i = 0; i < distance.length(); i++ )
        distance( i ) =
            std::sqrt( x( 0, i ) * x( 0, i ) + x( 1, i ) * x( 1, i ) + x( 2, i ) * x( 2, i ) );

    // Write the data to HDF5
    AMP::IO::writeHDF5( gid, "XYZ", x );
    AMP::IO::writeHDF5( gid, "dist", distance );

    // Register the data with XDMF
    auto center = AMP::Xdmf::Center::Node;
    auto prefix = filename + ":/" + meshname + "/";
    AMP::Xdmf::MeshData mesh;
    mesh.name = meshname;
    mesh.type = AMP::Xdmf::TopologyType::Polyline;
    mesh.size = { 3, N_elem, nodesPerElement };
    mesh.x    = prefix + "XYZ";
    mesh.addVariable(
        "distance", distance.size(), AMP::Xdmf::RankType::Scalar, center, prefix + "dist" );
    xmf.addMesh( "rays", mesh );
    AMP::IO::closeGroup( gid );
}


// 2D uniform mesh
void write_uniform( size_t Nx, size_t Ny, hid_t fid, const std::string &filename, AMP::Xdmf &xmf )
{
    // Create the scalar data
    AMP::Array<double> pressure( Nx, Ny );
    for ( size_t j = 0; j < Ny; j++ ) {
        for ( size_t i = 0; i < Nx; i++ )
            pressure( i, j ) = j;
    }
    AMP::Array<double> velocityx( Nx + 1, Ny + 1 );
    for ( size_t j = 0; j < Ny + 1; j++ ) {
        for ( size_t i = 0; i < Nx + 1; i++ )
            velocityx( i, j ) = i;
    }

    // Create the vector data
    AMP::Array<double> velocity( 2, Nx + 1, Ny + 1 );
    for ( size_t j = 0; j < Ny + 1; j++ ) {
        for ( size_t i = 0; i < Nx + 1; i++ ) {
            velocity( 0, i, j ) = i;
            velocity( 1, i, j ) = j;
        }
    }

    // Write the data to HDF5
    AMP::IO::writeHDF5( fid, "Uniform_Pressure", pressure );
    AMP::IO::writeHDF5( fid, "Uniform_VelocityX", velocityx );
    AMP::IO::writeHDF5( fid, "Uniform_Velocity", velocity );

    // Register the data with XDMF
    auto mesh = AMP::Xdmf::createUniformMesh( "uniform", { -1, 1, -1, 1 }, { Nx, Ny } );
    mesh.addVariable( "Pressure",
                      pressure.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Cell,
                      filename + ":/Uniform_Pressure" );
    mesh.addVariable( "VelocityX",
                      velocityx.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      filename + ":/Uniform_VelocityX" );
    mesh.addVariable( "Velocity",
                      velocity.size(),
                      AMP::Xdmf::RankType::Vector,
                      AMP::Xdmf::Center::Node,
                      filename + ":/Uniform_Velocity" );
    xmf.addMesh( "uniform", mesh );
}


// 2D curvilinear mesh
void write_curvilinear(
    size_t Nx, size_t Ny, hid_t fid, const std::string &filename, AMP::Xdmf &xmf )
{
    // Create the coordinate data
    AMP::Array<double> x( Nx + 1, Ny + 1 ), y( Nx + 1, Ny + 1 );
    constexpr double pi = 3.1415926535897932;
    for ( size_t j = 0; j < Ny + 1; j++ ) {
        float yt    = j * 1.0 / Ny;
        float angle = yt * pi;
        for ( size_t i = 0; i < Nx + 1; i++ ) {
            float xt  = i * 1.0 / Nx;
            float R   = ( 1. - xt ) * 2. + xt * 5.;
            x( i, j ) = R * cos( angle );
            y( i, j ) = R * sin( angle );
        }
    }

    // Create the scalar data
    AMP::Array<double> pressure( Nx, Ny );
    for ( size_t j = 0; j < Ny; j++ ) {
        for ( size_t i = 0; i < Nx; i++ )
            pressure( i, j ) = j;
    }
    AMP::Array<double> velocityx( Nx + 1, Ny + 1 );
    for ( size_t j = 0; j < Ny + 1; j++ ) {
        for ( size_t i = 0; i < Nx + 1; i++ )
            velocityx( i, j ) = i;
    }

    // Create the vector data
    AMP::Array<double> velocity( 2, Nx + 1, Ny + 1 );
    for ( size_t j = 0; j < Ny + 1; j++ ) {
        for ( size_t i = 0; i < Nx + 1; i++ ) {
            velocity( 0, i, j ) = i;
            velocity( 1, i, j ) = j;
        }
    }

    // Write the data to HDF5
    AMP::IO::writeHDF5( fid, "2D_X", x );
    AMP::IO::writeHDF5( fid, "2D_Y", y );
    AMP::IO::writeHDF5( fid, "2D_Pressure", pressure );
    AMP::IO::writeHDF5( fid, "2D_VelocityX", velocityx );
    AMP::IO::writeHDF5( fid, "2D_Velocity", velocity );

    // Register the data with XDMF
    auto mesh = AMP::Xdmf::createCurvilinearMesh(
        "curvilinear", { Nx, Ny }, filename + ":/2D_X", filename + ":/2D_Y" );
    mesh.addVariable( "Pressure",
                      pressure.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Cell,
                      filename + ":/2D_Pressure" );
    mesh.addVariable( "VelocityX",
                      velocityx.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      filename + ":/2D_VelocityX" );
    mesh.addVariable( "Velocity",
                      velocity.size(),
                      AMP::Xdmf::RankType::Vector,
                      AMP::Xdmf::Center::Node,
                      filename + ":/2D_Velocity" );
    xmf.addMesh( "curvilinear", mesh );
}


// 2D unstructured mesh
void write_unstructured(
    int NumElements, int NumNodes, hid_t fid, const std::string &filename, AMP::Xdmf &xmf )
{
    // Connectivity data
    AMP::Array<int> connectivity( 4, NumElements );
    connectivity( 0, 0 ) = 0;
    connectivity( 1, 0 ) = 1;
    connectivity( 2, 0 ) = 2;
    connectivity( 3, 0 ) = 3;
    connectivity( 0, 1 ) = 1;
    connectivity( 1, 1 ) = 4;
    connectivity( 2, 1 ) = 5;
    connectivity( 3, 1 ) = 2;
    connectivity( 0, 2 ) = 2;
    connectivity( 1, 2 ) = 5;
    connectivity( 2, 2 ) = 3;
    connectivity( 3, 2 ) = 3;

    // Node coordinates data
    AMP::Array<float> xyz( 2, NumNodes );
    xyz( 0, 0 ) = 0.0;
    xyz( 1, 0 ) = 0.0;
    xyz( 0, 1 ) = 1.0;
    xyz( 1, 1 ) = 0.0;
    xyz( 0, 2 ) = 1.0;
    xyz( 1, 2 ) = 1.0;
    xyz( 0, 3 ) = 0.0;
    xyz( 1, 3 ) = 1.0;
    xyz( 0, 4 ) = 2.5;
    xyz( 1, 4 ) = 0.0;
    xyz( 0, 5 ) = 2.0;
    xyz( 1, 5 ) = 2.0;

    // Scalar data
    AMP::Array<int> scalarInt( NumNodes );
    AMP::Array<float> scalarFloat( NumNodes );
    AMP::Array<double> scalarDouble( NumNodes );
    for ( int i = 0; i < NumNodes; i++ ) {
        scalarInt( i )    = i * 100;
        scalarFloat( i )  = i * 100 + 50.0;
        scalarDouble( i ) = i * 100 - 100.0;
    }

    // Write the data file
    AMP::IO::writeHDF5( fid, "Unstructured_Quadrilaterals", connectivity );
    AMP::IO::writeHDF5( fid, "Unstructured_XY", xyz );
    AMP::IO::writeHDF5( fid, "Unstructured_ScalarInt", scalarInt );
    AMP::IO::writeHDF5( fid, "Unstructured_ScalarFloat", scalarFloat );
    AMP::IO::writeHDF5( fid, "Unstructured_ScalarDouble", scalarDouble );

    // Write the xml
    auto mesh = AMP::Xdmf::createUnstructuredMesh( "2D Unstructured Mesh",
                                                   2,
                                                   AMP::Xdmf::TopologyType::Quadrilateral,
                                                   NumElements,
                                                   filename + ":/Unstructured_Quadrilaterals",
                                                   NumNodes,
                                                   filename + ":/Unstructured_XY" );
    mesh.addVariable( "ScalarInt",
                      scalarInt.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      filename + ":/Unstructured_ScalarInt" );
    mesh.addVariable( "ScalarFloat",
                      scalarFloat.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      filename + ":/Unstructured_ScalarFloat" );
    mesh.addVariable( "ScalarDouble",
                      scalarDouble.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      filename + ":/Unstructured_ScalarDouble" );
    mesh.addVariable( "VectorDouble",
                      xyz.size(),
                      AMP::Xdmf::RankType::Vector,
                      AMP::Xdmf::Center::Node,
                      filename + ":/Unstructured_XY" );
    xmf.addMesh( "2D Unstructured Mesh", mesh );
}


void writeTime( int i )
{

    // Open HDF5 file(s)
    int rank      = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    auto filename = "test_XDMF_" + std::to_string( i ) + "." + std::to_string( rank ) + ".h5";
    auto fid      = AMP::IO::openHDF5( filename, "w", AMP::IO::Compression::GZIP );

    // Create the Xdmf writer
    AMP::Xdmf xmf;

    // Write the serial meshes
    if ( rank == 0 ) {
        write_uniform( 16, 32, fid, filename, xmf );
        write_curvilinear( 32, 20, fid, filename, xmf );
        write_unstructured( 3, 6, fid, filename, xmf );
    }

    // Write the parallel meshes
    write_points<2>( 256, fid, filename, xmf );
    write_rays( 12, 32, fid, filename, xmf );

    // Write the multimesh
    xmf.addMultiMesh( "all", { "uniform", "curvilinear", "2D Unstructured Mesh" } );

    // Close the HDF5 file
    AMP::IO::closeHDF5( fid );

    // Gather the Xdmf info for all ranks and write the results
    xmf.gather( AMP_COMM_WORLD );
    xmf.write( "test_XDMF_" + std::to_string( i ) + ".xmf" );
    xmf.clear();
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );

    writeTime( 0 );
    writeTime( 1 );
    auto fid = fopen( "test_XDMF.visit", "w" );
    fprintf( fid, "test_XDMF_0.xmf\n" );
    fprintf( fid, "test_XDMF_1.xmf\n" );
    fclose( fid );

    // Finished
    AMP::AMPManager::shutdown();
    return 0;
}
