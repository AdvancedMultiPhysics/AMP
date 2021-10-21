#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Xdmf.h"

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>


// Point mesh
template<std::size_t NDIM>
void write_points( size_t N, hid_t fid, const std::string &filename, AMP::Xdmf &xmf )
{
    std::string meshname = NDIM == 2 ? "Points_2D" : "Points_3D";
    auto gid             = AMP::createGroup( fid, meshname );

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
        distance( i ) = sqrt( dist );
    }

    // Write the data to HDF5
    writeHDF5( gid, "XY", x );
    writeHDF5( gid, "dist", distance );

    // Register the data with XDMF
    auto center = AMP::Xdmf::Center::Node;
    auto prefix = filename + ":/" + meshname + "/";
    auto mesh   = AMP::Xdmf::createPointMesh( meshname, NDIM, N, prefix + "XY" );
    mesh.addVariable(
        "distance", distance.size(), AMP::Xdmf::RankType::Scalar, center, prefix + "dist" );
    mesh.addVariable( "vector", x.size(), AMP::Xdmf::RankType::Vector, center, prefix + "XY" );
    xmf.addMesh( "points", mesh );
    AMP::closeGroup( gid );
}


// 2D uniform mesh
void write_uniform( size_t Nx, size_t Ny, hid_t fid, AMP::Xdmf &xmf )
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
    writeHDF5( fid, "Uniform_Pressure", pressure );
    writeHDF5( fid, "Uniform_VelocityX", velocityx );
    writeHDF5( fid, "Uniform_Velocity", velocity );

    // Register the data with XDMF
    auto mesh = AMP::Xdmf::createUniformMesh( "uniform", { -1, 1, -1, 1 }, { Nx, Ny } );
    mesh.addVariable( "Pressure",
                      pressure.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Cell,
                      "test_XDMF.h5:/Uniform_Pressure" );
    mesh.addVariable( "VelocityX",
                      velocityx.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      "test_XDMF.h5:/Uniform_VelocityX" );
    mesh.addVariable( "Velocity",
                      velocity.size(),
                      AMP::Xdmf::RankType::Vector,
                      AMP::Xdmf::Center::Node,
                      "test_XDMF.h5:/Uniform_Velocity" );
    xmf.addMesh( "uniform", mesh );
}


// 2D curvilinear mesh
void write_curvilinear( size_t Nx, size_t Ny, hid_t fid, AMP::Xdmf &xmf )
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
    writeHDF5( fid, "2D_X", x );
    writeHDF5( fid, "2D_Y", y );
    writeHDF5( fid, "2D_Pressure", pressure );
    writeHDF5( fid, "2D_VelocityX", velocityx );
    writeHDF5( fid, "2D_Velocity", velocity );

    // Register the data with XDMF
    auto mesh = AMP::Xdmf::createCurvilinearMesh(
        "curvilinear", { Nx, Ny }, "test_XDMF.h5:/2D_X", "test_XDMF.h5:/2D_Y" );
    mesh.addVariable( "Pressure",
                      pressure.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Cell,
                      "test_XDMF.h5:/2D_Pressure" );
    mesh.addVariable( "VelocityX",
                      velocityx.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      "test_XDMF.h5:/2D_VelocityX" );
    mesh.addVariable( "Velocity",
                      velocity.size(),
                      AMP::Xdmf::RankType::Vector,
                      AMP::Xdmf::Center::Node,
                      "test_XDMF.h5:/2D_Velocity" );
    xmf.addMesh( "curvilinear", mesh );
}


// 2D unstructured mesh
void write_unstructured( int NumElements, int NumNodes, hid_t fid, AMP::Xdmf &xmf )
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
    writeHDF5( fid, "Unstructured_Quadrilaterals", connectivity );
    writeHDF5( fid, "Unstructured_XY", xyz );
    writeHDF5( fid, "Unstructured_ScalarInt", scalarInt );
    writeHDF5( fid, "Unstructured_ScalarFloat", scalarFloat );
    writeHDF5( fid, "Unstructured_ScalarDouble", scalarDouble );

    // Write the xml
    auto mesh = AMP::Xdmf::createUnstructuredMesh( "2D Unstructured Mesh",
                                                   2,
                                                   AMP::Xdmf::TopologyType::Quadrilateral,
                                                   NumElements,
                                                   "test_XDMF.h5:/Unstructured_Quadrilaterals",
                                                   NumNodes,
                                                   "test_XDMF.h5:/Unstructured_XY" );
    mesh.addVariable( "ScalarInt",
                      scalarInt.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      "test_XDMF.h5:/Unstructured_ScalarInt" );
    mesh.addVariable( "ScalarFloat",
                      scalarFloat.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      "test_XDMF.h5:/Unstructured_ScalarFloat" );
    mesh.addVariable( "ScalarDouble",
                      scalarDouble.size(),
                      AMP::Xdmf::RankType::Scalar,
                      AMP::Xdmf::Center::Node,
                      "test_XDMF.h5:/Unstructured_ScalarDouble" );
    mesh.addVariable( "VectorDouble",
                      xyz.size(),
                      AMP::Xdmf::RankType::Vector,
                      AMP::Xdmf::Center::Node,
                      "test_XDMF.h5:/Unstructured_XY" );
    xmf.addMesh( "2D Unstructured Mesh", mesh );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    int rank = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();

    // Open HDF5 file(s)
    auto filename = "test_XDMF." + std::to_string( rank ) + ".h5";
    auto fid      = AMP::openHDF5( filename, "w", AMP::Compression::GZIP );

    // Create the Xdmf writer
    AMP::Xdmf xmf;

    // Write the serial meshes
    if ( rank == 0 ) {
        write_points<2>( 256, fid, filename, xmf );
        write_uniform( 16, 32, fid, xmf );
        write_curvilinear( 32, 20, fid, xmf );
        write_unstructured( 3, 6, fid, xmf );
    }

    // Write the parallel meshes

    // Close the HDF5 file
    AMP::closeHDF5( fid );

    // Gather the Xdmf info for all ranks and write the results
    xmf.gather( AMP_COMM_WORLD );
    xmf.write( "test_XDMF.xmf" );
    xmf.clear();

    // Finished
    AMP::AMPManager::shutdown();
    return 0;
}