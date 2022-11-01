#include "AMP/IO/HDF5writer.h"
#include "AMP/IO/HDF5.h"
#include "AMP/IO/Xdmf.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshIterator.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/ArrayVectorData.h"

#include "ProfilerApp.h"

#include <chrono>


namespace AMP::IO {


/************************************************************
 * Helper functions                                          *
 ************************************************************/
AMP::Array<double> getArrayData( std::shared_ptr<const AMP::LinearAlgebra::Vector> vec )
{
    auto multivec = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( vec );
    if ( multivec ) {
        AMP_ASSERT( multivec->getNumberOfSubvectors() == 1 );
        vec = multivec->getVector( 0 );
    }
    auto data = vec->getVectorData();
    auto arrayData =
        std::dynamic_pointer_cast<const AMP::LinearAlgebra::ArrayVectorData<double>>( data );
    if ( arrayData )
        return arrayData->getArray();
    size_t N = data->getLocalSize();
    AMP::Array<double> data2( N );
    auto it = data->constBegin();
    for ( size_t i = 0; i < N; i++, ++it )
        data2( i ) = *it;
    return data2;
}


/************************************************************
 * Constructor/Destructor                                    *
 ************************************************************/
HDF5writer::HDF5writer() : AMP::IO::Writer() {}
HDF5writer::~HDF5writer() = default;


/************************************************************
 * Some basic functions                                      *
 ************************************************************/
Writer::WriterProperties HDF5writer::getProperties() const
{
    WriterProperties properties;
    properties.type                   = "HDF5";
    properties.extension              = "hdf5";
    properties.registerVector         = true;
    properties.registerMesh           = true;
    properties.registerVectorWithMesh = true;
    properties.registerMatrix         = false;
    return properties;
}


/************************************************************
 * Register arbitrary user data                              *
 ************************************************************/
void HDF5writer::registerData( std::function<void( hid_t, std::string, Xdmf & )> fun )
{
    d_fun.push_back( fun );
}


/************************************************************
 * Function to read a file                                   *
 ************************************************************/
void HDF5writer::readFile( const std::string & ) { AMP_ERROR( "readFile is not implemented yet" ); }


/************************************************************
 * Function to write the hdf5 file                           *
 * Note: it appears that only one processor may write to a   *
 * file at a time, and that once a processor closes the file *
 * it cannot reopen it (or at least doing this on the        *
 * processor that created the file creates problems).        *
 ************************************************************/
void HDF5writer::writeFile( const std::string &fname_in, size_t cycle, double time )
{
    PROFILE_SCOPED( timer, "writeFile" );
#ifdef AMP_USE_HDF5
    Xdmf xmf;
    AMP_ASSERT( d_comm.getSize() == 1 );
    // Create the file
    auto filename  = fname_in + "_" + std::to_string( cycle ) + ".hdf5";
    auto fid       = openHDF5( filename, "w", Compression::GZIP );
    auto filename2 = AMP::Utilities::filename( filename );
    writeHDF5( fid, "time", time );
    // Synchronize the vectors
    syncVectors();
    // Add the mesh
    std::map<GlobalID, Xdmf::MeshData> baseMeshData;
    auto gid = createGroup( fid, "meshes" );
    for ( const auto &[id, mesh] : d_baseMeshes )
        baseMeshData[id] = writeMesh( gid, mesh, filename2 + ":/meshes" );
    for ( const auto &[id, mesh] : d_multiMeshes ) {
        NULL_USE( id );
        std::vector<Xdmf::MeshData> data;
        for ( const auto &id2 : mesh.meshes ) {
            auto it = baseMeshData.find( id2 );
            if ( it != baseMeshData.end() )
                data.push_back( it->second );
        }
        xmf.addMultiMesh( mesh.name, data );
    }
    closeGroup( gid );
    // Add the vectors
    for ( const auto &[id, data] : d_vectors ) {
        NULL_USE( id );
        auto data2 = getArrayData( data.vec );
        writeHDF5( fid, data.name, data2 );
    }
    // Add the matricies
    for ( size_t i = 0; i < d_matrices.size(); i++ ) {
        AMP_ERROR( "Not finished" );
    }
    // Add user data
    for ( auto fun : d_fun )
        fun( fid, filename2 + ":", xmf );
    // Close the file
    closeHDF5( fid );
    // Open summary file
    // Write the Xdmf file
    xmf.gather( d_comm );
    if ( !xmf.empty() ) {
        auto fname = fname_in + "_" + std::to_string( cycle ) + ".xmf";
        xmf.write( fname );
        auto sname = fname_in + ".visit";
        FILE *sid  = nullptr;
        if ( cycle == 0 )
            sid = fopen( sname.data(), "w" );
        else
            sid = fopen( sname.data(), "a" );
        fprintf( sid, "%s\n", AMP::Utilities::filename( fname ).data() );
        fclose( sid );
    }

#else
    // No HDF5
    NULL_USE( fname_in );
    NULL_USE( cycle );
    NULL_USE( time );
#endif
}


/************************************************************
 * Function to write a base mesh                             *
 ************************************************************/
static AMP::Xdmf::RankType getRankType( int numDOFs, int ndim )
{
    if ( numDOFs == 1 )
        return AMP::Xdmf::RankType::Scalar;
    if ( numDOFs == ndim )
        return AMP::Xdmf::RankType::Vector;
    if ( numDOFs == 9 )
        return AMP::Xdmf::RankType::Tensor;
    return AMP::Xdmf::RankType::Matrix;
}
static AMP::Xdmf::Center getCenter( AMP::Mesh::GeomType meshType, AMP::Mesh::GeomType vecType )
{
    if ( vecType == AMP::Mesh::GeomType::Vertex )
        return AMP::Xdmf::Center::Node;
    if ( meshType == vecType )
        return AMP::Xdmf::Center::Cell;
    if ( vecType == AMP::Mesh::GeomType::Edge )
        return AMP::Xdmf::Center::Edge;
    if ( vecType == AMP::Mesh::GeomType::Face )
        return AMP::Xdmf::Center::Face;
    if ( vecType == AMP::Mesh::GeomType::Cell )
        return AMP::Xdmf::Center::Cell;
    return AMP::Xdmf::Center::Null;
}
Xdmf::MeshData HDF5writer::writeDefaultMesh( hid_t fid,
                                             const baseMeshData &mesh,
                                             const std::string &name,
                                             const std::string &path ) const
{
    PROFILE_SCOPED( timer, "writeDefaultMesh", 1 );
    // Treat the mesh as an unstructured mesh
    const int ndim      = mesh.mesh->getDim();
    const auto type     = mesh.mesh->getGeomType();
    const auto elements = mesh.mesh->getIterator( type, 0 );
    AMP::Array<double> x[3];
    AMP::Array<int> nodelist;
    std::vector<AMP::Mesh::MeshElementID> nodelist_ids;
    getNodeElemList( mesh.mesh, elements, x, nodelist, nodelist_ids );
    auto shapetype = AMP::Xdmf::TopologyType::Null;
    int shapesize  = nodelist.size( 0 );
    if ( shapesize == 8 && type == AMP::Mesh::GeomType::Cell )
        shapetype = AMP::Xdmf::TopologyType::Hexahedron;
    else if ( shapesize == 4 && type == AMP::Mesh::GeomType::Cell )
        shapetype = AMP::Xdmf::TopologyType::Tetrahedron;
    else if ( shapesize == 4 && type == AMP::Mesh::GeomType::Face )
        shapetype = AMP::Xdmf::TopologyType::Quadrilateral;
    else if ( shapesize == 3 && type == AMP::Mesh::GeomType::Face )
        shapetype = AMP::Xdmf::TopologyType::Triangle;
    else if ( shapesize == 2 && type == AMP::Mesh::GeomType::Edge )
        shapetype = AMP::Xdmf::TopologyType::Polyline;
    else
        AMP_ERROR( "Unknown element type" );
    const char *x_names[3] = { "x", "y", "z" };
    std::string x_path[3];
    for ( int d = 0; d < ndim; d++ ) {
        writeHDF5( fid, x_names[d], x[d] );
        x_path[d] = path + "/" + x_names[d];
    }
    writeHDF5( fid, "type", static_cast<int>( type ) );
    writeHDF5( fid, "elements", nodelist );
    auto name2    = mesh.mesh->getName() + "_" + name;
    auto XdmfData = AMP::Xdmf::createUnstructuredMesh( name2,
                                                       ndim,
                                                       shapetype,
                                                       elements.size(),
                                                       path + "/elements",
                                                       x[0].length(),
                                                       x_path[0],
                                                       x_path[1],
                                                       x_path[2] );
    // Write the vectors
    for ( const auto &vec : mesh.vectors ) {
        auto DOFs = vec.vec->getDOFManager();
        AMP::Array<double> data;
        std::vector<size_t> dofs;
        if ( vec.type == AMP::Mesh::GeomType::Vertex ) {
            data.resize( vec.numDOFs, nodelist_ids.size() );
            data.fill( 0 );
            for ( size_t i = 0; i < nodelist_ids.size(); i++ ) {
                DOFs->getDOFs( nodelist_ids[i], dofs );
                AMP_ASSERT( (int) dofs.size() == vec.numDOFs );
                vec.vec->getValuesByGlobalID( vec.numDOFs, dofs.data(), &data( 0, i ) );
            }
        } else {
            auto it  = mesh.mesh->getIterator( vec.type, 0 );
            size_t N = it.size();
            data.resize( vec.numDOFs, N );
            data.fill( 0 );
            for ( size_t i = 0; i < N; i++, ++it ) {
                DOFs->getDOFs( it->globalID(), dofs );
                AMP_ASSERT( (int) dofs.size() == vec.numDOFs );
                vec.vec->getValuesByGlobalID( vec.numDOFs, dofs.data(), &data( 0, i ) );
            }
        }
        if ( vec.numDOFs == 1 )
            data.reshape( data.length() );
        writeHDF5( fid, vec.name, data );
        AMP::Xdmf::VarData var;
        var.name     = vec.name;
        var.rankType = getRankType( vec.numDOFs, mesh.mesh->getDim() );
        var.center   = getCenter( mesh.mesh->getGeomType(), vec.type );
        var.size     = data.size();
        var.data     = path + "/" + vec.name;
        XdmfData.vars.push_back( var );
    }
    return XdmfData;
}
Xdmf::MeshData HDF5writer::writeBoxMesh( hid_t fid,
                                         const baseMeshData &mesh,
                                         const std::string &name,
                                         const std::string &path ) const
{
    PROFILE_SCOPED( timer, "writeBoxMesh", 1 );
    using AMP::Mesh::GeomType;
    using MeshElementIndex = AMP::Mesh::BoxMesh::MeshElementIndex;
    auto mesh2             = std::dynamic_pointer_cast<const AMP::Mesh::BoxMesh>( mesh.mesh );
    Xdmf::MeshData XdmfData;
    AMP::ArraySize size( mesh2->size() );
    if ( size.ndim() != mesh2->getDim() )
        return writeDefaultMesh( fid, mesh, name, path ); // We have issues with surface meshes
    auto size2      = size + (size_t) 1;
    int ndim        = size2.ndim();
    auto isPeriodic = mesh2->periodic();
    AMP::Array<double> x( size2 ), y( size2 ), z( size2 );
    for ( size_t k = 0; k < size2[2]; k++ ) {
        size_t k2 = isPeriodic[2] ? k % size[2] : k;
        for ( size_t j = 0; j < size2[1]; j++ ) {
            size_t j2 = isPeriodic[1] ? j % size[1] : j;
            for ( size_t i = 0; i < size2[0]; i++ ) {
                size_t i2     = isPeriodic[0] ? i % size[0] : i;
                double pos[3] = { 0 };
                auto index    = MeshElementIndex( GeomType::Vertex, 0, i2, j2, k2 );
                mesh2->coord( index, pos );
                x( i, j, k ) = pos[0];
                y( i, j, k ) = pos[1];
                z( i, j, k ) = pos[2];
            }
        }
    }
    auto name2 = mesh2->getName() + "_" + name;
    if ( mesh2->getDim() == 1 ) {
        writeHDF5( fid, "x", x );
        XdmfData = AMP::Xdmf::createCurvilinearMesh( name2, size, path + "/x" );
    } else if ( mesh2->getDim() == 2 ) {
        writeHDF5( fid, "x", x );
        writeHDF5( fid, "y", y );
        XdmfData = AMP::Xdmf::createCurvilinearMesh( name2, size, path + "/x", path + "/y" );
    } else if ( mesh2->getDim() == 3 ) {
        writeHDF5( fid, "x", x );
        writeHDF5( fid, "y", y );
        writeHDF5( fid, "z", z );
        XdmfData =
            AMP::Xdmf::createCurvilinearMesh( name2, size, path + "/x", path + "/y", path + "/z" );
    } else {
        AMP_ERROR( "Not finished" );
    }
    // Write the vectors
    for ( const auto &vec : mesh.vectors ) {
        auto DOFs = vec.vec->getDOFManager();
        AMP::Array<double> data;
        std::vector<size_t> dofs;
        AMP::ArraySize size0;
        if ( vec.type == GeomType::Vertex ) {
            size2         = size + 1;
            size0         = size;
            auto periodic = mesh2->periodic();
            for ( size_t d = 0; d < periodic.size(); d++ )
                size0.resize( d, periodic[d] ? size[d] : size2[d] );
        } else if ( vec.type == mesh2->getGeomType() ) {
            size2 = size;
            size0 = size;
        } else {
            AMP_ERROR( "Not finished" );
        }
        data.resize(
            ArraySize( { (uint8_t) vec.numDOFs, size2[0], size2[1], size2[2] }, ndim + 1 ) );
        data.fill( 0 );
        for ( size_t k = 0; k < size2[2]; k++ ) {
            size_t k2 = k % size0[2];
            for ( size_t j = 0; j < size2[1]; j++ ) {
                size_t j2 = j % size0[1];
                for ( size_t i = 0; i < size2[0]; i++ ) {
                    size_t i2  = i % size0[0];
                    auto index = MeshElementIndex( vec.type, 0, i2, j2, k2 );
                    auto id    = mesh2->convert( index );
                    DOFs->getDOFs( id, dofs );
                    AMP_ASSERT( (int) dofs.size() == vec.numDOFs );
                    vec.vec->getValuesByGlobalID( vec.numDOFs, dofs.data(), &data( 0, i, j, k ) );
                }
            }
        }
        if ( vec.numDOFs == 1 )
            data.reshape( size2 );
        writeHDF5( fid, vec.name, data );
        AMP::Xdmf::VarData var;
        var.name     = vec.name;
        var.rankType = getRankType( vec.numDOFs, mesh.mesh->getDim() );
        var.center   = getCenter( mesh.mesh->getGeomType(), vec.type );
        var.size     = data.size();
        var.data     = path + "/" + vec.name;
        XdmfData.vars.push_back( var );
    }
    return XdmfData;
}
static std::vector<std::string> splitPath( const std::string &path )
{
    if ( path.empty() )
        return std::vector<std::string>();
    std::vector<std::string> data;
    for ( size_t i = 0; i < path.size(); ) {
        size_t j = std::min( { path.find( '/', i ), path.find( 92, i ), path.size() } );
        data.push_back( path.substr( i, j - i ) );
        i = j + 1;
    }
    return data;
}
Xdmf::MeshData HDF5writer::writeMesh( hid_t fid, const baseMeshData &mesh, std::string path )
{
    Xdmf::MeshData XdmfData;
    if ( !mesh.mesh )
        return XdmfData;
    // Update the path
    auto gid = fid;
    std::vector<hid_t> groups;
    for ( auto dir : splitPath( mesh.path ) ) {
        if ( H5Gexists( gid, dir ) )
            gid = openGroup( gid, dir );
        else
            gid = createGroup( gid, dir );
        path = path + "/" + dir;
        groups.push_back( gid );
    }
    // Write the mesh and variables
    writeHDF5( gid, "ndim", (int) mesh.mesh->getDim() );
    writeHDF5( gid, "meshClass", mesh.mesh->meshClass() );
    if ( std::dynamic_pointer_cast<AMP::Mesh::BoxMesh>( mesh.mesh ) ) {
        // We are dealing with a box mesh
        XdmfData = writeBoxMesh( gid, mesh, mesh.meshName, path );
    } else {
        XdmfData = writeDefaultMesh( gid, mesh, mesh.meshName, path );
    }
    // Write the geometry (if it exists)

    // Close the groups
    for ( int i = static_cast<int>( groups.size() ) - 1; i >= 0; i-- )
        closeGroup( groups[i] );
    return XdmfData;
}


} // namespace AMP::IO
