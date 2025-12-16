#include "AMP/IO/HDF5writer.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/IO/HDF.h"
#include "AMP/IO/Xdmf.h"
#include "AMP/discretization/boxMeshDOFManager.h"
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
[[maybe_unused]] static inline void
writeHDF5( hid_t fid, const std::string &name, const Array<double> &data, Writer::VectorType type )
{
    if ( type == Writer::VectorType::DOUBLE )
        writeHDF5( fid, name, data );
    else if ( type == Writer::VectorType::SINGLE )
        writeHDF5( fid, name, data.cloneTo<float>() );
    else if ( type == Writer::VectorType::INT )
        writeHDF5( fid, name, data.cloneTo<int>() );
    else
        AMP_ERROR( "Unknown vector type" );
}
[[maybe_unused]] static AMP::Array<double>
getArrayData( std::shared_ptr<const AMP::LinearAlgebra::Vector> vec )
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
#ifdef AMP_USE_HDF5
    properties.enabled = true;
#else
    properties.enabled = false;
#endif
    properties.isNull = false;
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
void HDF5writer::writeFile( [[maybe_unused]] const std::string &fname_in,
                            [[maybe_unused]] size_t cycle,
                            [[maybe_unused]] double time )
{
    PROFILE( "writeFile" );
#ifdef AMP_USE_HDF5
    int rank = d_comm.getRank();
    Xdmf xmf;
    // AMP_ASSERT( d_comm.getSize() == 1 );
    //  Create the file
    hid_t fid = -1;
    std::string filename;
    if ( d_decomposition == DecompositionType::SINGLE ) {
        filename = fname_in + "_" + std::to_string( cycle ) + ".hdf5";
        if ( rank == 0 ) {
            auto fid2 = openHDF5( filename, "w", Compression::GZIP );
            writeHDF5( fid2, "time", time );
            auto gid = createGroup( fid2, "meshes" );
            closeGroup( gid );
            closeHDF5( fid2 );
        }
    } else {
        int rank = d_comm.getRank();
        filename =
            fname_in + "_" + std::to_string( cycle ) + "." + std::to_string( rank ) + ".hdf5";
        fid = openHDF5( filename, "w", Compression::GZIP );
        writeHDF5( fid, "time", time );
        auto gid = createGroup( fid, "meshes" );
        closeGroup( gid );
    }
    // Synchronize the data
    syncMultiMeshData();
    syncVectors();
    // Add the mesh based data
    std::set<uint64_t> meshIDs;
    for ( [[maybe_unused]] auto &[id, mesh] : d_baseMeshes )
        meshIDs.insert( id.objID );
    d_comm.setGather( meshIDs );
    std::map<GlobalID, Xdmf::MeshData> baseMeshData;
    for ( auto meshID : meshIDs ) {
        GlobalID id( meshID, rank );
        auto it = d_baseMeshes.find( id );
        if ( it != d_baseMeshes.end() ) {
            auto data = writeMesh( fid, filename, it->second );
            if ( data.type != Xdmf::TopologyType::Null )
                baseMeshData[id] = data;
        }
        if ( d_decomposition == DecompositionType::SINGLE )
            d_comm.barrier();
    }
    for ( const auto &[id, mesh] : d_multiMeshes ) {
        AMP::Utilities::nullUse( &id );
        std::vector<Xdmf::MeshData> data;
        for ( const auto &id2 : mesh.meshes ) {
            auto it = baseMeshData.find( id2 );
            if ( it != baseMeshData.end() )
                data.push_back( it->second );
        }
        xmf.addMultiMesh( mesh.name, data );
    }
    // Open HDF5 to write serial data
    auto gid = fid;
    if ( d_decomposition == DecompositionType::SINGLE ) {
        if ( fid != -1 )
            closeHDF5( fid );
        d_comm.serializeStart();
        fid = openHDF5( filename, "rw", Compression::GZIP );
        gid = createGroup( fid, "rank_" + std::to_string( rank ) );
    }
    // Add the vectors
    for ( const auto &[id, data] : d_vectors ) {
        AMP::Utilities::nullUse( &id );
        auto data2 = getArrayData( data.vec );
        writeHDF5( gid, data.name, data2, data.dataType );
    }
    // Add the matricies
    for ( size_t i = 0; i < d_matrices.size(); i++ ) {
        AMP_ERROR( "Not finished" );
    }
    // Add user data
    auto path = AMP::IO::filename( filename ) + ":";
    for ( auto fun : d_fun )
        fun( gid, path, xmf );
    // Close the file
    if ( gid != fid )
        closeGroup( gid );
    closeHDF5( fid );
    if ( d_decomposition == DecompositionType::SINGLE )
        d_comm.serializeStop();
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
        fprintf( sid, "%s\n", AMP::IO::filename( fname ).data() );
        fclose( sid );
    }
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
                                             const std::string &filename,
                                             const baseMeshData &mesh ) const
{
    PROFILE( "writeDefaultMesh", 1 );
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
    // Update the path
    auto path     = AMP::IO::filename( filename ) + ":/meshes";
    auto path2    = mesh.path;
    auto elemPath = path + "/elements";
    auto comm     = mesh.mesh->getComm();
    if ( d_decomposition == DecompositionType::SINGLE ) {
        comm.serializeStart();
        fid = openHDF5( filename, "rw", Compression::GZIP );
        if ( fid == -1 )
            fid = openHDF5( filename, "rw", Compression::GZIP );
        path2 += "rank_" + std::to_string( mesh.mesh->getComm().getRank() ) + "/";
    }
    AMP_ASSERT( fid != -1 );
    auto gid0 = openGroup( fid, "meshes" );
    auto gid  = gid0;
    std::vector<hid_t> groups;
    for ( auto &dir : splitPath( path2 ) ) {
        gid  = openGroup( gid, dir, true );
        path = path + "/" + dir;
        groups.push_back( gid );
    }
    // Write the mesh and variables
    writeHDF5( gid, "ndim", (int) mesh.mesh->getDim() );
    writeHDF5( gid, "meshClass", mesh.mesh->meshClass() );
    const char *x_names[3] = { "x", "y", "z" };
    std::string x_path[3];
    for ( int d = 0; d < ndim; d++ ) {
        writeHDF5( gid, x_names[d], x[d], VectorType::SINGLE );
        x_path[d] = path + "/" + x_names[d];
    }
    writeHDF5( gid, "type", static_cast<int>( type ) );
    writeHDF5( gid, "elements", nodelist );
    auto name2    = mesh.mesh->getName() + "_" + mesh.meshName;
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
        writeHDF5( gid, vec.name, data, vec.dataType );
        AMP::Xdmf::VarData var;
        var.name     = vec.name;
        var.rankType = getRankType( vec.numDOFs, ndim );
        var.center   = getCenter( mesh.mesh->getGeomType(), vec.type );
        var.size     = data.size();
        var.data     = path + "/" + vec.name;
        XdmfData.vars.push_back( var );
    }
    // Close the groups
    for ( int i = static_cast<int>( groups.size() ) - 1; i >= 0; i-- )
        closeGroup( groups[i] );
    closeGroup( gid0 );
    if ( d_decomposition == DecompositionType::SINGLE ) {
        closeHDF5( fid );
        comm.serializeStop();
    }
    return XdmfData;
}
static Array<double> getBoxMeshVar( const AMP::Mesh::BoxMesh &mesh,
                                    const AMP::LinearAlgebra::Vector &vec,
                                    AMP::Mesh::GeomType type,
                                    int numDOFs )
{
    AMP::Array<double> data;
    auto DOFs = vec.getDOFManager();
    auto size = mesh.localSize();
    auto box  = mesh.getLocalBox();
    if ( type == AMP::Mesh::GeomType::Vertex )
        size = size + 1;
    ArraySize size2( { (size_t) numDOFs, size[0], size[1], size[2] }, size.ndim() + 1 );
    // Check if the internal data is compatible with a raw Array
    auto boxMeshDOFs =
        std::dynamic_pointer_cast<const AMP::Discretization::boxMeshDOFManager>( DOFs );
    if ( boxMeshDOFs && vec.numberOfDataBlocks() == 1 ) {
        if ( boxMeshDOFs->getArraySize() == size2 ) {
            auto ptr = vec.getRawDataBlock<double>();
            data.viewRaw( size2, const_cast<double *>( ptr ) );
            return data;
        }
    }
    // Copy the data
    PROFILE( "convertData", 1 );
    data.resize( size2 );
    data.fill( 0 );
    std::vector<size_t> dofs;
    for ( size_t k = 0; k < size[2]; k++ ) {
        size_t k2 = k + box.first[2];
        for ( size_t j = 0; j < size[1]; j++ ) {
            size_t j2 = j + box.first[1];
            for ( size_t i = 0; i < size[0]; i++ ) {
                size_t i2  = i + box.first[0];
                auto index = AMP::Mesh::BoxMesh::MeshElementIndex( type, 0, i2, j2, k2 );
                mesh.fixPeriodic( index );
                auto id = mesh.convert( index );
                DOFs->getDOFs( id, dofs );
                AMP_ASSERT( (int) dofs.size() == numDOFs );
                vec.getValuesByGlobalID( numDOFs, dofs.data(), &data( 0, i, j, k ) );
            }
        }
    }
    if ( numDOFs == 1 )
        data.reshape( size );
    return data;
}
static Array<double> getGlobalBoxMeshVar( const AMP::Mesh::BoxMesh &mesh,
                                          const AMP::LinearAlgebra::Vector &vec,
                                          AMP::Mesh::GeomType type,
                                          int numDOFs )
{
    auto comm = mesh.getComm();
    auto data = getBoxMeshVar( mesh, vec, type, numDOFs );
    if ( comm.getSize() == 1 )
        return data;
    int tag = comm.newTag();
    if ( comm.getRank() != 0 ) {
        comm.send( data.data(), data.length(), 0, tag );
        return {};
    }
    if ( numDOFs == 1 )
        data.reshape( ArraySize( 1, data.size( 0 ), data.size( 1 ), data.size( 2 ) ) );
    auto size = mesh.size();
    if ( type == AMP::Mesh::GeomType::Vertex )
        size = size + 1;
    size_t s2[4] = { (size_t) numDOFs, size[0], size[1], size[2] };
    ArraySize size2( size.ndim() + 1, s2 );
    Array<double> data3( size2 );
    data3.fill( 0 );
    data3.copySubset( { 0,
                        (size_t) numDOFs - 1,
                        0,
                        data.size( 1 ) - 1,
                        0,
                        data.size( 2 ) - 1,
                        0,
                        data.size( 3 ) - 1 },
                      data );
    for ( int r = 1; r < comm.getSize(); r++ ) {
        auto block = mesh.getLocalBlock( r );
        if ( type == AMP::Mesh::GeomType::Vertex )
            block = { block[0], block[1] + 1, block[2], block[3] + 1, block[4], block[5] + 1 };
        ArraySize remoteSize(
            numDOFs, block[1] - block[0] + 1, block[3] - block[2] + 1, block[5] - block[4] + 1 );
        AMP::Array<double> remote( remoteSize );
        comm.recv( remote.data(), remote.length(), r, tag );
        std::vector<size_t> index = { 0,
                                      (size_t) numDOFs - 1,
                                      (size_t) block[0],
                                      (size_t) block[1],
                                      (size_t) block[2],
                                      (size_t) block[3],
                                      (size_t) block[4],
                                      (size_t) block[5] };
        data3.copySubset( index, remote );
    }
    if ( numDOFs == 1 )
        data3.reshape( size );
    return data3;
}
Xdmf::MeshData
HDF5writer::writeBoxMesh( hid_t fid, const std::string &filename, const baseMeshData &mesh ) const
{
    PROFILE( "writeBoxMesh", 1 );
    auto mesh2 = std::dynamic_pointer_cast<const AMP::Mesh::BoxMesh>( mesh.mesh );
    // Check for surface meshes (they have issues)
    if ( static_cast<int>( mesh2->getGeomType() ) != mesh2->getDim() )
        return writeDefaultMesh( fid, filename, mesh );
    // Local the local/global data
    ArraySize meshSize;
    std::array<AMP::Array<double>, 3> x;
    std::vector<AMP::Array<double>> data( mesh.vectors.size() );
    if ( d_decomposition == DecompositionType::SINGLE ) {
        fid = -1;
        if ( mesh2->getComm().getRank() == 0 ) {
            fid = openHDF5( filename, "rw", Compression::GZIP );
        }
        x = mesh2->globalCoord();
        for ( size_t i = 0; i < data.size(); i++ ) {
            auto &vec = mesh.vectors[i];
            data[i]   = getGlobalBoxMeshVar( *mesh2, *vec.vec, vec.type, vec.numDOFs );
        }
        meshSize = mesh2->size();
    } else {
        x = mesh2->localCoord();
        for ( size_t i = 0; i < data.size(); i++ ) {
            auto &vec = mesh.vectors[i];
            data[i]   = getBoxMeshVar( *mesh2, *vec.vec, vec.type, vec.numDOFs );
        }
        meshSize = mesh2->localSize();
    }
    if ( fid == -1 )
        return Xdmf::MeshData();
    // Update the path
    auto path = AMP::IO::filename( filename ) + ":/meshes";
    auto gid0 = openGroup( fid, "meshes" );
    auto gid  = gid0;
    std::vector<hid_t> groups;
    for ( auto &dir : splitPath( mesh.path ) ) {
        gid  = openGroup( gid, dir, true );
        path = path + "/" + dir;
        groups.push_back( gid );
    }
    // Write the mesh and variables
    writeHDF5( gid, "ndim", (int) mesh.mesh->getDim() );
    writeHDF5( gid, "meshClass", mesh.mesh->meshClass() );
    AMP_ASSERT( mesh2 );
    Xdmf::MeshData XdmfData;
    auto name2 = mesh2->getName() + "_" + mesh.meshName;
    if ( mesh2->getDim() == 1 ) {
        writeHDF5( gid, "x", x[0], VectorType::SINGLE );
        XdmfData = AMP::Xdmf::createCurvilinearMesh( name2, meshSize, path + "/x" );
    } else if ( mesh2->getDim() == 2 ) {
        writeHDF5( gid, "x", x[0], VectorType::SINGLE );
        writeHDF5( gid, "y", x[1], VectorType::SINGLE );
        XdmfData = AMP::Xdmf::createCurvilinearMesh( name2, meshSize, path + "/x", path + "/y" );
    } else if ( mesh2->getDim() == 3 ) {
        writeHDF5( gid, "x", x[0], VectorType::SINGLE );
        writeHDF5( gid, "y", x[1], VectorType::SINGLE );
        writeHDF5( gid, "z", x[2], VectorType::SINGLE );
        XdmfData = AMP::Xdmf::createCurvilinearMesh(
            name2, meshSize, path + "/x", path + "/y", path + "/z" );
    } else {
        AMP_ERROR( "Not finished" );
    }
    // Write the vectors
    PROFILE( "writeBoxMeshVars", 1 );
    for ( size_t i = 0; i < data.size(); i++ ) {
        const auto &vec = mesh.vectors[i];
        writeHDF5( gid, vec.name, data[i], vec.dataType );
        AMP::Xdmf::VarData var;
        var.name     = vec.name;
        var.rankType = getRankType( vec.numDOFs, mesh.mesh->getDim() );
        var.center   = getCenter( mesh.mesh->getGeomType(), vec.type );
        var.size     = data[i].size();
        var.data     = path + "/" + vec.name;
        XdmfData.vars.push_back( var );
    }
    // Close the groups
    for ( int i = static_cast<int>( groups.size() ) - 1; i >= 0; i-- )
        closeGroup( groups[i] );
    closeGroup( gid0 );
    if ( d_decomposition == DecompositionType::SINGLE )
        closeHDF5( fid );
    return XdmfData;
}
Xdmf::MeshData
HDF5writer::writeMesh( hid_t fid, const std::string &filename, const baseMeshData &mesh ) const
{
    Xdmf::MeshData XdmfData;
    if ( !mesh.mesh )
        return XdmfData;
    if ( std::dynamic_pointer_cast<AMP::Mesh::BoxMesh>( mesh.mesh ) ) {
        // We are dealing with a box mesh
        XdmfData = writeBoxMesh( fid, filename, mesh );
    } else {
        XdmfData = writeDefaultMesh( fid, filename, mesh );
    }
    // Write the geometry (if it exists)
    return XdmfData;
}


} // namespace AMP::IO
