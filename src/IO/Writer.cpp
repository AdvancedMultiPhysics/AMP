#include "AMP/AMP_TPLs.h"
#include "AMP/IO/AsciiWriter.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/IO/HDF5writer.h"
#include "AMP/IO/NullWriter.h"
#include "AMP/IO/SiloWriter.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorSelector.h"

#include <algorithm>

#include "ProfilerApp.h"


namespace AMP::IO {


/************************************************************
 * Functions to pack/unpack data to a char array             *
 ************************************************************/
template<class TYPE>
static inline void packData( char *ptr, size_t &pos, const TYPE &data )
{
    if constexpr ( std::is_trivially_copyable_v<TYPE> ) {
        memcpy( &ptr[pos], &data, sizeof( TYPE ) );
        pos += sizeof( TYPE );
    } else if constexpr ( std::is_same_v<TYPE, std::string> ) {
        int N = data.size();
        memcpy( &ptr[pos], data.c_str(), N + 1 );
        pos += N + 1;
    } else {
        throw std::logic_error( "Error packing data" );
    }
}
template<class TYPE>
static inline TYPE unpackData( const char *ptr, size_t &pos )
{
    if constexpr ( std::is_trivially_copyable_v<TYPE> ) {
        TYPE data;
        memcpy( &data, &ptr[pos], sizeof( TYPE ) );
        pos += sizeof( TYPE );
        return data;
    } else if constexpr ( std::is_same_v<TYPE, std::string> ) {
        std::string data( &ptr[pos] );
        pos += data.size() + 1;
        return data;
    } else {
        throw std::logic_error( "Error packing data" );
    }
}


/************************************************************
 * Helper function to get a unique id for each vector        *
 ************************************************************/
Writer::GlobalID Writer::getID( const AMP_MPI &comm ) const
{
    GlobalID id;
    if ( comm.getRank() == 0 ) {
        static uint64_t localID = 0;
        id.objID                = localID++;
        id.ownerRank            = d_comm.getRank();
    }
    return comm.bcast( id, 0 );
}


/************************************************************
 * WriterProperties                                          *
 ************************************************************/
Writer::WriterProperties::WriterProperties()
    : registerMesh( false ),
      registerVector( false ),
      registerVectorWithMesh( false ),
      registerMatrix( false ),
      enabled( false ),
      isNull( true )
{
}


/************************************************************
 * Writer::VectorData                                        *
 ************************************************************/
Writer::VectorData::VectorData( std::shared_ptr<AMP::LinearAlgebra::Vector> vec_,
                                const std::string &name_ )
    : isStatic( false ),
      dataType( VectorType::DOUBLE ),
      numDOFs( 0 ),
      type( static_cast<AMP::Mesh::GeomType>( 0xFF ) ),
      name( name_ ),
      vec( vec_ )
{
    if ( !vec )
        return;
    if ( name.empty() )
        name = vec->getName();
    if ( name.empty() )
        name = vec->type();
}


/************************************************************
 * Writer::MatrixData                                        *
 ************************************************************/
Writer::MatrixData::MatrixData( std::shared_ptr<AMP::LinearAlgebra::Matrix> mat_,
                                const std::string &name_ )
    : name( name_ ), mat( mat_ )
{
    if ( !mat )
        return;
    if ( name.empty() )
        name = mat->createOutputVector()->getName() + " - " + mat->createInputVector()->getName();
}


/************************************************************
 * Builder                                                   *
 ************************************************************/
std::shared_ptr<AMP::IO::Writer> Writer::buildWriter( std::string type, AMP_MPI comm )
{
    std::for_each( type.begin(), type.end(), []( char &c ) { c = ::tolower( c ); } );
    std::shared_ptr<AMP::IO::Writer> writer;
    if ( type == "none" || type == "null" ) {
        writer.reset( new AMP::IO::NullWriter() );
    } else if ( type == "silo" ) {
        writer.reset( new AMP::IO::SiloIO() );
    } else if ( type == "hdf5" ) {
        writer.reset( new AMP::IO::HDF5writer() );
    } else if ( type == "ascii" ) {
        writer.reset( new AMP::IO::AsciiWriter() );
    } else if ( type == "auto" ) {
#ifdef AMP_USE_HDF5
        if ( comm.getSize() == 1 )
            return buildWriter( "hdf5", comm );
#endif
#ifdef AMP_USE_SILO
        return buildWriter( "silo", comm );
#else
        return buildWriter( "null", comm );
#endif
    } else {
        AMP_ERROR( "Unknown writer: " + type );
    }
    writer->d_comm = std::move( comm );
    return writer;
}
std::shared_ptr<AMP::IO::Writer> Writer::buildWriter( std::shared_ptr<AMP::Database> db )
{
    auto type   = db->getString( "Name" );
    auto writer = Writer::buildWriter( type );
    if ( db->keyExists( "Decomposition" ) )
        writer->setDecomposition( db->getScalar<int>( "Decomposition" ) );
    return writer;
}


/************************************************************
 * Some basic functions                                      *
 ************************************************************/
std::string Writer::getExtension() const { return getProperties().extension; }
void Writer::setDecomposition( int d )
{
    if ( d == 1 )
        d_decomposition = DecompositionType::SINGLE;
    else
        d_decomposition = DecompositionType::MULTIPLE;
}
void Writer::createDirectories( const std::string &filename )
{
    size_t i = filename.rfind( '/' );
    if ( i != std::string::npos && d_comm.getRank() == 0 )
        recursiveMkdir( filename.substr( 0, i ) );
    d_comm.barrier();
}


/************************************************************
 * Register a mesh                                           *
 ************************************************************/
void Writer::registerMesh( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                           int level,
                           const std::string &path )
{
    AMP_INSIST( level >= 0 && level <= 3, "Invalid value for level" );
    std::set<GlobalID> base_ids;
    registerMesh2( mesh, level, path, base_ids );
}
void Writer::registerMesh2( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                            int level,
                            const std::string &path,
                            std::set<GlobalID> &base_ids )
{
    if ( !mesh )
        return;
    if ( !getProperties().registerMesh )
        AMP_ERROR( "registerMesh is not supported for " + getProperties().type );
    PROFILE( "registerMesh" );
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( !multimesh ) {
        // Create a unique id for each rank
        int rank = d_comm.getRank();
        GlobalID id( mesh->meshID().getData(), rank );
        base_ids.insert( id );
        // Register the base mesh
        auto it = d_baseMeshes.find( id );
        if ( it == d_baseMeshes.end() ) {
            auto path2 = path + mesh->getName() + "_/";
            for ( const auto &[id2, mesh2] : d_baseMeshes ) {
                AMP::Utilities::nullUse( &id2 );
                if ( mesh2.path == path2 )
                    AMP_ERROR( "Registering multiple meshes with the same name: " +
                               mesh->getName() );
            }
            baseMeshData data;
            data.id        = id;
            data.mesh      = mesh;
            data.rank      = rank;
            data.ownerRank = rank;
            data.meshName  = "rank_" + std::to_string( data.rank );
            data.path      = path2;
            if ( d_baseMeshes.find( id ) == d_baseMeshes.end() )
                d_baseMeshes.insert( std::make_pair( id, data ) );
        }
        // Create and register a multimesh for the current mesh
        if ( level > -1 ) {
            multiMeshData data2;
            data2.id        = GlobalID( mesh->meshID().getData(), 0 );
            data2.mesh      = mesh;
            data2.name      = path + mesh->getName();
            data2.meshes    = { id };
            data2.ownerRank = mesh->getComm().bcast<int>( rank, 0 );
            d_multiMeshes.insert( std::make_pair( data2.id, data2 ) );
        }
        // Create and register a multimesh for the rank
        if ( level == 3 ) {
            const auto &data = d_baseMeshes[id];
            multiMeshData data2;
            data2.id        = id;
            data2.mesh      = mesh;
            data2.name      = path + mesh->getName() + "_/rank_" + std::to_string( data.rank );
            data2.meshes    = { id };
            data2.ownerRank = rank;
            d_multiMeshes.insert( std::make_pair( data2.id, data2 ) );
        }
    } else {
        // Check if we previously registered the mesh
        for ( const auto &[id, mesh2] : d_multiMeshes ) {
            AMP::Utilities::nullUse( &id );
            if ( mesh2.mesh->meshID() == mesh->meshID() )
                return;
            if ( mesh2.mesh->getName() == mesh->getName() )
                AMP_WARNING( "Registering multiple meshes with the same name: " + mesh->getName() );
        }
        // We are dealing with a multimesh, register the current mesh and sub meshes
        GlobalID id( mesh->meshID().getData(), 0 );
        int level2 = level;
        if ( level == 1 )
            level2 = -1;
        auto new_path  = path + mesh->getName() + "_/";
        auto submeshes = multimesh->getMeshes();
        std::set<GlobalID> ids;
        for ( auto &submesh : submeshes )
            registerMesh2( submesh, level2, new_path, ids );
        base_ids.insert( ids.begin(), ids.end() );
        if ( level > 0 ) {
            multiMeshData data;
            data.id        = id;
            data.mesh      = mesh;
            data.name      = path + mesh->getName();
            data.ownerRank = mesh->getComm().bcast<int>( d_comm.getRank(), 0 );
            data.meshes.insert( data.meshes.begin(), ids.begin(), ids.end() );
            d_multiMeshes.insert( std::make_pair( data.id, data ) );
        }
    }
}


/************************************************************
 * Register a vector without a mesh                          *
 ************************************************************/
void Writer::registerVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                             const std::string &name )
{
    auto id = getID( vec->getComm() );
    VectorData data( vec, name );
    d_vectors.insert( std::make_pair( id, std::move( data ) ) );
}


/************************************************************
 * Register a vector with a mesh                             *
 ************************************************************/
static int getDOFsPerPoint( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                            std::shared_ptr<AMP::Mesh::Mesh> mesh,
                            AMP::Mesh::GeomType type )
{
    std::vector<size_t> dofs;
    auto DOFs = vec->getDOFManager();
    auto it1  = mesh->getIterator( type, 0 );
    DOFs->getDOFs( it1->globalID(), dofs );
    int DOFsPerPoint = dofs.size();
    if ( type == AMP::Mesh::GeomType::Vertex )
        it1 = mesh->getIterator( type, 1 );
    for ( const auto &elem : it1 ) {
        DOFs->getDOFs( elem.globalID(), dofs );
        AMP_ASSERT( (int) dofs.size() == DOFsPerPoint );
    }
    return DOFsPerPoint;
}
void Writer::registerVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                             std::shared_ptr<AMP::Mesh::Mesh> mesh,
                             AMP::Mesh::GeomType type,
                             const std::string &name_in,
                             VectorType precision,
                             bool isStatic )
{
    // Return if the vector or mesh is empty
    if ( !vec || !mesh )
        return;
    PROFILE( "registerVector" );
    // Make sure the mesh has been registered
    std::set<GlobalID> base_ids;
    registerMesh2( mesh, 1, "", base_ids );
    // Perform some error checking
    auto DOFs = vec->getDOFManager();
    if ( !DOFs )
        return;
    auto it1 = mesh->getIterator( type, 0 );
    auto it2 = DOFs->getIterator();
    auto it3 = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Intersection, it1, it2 );
    if ( it1.size() == 0 || it2.size() == 0 )
        return;
    if ( it1.size() != it3.size() )
        AMP_WARNING( "vector does not cover the entire mesh for the given entity type" );
    // Register the vector with the appropriate base meshes
    auto ids = getMeshIDs( mesh );
    for ( auto id : ids ) {
        for ( auto &[id0, mesh2] : d_baseMeshes ) {
            if ( id0.objID == id.getData() ) {
                AMP::LinearAlgebra::VS_Mesh meshSelector( mesh2.mesh );
                auto vec2 = vec->selectInto( meshSelector );
                if ( vec2 ) {
                    VectorData data( vec2, name_in );
                    data.type     = type;
                    data.numDOFs  = getDOFsPerPoint( vec2, mesh2.mesh, type );
                    data.isStatic = isStatic;
                    data.dataType = precision;
                    mesh2.vectors.push_back( data );
                }
            }
        }
    }
    // Register the vector with the appropriate multi-meshes
    VectorData data( vec, name_in );
    for ( auto &[id0, mesh2] : d_multiMeshes ) {
        if ( id0.objID == mesh->meshID().getData() )
            mesh2.varName.push_back( data.name );
    }
    // Add the vector to the list of vectors so we can perform makeConsistent
    d_vectorsMesh.push_back( vec );
}


/************************************************************
 * Register a matrix                                         *
 ************************************************************/
void Writer::registerMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> mat,
                             const std::string &name )
{
    auto id = getID( mat->createOutputVector()->getComm() );
    MatrixData data( mat, name );
    d_matrices.insert( std::make_pair( id, std::move( data ) ) );
}


/****************************************************
 * Synchronize all vectors                           *
 ****************************************************/
void Writer::syncVectors()
{
    // Syncronize all vectors
    PROFILE( "makeConsistent", 1 );
    for ( auto &elem : d_vectorsMesh ) {
        auto localState = elem->getUpdateStatus();
        if ( localState == AMP::LinearAlgebra::UpdateState::ADDING )
            elem->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
        else
            elem->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
}


/************************************************************
 * Function to get the mesh ids to use for registering       *
 ************************************************************/
std::vector<AMP::Mesh::MeshID> Writer::getMeshIDs( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    std::vector<AMP::Mesh::MeshID> ids;
    auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
    if ( !multimesh ) {
        // We are dealing with a single mesh
        ids = std::vector<AMP::Mesh::MeshID>( 1, mesh->meshID() );
    } else {
        // We are dealing with a multimesh
        auto meshes = multimesh->getMeshes();
        for ( auto &meshe : meshes ) {
            auto ids2 = getMeshIDs( meshe );
            for ( auto &elem : ids2 )
                ids.push_back( elem );
        }
    }
    return ids;
}


/************************************************************
 * Functions for baseMeshData                                *
 ************************************************************/
size_t Writer::baseMeshData::size() const
{
    size_t N_bytes = sizeof( id );  // Store the mesh id
    N_bytes += sizeof( int );       // Store the processor rank
    N_bytes += sizeof( int );       // Store the owner rank
    N_bytes += meshName.size() + 1; // Store the mesh name
    N_bytes += path.size() + 1;     // Store the mesh path
    N_bytes += file.size() + 1;     // Store the mesh file
    N_bytes += sizeof( int );       // Store the number of variables
    for ( auto &vec : vectors ) {
        N_bytes += vec.name.size() + 1;           // Store the variable name
        N_bytes += sizeof( AMP::Mesh::GeomType ); // Store the variable type
        N_bytes += sizeof( int );                 // Store the number of unknowns per point
    }
    return N_bytes;
}
void Writer::baseMeshData::pack( char *ptr ) const
{
    size_t pos = 0;
    packData<GlobalID>( ptr, pos, id );
    packData<int>( ptr, pos, rank );
    packData<int>( ptr, pos, ownerRank );
    packData<std::string>( ptr, pos, meshName );
    packData<std::string>( ptr, pos, path );
    packData<std::string>( ptr, pos, file );
    packData<int>( ptr, pos, vectors.size() );
    for ( auto &vec : vectors ) {
        packData<std::string>( ptr, pos, vec.name );
        packData<AMP::Mesh::GeomType>( ptr, pos, vec.type );
        packData<int>( ptr, pos, vec.numDOFs );
    }
    AMP_ASSERT( pos == size() );
}
Writer::baseMeshData Writer::baseMeshData::unpack( const char *ptr )
{
    baseMeshData data;
    size_t pos     = 0;
    data.id        = unpackData<GlobalID>( ptr, pos );
    data.rank      = unpackData<int>( ptr, pos );
    data.ownerRank = unpackData<int>( ptr, pos );
    data.meshName  = unpackData<std::string>( ptr, pos );
    data.path      = unpackData<std::string>( ptr, pos );
    data.file      = unpackData<std::string>( ptr, pos );
    // Store the variables
    size_t N_var = unpackData<int>( ptr, pos );
    data.vectors.resize( N_var );
    for ( size_t i = 0; i < N_var; ++i ) {
        data.vectors[i].name    = unpackData<std::string>( ptr, pos );
        data.vectors[i].type    = unpackData<AMP::Mesh::GeomType>( ptr, pos );
        data.vectors[i].numDOFs = unpackData<int>( ptr, pos );
    }
    AMP_ASSERT( pos == data.size() );
    return data;
}


/************************************************************
 * Functions for multiMeshData                               *
 ************************************************************/
size_t Writer::multiMeshData::size() const
{
    size_t N_bytes = sizeof( id );                 // Store the mesh id
    N_bytes += sizeof( int );                      // Store the owner rank
    N_bytes += name.size() + 1;                    // Store the mesh name
    N_bytes += sizeof( int );                      // Store the number of sub meshes
    N_bytes += meshes.size() * sizeof( GlobalID ); // Store the sub meshes
    N_bytes += sizeof( int );                      // Store the number of variables
    for ( const auto &name : varName )
        N_bytes += name.size() + 1; // Store the variable name
    return N_bytes;
}
void Writer::multiMeshData::pack( char *ptr ) const
{
    size_t pos = 0;
    packData<GlobalID>( ptr, pos, id );
    packData<int>( ptr, pos, ownerRank );
    packData<std::string>( ptr, pos, name );
    packData<int>( ptr, pos, meshes.size() );
    for ( auto &mesh : meshes )
        packData<GlobalID>( ptr, pos, mesh );
    packData<int>( ptr, pos, varName.size() );
    for ( auto &var : varName )
        packData<std::string>( ptr, pos, var );
    AMP_ASSERT( pos == size() );
}
Writer::multiMeshData Writer::multiMeshData::unpack( const char *ptr )
{
    Writer::multiMeshData data;
    size_t pos     = 0;
    data.id        = unpackData<GlobalID>( ptr, pos );
    data.ownerRank = unpackData<int>( ptr, pos );
    data.name      = unpackData<std::string>( ptr, pos );
    int N_meshes   = unpackData<int>( ptr, pos );
    data.meshes.resize( N_meshes );
    for ( auto &mesh : data.meshes )
        mesh = unpackData<GlobalID>( ptr, pos );
    int N_vars = unpackData<int>( ptr, pos );
    data.varName.resize( N_vars );
    for ( auto &var : data.varName )
        var = unpackData<std::string>( ptr, pos );
    AMP_ASSERT( pos == data.size() );
    return data;
}


/************************************************************
 * Function to synchronize the multimesh data                *
 * If root==-1, the data will be synced across all procs     *
 ************************************************************/
template<class TYPE>
void Writer::syncData( std::vector<TYPE> &data, int root ) const
{
    if ( d_comm.getSize() == 1 )
        return;
    PROFILE( "syncData" );
    // Create buffers to store the data
    size_t sendcount = 0;
    for ( size_t i = 0; i < data.size(); ++i )
        sendcount += data[i].size();
    std::vector<char> sendbuf( sendcount );
    char *ptr = sendbuf.data();
    for ( auto &elem : data ) {
        elem.pack( ptr );
        ptr = &ptr[elem.size()];
    }
    // Send the data and unpack the buffer to a vector
    std::vector<char> recvbuf;
    if ( root == -1 ) {
        recvbuf = d_comm.allGather( sendbuf );
    } else {
        recvbuf = d_comm.gather( sendbuf, root );
    }
    // Unpack the data
    ptr          = recvbuf.data();
    auto end_ptr = ptr + recvbuf.size();
    data.clear();
    while ( ptr < end_ptr ) {
        data.push_back( TYPE::unpack( ptr ) );
        ptr += data.back().size();
    }
}
std::tuple<std::vector<Writer::multiMeshData>, std::map<Writer::GlobalID, Writer::baseMeshData>>
Writer::syncMultiMeshData( int root ) const
{
    PROFILE( "syncMultiMeshData" );
    // Convert the data to vectors
    std::vector<multiMeshData> multiMesh;
    multiMesh.reserve( d_multiMeshes.size() );
    for ( const auto &tmp : d_multiMeshes )
        multiMesh.push_back( tmp.second );
    // Convert the data to vectors
    std::vector<baseMeshData> baseMesh;
    baseMesh.reserve( d_baseMeshes.size() );
    for ( const auto &tmp : d_baseMeshes )
        baseMesh.push_back( tmp.second );
    // Sync the data
    syncData( baseMesh, root );
    syncData( multiMesh, root );
    // Create the map for base meshes
    std::map<GlobalID, baseMeshData> baseMeshMap;
    for ( const auto &tmp : baseMesh ) {
        AMP_ASSERT( baseMeshMap.find( tmp.id ) == baseMeshMap.end() );
        baseMeshMap[tmp.id] = tmp;
    }
    // Combine the multimesh data
    std::vector<multiMeshData> multiMesh2;
    multiMesh2.reserve( multiMesh.size() );
    for ( auto mesh : multiMesh ) {
        auto id  = mesh.id;
        auto fun = [id]( const multiMeshData &data ) { return data.id == id; };
        auto it  = std::find_if( multiMesh2.begin(), multiMesh2.end(), fun );
        if ( it == multiMesh2.end() ) {
            multiMesh2.push_back( mesh );
        } else {
            it->meshes.insert( it->meshes.end(), mesh.meshes.begin(), mesh.meshes.end() );
            it->varName.insert( it->varName.end(), mesh.varName.begin(), mesh.varName.end() );
        }
    }
    std::swap( multiMesh, multiMesh2 );
    for ( auto &mesh : multiMesh ) {
        AMP::Utilities::unique( mesh.meshes );
        AMP::Utilities::unique( mesh.varName );
        for ( auto id : mesh.meshes )
            AMP_ASSERT( baseMeshMap.find( id ) != baseMeshMap.end() );
    }
    return std::tie( multiMesh, baseMeshMap );
}


/************************************************************
 * Helper function to get node and element lists for a mesh  *
 ************************************************************/
void Writer::getNodeElemList( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                              const AMP::Mesh::MeshIterator &elements,
                              AMP::Array<double> *x,
                              AMP::Array<int> &nodelist,
                              std::vector<AMP::Mesh::MeshElementID> &nodelist_ids )
{
    PROFILE( "getNodeElemList" );
    AMP_ASSERT( elements.size() > 0 );
    int ndim = mesh->getDim();
    // Get the element list
    auto elem_iterator = elements.begin();
    auto nodes         = elem_iterator->getElements( AMP::Mesh::GeomType::Vertex );
    int shapesize      = nodes.size();
    // Get the node list (unique integer for each node) and coordinates
    auto node_iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 1 );
    nodelist_ids.resize( node_iterator.size() );
    for ( size_t i = 0; i < node_iterator.size(); ++i, ++node_iterator )
        nodelist_ids[i] = node_iterator->globalID();
    AMP::Utilities::quicksort( nodelist_ids );
    double *coord[3] = { nullptr };
    for ( int d = 0; d < ndim; ++d ) {
        x[d].resize( node_iterator.size() );
        coord[d] = x[d].data();
    }
    node_iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 1 );
    for ( size_t i = 0; i < node_iterator.size(); ++i ) {
        size_t index = AMP::Utilities::findfirst( nodelist_ids, node_iterator->globalID() );
        AMP_ASSERT( nodelist_ids[index] == node_iterator->globalID() );
        auto elem_coord = node_iterator->coord();
        for ( int d = 0; d < ndim; ++d )
            coord[d][index] = elem_coord[d];
        ++node_iterator;
    }
    elem_iterator = elements.begin();
    nodelist.resize( shapesize, elem_iterator.size() );
    AMP::Mesh::MeshElementID nodeids[32];
    size_t i = 0;
    for ( const auto &elem : elem_iterator ) {
        auto N_nodes = elem.getElementsID( AMP::Mesh::GeomType::Vertex, nodeids );
        AMP_INSIST( N_nodes == shapesize, "Mixed element types is currently not supported" );
        for ( int j = 0; j < N_nodes; j++ ) {
            int index = AMP::Utilities::findfirst( nodelist_ids, nodeids[j] );
            AMP_ASSERT( nodelist_ids[index] == nodeids[j] );
            nodelist( i++ ) = index;
        }
        ++elem_iterator;
    }
    AMP_ASSERT( i == nodelist.length() );
}


} // namespace AMP::IO


/****************************************************************************
 * Explicit instantiations                                                   *
 ****************************************************************************/
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/Utilities.hpp"
AMP_INSTANTIATE_SORT( AMP::IO::Writer::GlobalID );
INSTANTIATE_MPI_GATHER( AMP::IO::Writer::GlobalID );
#if !( defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 ) || \
       defined( _MSC_VER ) )
INSTANTIATE_MPI_BCAST( AMP::IO::Writer::GlobalID );
INSTANTIATE_MPI_SENDRECV( AMP::IO::Writer::GlobalID );
#endif
