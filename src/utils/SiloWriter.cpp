#include "AMP/utils/SiloWriter.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <chrono>

#ifdef USE_AMP_MESH
#include "AMP/ampmesh/Mesh.h"
#include "AMP/ampmesh/MultiMesh.h"
#endif
#ifdef USE_AMP_VECTORS
#include "AMP/vectors/Vector.h"
#endif
#ifdef USE_AMP_MATRICES
#include "AMP/matrices/Matrix.h"
#endif

namespace AMP::Utilities {


static inline size_t find_slash( const std::string &filename )
{
    size_t i1 = filename.find_last_of( 47 );
    size_t i2 = filename.find_last_of( 92 );
    size_t i  = std::string::npos;
    if ( i1 == std::string::npos )
        i = i2;
    else if ( i2 == std::string::npos )
        i = i1;
    else if ( i1 != std::string::npos && i2 != std::string::npos )
        i = std::max( i1, i2 );
    return i;
}


// Function to replace all instances of a string with another
static inline void strrep( std::string &str, const std::string &s, const std::string &r )
{
    size_t i = 0;
    while ( i < str.length() ) {
        i = str.find( s, i );
        if ( i == std::string::npos ) {
            break;
        }
        str.replace( i, s.length(), r );
        i += r.length();
    }
}


/************************************************************
 * Constructor/Destructor                                    *
 ************************************************************/
SiloIO::SiloIO() : AMP::Utilities::Writer()
{
#ifdef USE_AMP_MESH
    d_dim = -1;
#ifdef USE_EXT_SILO
    DBSetAllowEmptyObjects( true );
#endif
#endif
}
SiloIO::~SiloIO() = default;


/************************************************************
 * Some basic functions                                      *
 ************************************************************/
Writer::WriterProperties SiloIO::getProperties() const
{
    WriterProperties properties;
    properties.type                   = "Silo";
    properties.extension              = "silo";
    properties.registerMesh           = true;
    properties.registerVectorWithMesh = true;
    return properties;
}


#if defined( USE_EXT_SILO ) && defined( USE_AMP_MESH )

// Some internal functions
static void createSiloDirectory( DBfile *FileHandle, const std::string &path );


/************************************************************
 * Function to read a silo file                              *
 ************************************************************/
void SiloIO::readFile( const std::string & ) { AMP_ERROR( "readFile is not implimented yet" ); }


/************************************************************
 * Function to write a silo file                             *
 * Note: it appears that only one prcoessor may write to a   *
 * file at a time, and that once a processor closes the file *
 * it cannot reopen it (or at least doing this on the        *
 * processor that created the file creates problems).        *
 ************************************************************/
void SiloIO::writeFile( const std::string &fname_in, size_t cycle, double time )
{
    PROFILE_START( "writeFile" );
    // Create the directory (if needed)
    createDirectories( fname_in );
    // Create the file name
    std::string fname = fname_in + "_" + std::to_string( cycle ) + "." + getExtension();
    // Check that the dimension is matched across all processors
    PROFILE_START( "sync dim", 1 );
    d_dim = -1;
    for ( auto tmp : d_baseMeshes ) {
        int dim = tmp.second.mesh->getDim();
        if ( d_dim == -1 )
            d_dim = tmp.second.mesh->getDim();
        AMP_INSIST( d_dim == dim, "All meshes must have the same number of physical dimensions" );
    }
    int dim = d_comm.maxReduce( d_dim );
    if ( d_dim == -1 )
        d_dim = dim;
    AMP_INSIST( d_dim == dim, "All meshes must have the same number of physical dimensions" );
    d_comm.barrier();
    PROFILE_STOP( "sync dim", 1 );
// Syncronize all vectors
#ifdef USE_AMP_VECTORS
    PROFILE_START( "makeConsistent", 1 );
    for ( auto &elem : d_vectorsMesh ) {
        auto localState = elem->getUpdateStatus();
        if ( localState == AMP::LinearAlgebra::VectorData::UpdateState::ADDING )
            elem->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_ADD );
        else
            elem->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );
    }
    PROFILE_STOP( "makeConsistent", 1 );
#endif
    // Write the data for each base mesh
    if ( d_decomposition == 1 ) {
        // Write all mesh data to the main file
        for ( int i = 0; i < d_comm.getSize(); ++i ) {
            if ( d_comm.getRank() == i ) {
                // Open the file
                DBfile *FileHandle;
                if ( d_comm.getRank() == 0 ) {
                    FileHandle = DBCreate( fname.c_str(), DB_CLOBBER, DB_LOCAL, nullptr, DB_HDF5 );
                } else {
                    FileHandle = DBOpen( fname.c_str(), DB_HDF5, DB_APPEND );
                }
                // Write the base meshes
                for ( auto &baseMesh : d_baseMeshes ) {
                    auto &data = baseMesh.second;
                    data.file  = fname.c_str();
                    AMP_ASSERT( data.id == baseMesh.first );
                    writeMesh( FileHandle, baseMesh.second, cycle, time );
                }
                // Close the file
                DBClose( FileHandle );
            }
            d_comm.barrier();
        }
    } else if ( d_decomposition == 2 ) {
        // Every rank will write a seperate file
        if ( d_comm.getRank() == 0 )
            Utilities::recursiveMkdir( fname_in + "_silo", ( S_IRUSR | S_IWUSR | S_IXUSR ), false );
        d_comm.barrier();
        auto fname_rank = fname_in + "_silo/" + std::to_string( cycle ) + "." +
                          std::to_string( d_comm.getRank() + 1 ) + "." + getExtension();
        DBfile *FileHandle = DBCreate( fname_rank.c_str(), DB_CLOBBER, DB_LOCAL, nullptr, DB_HDF5 );
        // Write the base meshes
        for ( auto &baseMesh : d_baseMeshes ) {
            auto &data = baseMesh.second;
            data.file  = fname_rank.c_str();
            AMP_ASSERT( data.id == baseMesh.first );
            writeMesh( FileHandle, baseMesh.second, cycle, time );
        }
        // Close the file
        DBClose( FileHandle );
    } else {
        AMP_ERROR( "Unknown file decomposition" );
    }
    // Write the summary results (multimeshes, multivariables, etc.)
    if ( d_decomposition != 1 ) {
        if ( d_comm.getRank() == 0 ) {
            DBfile *FileHandle = DBCreate( fname.c_str(), DB_CLOBBER, DB_LOCAL, nullptr, DB_HDF5 );
            DBClose( FileHandle );
        }
        d_comm.barrier();
    }
    writeSummary( fname, cycle, time );
    PROFILE_STOP( "writeFile" );
}


/************************************************************
 * Function to write a mesh                                  *
 ************************************************************/
void SiloIO::writeMesh( DBfile *FileHandle, const baseMeshData &data, int cycle, double time )
{
    NULL_USE( cycle );
    NULL_USE( time );
    PROFILE_START( "writeMesh", 1 );
    auto mesh = data.mesh;
    // Get the zone (element) lists
    PROFILE_START( "writeMesh - get-elements", 2 );
    auto elem_iterator = mesh->getIterator( mesh->getGeomType(), 0 );
    AMP_ASSERT( elem_iterator.size() > 0 );
    auto type     = elem_iterator->globalID().type();
    auto nodes    = elem_iterator->getElements( AMP::Mesh::GeomType::Vertex );
    int shapesize = nodes.size();
    int shapetype;
    if ( shapesize == 8 && type == AMP::Mesh::GeomType::Volume )
        shapetype = DB_ZONETYPE_HEX;
    else if ( shapesize == 4 && type == AMP::Mesh::GeomType::Volume )
        shapetype = DB_ZONETYPE_TET;
    else if ( shapesize == 4 && type == AMP::Mesh::GeomType::Face )
        shapetype = DB_ZONETYPE_QUAD;
    else if ( shapesize == 3 && type == AMP::Mesh::GeomType::Face )
        shapetype = DB_ZONETYPE_TRIANGLE;
    else if ( shapesize == 2 && type == AMP::Mesh::GeomType::Edge )
        shapetype = DB_ZONETYPE_BEAM;
    else
        AMP_ERROR( "Unknown element type" );
    int shapecnt = elem_iterator.size();
    PROFILE_STOP( "writeMesh - get-elements", 2 );
    // Get the node list (unique integer for each node) and coordinates
    PROFILE_START( "writeMesh - get-nodelist", 2 );
    PROFILE_START( "writeMesh - get-nodelist-1", 3 );
    auto node_iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 1 );
    std::vector<AMP::Mesh::MeshElementID> nodelist_ids( node_iterator.size() );
    for ( size_t i = 0; i < node_iterator.size(); ++i, ++node_iterator )
        nodelist_ids[i] = node_iterator->globalID();
    AMP::Utilities::quicksort( nodelist_ids );
    PROFILE_STOP( "writeMesh - get-nodelist-1", 3 );
    PROFILE_START( "writeMesh - get-nodelist-2", 3 );
    double *coord[3];
    for ( int i = 0; i < d_dim; ++i )
        coord[i] = new double[node_iterator.size()];
    node_iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 1 );
    for ( size_t i = 0; i < node_iterator.size(); ++i ) {
        size_t index = AMP::Utilities::findfirst( nodelist_ids, node_iterator->globalID() );
        AMP_ASSERT( nodelist_ids[index] == node_iterator->globalID() );
        auto elem_coord = node_iterator->coord();
        for ( int j = 0; j < d_dim; ++j )
            coord[j][index] = elem_coord[j];
        ++node_iterator;
    }
    PROFILE_STOP( "writeMesh - get-nodelist-2", 3 );
    PROFILE_START( "writeMesh - get-nodelist-3", 3 );
    elem_iterator = mesh->getIterator( mesh->getGeomType(), 0 );
    std::vector<int> nodelist;
    nodelist.reserve( shapesize * elem_iterator.size() );
    std::vector<AMP::Mesh::MeshElementID> nodeids;
    for ( const auto &elem : elem_iterator ) {
        elem.getElementsID( AMP::Mesh::GeomType::Vertex, nodeids );
        AMP_INSIST( (int) nodeids.size() == shapesize,
                    "Mixed element types is currently not supported" );
        for ( auto &nodeid : nodeids ) {
            size_t index = AMP::Utilities::findfirst( nodelist_ids, nodeid );
            AMP_ASSERT( nodelist_ids[index] == nodeid );
            nodelist.push_back( (int) index );
        }
        ++elem_iterator;
    }
    PROFILE_STOP( "writeMesh - get-nodelist-3", 3 );
    PROFILE_STOP( "writeMesh - get-nodelist", 2 );
    // Create the directory for the mesh
    PROFILE_START( "writeMesh - directory", 2 );
    std::string tmp_path = data.path;
    while ( tmp_path.size() > 0 ) {
        if ( tmp_path[0] == '/' ) {
            tmp_path.erase( 0, 1 );
            continue;
        }
        size_t pos = tmp_path.find_first_of( '/' );
        if ( pos == std::string::npos ) {
            pos = tmp_path.size();
        }
        auto subdir       = tmp_path.substr( 0, pos );
        DBtoc *toc        = DBGetToc( FileHandle );
        bool subdir_found = false;
        for ( int i = 0; i < toc->ndir; ++i ) {
            if ( subdir.compare( toc->dir_names[i] ) == 0 )
                subdir_found = true;
        }
        if ( !subdir_found )
            DBMkDir( FileHandle, subdir.c_str() );
        DBSetDir( FileHandle, subdir.c_str() );
        tmp_path.erase( 0, pos );
    }
    DBSetDir( FileHandle, "/" );
    DBSetDir( FileHandle, data.path.c_str() );
    PROFILE_STOP( "writeMesh - directory", 2 );
    // Write the elements (connectivity)
    PROFILE_START( "writeMesh - elements", 2 );
    std::string meshName  = data.meshName;
    std::string zoneName  = "zone_" + std::to_string( data.rank );
    auto element_iterator = mesh->getIterator( mesh->getGeomType(), 0 );
    auto num_elems        = (int) element_iterator.size();
    DBPutZonelist2( FileHandle,
                    zoneName.c_str(),
                    num_elems,
                    d_dim,
                    &nodelist[0],
                    nodelist.size(),
                    0,
                    0,
                    0,
                    &shapetype,
                    &shapesize,
                    &shapecnt,
                    1,
                    nullptr );
    PROFILE_STOP( "writeMesh - elements", 2 );
    // Write the mesh
    PROFILE_START( "writeMesh - mesh", 2 );
    DBPutUcdmesh( FileHandle,
                  meshName.c_str(),
                  d_dim,
                  nullptr,
                  coord,
                  node_iterator.size(),
                  nodelist.size(),
                  zoneName.c_str(),
                  nullptr,
                  DB_DOUBLE,
                  nullptr );
    for ( int i = 0; i < d_dim; ++i )
        delete[] coord[i];
    PROFILE_STOP( "writeMesh - mesh", 2 );
    // Write the variables
    PROFILE_START( "writeMesh - variables", 2 );
#ifdef USE_AMP_VECTORS
    float ftime        = time;
    DBoptlist *optlist = DBMakeOptlist( 10 );
    DBAddOption( optlist, DBOPT_CYCLE, &cycle );
    DBAddOption( optlist, DBOPT_TIME, &ftime );
    DBAddOption( optlist, DBOPT_DTIME, &time );
    // DBAddOption(optlist, DBOPT_UNITS, (void *)units);
    for ( const auto &vector : data.vectors ) {
        int varSize   = vector.numDOFs;
        auto varType  = vector.type;
        auto DOFs     = vector.vec->getDOFManager();
        int nvar      = 0;
        int centering = 0;
        auto var      = new double *[varSize];
        for ( int j = 0; j < varSize; ++j )
            var[j] = nullptr;
        const char *varnames[] = { "1", "2", "3" };
        if ( varType > mesh->getGeomType() ) {
            // We have a mixed mesh type and there will be no data of the given type for this mesh
            continue;
        } else if ( varType == AMP::Mesh::GeomType::Vertex ) {
            // We are saving node-centered data
            centering = DB_NODECENT;
            nvar      = (int) nodelist_ids.size();
            for ( int j = 0; j < varSize; ++j )
                var[j] = new double[nvar];
            std::vector<size_t> dofs( varSize );
            std::vector<double> vals( varSize );
            for ( int j = 0; j < nvar; ++j ) {
                DOFs->getDOFs( nodelist_ids[j], dofs );
                AMP_ASSERT( (int) dofs.size() == varSize );
                vector.vec->getValuesByGlobalID( varSize, &dofs[0], &vals[0] );
                for ( int k = 0; k < varSize; ++k )
                    var[k][j] = vals[k];
            }
        } else if ( varType == mesh->getGeomType() ) {
            // We are saving cell-centered data
            centering = DB_ZONECENT;
            nvar      = (int) num_elems;
            for ( int j = 0; j < varSize; ++j )
                var[j] = new double[nvar];
            std::vector<size_t> dofs( varSize );
            std::vector<double> vals( varSize );
            auto it = element_iterator.begin();
            for ( int j = 0; j < nvar; ++j, ++it ) {
                DOFs->getDOFs( it->globalID(), dofs );
                vector.vec->getValuesByGlobalID( varSize, &dofs[0], &vals[0] );
                for ( int k = 0; k < varSize; ++k )
                    var[k][j] = vals[k];
            }
        } else {
            // We are storing edge or face data
            AMP_ERROR( "The silo writer currently only supports GeomType::Vertex and Cell data" );
        }
        std::string varNameRank = vector.name + "P" + std::to_string( data.rank );
        if ( varSize == 1 || varSize == d_dim || varSize == d_dim * d_dim ) {
            // We are writing a scalar, vector, or tensor variable
            DBPutUcdvar( FileHandle,
                         varNameRank.c_str(),
                         meshName.c_str(),
                         varSize,
                         (char **) varnames,
                         var,
                         nvar,
                         nullptr,
                         0,
                         DB_DOUBLE,
                         centering,
                         optlist );
        } else {
            // Write each component
            for ( int j = 0; j < varSize; ++j ) {
                auto vname = varNameRank + "_" + std::to_string( j );
                DBPutUcdvar( FileHandle,
                             vname.c_str(),
                             meshName.c_str(),
                             1,
                             (char **) varnames,
                             &var[j],
                             nvar,
                             nullptr,
                             0,
                             DB_DOUBLE,
                             centering,
                             optlist );
            }
        }
        for ( int j = 0; j < varSize; ++j ) {
            if ( var[j] )
                delete[] var[j];
        }
        delete[] var;
    }
    DBFreeOptlist( optlist );
    PROFILE_STOP( "writeMesh - variables", 2 );
#endif
    // Change the directory back to root
    DBSetDir( FileHandle, "/" );
    PROFILE_STOP( "writeMesh", 1 );
}


/************************************************************
 * Function to syncronize the multimesh data                 *
 * If root==-1, the data will be synced across all procs     *
 ************************************************************/
void SiloIO::syncMultiMeshData( std::map<uint64_t, multiMeshData> &data, int root ) const
{
    PROFILE_START( "syncMultiMeshData", 1 );
    // Convert the data to vectors
    std::vector<uint64_t> ids;
    std::vector<multiMeshData> meshdata;
    int myRank = d_comm.getRank();
    for ( const auto &it : data ) {
        // Only send the base meshes that I own
        auto tmp = it.second;
        tmp.meshes.resize( 0 );
        for ( auto &elem : it.second.meshes ) {
            if ( elem.ownerRank == myRank )
                tmp.meshes.push_back( elem );
        }
        // Only the owner rank will send the variable list
        if ( tmp.ownerRank != myRank )
            tmp.varName.resize( 0 );
        // Only send the multimesh if there are base meshes that need to be sent or I own the mesh
        if ( !tmp.meshes.empty() || tmp.ownerRank == myRank ) {
            ids.push_back( it.first );
            meshdata.push_back( it.second );
        }
    }
    // Create buffers to store the data
    size_t send_size = 0;
    for ( size_t i = 0; i < meshdata.size(); ++i ) {
        AMP_ASSERT( ids[i] == meshdata[i].id );
        send_size += meshdata[i].size();
    }
    auto send_buf = new char[send_size];
    char *ptr     = send_buf;
    for ( auto &elem : meshdata ) {
        elem.pack( ptr );
        ptr = &ptr[elem.size()];
    }
    // Send the data and unpack the buffer to a vector
    size_t tot_num = d_comm.sumReduce( meshdata.size() );
    if ( root == -1 ) {
        // Everybody gets a copy
        size_t tot_size = d_comm.sumReduce( send_size );
        auto recv_buf   = new char[tot_size];
        meshdata.resize( tot_num );
        d_comm.allGather( send_buf, send_size, recv_buf );
        ptr = recv_buf;
        for ( size_t i = 0; i < tot_num; ++i ) {
            meshdata[i] = multiMeshData::unpack( ptr );
            ptr         = &ptr[meshdata[i].size()];
        }
        delete[] recv_buf;
    } else {
        AMP_ASSERT( root >= 0 && root < d_comm.getSize() );
        // Only the root gets a copy
        // Note: the root already has his own data
        size_t max_size = d_comm.maxReduce( send_size );
        std::vector<int> recv_num( d_comm.getSize() );
        d_comm.allGather( (int) meshdata.size(), &recv_num[0] );
        if ( root == d_comm.getRank() ) {
            // Recieve all data
            meshdata.resize( 0 );
            meshdata.reserve( tot_num );
            auto recv_buf = new char[max_size];
            for ( int i = 0; i < d_comm.getSize(); ++i ) {
                if ( i == root )
                    continue;
                int recv_size = d_comm.probe( i, 24987 );
                AMP_ASSERT( recv_size <= (int) max_size );
                d_comm.recv( recv_buf, recv_size, i, false, 24987 );
                char *cptr = recv_buf;
                for ( int j = 0; j < recv_num[i]; ++j ) {
                    auto tmp = multiMeshData::unpack( cptr );
                    cptr     = &cptr[tmp.size()];
                    meshdata.push_back( tmp );
                }
            }
            delete[] recv_buf;
        } else {
            // Send my data
            d_comm.send( send_buf, send_size, root, 24987 );
        }
    }
    delete[] send_buf;
    // Add the meshes from other processors (keeping the existing meshes)
    for ( auto &elem : meshdata ) {
        auto iterator = data.find( elem.id );
        if ( iterator == data.end() ) {
            // Add the multimesh
            data.insert( std::make_pair( elem.id, elem ) );
        } else {
            // Add the submeshes
            for ( auto &meshe : elem.meshes ) {
                bool found = false;
                for ( auto &_k : iterator->second.meshes ) {
                    if ( meshe.id == _k.id && meshe.meshName == _k.meshName &&
                         meshe.path == _k.path && meshe.path == _k.file )
                        found = true;
                }
                if ( !found )
                    iterator->second.meshes.push_back( meshe );
            }
            // Add the variables if we don't have them yet
            if ( elem.varName.size() > 0 ) {
                if ( !iterator->second.varName.empty() )
                    AMP_ASSERT( iterator->second.varName.size() == elem.varName.size() );
                iterator->second.varName = elem.varName;
            }
        }
    }
    PROFILE_STOP( "syncMultiMeshData", 1 );
}


/************************************************************
 * Function to syncronize a variable list                    *
 * If root==-1, the data will be synced across all procs     *
 ************************************************************/
void SiloIO::syncVariableList( std::set<std::string> &data_set, int root ) const
{
    PROFILE_START( "syncVariableList", 1 );
    std::vector<std::string> data( data_set.begin(), data_set.end() );
    size_t N_local  = data.size();
    size_t N_global = d_comm.sumReduce( N_local );
    auto size_local = new size_t[N_local];
    for ( size_t i = 0; i < N_local; ++i )
        size_local[i] = data[i].size();
    auto size_global = new size_t[N_global];
    d_comm.allGather( size_local, N_local, size_global );
    size_t tot_size_local = 0;
    for ( size_t i = 0; i < N_local; ++i )
        tot_size_local += size_local[i];
    size_t tot_size_global = 0;
    for ( size_t i = 0; i < N_global; ++i )
        tot_size_global += size_global[i];
    auto send_buf = new char[tot_size_local];
    auto recv_buf = new char[tot_size_global];
    size_t k      = 0;
    for ( size_t i = 0; i < N_local; ++i ) {
        data[i].copy( &send_buf[k], data[i].size(), 0 );
        k += size_local[i];
    }
    if ( root == -1 ) {
        // Everybody gets a copy
        d_comm.allGather( send_buf, tot_size_local, recv_buf );
        k = 0;
        for ( size_t i = 0; i < N_global; ++i ) {
            std::string tmp( &recv_buf[k], size_global[i] );
            data_set.insert( tmp );
            k += size_global[i];
        }
    } else {
        // Only the root gets a copy
        // Note: the root already has his own data
        AMP_ASSERT( root >= 0 && root < d_comm.getSize() );
        std::vector<int> recv_num( d_comm.getSize() );
        d_comm.allGather( (int) N_local, &recv_num[0] );
        if ( root == d_comm.getRank() ) {
            // Recieve all data
            int index = 0;
            for ( int i = 0; i < d_comm.getSize(); ++i ) {
                if ( i == root ) {
                    index += recv_num[i];
                    continue;
                }
                int recv_size = d_comm.probe( i, 24987 );
                d_comm.recv( recv_buf, recv_size, i, false, 24987 );
                k = 0;
                for ( int j = 0; j < recv_num[i]; ++j ) {
                    std::string tmp( &recv_buf[k], size_global[index] );
                    data_set.insert( tmp );
                    k += size_global[index];
                    index++;
                }
                AMP_ASSERT( (int) k == recv_size );
            }
        } else {
            // Send my data
            d_comm.send( send_buf, tot_size_local, root, 24987 );
        }
    }
    delete[] send_buf;
    delete[] recv_buf;
    delete[] size_local;
    delete[] size_global;
    PROFILE_STOP( "syncVariableList", 1 );
}


/************************************************************
 * Function to write the summary data                        *
 ************************************************************/
static inline std::string getFile( const std::string &file, const std::string &root )
{
    AMP_ASSERT( !file.empty() );
    if ( file.compare( 0, root.size(), root ) == 0 )
        return file.substr( root.size() );
    return file;
}
void SiloIO::writeSummary( std::string filename, int cycle, double time )
{
    PROFILE_START( "writeSummary", 1 );
    AMP_ASSERT( !filename.empty() );
    // Add the baseMeshData to the multimeshes
    auto multiMeshes = d_multiMeshes;
    for ( auto &tmp : multiMeshes ) {
        auto mesh     = tmp.second.mesh;
        auto base_ids = getMeshIDs( mesh );
        for ( auto id : base_ids ) {
            const auto &it = d_baseMeshes.find( id.getData() );
            if ( it != d_baseMeshes.end() ) {
                baseMeshData data = it->second;
                AMP_ASSERT( it->first == data.id );
                tmp.second.meshes.push_back( data );
            }
        }
    }
    // Add the whole mesh
    /*if ( multiMeshes.size()==0 ) {
        multiMeshData wholemesh;
        wholemesh.id = AMP::Mesh::MeshID((unsigned int)-1,0);
        wholemesh.name = "whole_mesh";
        for (auto it=d_baseMeshes.begin(); it!=d_baseMeshes.end(); ++it) {
            baseMeshData data = it->second;
            AMP_ASSERT(it->first==data.id);
            wholemesh.meshes.push_back(data);
        }
        wholemesh.owner_rank = 0;
        multimeshes.insert( std::make_pair(wholemesh.id,wholemesh)
    );
    }*/
    // Gather the results
    // Note: we only need to guarantee that rank 0 has all the data
    syncMultiMeshData( multiMeshes, 0 );
    syncVariableList( d_varNames, 0 );
    // Write the multimeshes and multivariables
    std::string base_path;
    if ( find_slash( filename ) != std::string::npos )
        base_path = filename.substr( 0, find_slash( filename ) + 1 );
    if ( d_comm.getRank() == 0 ) {
        DBfile *FileHandle = DBOpen( filename.c_str(), DB_HDF5, DB_APPEND );
        // Create the subdirectories
        PROFILE_START( "create directories", 2 );
        std::set<std::string> subdirs;
        for ( const auto &tmp : multiMeshes ) {
            auto data  = tmp.second;
            auto file  = getFile( data.name, base_path );
            size_t pos = find_slash( file );
            if ( pos != std::string::npos )
                subdirs.insert( file.substr( 0, pos ) );
        }
        for ( const auto &subdir : subdirs )
            createSiloDirectory( FileHandle, subdir );
        PROFILE_STOP( "create directories", 2 );
        // Create the multimeshes
        PROFILE_START( "write multimeshes", 2 );
        for ( const auto &tmp : multiMeshes ) {
            const auto &data = tmp.second;
            size_t N         = data.meshes.size();
            std::vector<std::string> meshNames( N );
            for ( size_t i = 0; i < N; ++i ) {
                auto file    = getFile( data.meshes[i].file, base_path );
                meshNames[i] = file + ":" + data.meshes[i].path + "/" + data.meshes[i].meshName;
                strrep( meshNames[i], "//", "/" );
            }
            auto meshnames = new char *[N];
            auto meshtypes = new int[N];
            for ( size_t i = 0; i < N; ++i ) {
                meshnames[i] = (char *) meshNames[i].c_str();
                meshtypes[i] = DB_UCDMESH;
            }
            std::string tree_name = data.name + "_tree";
            DBoptlist *optList    = DBMakeOptlist( 10 );
            DBAddOption( optList, DBOPT_MRGTREE_NAME, (char *) tree_name.c_str() );
            DBPutMultimesh(
                FileHandle, data.name.c_str(), meshNames.size(), meshnames, meshtypes, nullptr );
            DBFreeOptlist( optList );
            delete[] meshnames;
            delete[] meshtypes;
        }
        PROFILE_STOP( "write multimeshes", 2 );
        // Generate the multi-variables
        PROFILE_START( "write multivariables", 2 );
        for ( const auto &tmp : multiMeshes ) {
            const auto &data = tmp.second;
            size_t N         = data.meshes.size();
            // std::cout << data.name << std::endl;
            for ( const auto &varName : data.varName ) {
                std::vector<std::string> varNames( N );
                auto varnames = new char *[N];
                auto vartypes = new int[N];
                for ( size_t i = 0; i < N; ++i ) {
                    std::string rankStr = std::to_string( data.meshes[i].rank );
                    auto file           = getFile( data.meshes[i].file, base_path );
                    varNames[i] = file + ":" + data.meshes[i].path + "/" + varName + "P" + rankStr;
                    strrep( varNames[i], "//", "/" );
                    varnames[i] = (char *) varNames[i].c_str();
                    vartypes[i] = DB_UCDVAR;
                }
                int varSize = 0;
                for ( size_t i = 0; i < data.meshes[0].vectors.size(); ++i ) {
                    if ( data.meshes[0].vectors[i].name == varName ) {
                        varSize = data.meshes[0].vectors[i].numDOFs;
                        break;
                    }
                }
                auto multiMeshName = data.name;
                auto visitVarName  = multiMeshName + "_" + varName;
                float ftime        = time;
                DBoptlist *opts    = DBMakeOptlist( 10 );
                DBAddOption( opts, DBOPT_CYCLE, &cycle );
                DBAddOption( opts, DBOPT_TIME, &ftime );
                DBAddOption( opts, DBOPT_DTIME, &time );
                // DBAddOption( opts, DBOPT_MMESH_NAME, (char*) multiMeshName.c_str() );
                if ( varSize == 1 || varSize == d_dim || varSize == d_dim * d_dim ) {
                    // We are writing a scalar, vector, or tensor variable
                    DBPutMultivar( FileHandle,
                                   visitVarName.c_str(),
                                   varNames.size(),
                                   varnames,
                                   vartypes,
                                   opts );
                } else {
                    // Write each component
                    for ( int j = 0; j < varSize; ++j ) {
                        std::string postfix = "_" + std::to_string( j );
                        std::vector<std::string> varNames2( data.meshes.size() );
                        for ( size_t k = 0; k < data.meshes.size(); ++k ) {
                            varNames2[k] = varNames[k] + postfix;
                            varnames[k]  = (char *) varNames2[k].c_str();
                        }
                        DBPutMultivar( FileHandle,
                                       ( visitVarName + postfix ).c_str(),
                                       varNames.size(),
                                       varnames,
                                       vartypes,
                                       opts );
                    }
                }
                DBFreeOptlist( opts );
                delete[] varnames;
                delete[] vartypes;
            }
        }
        PROFILE_STOP( "write multivariables", 2 );
        DBClose( FileHandle );
    }
    PROFILE_STOP( "writeSummary", 1 );
}


/************************************************************
 * Functions to pack/unpack data to a char array             *
 ************************************************************/
template<class TYPE>
static inline void packData( char *ptr, size_t &pos, const TYPE &data );
template<class TYPE>
static inline TYPE unpackData( const char *ptr, size_t &pos );
template<>
inline void packData<std::string>( char *ptr, size_t &pos, const std::string &data )
{
    int N = data.size();
    memcpy( &ptr[pos], data.c_str(), N + 1 );
    pos += N + 1;
}
template<>
inline std::string unpackData<std::string>( const char *ptr, size_t &pos )
{
    std::string data( &ptr[pos] );
    pos += data.size() + 1;
    return data;
}
template<class TYPE>
static inline void packData( char *ptr, size_t &pos, const TYPE &data )
{
    memcpy( &ptr[pos], &data, sizeof( TYPE ) );
    pos += sizeof( TYPE );
}
template<class TYPE>
static inline TYPE unpackData( const char *ptr, size_t &pos )
{
    TYPE data;
    memcpy( &data, &ptr[pos], sizeof( TYPE ) );
    pos += sizeof( TYPE );
    return data;
}


/************************************************************
 * Some utility functions                                    *
 ************************************************************/
void createSiloDirectory( DBfile *FileHandle, const std::string &path )
{
    // Create a subdirectory tree from the current working path if it does not exist
    char current_dir[256];
    DBGetDir( FileHandle, current_dir );
    // Get the list of directories that may need to be created
    std::vector<std::string> subdirs;
    std::string path2 = path + "/";
    while ( !path2.empty() ) {
        size_t pos = path2.find( "/" );
        if ( pos > 0 ) {
            subdirs.push_back( path2.substr( 0, pos ) );
        }
        path2.erase( 0, pos + 1 );
    }
    // Create the directories as necessary
    for ( auto &subdir : subdirs ) {
        DBtoc *toc  = DBGetToc( FileHandle );
        bool exists = false;
        for ( int j = 0; j < toc->ndir; ++j ) {
            if ( subdir.compare( toc->dir_names[j] ) == 0 )
                exists = true;
        }
        if ( !exists )
            DBMkDir( FileHandle, subdir.c_str() );
        DBSetDir( FileHandle, subdir.c_str() );
    }
    // Return back to the original working directory
    DBSetDir( FileHandle, current_dir );
}


#else
void SiloIO::readFile( const std::string & ) {}
void SiloIO::writeFile( const std::string &, size_t, double ) {}
#endif


} // namespace AMP::Utilities
