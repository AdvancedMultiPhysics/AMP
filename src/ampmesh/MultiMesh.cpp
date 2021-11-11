#include "AMP/ampmesh/MultiMesh.h"
#include "AMP/ampmesh/MeshElement.h"
#include "AMP/ampmesh/MeshParameters.h"
#include "AMP/ampmesh/MultiGeometry.h"
#include "AMP/ampmesh/MultiIterator.h"
#include "AMP/ampmesh/SubsetMesh.h"
#include "AMP/ampmesh/loadBalance/loadBalanceSimulator.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#ifdef USE_AMP_VECTORS
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"
#endif

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace AMP {
namespace Mesh {


// Function to check if a string contains another string as the prefix
static bool check_prefix( std::string prefix, std::string str )
{
    if ( str.size() < prefix.size() )
        return false;
    if ( str.compare( 0, prefix.size(), prefix ) == 0 )
        return true;
    return false;
}


// Misc function declerations
static void copyKey( std::shared_ptr<const AMP::Database>,
                     std::vector<std::shared_ptr<AMP::Database>> &,
                     const std::string &,
                     bool,
                     const std::string &,
                     const std::vector<std::string> & );

/********************************************************
 * Constructors                                          *
 ********************************************************/
MultiMesh::MultiMesh( std::shared_ptr<const MeshParameters> params_in ) : Mesh( params_in )
{

    AMP_ASSERT( d_db != nullptr );
    // Create an array of MeshParameters for each submesh
    auto meshDatabases = MultiMesh::createDatabases( d_db );
    // Create the load balancer and comms
    loadBalanceSimulator loadBalance( d_db );
    loadBalance.setProcs( d_comm.getSize() );
    const auto &submeshes = loadBalance.getSubmeshes();
    std::vector<std::vector<int>> groups( submeshes.size() );
    for ( size_t i = 0; i < submeshes.size(); i++ )
        groups[i] = submeshes[i].getRanks();
    auto comms = createComms( d_comm, groups );
    // Check that every mesh exist on some comm
    std::vector<int> onComm( comms.size(), 0 );
    for ( size_t i = 0; i < comms.size(); i++ ) {
        if ( !comms[i].isNull() )
            onComm[i] = 1;
    }
    d_comm.maxReduce( &onComm[0], (int) onComm.size() );
    for ( auto &elem : onComm ) {
        AMP_ASSERT( elem == 1 );
    }
    // Create the meshes
    d_meshes = std::vector<AMP::Mesh::Mesh::shared_ptr>( 0 );
    for ( size_t i = 0; i < comms.size(); i++ ) {
        if ( comms[i].isNull() )
            continue;
        auto params = std::make_shared<MeshParameters>( meshDatabases[i] );
        params->setComm( comms[i] );
        auto new_mesh = AMP::Mesh::Mesh::buildMesh( params );
        d_meshes.push_back( new_mesh );
    }
    // Get the physical dimension
    PhysicalDim = 0;
    if ( !d_meshes.empty() )
        PhysicalDim = d_meshes[0]->getDim();
    PhysicalDim = d_comm.maxReduce( PhysicalDim );
    for ( size_t i = 0; i < d_meshes.size(); i++ )
        AMP_INSIST( PhysicalDim == d_meshes[i]->getDim(),
                    "Physical dimension must match for all meshes in multimesh" );
    // Get the highest geometric type
    GeomDim   = AMP::Mesh::GeomType::Vertex;
    d_max_gcw = 0;
    for ( size_t i = 0; i < d_meshes.size(); i++ ) {
        AMP_INSIST( PhysicalDim == d_meshes[i]->getDim(),
                    "Physical dimension must match for all meshes in multimesh" );
        if ( d_meshes[i]->getGeomType() > GeomDim )
            GeomDim = d_meshes[i]->getGeomType();
        if ( d_meshes[i]->getMaxGhostWidth() > d_max_gcw )
            d_max_gcw = d_meshes[i]->getMaxGhostWidth();
    }
    GeomDim   = (GeomType) d_comm.maxReduce( (int) GeomDim );
    d_max_gcw = d_comm.maxReduce( d_max_gcw );
    // Compute the bounding box of the multimesh
    d_box_local = { 1e200, -1e200, 1e200, -1e200, 1e200, -1e200 };
    d_box_local.resize( 2 * PhysicalDim );
    for ( size_t i = 0; i < d_meshes.size(); i++ ) {
        auto meshBox = d_meshes[i]->getBoundingBox();
        for ( int j = 0; j < PhysicalDim; j++ ) {
            if ( meshBox[2 * j + 0] < d_box_local[2 * j + 0] ) {
                d_box_local[2 * j + 0] = meshBox[2 * j + 0];
            }
            if ( meshBox[2 * j + 1] > d_box_local[2 * j + 1] ) {
                d_box_local[2 * j + 1] = meshBox[2 * j + 1];
            }
        }
    }
    d_box = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] = d_comm.minReduce( d_box_local[2 * i + 0] );
        d_box[2 * i + 1] = d_comm.maxReduce( d_box_local[2 * i + 1] );
    }
    // Displace the meshes
    std::vector<double> displacement( PhysicalDim, 0.0 );
    if ( d_db->keyExists( "x_offset" ) )
        displacement[0] = d_db->getScalar<double>( "x_offset" );
    if ( d_db->keyExists( "y_offset" ) )
        displacement[1] = d_db->getScalar<double>( "y_offset" );
    if ( d_db->keyExists( "z_offset" ) )
        displacement[2] = d_db->getScalar<double>( "z_offset" );
    bool test = false;
    for ( auto &elem : displacement ) {
        if ( elem != 0.0 )
            test = true;
    }
    if ( test )
        displaceMesh( displacement );
    // Create additional multi-mesh views
    for ( int i = 1; d_db->keyExists( "MeshView_" + std::to_string( i ) ); i++ ) {
        auto db   = d_db->getDatabase( "MeshView_" + std::to_string( i ) );
        auto name = db->getString( "MeshName" );
        auto op   = db->getWithDefault<std::string>( "Operation", "" );
        auto list = db->getVector<std::string>( "MeshList" );
        std::vector<Mesh::shared_ptr> meshes;
        for ( const auto &tmp : list ) {
            auto mesh = this->Subset( tmp );
            if ( mesh )
                meshes.push_back( mesh );
        }
        auto comm = d_comm.split( meshes.empty() ? 0 : 1 );
        if ( meshes.empty() )
            continue;
        auto mesh = std::make_shared<MultiMesh>( name, comm, meshes );
        if ( op == "" ) {
            d_meshes.push_back( mesh );
        } else if ( op == "SurfaceIterator" ) {
            auto type =
                static_cast<AMP::Mesh::GeomType>( static_cast<int>( mesh->getGeomType() ) - 1 );
            auto mesh2 = mesh->Subset( mesh->getSurfaceIterator( type ) );
            mesh2->setName( name );
            d_meshes.push_back( mesh2 );
        } else {
            AMP_ERROR( "Unknown operation" );
        }
    }
    // Construct the geometry object for the multimesh
    std::vector<std::shared_ptr<AMP::Geometry::Geometry>> geom;
    for ( auto &mesh : d_meshes ) {
        auto tmp = mesh->getGeometry();
        if ( tmp )
            geom.push_back( tmp );
    }
    if ( !geom.empty() )
        d_geometry.reset( new AMP::Geometry::MultiGeometry( geom ) );
}
MultiMesh::MultiMesh( const std::string &name,
                      const AMP_MPI &comm,
                      const std::vector<Mesh::shared_ptr> &meshes )
{
    d_name = name;
    d_comm = comm;
    this->setMeshID();
    // Get the list of non-null meshes
    d_meshes = std::vector<Mesh::shared_ptr>();
    for ( auto &meshe : meshes ) {
        if ( meshe )
            d_meshes.push_back( meshe );
    }
    if ( d_comm.sumReduce( d_meshes.size() ) == 0 ) {
        AMP_ERROR( "Empty multimeshes have not been tested yet" );
    }
    // Check the comm (note: the order for the comparison matters)
    for ( auto &elem : d_meshes ) {
        AMP_ASSERT( elem->getComm() <= d_comm );
    }
    // Get the physical dimension and the highest geometric type
    PhysicalDim = d_meshes[0]->getDim();
    GeomDim     = d_meshes[0]->getGeomType();
    d_max_gcw   = 0;
    for ( size_t i = 1; i < d_meshes.size(); i++ ) {
        AMP_INSIST( PhysicalDim == d_meshes[i]->getDim(),
                    "Physical dimension must match for all meshes in multimesh" );
        if ( d_meshes[i]->getGeomType() > GeomDim )
            GeomDim = d_meshes[i]->getGeomType();
        if ( d_meshes[i]->getMaxGhostWidth() > d_max_gcw )
            d_max_gcw = d_meshes[i]->getMaxGhostWidth();
    }
    GeomDim   = (GeomType) d_comm.maxReduce( (int) GeomDim );
    d_max_gcw = d_comm.maxReduce( d_max_gcw );
    // Compute the bounding box of the multimesh
    d_box_local = d_meshes[0]->getBoundingBox();
    for ( size_t i = 1; i < d_meshes.size(); i++ ) {
        auto meshBox = d_meshes[i]->getBoundingBox();
        for ( int j = 0; j < PhysicalDim; j++ ) {
            if ( meshBox[2 * j + 0] < d_box_local[2 * j + 0] ) {
                d_box_local[2 * j + 0] = meshBox[2 * j + 0];
            }
            if ( meshBox[2 * j + 1] > d_box_local[2 * j + 1] ) {
                d_box_local[2 * j + 1] = meshBox[2 * j + 1];
            }
        }
    }
    d_box = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] = d_comm.minReduce( d_box_local[2 * i + 0] );
        d_box[2 * i + 1] = d_comm.maxReduce( d_box_local[2 * i + 1] );
    }
    // Construct the geometry object for the multimesh
    std::vector<std::shared_ptr<AMP::Geometry::Geometry>> geom;
    for ( auto &mesh : d_meshes ) {
        auto tmp = mesh->getGeometry();
        if ( tmp )
            geom.push_back( tmp );
    }
    d_geometry.reset( new AMP::Geometry::MultiGeometry( geom ) );
}
MultiMesh::MultiMesh( const MultiMesh &rhs ) : Mesh( rhs )
{
    for ( const auto &mesh : rhs.d_meshes )
        d_meshes.push_back( mesh->clone() );
    std::vector<std::shared_ptr<AMP::Geometry::Geometry>> geom;
    for ( auto &mesh : d_meshes ) {
        auto tmp = mesh->getGeometry();
        if ( tmp )
            geom.push_back( tmp );
    }
    d_geometry.reset( new AMP::Geometry::MultiGeometry( geom ) );
}


/********************************************************
 * De-constructor                                        *
 ********************************************************/
MultiMesh::~MultiMesh() = default;


/********************************************************
 * Return the class name                                 *
 ********************************************************/
std::string MultiMesh::meshClass() const
{
    if ( d_meshes.empty() )
        return "MultiMesh<>";
    std::string name = "MultiMesh<" + d_meshes[0]->meshClass();
    for ( size_t i = 1; i < d_meshes.size(); i++ ) {
        name += "," + d_meshes[i]->meshClass();
    }
    name += ">";
    return name;
}


/********************************************************
 * Function to copy the mesh                             *
 ********************************************************/
std::unique_ptr<Mesh> MultiMesh::clone() const { return std::make_unique<MultiMesh>( *this ); }


/********************************************************
 * Function to estimate the mesh size                    *
 ********************************************************/
size_t MultiMesh::estimateMeshSize( std::shared_ptr<const MeshParameters> params_in )
{
    auto db = params_in->getDatabase();
    // Create an array of MeshParameters for each submesh
    auto meshDatabases = MultiMesh::createDatabases( db );
    std::vector<std::shared_ptr<MeshParameters>> params( meshDatabases.size() );
    for ( size_t i = 0; i < meshDatabases.size(); i++ )
        params[i] = std::make_shared<AMP::Mesh::MeshParameters>( meshDatabases[i] );
    // Get the approximate number of elements for each mesh
    size_t totalMeshSize = 0;
    for ( auto &elem : params ) {
        size_t localMeshSize = AMP::Mesh::Mesh::estimateMeshSize( elem );
        AMP_ASSERT( localMeshSize > 0 );
        totalMeshSize += localMeshSize;
    }
    // Adjust the number of elements by a weight if desired
    if ( db->keyExists( "Weight" ) ) {
        double weight = db->getScalar<double>( "Weight" );
        totalMeshSize = (size_t) ceil( weight * ( (double) totalMeshSize ) );
    }
    return totalMeshSize;
}


/********************************************************
 * Function to estimate the mesh size                    *
 ********************************************************/
size_t MultiMesh::maxProcs( std::shared_ptr<const MeshParameters> params_in )
{
    auto db = params_in->getDatabase();
    // Create an array of MeshParameters for each submesh
    auto meshDatabases = MultiMesh::createDatabases( db );
    std::vector<std::shared_ptr<MeshParameters>> params( meshDatabases.size() );
    for ( size_t i = 0; i < meshDatabases.size(); i++ )
        params[i] = std::make_shared<AMP::Mesh::MeshParameters>( meshDatabases[i] );
    // Get the approximate number of elements for each mesh
    size_t totalMaxSize = 0;
    int method          = db->getWithDefault<int>( "LoadBalanceMethod", 1 );
    for ( auto &elem : params ) {
        size_t localMaxSize = AMP::Mesh::Mesh::maxProcs( elem );
        AMP_ASSERT( localMaxSize > 0 );
        if ( method == 1 ) {
            totalMaxSize += localMaxSize;
        } else if ( method == 2 ) {
            totalMaxSize = std::max( totalMaxSize, localMaxSize );
        }
    }
    return totalMaxSize;
}


/********************************************************
 * Function to create the databases for the meshes       *
 * within the multimesh.                                 *
 ********************************************************/
std::vector<std::shared_ptr<AMP::Database>>
MultiMesh::createDatabases( std::shared_ptr<const AMP::Database> database )
{
    // We might have already created and stored the databases for each mesh
    if ( database->keyExists( "submeshDatabases" ) ) {
        auto databaseNames = database->getVector<std::string>( "submeshDatabases" );
        AMP_ASSERT( !databaseNames.empty() );
        std::vector<std::shared_ptr<AMP::Database>> meshDatabases( databaseNames.size() );
        for ( size_t i = 0; i < databaseNames.size(); i++ )
            meshDatabases[i] = database->getDatabase( databaseNames[i] )->cloneDatabase();
        return meshDatabases;
    }
    // Find all of the meshes in the database
    AMP_ASSERT( database != nullptr );
    AMP_INSIST( database->keyExists( "MeshDatabasePrefix" ),
                "MeshDatabasePrefix must exist in input database" );
    AMP_INSIST( database->keyExists( "MeshArrayDatabasePrefix" ),
                "MeshArrayDatabasePrefix must exist in input database" );
    std::string MeshPrefix      = database->getString( "MeshDatabasePrefix" );
    std::string MeshArrayPrefix = database->getString( "MeshArrayDatabasePrefix" );
    AMP_ASSERT( !check_prefix( MeshPrefix, MeshArrayPrefix ) );
    AMP_ASSERT( !check_prefix( MeshArrayPrefix, MeshPrefix ) );
    std::vector<std::string> keys = database->getAllKeys();
    std::vector<std::string> meshes, meshArrays;
    for ( auto &key : keys ) {
        if ( check_prefix( MeshPrefix, key ) ) {
            meshes.push_back( key );
        } else if ( check_prefix( MeshArrayPrefix, key ) ) {
            meshArrays.push_back( key );
        }
    }
    // Create the basic databases for each mesh
    std::vector<std::shared_ptr<AMP::Database>> meshDatabases;
    for ( auto &meshe : meshes ) {
        // We are dealing with a single mesh object, use the existing database
        auto database2 = database->getDatabase( meshe )->cloneDatabase();
        meshDatabases.push_back( std::move( database2 ) );
    }
    for ( auto &meshArray : meshArrays ) {
        // We are dealing with an array of meshes, create a database for each
        auto database1 = database->getDatabase( meshArray );
        int N          = database1->getScalar<int>( "N" );
        // Get the iterator and indicies
        std::string iterator;
        std::vector<std::string> index( N );
        if ( database1->keyExists( "iterator" ) ) {
            iterator = database1->getString( "iterator" );
            AMP_ASSERT( database1->keyExists( "indicies" ) );
            if ( database1->isType<int>( "indicies" ) ) {
                auto array = database1->getVector<int>( "indicies" );
                AMP_ASSERT( (int) array.size() == N );
                for ( int j = 0; j < N; j++ ) {
                    std::stringstream ss;
                    ss << array[j];
                    index[j] = ss.str();
                }
            } else if ( database1->isType<std::string>( "indicies" ) ) {
                index = database1->getVector<std::string>( "indicies" );
            } else {
                AMP_ERROR( "Unknown type for indicies" );
            }
        }
        // Create the new databases
        std::vector<std::shared_ptr<AMP::Database>> databaseArray( N );
        for ( int j = 0; j < N; j++ )
            databaseArray[j] = std::make_shared<AMP::Database>( meshArray );
        // Populate the databases with the proper keys
        keys = database1->getAllKeys();
        for ( auto &key : keys ) {
            if ( key.compare( "N" ) == 0 || key.compare( "iterator" ) == 0 ||
                 key.compare( "indicies" ) == 0 ) {
                // These keys are used by the mesh-array and should not be copied
            } else if ( key.compare( "Size" ) == 0 || key.compare( "Range" ) == 0 ) {
                // These are special keys that should not be divided
                copyKey( database1, databaseArray, key, false, std::string(), index );
            } else {
                // We need to copy the key (and possibly replace the iterator)
                copyKey( database1, databaseArray, key, true, iterator, index );
            }
        }
        // Add the new databases to meshDatabases
        for ( int j = 0; j < N; j++ )
            meshDatabases.push_back( databaseArray[j] );
    }
    return meshDatabases;
}


/********************************************************
 * Return basic mesh info                                *
 ********************************************************/
size_t MultiMesh::numLocalElements( const GeomType type ) const
{
    // Should we cache this?
    size_t N = 0;
    for ( auto &elem : d_meshes )
        N += elem->numLocalElements( type );
    return N;
}
size_t MultiMesh::numGlobalElements( const GeomType type ) const
{
    // Should we cache this?
    size_t N = numLocalElements( type );
    return d_comm.sumReduce( N );
}
size_t MultiMesh::numGhostElements( const GeomType type, int gcw ) const
{
    // Should we cache this?
    size_t N = 0;
    for ( auto &elem : d_meshes )
        N += elem->numGhostElements( type, gcw );
    return N;
}
std::vector<Mesh::shared_ptr> MultiMesh::getMeshes() { return d_meshes; }
std::vector<Mesh::const_shared_ptr> MultiMesh::getMeshes() const
{
    std::vector<Mesh::const_shared_ptr> list( d_meshes.size() );
    for ( size_t i = 0; i < d_meshes.size(); i++ )
        list[i] = d_meshes[i];
    return list;
}


/********************************************************
 * Return mesh iterators                                 *
 ********************************************************/
MeshIterator MultiMesh::getIterator( const GeomType type, const int gcw ) const
{
    std::vector<MeshIterator> iterators( d_meshes.size() );
    for ( size_t i = 0; i < d_meshes.size(); i++ )
        iterators[i] = MeshIterator( d_meshes[i]->getIterator( type, gcw ) );
    return MultiIterator( iterators );
}
MeshIterator MultiMesh::getSurfaceIterator( const GeomType type, const int gcw ) const
{
    std::vector<MeshIterator> iterators( d_meshes.size() );
    for ( size_t i = 0; i < d_meshes.size(); i++ )
        iterators[i] = MeshIterator( d_meshes[i]->getSurfaceIterator( type, gcw ) );
    return MultiIterator( iterators );
}
std::vector<int> MultiMesh::getBoundaryIDs() const
{
    // Get all local id sets
    std::set<int> ids_set;
    for ( auto &elem : d_meshes ) {
        std::vector<int> mesh_idSet = elem->getBoundaryIDs();
        ids_set.insert( mesh_idSet.begin(), mesh_idSet.end() );
    }
    std::vector<int> local_ids( ids_set.begin(), ids_set.end() );
    // Perform a global communication to syncronize the id sets across all processors
    auto N_id_local = (int) local_ids.size();
    std::vector<int> count( d_comm.getSize(), 0 );
    std::vector<int> disp( d_comm.getSize(), 0 );
    d_comm.allGather( N_id_local, &count[0] );
    for ( int i = 1; i < d_comm.getSize(); i++ )
        disp[i] = disp[i - 1] + count[i - 1];
    int N_id_global = disp[d_comm.getSize() - 1] + count[d_comm.getSize() - 1];
    if ( N_id_global == 0 )
        return std::vector<int>();
    std::vector<int> global_id_list( N_id_global, 0 );
    int *ptr = nullptr;
    if ( N_id_local > 0 )
        ptr = &local_ids[0];
    d_comm.allGather( ptr, N_id_local, &global_id_list[0], &count[0], &disp[0], true );
    // Get the unique set
    for ( auto &elem : global_id_list )
        ids_set.insert( elem );
    // Return the final vector of ids
    return std::vector<int>( ids_set.begin(), ids_set.end() );
}
MeshIterator
MultiMesh::getBoundaryIDIterator( const GeomType type, const int id, const int gcw ) const
{
    std::vector<MeshIterator> iterators;
    iterators.reserve( d_meshes.size() );
    for ( auto &elem : d_meshes ) {
        MeshIterator it = elem->getBoundaryIDIterator( type, id, gcw );
        if ( it.size() > 0 )
            iterators.push_back( it );
    }
    return MultiIterator( iterators );
}
std::vector<int> MultiMesh::getBlockIDs() const
{
    // Get all local id sets
    std::set<int> ids_set;
    for ( auto &elem : d_meshes ) {
        std::vector<int> mesh_idSet = elem->getBlockIDs();
        ids_set.insert( mesh_idSet.begin(), mesh_idSet.end() );
    }
    std::vector<int> local_ids( ids_set.begin(), ids_set.end() );
    // Perform a global communication to syncronize the id sets across all processors
    auto N_id_local = (int) local_ids.size();
    std::vector<int> count( d_comm.getSize(), 0 );
    std::vector<int> disp( d_comm.getSize(), 0 );
    d_comm.allGather( N_id_local, &count[0] );
    for ( int i = 1; i < d_comm.getSize(); i++ )
        disp[i] = disp[i - 1] + count[i - 1];
    int N_id_global = disp[d_comm.getSize() - 1] + count[d_comm.getSize() - 1];
    if ( N_id_global == 0 )
        return std::vector<int>();
    std::vector<int> global_id_list( N_id_global, 0 );
    int *ptr = nullptr;
    if ( N_id_local > 0 )
        ptr = &local_ids[0];
    d_comm.allGather( ptr, N_id_local, &global_id_list[0], &count[0], &disp[0], true );
    // Get the unique set
    for ( auto &elem : global_id_list )
        ids_set.insert( elem );
    // Return the final vector of ids
    return std::vector<int>( ids_set.begin(), ids_set.end() );
}
MeshIterator MultiMesh::getBlockIDIterator( const GeomType type, const int id, const int gcw ) const
{
    std::vector<MeshIterator> iterators;
    iterators.reserve( d_meshes.size() );
    for ( auto &elem : d_meshes ) {
        MeshIterator it = elem->getBlockIDIterator( type, id, gcw );
        if ( it.size() > 0 )
            iterators.push_back( it );
    }
    return MultiIterator( iterators );
}


/********************************************************
 * Function to return the meshID composing the mesh      *
 ********************************************************/
std::vector<MeshID> MultiMesh::getAllMeshIDs() const
{
    std::vector<MeshID> tmp = this->getLocalMeshIDs();
    std::set<MeshID> ids( tmp.begin(), tmp.end() );
    d_comm.setGather( ids );
    return std::vector<MeshID>( ids.begin(), ids.end() );
}
std::vector<MeshID> MultiMesh::getBaseMeshIDs() const
{
    std::vector<MeshID> tmp = this->getLocalBaseMeshIDs();
    std::set<MeshID> ids( tmp.begin(), tmp.end() );
    d_comm.setGather( ids );
    return std::vector<MeshID>( ids.begin(), ids.end() );
}
std::vector<MeshID> MultiMesh::getLocalMeshIDs() const
{
    std::set<MeshID> ids;
    ids.insert( d_meshID );
    for ( auto &elem : d_meshes ) {
        std::vector<MeshID> mesh_ids = elem->getLocalMeshIDs();
        for ( auto &mesh_id : mesh_ids )
            ids.insert( mesh_id );
    }
    return std::vector<MeshID>( ids.begin(), ids.end() );
}
std::vector<MeshID> MultiMesh::getLocalBaseMeshIDs() const
{
    std::set<MeshID> ids;
    for ( auto &elem : d_meshes ) {
        std::vector<MeshID> mesh_ids = elem->getLocalBaseMeshIDs();
        for ( auto &mesh_id : mesh_ids )
            ids.insert( mesh_id );
    }
    return std::vector<MeshID>( ids.begin(), ids.end() );
}


/********************************************************
 * Check if the element is a member of the mesh          *
 ********************************************************/
bool MultiMesh::isMember( const MeshElementID &id ) const
{
    for ( auto &elem : d_meshes ) {
        if ( elem->isMember( id ) )
            return true;
    }
    return false;
}


/********************************************************
 * Function to return the element given an ID            *
 ********************************************************/
MeshElement MultiMesh::getElement( const MeshElementID &elem_id ) const
{
    MeshID mesh_id = elem_id.meshID();
    for ( auto &elem : d_meshes ) {
        auto ids        = elem->getLocalBaseMeshIDs();
        bool mesh_found = false;
        for ( auto &id : ids ) {
            if ( id == mesh_id )
                mesh_found = true;
        }
        if ( mesh_found )
            return elem->getElement( elem_id );
    }
    AMP_ERROR( "A mesh matching the element's mesh id was not found" );
    return MeshElement();
}


/********************************************************
 * Function to return parents of an element              *
 ********************************************************/
std::vector<MeshElement> MultiMesh::getElementParents( const MeshElement &elem,
                                                       const GeomType type ) const
{
    MeshID mesh_id = elem.globalID().meshID();
    for ( auto &_i : d_meshes ) {
        std::vector<MeshID> ids = _i->getLocalBaseMeshIDs();
        bool mesh_found         = false;
        for ( auto &id : ids ) {
            if ( id == mesh_id )
                mesh_found = true;
        }
        if ( mesh_found )
            return _i->getElementParents( elem, type );
    }
    AMP_ERROR( "A mesh matching the element's mesh id was not found" );
    return std::vector<MeshElement>();
}


/********************************************************
 * Function to return the mesh with the given ID         *
 ********************************************************/
std::shared_ptr<Mesh> MultiMesh::Subset( MeshID meshID ) const
{
    if ( d_meshID == meshID )
        return std::const_pointer_cast<Mesh>( shared_from_this() );
    for ( auto &elem : d_meshes ) {
        std::shared_ptr<Mesh> mesh = elem->Subset( meshID );
        if ( mesh )
            return mesh;
    }
    return std::shared_ptr<Mesh>();
}


/********************************************************
 * Function to subset a mesh using a mesh iterator       *
 ********************************************************/
std::shared_ptr<Mesh> MultiMesh::Subset( const MeshIterator &iterator_in, bool isGlobal ) const
{
    if ( !isGlobal && iterator_in.size() == 0 )
        return std::shared_ptr<Mesh>();
    // Check the iterator
    auto type = AMP::Mesh::GeomType::null;
    if ( iterator_in.size() > 0 ) {
        type          = iterator_in->elementType();
        auto iterator = iterator_in.begin();
        for ( size_t i = 0; i < iterator.size(); i++ ) {
            if ( type != iterator->elementType() )
                AMP_ERROR( "Subset mesh requires all of the elements to be the same type" );
            ++iterator;
        }
    }
    // Subset for the iterator in each submesh
    std::vector<Mesh::shared_ptr> subset;
    std::set<MeshID> subsetID;
    for ( auto &elem : d_meshes ) {
        MeshIterator iterator;
        if ( iterator_in.size() > 0 ) {
            iterator = Mesh::getIterator( SetOP::Intersection,
                                          iterator_in,
                                          elem->getIterator( type, elem->getMaxGhostWidth() ) );
        }
        auto mesh = elem->Subset( iterator, isGlobal );
        if ( mesh ) {
            subset.push_back( mesh );
            subsetID.insert( mesh->meshID() );
        }
    }
    // Count the number of globally unique sub-meshes
    AMP::AMP_MPI new_comm( AMP_COMM_SELF );
    if ( isGlobal )
        d_comm.setGather( subsetID );
    if ( subsetID.size() <= 1 ) {
        if ( subset.empty() ) {
            return std::shared_ptr<Mesh>();
        } else {
            if ( isGlobal )
                new_comm = subset[0]->getComm();
            return std::make_shared<MultiMesh>( d_name + "_subset", new_comm, subset );
        }
    }
    // Create a new multi-mesh to contain the subset
    if ( isGlobal ) {
        int color = subset.empty() ? -1 : 0;
        new_comm  = d_comm.split( color );
    }
    if ( new_comm.isNull() )
        return std::shared_ptr<Mesh>();
    return std::make_shared<MultiMesh>( d_name + "_subset", new_comm, subset );
}


/********************************************************
 * Function to return the mesh with the given name       *
 ********************************************************/
std::shared_ptr<Mesh> MultiMesh::Subset( std::string name ) const
{
    if ( d_name == name )
        return std::const_pointer_cast<Mesh>( shared_from_this() );
    // Subset for the name in each submesh
    std::vector<Mesh::shared_ptr> subset;
    std::set<MeshID> subsetID;
    for ( auto &elem : d_meshes ) {
        Mesh::shared_ptr mesh = elem->Subset( name );
        if ( mesh ) {
            subset.push_back( mesh );
            subsetID.insert( mesh->meshID() );
        }
    }
    // Count the number of globally unique sub-meshes
    d_comm.setGather( subsetID );
    if ( subsetID.size() <= 1 ) {
        if ( subset.size() == 0 ) {
            return std::shared_ptr<Mesh>();
        } else {
            return subset[0];
        }
    }
    // Create a new multi-mesh to contain the subset
    int color             = subset.empty() ? -1 : 0;
    AMP::AMP_MPI new_comm = d_comm.split( color );
    if ( new_comm.isNull() )
        return std::shared_ptr<Mesh>();
    return std::make_shared<MultiMesh>( name, new_comm, subset );
}


/********************************************************
 * Displace a mesh                                       *
 ********************************************************/
Mesh::Movable MultiMesh::isMeshMovable() const
{
    int value = 2;
    for ( auto &elem : d_meshes )
        value = std::min( value, static_cast<int>( elem->isMeshMovable() ) );
    return static_cast<Mesh::Movable>( value );
}
uint64_t MultiMesh::positionHash() const
{
    uint64_t hash = 0;
    for ( uint64_t i = 0; i < d_meshes.size(); i++ ) {
        auto h = d_meshes[i]->positionHash();
        hash   = hash ^ ( ( h * 0x9E3779B97F4A7C15 ) << i );
    }
    return hash;
}
void MultiMesh::displaceMesh( const std::vector<double> &x_in )
{
    // Check x
    AMP_INSIST( (short int) x_in.size() == PhysicalDim,
                "Displacement vector size should match PhysicalDim" );
    std::vector<double> x = x_in;
    d_comm.minReduce( &x[0], (int) x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        AMP_INSIST( fabs( x[i] - x_in[i] ) < 1e-12, "x does not match on all processors" );
    // Displace the meshes
    for ( auto &mesh : d_meshes )
        mesh->displaceMesh( x );
    // Update the bounding box
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] += x[i];
        d_box[2 * i + 1] += x[i];
        d_box_local[2 * i + 0] += x[i];
        d_box_local[2 * i + 1] += x[i];
    }
}
#ifdef USE_AMP_VECTORS
void MultiMesh::displaceMesh( const AMP::LinearAlgebra::Vector::const_shared_ptr x )
{
    // Displace the individual meshes
    for ( auto &elem : d_meshes )
        elem->displaceMesh( x );
    // Compute the bounding box of the multimesh
    d_box_local = d_meshes[0]->getBoundingBox();
    for ( size_t i = 1; i < d_meshes.size(); i++ ) {
        std::vector<double> meshBox = d_meshes[i]->getBoundingBox();
        for ( int j = 0; j < PhysicalDim; j++ ) {
            if ( meshBox[2 * j + 0] < d_box_local[2 * j + 0] ) {
                d_box_local[2 * j + 0] = meshBox[2 * j + 0];
            }
            if ( meshBox[2 * j + 1] > d_box_local[2 * j + 1] ) {
                d_box_local[2 * j + 1] = meshBox[2 * j + 1];
            }
        }
    }
    d_box = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] = d_comm.minReduce( d_box_local[2 * i + 0] );
        d_box[2 * i + 1] = d_comm.maxReduce( d_box_local[2 * i + 1] );
    }
}
#endif


/****************************************************************
 * Check if two meshes are equal                                 *
 ****************************************************************/
bool MultiMesh::operator==( const Mesh &rhs ) const
{
    // Check if &rhs == this
    if ( this == &rhs )
        return true;
    // Check if we can cast to a MultiMesh
    auto mesh = dynamic_cast<const MultiMesh *>( &rhs );
    if ( !mesh )
        return false;
    // Perform comparison on sub-meshes
    return d_meshes == mesh->d_meshes;
}


/********************************************************
 * Function to copy a key from database 1 to database 2  *
 * If the key is an array of size N, it will only copy   *
 * the ith value.                                        *
 ********************************************************/
template<class TYPE>
static inline void putEntry( std::shared_ptr<const AMP::Database> database1,
                             std::vector<std::shared_ptr<AMP::Database>> &database2,
                             const std::string &key )
{
    auto N    = database2.size();
    auto data = database1->getVector<TYPE>( key );
    for ( size_t i = 0; i < database2.size(); i++ ) {
        if ( N == data.size() )
            database2[i]->putScalar( key, data[i] );
        else
            database2[i]->putVector( key, data );
    }
}
static std::string
strrep( const std::string &in, const std::string_view &s, const std::string_view &r )
{
    std::string str( in );
    size_t pos = str.find( s.data(), 0, s.size() );
    while ( pos != std::string::npos ) {
        str.replace( pos, s.size(), r.data(), r.size() );
        pos = str.find( s.data(), 0, s.size() );
    }
    return str;
}
static void copyKey( std::shared_ptr<const AMP::Database> database1,
                     std::vector<std::shared_ptr<AMP::Database>> &database2,
                     const std::string &key,
                     bool select,
                     const std::string &iterator,
                     const std::vector<std::string> &index )
{
    if ( database1->isDatabase( key ) ) {
        // Copy the database
        auto subDatabase1 = database1->getDatabase( key );
        for ( size_t i = 0; i < database2.size(); i++ ) {
            std::vector<std::shared_ptr<AMP::Database>> subDatabase2(
                1, database2[i]->putDatabase( key ) );
            std::vector<std::string> index2( 1, index[i] );
            auto subKeys = subDatabase1->getAllKeys();
            for ( auto &subKey : subKeys )
                copyKey( subDatabase1, subDatabase2, subKey, false, iterator, index2 );
        }
    } else if ( !select ) {
        for ( size_t i = 0; i < database2.size(); i++ )
            database2[i]->putData( key, database1->getData( key )->clone() );
    } else if ( database1->isType<bool>( key ) ) {
        // Copy a bool
        putEntry<bool>( database1, database2, key );
    } else if ( database1->isType<int>( key ) ) {
        // Copy a int
        putEntry<int>( database1, database2, key );
    } else if ( database1->isType<float>( key ) ) {
        // Copy a float
        putEntry<float>( database1, database2, key );
    } else if ( database1->isType<double>( key ) ) {
        // Copy a double
        putEntry<double>( database1, database2, key );
    } else if ( database1->isType<std::complex<double>>( key ) ) {
        // Copy a std::complex<double>
        putEntry<std::complex<double>>( database1, database2, key );
    } else if ( database1->isType<std::string>( key ) ) {
        // Copy a std::string (checking for the index)
        auto data = database1->getVector<std::string>( key );
        AMP_ASSERT( !data.empty() );
        if ( data.size() == database2.size() ) {
            for ( size_t i = 0; i < database2.size(); i++ )
                database2[i]->putScalar( key, data[i] );
        } else if ( data.size() == 1 ) {
            for ( size_t i = 0; i < database2.size(); i++ ) {
                auto data2 = strrep( data[0], iterator, index[i] );
                database2[i]->putScalar( key, data2 );
            }
        } else {
            for ( size_t i = 0; i < database2.size(); i++ )
                database2[i]->putVector( key, data );
        }
    } else {
        AMP_ERROR( "Unknown key type" );
    }
}


/********************************************************
 * Function to create the sub-communicators              *
 ********************************************************/
std::vector<AMP_MPI> MultiMesh::createComms( const AMP_MPI &comm,
                                             const std::vector<std::vector<int>> &groups )
{
    int myRank = comm.getRank();
    std::vector<AMP_MPI> comms( groups.size() );
    for ( size_t i = 0; i < groups.size(); i++ ) {
        int color = -1;
        for ( auto &elem : groups[i] ) {
            if ( elem == myRank )
                color = 0;
        }
        comms[i] = comm.split( color, comm.getRank() );
        if ( color != -1 )
            AMP_ASSERT( comms[i].getSize() == (int) groups[i].size() );
    }
    return comms;
}


} // namespace Mesh
} // namespace AMP
