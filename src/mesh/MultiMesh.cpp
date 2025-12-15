#include "AMP/mesh/MultiMesh.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/geometry/MultiGeometry.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/SubsetMesh.h"
#include "AMP/mesh/loadBalance/loadBalanceSimulator.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace AMP::Mesh {


// Function to check if a string contains another string as the prefix
static bool check_prefix( const std::string &prefix, const std::string &str )
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
    auto db = params_in->getDatabase();
    AMP_INSIST( db, "Database must exist" );
    // Create an array of MeshParameters for each submesh
    auto meshDatabases = MultiMesh::createDatabases( db );
    // Create the load balancer and comms
    loadBalanceSimulator loadBalance( db );
    loadBalance.setProcs( d_comm.getSize() );
    AMP_ASSERT( loadBalance.maxProcs() >= d_comm.getSize() );
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
    d_meshes = std::vector<std::shared_ptr<AMP::Mesh::Mesh>>( 0 );
    for ( size_t i = 0; i < comms.size(); i++ ) {
        if ( comms[i].isNull() )
            continue;
        auto params = std::make_shared<MeshParameters>( meshDatabases[i] );
        params->setComm( comms[i] );
        auto new_mesh = AMP::Mesh::MeshFactory::create( params );
        AMP_ASSERT( new_mesh );
        d_meshes.push_back( new_mesh );
    }
    // Finish initializing the mesh
    initialize();
    // Displace the meshes
    std::vector<double> displacement( PhysicalDim, 0.0 );
    if ( db->keyExists( "x_offset" ) )
        displacement[0] = db->getScalar<double>( "x_offset" );
    if ( db->keyExists( "y_offset" ) )
        displacement[1] = db->getScalar<double>( "y_offset" );
    if ( db->keyExists( "z_offset" ) )
        displacement[2] = db->getScalar<double>( "z_offset" );
    bool test = false;
    for ( auto &elem : displacement ) {
        if ( elem != 0.0 )
            test = true;
    }
    if ( test )
        displaceMesh( displacement );
    // Create additional multi-mesh views
    for ( int i = 1; db->keyExists( "MeshView_" + std::to_string( i ) ); i++ ) {
        auto db2  = db->getDatabase( "MeshView_" + std::to_string( i ) );
        auto view = createView( *this, *db2 );
        if ( view )
            d_meshes.push_back( view );
    }
}
MultiMesh::MultiMesh( const std::string &name,
                      const AMP_MPI &comm,
                      const std::vector<std::shared_ptr<Mesh>> &meshes,
                      bool checkMeshComm )
{
    d_name = name;
    d_comm = comm;
    this->setMeshID();
    // Get the list of non-null meshes
    d_meshes = std::vector<std::shared_ptr<Mesh>>();
    for ( auto &mesh : meshes ) {
        if ( mesh )
            d_meshes.push_back( mesh );
    }
    if ( d_comm.sumReduce( d_meshes.size() ) == 0 ) {
        AMP_ERROR( "Empty multimeshes have not been tested yet" );
    }
    // Check the comm (note: the order for the comparison matters)
    if ( checkMeshComm ) {
        for ( auto &mesh : d_meshes )
            AMP_ASSERT( mesh->getComm() <= d_comm );
    }
    // Finish initializing the mesh
    initialize();
}
MultiMesh::MultiMesh( const MultiMesh &rhs ) : Mesh( rhs )
{
    for ( const auto &mesh : rhs.d_meshes ) {
        d_meshes.push_back( mesh->clone() );
        AMP_ASSERT( d_meshes.back() );
    }
    std::vector<std::shared_ptr<AMP::Geometry::Geometry>> geom;
    for ( auto &mesh : d_meshes ) {
        auto tmp = mesh->getGeometry();
        if ( tmp )
            geom.push_back( tmp );
    }
    d_geometry.reset( new AMP::Geometry::MultiGeometry( geom ) );
}
void MultiMesh::initialize()
{
    // Get the physical dimension and the highest geometric type
    AMP_ASSERT( !d_meshes.empty() );
    PhysicalDim = d_meshes[0]->getDim();
    GeomDim     = d_meshes[0]->getGeomType();
    d_max_gcw   = d_meshes[0]->getMaxGhostWidth();
    for ( size_t i = 1; i < d_meshes.size(); i++ ) {
        AMP_INSIST( PhysicalDim == d_meshes[i]->getDim(),
                    "Physical dimension must match for all meshes in multimesh" );
        GeomDim   = std::max( GeomDim, d_meshes[i]->getGeomType() );
        d_max_gcw = std::min( d_max_gcw, d_meshes[i]->getMaxGhostWidth() );
    }
    GeomDim   = (GeomType) d_comm.maxReduce( (int) GeomDim );
    d_max_gcw = d_comm.minReduce( d_max_gcw );
    // Compute the bounding box of the multimesh
    d_box       = d_meshes[0]->getBoundingBox();
    d_box_local = d_meshes[0]->getLocalBoundingBox();
    AMP_ASSERT( d_box.size() == 2 * PhysicalDim );
    AMP_ASSERT( d_box_local.size() == 2 * PhysicalDim );
    for ( size_t i = 1; i < d_meshes.size(); i++ ) {
        auto meshBox  = d_meshes[i]->getBoundingBox();
        auto localBox = d_meshes[i]->getLocalBoundingBox();
        AMP_ASSERT( meshBox.size() == 2 * PhysicalDim );
        AMP_ASSERT( localBox.size() == 2 * PhysicalDim );
        for ( int j = 0; j < PhysicalDim; j++ ) {
            d_box[2 * j + 0]       = std::min( d_box[2 * j + 0], meshBox[2 * j + 0] );
            d_box[2 * j + 1]       = std::max( d_box[2 * j + 1], meshBox[2 * j + 1] );
            d_box_local[2 * j + 0] = std::min( d_box_local[2 * j + 0], localBox[2 * j + 0] );
            d_box_local[2 * j + 1] = std::max( d_box_local[2 * j + 1], localBox[2 * j + 1] );
        }
    }
    d_box = Mesh::reduceBox( d_box, d_comm );
    // Construct the geometry object for the multimesh
    std::vector<std::shared_ptr<AMP::Geometry::Geometry>> geom;
    for ( auto &mesh : d_meshes ) {
        auto tmp = mesh->getGeometry();
        if ( tmp )
            geom.push_back( tmp );
    }
    if ( !geom.empty() )
        d_geometry.reset( new AMP::Geometry::MultiGeometry( geom ) );
    // Cache key data
    int value = 2;
    for ( auto &mesh : d_meshes )
        value = std::min( value, static_cast<int>( mesh->isMeshMovable() ) );
    value       = d_comm.minReduce( value );
    d_isMovable = static_cast<Mesh::Movable>( value );
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
        auto weight   = db->getScalar<double>( "Weight" );
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
    for ( auto &param : params ) {
        size_t localMaxSize = AMP::Mesh::Mesh::maxProcs( param );
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
    AMP_ASSERT( database );
    if ( !database->keyExists( "MeshDatabasePrefix" ) ||
         !database->keyExists( "MeshArrayDatabasePrefix" ) ) {
        std::string msg = "Missing field MeshDatabasePrefix/MeshArrayDatabasePrefix in database:\n";
        msg += database->print( "   " );
        AMP_ERROR( msg );
    }
    auto MeshPrefix      = database->getString( "MeshDatabasePrefix" );
    auto MeshArrayPrefix = database->getString( "MeshArrayDatabasePrefix" );
    AMP_ASSERT( !check_prefix( MeshPrefix, MeshArrayPrefix ) );
    AMP_ASSERT( !check_prefix( MeshArrayPrefix, MeshPrefix ) );
    auto keys = database->getAllKeys();
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
    for ( auto &mesh : meshes ) {
        // We are dealing with a single mesh object, use the existing database
        auto database2 = database->getDatabase( mesh )->cloneDatabase();
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
            if ( key == "N" || key == "iterator" || key == "indicies" ) {
                // These keys are used by the mesh-array and should not be copied
            } else if ( key == "Size" || key == "Range" ) {
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
    for ( auto &mesh : d_meshes )
        N += mesh->numLocalElements( type );
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
    for ( auto &mesh : d_meshes )
        N += mesh->numGhostElements( type, gcw );
    return N;
}
std::vector<std::shared_ptr<Mesh>> MultiMesh::getMeshes() { return d_meshes; }
std::vector<std::shared_ptr<const Mesh>> MultiMesh::getMeshes() const
{
    std::vector<std::shared_ptr<const Mesh>> list( d_meshes.size() );
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
    for ( auto &mesh : d_meshes ) {
        std::vector<int> mesh_idSet = mesh->getBoundaryIDs();
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
    for ( auto &id : global_id_list )
        ids_set.insert( id );
    // Return the final vector of ids
    return std::vector<int>( ids_set.begin(), ids_set.end() );
}
MeshIterator
MultiMesh::getBoundaryIDIterator( const GeomType type, const int id, const int gcw ) const
{
    std::vector<MeshIterator> iterators;
    iterators.reserve( d_meshes.size() );
    for ( auto &mesh : d_meshes ) {
        MeshIterator it = mesh->getBoundaryIDIterator( type, id, gcw );
        if ( it.size() > 0 )
            iterators.push_back( it );
    }
    return MultiIterator( iterators );
}
std::vector<int> MultiMesh::getBlockIDs() const
{
    std::set<int> ids;
    for ( auto &mesh : d_meshes ) {
        auto ids2 = mesh->getBlockIDs();
        ids.insert( ids2.begin(), ids2.end() );
    }
    d_comm.setGather( ids );
    return std::vector<int>( ids.begin(), ids.end() );
}
MeshIterator MultiMesh::getBlockIDIterator( const GeomType type, const int id, const int gcw ) const
{
    std::vector<MeshIterator> iterators;
    iterators.reserve( d_meshes.size() );
    for ( auto &mesh : d_meshes ) {
        auto it = mesh->getBlockIDIterator( type, id, gcw );
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
    for ( auto &mesh : d_meshes ) {
        for ( auto &id : mesh->getLocalMeshIDs() )
            ids.insert( id );
    }
    return std::vector<MeshID>( ids.begin(), ids.end() );
}
std::vector<MeshID> MultiMesh::getLocalBaseMeshIDs() const
{
    std::set<MeshID> ids;
    for ( auto &mesh : d_meshes ) {
        for ( auto &id : mesh->getLocalBaseMeshIDs() )
            ids.insert( id );
    }
    return std::vector<MeshID>( ids.begin(), ids.end() );
}


/********************************************************
 * Check if the element is a member of the mesh          *
 ********************************************************/
bool MultiMesh::isMember( const MeshElementID &id ) const
{
    for ( auto &mesh : d_meshes ) {
        if ( mesh->isMember( id ) )
            return true;
    }
    return false;
}
MeshIterator MultiMesh::isMember( const MeshIterator &iterator ) const
{
    std::vector<MeshIterator> iterators;
    iterators.reserve( d_meshes.size() );
    for ( auto &mesh : d_meshes ) {
        auto it = mesh->isMember( iterator );
        if ( it.size() > 0 )
            iterators.push_back( it );
    }
    return MultiIterator( iterators );
}


/********************************************************
 * Function to return the element given an ID            *
 ********************************************************/
std::unique_ptr<MeshElement> MultiMesh::getElement( const MeshElementID &elem_id ) const
{
    MeshID mesh_id = elem_id.meshID();
    for ( auto &mesh : d_meshes ) {
        auto ids        = mesh->getLocalBaseMeshIDs();
        bool mesh_found = false;
        for ( auto &id : ids ) {
            if ( id == mesh_id )
                mesh_found = true;
        }
        if ( mesh_found )
            return mesh->getElement( elem_id );
    }
    AMP_ERROR( "A mesh matching the element's mesh id was not found" );
    return std::make_unique<MeshElement>();
}


/********************************************************
 * Function to return parents of an element              *
 ********************************************************/
std::vector<std::unique_ptr<MeshElement>> MultiMesh::getElementParents( const MeshElement &elem,
                                                                        const GeomType type ) const
{
    MeshID mesh_id = elem.globalID().meshID();
    for ( auto &mesh : d_meshes ) {
        auto ids        = mesh->getLocalBaseMeshIDs();
        bool mesh_found = false;
        for ( auto &id : ids ) {
            if ( id == mesh_id )
                mesh_found = true;
        }
        if ( mesh_found )
            return mesh->getElementParents( elem, type );
    }
    AMP_ERROR( "A mesh matching the element's mesh id was not found" );
    return {};
}


/********************************************************
 * Function to return the mesh with the given ID         *
 ********************************************************/
std::shared_ptr<Mesh> MultiMesh::Subset( MeshID meshID ) const
{
    if ( d_meshID == meshID )
        return std::const_pointer_cast<Mesh>( shared_from_this() );
    for ( auto &mesh : d_meshes ) {
        auto mesh2 = mesh->Subset( meshID );
        if ( mesh2 )
            return mesh2;
    }
    return nullptr;
}


/********************************************************
 * Function to subset a mesh using a mesh iterator       *
 ********************************************************/
std::shared_ptr<Mesh> MultiMesh::Subset( const MeshIterator &iterator_in, bool isGlobal ) const
{
    if ( !isGlobal && iterator_in.size() == 0 )
        return std::shared_ptr<Mesh>();
    // Check the iterator
    auto type = AMP::Mesh::GeomType::Nullity;
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
    std::vector<std::shared_ptr<Mesh>> subset;
    std::set<MeshID> subsetID;
    for ( auto &mesh : d_meshes ) {
        MeshIterator iterator;
        if ( iterator_in.size() > 0 ) {
            iterator = Mesh::getIterator( SetOP::Intersection,
                                          iterator_in,
                                          mesh->getIterator( type, mesh->getMaxGhostWidth() ) );
        }
        auto mesh2 = mesh->Subset( iterator, isGlobal );
        if ( mesh2 ) {
            subset.push_back( mesh2 );
            subsetID.insert( mesh2->meshID() );
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
    std::vector<std::shared_ptr<Mesh>> subset;
    std::set<MeshID> subsetID;
    for ( auto &mesh : d_meshes ) {
        AMP_ASSERT( mesh );
        auto mesh2 = mesh->Subset( name );
        if ( mesh2 ) {
            subset.push_back( mesh2 );
            subsetID.insert( mesh2->meshID() );
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
Mesh::Movable MultiMesh::isMeshMovable() const { return d_isMovable; }
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
void MultiMesh::displaceMesh( const AMP::LinearAlgebra::Vector::const_shared_ptr x )
{
    // Displace the individual meshes
    for ( auto &mesh : d_meshes )
        mesh->displaceMesh( x );
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
    if ( d_meshes.size() != mesh->d_meshes.size() )
        return false;
    bool equal = true;
    for ( size_t i = 0; i < d_meshes.size(); i++ )
        equal = equal && *d_meshes[i] == *mesh->d_meshes[i];
    return equal;
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
    } else if ( database1->isType<std::string>( key ) ) {
        // Copy a std::string (checking for the iterator)
        const auto data = database1->getVector<std::string>( key );
        AMP_ASSERT( !data.empty() );
        for ( size_t i = 0; i < database2.size(); i++ ) {
            if ( data.size() == 1 ) {
                auto data2 = AMP::Utilities::strrep( data[0], iterator, index[i] );
                database2[i]->putScalar( key, data2 );
            } else if ( data.size() == database2.size() && select ) {
                auto data2 = AMP::Utilities::strrep( data[i], iterator, index[i] );
                database2[i]->putScalar( key, data2 );
            } else {
                auto data2 = data;
                for ( auto &str : data2 )
                    str = AMP::Utilities::strrep( str, iterator, index[i] );
                database2[i]->putVector( key, data2 );
            }
        }
    } else if ( !select ) {
        for ( auto &db : database2 )
            db->putData( key, database1->getData( key )->clone() );
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
        for ( auto &group : groups[i] ) {
            if ( group == myRank )
                color = 0;
        }
        comms[i] = comm.split( color, comm.getRank() );
        if ( color != -1 )
            AMP_ASSERT( comms[i].getSize() == (int) groups[i].size() );
    }
    return comms;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MultiMesh::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    Mesh::registerChildObjects( manager );
    for ( auto mesh : d_meshes )
        manager->registerObject( mesh );
}
void MultiMesh::writeRestart( int64_t fid ) const
{
    Mesh::writeRestart( fid );
    std::vector<MeshID> meshIDs;
    for ( auto &mesh : d_meshes )
        meshIDs.push_back( mesh->meshID() );
    IO::writeHDF5( fid, "meshIDs", meshIDs );
}
MultiMesh::MultiMesh( int64_t fid, AMP::IO::RestartManager *manager ) : Mesh( fid, manager )
{
    std::vector<MeshID> meshIDs;
    IO::readHDF5( fid, "meshIDs", meshIDs );
    d_meshes.resize( meshIDs.size() );
    for ( size_t i = 0; i < meshIDs.size(); i++ )
        d_meshes[i] = manager->getData<Mesh>( meshIDs[i].getHash() );
}


} // namespace AMP::Mesh
