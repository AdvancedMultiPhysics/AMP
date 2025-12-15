#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"

#include <set>
#include <vector>

#include "ProfilerApp.h"

namespace AMP::Discretization {


/****************************************************************
 * Find a value, returning the index if found or -1              *
 ****************************************************************/
template<class T>
static int64_t find( int64_t N, const T *x, const T &value )
{
    if ( N == 0 )
        return -1;
    if ( x[0] == value )
        return 0;
    int64_t lower = 0;
    int64_t upper = N - 1;
    while ( ( upper - lower ) > 1 ) {
        auto index = ( upper + lower ) / 2;
        if ( x[index] >= value )
            upper = index;
        else
            lower = index;
    }
    if ( x[upper] == value )
        return upper;
    return -1;
}
template<class T>
static inline int64_t find( const std::vector<T> &x, const T &value )
{
    return find( x.size(), x.data(), value );
}


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
std::shared_ptr<DOFManager> simpleDOFManager::create( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                                      AMP::Mesh::GeomType type,
                                                      int gcw,
                                                      int DOFsPerObject,
                                                      bool split )
{
    PROFILE( "simpleDOFManager::create" );

    if ( !mesh )
        return std::shared_ptr<DOFManager>();
    if ( split && std::dynamic_pointer_cast<const AMP::Mesh::MultiMesh>( mesh ) ) {
        // We want to split the DOFs by the mesh
        auto meshIDs = mesh->getLocalBaseMeshIDs();
        std::vector<std::shared_ptr<DOFManager>> managers;
        for ( auto &meshID : meshIDs ) {
            auto subMesh = mesh->Subset( meshID );
            if ( subMesh )
                managers.push_back( create( subMesh, type, gcw, DOFsPerObject, false ) );
        }
        auto rtn = std::make_shared<multiDOFManager>( mesh->getComm(), managers, mesh );
        return rtn;
    }
    // Check if the mesh is a BoxMesh
    if ( std::dynamic_pointer_cast<const AMP::Mesh::BoxMesh>( mesh ) ) {
        return std::make_shared<boxMeshDOFManager>( mesh, type, gcw, DOFsPerObject );
    }
    // Create a simpleDOFManager
    auto it1 = mesh->getIterator( type, 0 );
    auto it2 = mesh->getIterator( type, gcw );
    return std::make_shared<simpleDOFManager>( mesh, it1, it2, type, DOFsPerObject );
}
std::shared_ptr<DOFManager> simpleDOFManager::create( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                                      const AMP::Mesh::MeshIterator &it1,
                                                      const AMP::Mesh::MeshIterator &it2,
                                                      int DOFsPerObject )
{
    PROFILE( "simpleDOFManager::create" );

    // Check the iterators
    for ( auto &elem : it2 ) {
        auto id = elem.globalID();
        AMP_INSIST( id.is_local(), "it2 may not contain any ghost elements" );
    }
    auto type      = it1->globalID().type();
    size_t N_local = 0;
    for ( auto &elem : it1 ) {
        auto id = elem.globalID();
        AMP_INSIST( id.type() == type, "All elements in the iterator must be the same type" );
        if ( id.is_local() )
            N_local++;
    }
    AMP_INSIST( N_local == it2.size(), "it1 must contain it2" );
    // Create the simpleDOFManager
    return std::make_shared<simpleDOFManager>( mesh, it2, it1, type, DOFsPerObject );
}
std::shared_ptr<DOFManager> simpleDOFManager::create( const AMP::Mesh::MeshIterator &it,
                                                      int DOFsPerObject )
{
    PROFILE( "simpleDOFManager::create" );

    // Check the iterator
    auto type = it->globalID().type();
    for ( auto &elem : it ) {
        auto id = elem.globalID();
        AMP_INSIST( id.type() == type, "All elements in the iterator must be the same type" );
    }
    // Create the simpleDOFManager
    return std::make_shared<simpleDOFManager>( nullptr, it, it, type, DOFsPerObject );
}
simpleDOFManager::simpleDOFManager( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                    const AMP::Mesh::MeshIterator &it1,
                                    const AMP::Mesh::MeshIterator &it2,
                                    AMP::Mesh::GeomType type,
                                    int DOFsPerObject )
    : DOFManager(),
      d_type( type ),
      d_DOFsPerElement( DOFsPerObject ),
      d_mesh( mesh ),
      d_localIterator( it1 ),
      d_ghostIterator( it2 )
{
    AMP_ASSERT( d_ghostIterator.size() >= d_localIterator.size() );
    d_comm = AMP_MPI( AMP_COMM_SELF );
    if ( mesh ) {
        d_comm        = mesh->getComm();
        d_baseMeshIDs = mesh->getBaseMeshIDs();
    }
    initialize();
}


/****************************************************************
 * Destructor                                                    *
 ****************************************************************/
simpleDOFManager::~simpleDOFManager() = default;


/****************************************************************
 * Initialize the data                                           *
 ****************************************************************/
void simpleDOFManager::initialize()
{
    PROFILE( "simpleDOFManager::initialize" );

    // Get the mesh ids
    if ( d_mesh != nullptr ) {
        d_meshID     = d_mesh->meshID();
        d_isBaseMesh = d_mesh->isBaseMesh();
        // Get the list of global mesh ids (use communication on this->comm)
        d_baseMeshIDs = d_mesh->getBaseMeshIDs();
        std::set<AMP::Mesh::MeshID> set( d_baseMeshIDs.begin(), d_baseMeshIDs.end() );
        d_comm.setGather( set );
        d_baseMeshIDs = std::vector<AMP::Mesh::MeshID>( set.begin(), set.end() );
    } else {
        d_meshID     = AMP::Mesh::MeshID();
        d_isBaseMesh = false;
        d_baseMeshIDs.clear();
    }
    // Create a sorted list of the local and remote types
    d_local_id.clear();
    d_remote_id.clear();
    d_local_id.reserve( d_localIterator.size() );
    d_remote_id.reserve( d_ghostIterator.size() - d_localIterator.size() );
    for ( const auto &elem : d_ghostIterator ) {
        auto id = elem.globalID();
        if ( id.is_local() )
            d_local_id.push_back( id );
        else
            d_remote_id.push_back( id );
    }
    AMP_ASSERT( d_local_id.size() == d_localIterator.size() );
    AMP_ASSERT( d_remote_id.size() == d_ghostIterator.size() - d_localIterator.size() );
    // Sort the elements (they will be sorted by the meshID, then the rank on the
    // comm of the given mesh, then the element type, and finally the local id)
    AMP::Utilities::quicksort( d_local_id );
    AMP::Utilities::quicksort( d_remote_id );
    // Get the number of local elements per processor and the global number of DOFs
    size_t N_local = d_local_id.size() * d_DOFsPerElement;
    d_comm.sumScan<size_t>( &N_local, &d_end, 1 );
    d_begin  = d_end - N_local;
    d_global = d_comm.bcast( d_end, d_comm.getSize() - 1 );
    // Determine the remote DOFs (assuming 1 DOF per node)
    // Note: this must be done after d_local_id is set, d_begin and d_global are set, and remote_ids
    // must be sorted.
    d_remote_dof = getRemoteDOF( d_remote_id );
    AMP_ASSERT( d_remote_dof.size() == d_remote_id.size() );
}


/****************************************************************
 * Subset the DOF manager                                        *
 ****************************************************************/
static bool containsMesh( std::shared_ptr<const AMP::Mesh::Mesh> mesh, AMP::Mesh::MeshID id )
{
    if ( mesh->meshID() == id )
        return true;
    auto multimesh = std::dynamic_pointer_cast<const AMP::Mesh::MultiMesh>( mesh );
    if ( multimesh ) {
        auto list = multimesh->getMeshes();
        for ( auto &mesh2 : list ) {
            auto mesh3 = containsMesh( mesh2, id );
            if ( mesh3 )
                return true;
        }
    }
    return false;
}
std::shared_ptr<DOFManager>
simpleDOFManager::subset( const std::shared_ptr<const AMP::Mesh::Mesh> mesh, bool useMeshComm )
{
    // Check if we are dealing with a single mesh for both the internal and desired mesh
    if ( mesh->meshID() == d_meshID ) {
        // The mesh IDs match
        return shared_from_this();
    } else if ( mesh->isBaseMesh() && d_isBaseMesh ) {
        // Both meshes are base meshes and the ids do not match
        return std::shared_ptr<DOFManager>();
    } else if ( d_baseMeshIDs.size() == 1 && mesh->isBaseMesh() ) {
        // The subsetting mesh is a base mesh and we only contain one mesh
        if ( d_baseMeshIDs[0] == mesh->meshID() )
            return shared_from_this();
        else
            return std::shared_ptr<DOFManager>();
    }
    // Check if the desired mesh is a multimesh that contains the current mesh
    if ( std::dynamic_pointer_cast<const AMP::Mesh::MultiMesh>( mesh ) ) {
        bool found_local = containsMesh( mesh, d_meshID );
        AMP_MPI comm( AMP_COMM_NULL );
        if ( useMeshComm ) {
            comm = AMP_MPI::intersect( d_comm, mesh->getComm() );
        } else {
            comm = d_comm;
        }
        found_local = comm.allReduce( found_local );
        if ( found_local )
            return shared_from_this();
    }
    // We were not able to use an efficient subset, use the generic base function
    return DOFManager::subset( mesh, useMeshComm );
}


/****************************************************************
 * Get the DOFs for the element                                  *
 ****************************************************************/
size_t simpleDOFManager::appendDOFs( const AMP::Mesh::MeshElementID &id,
                                     size_t *dofs,
                                     size_t N0,
                                     size_t capacity ) const
{
    if ( id.is_local() ) {
        // Search for the dof locally
        auto index = find( d_local_id, id );
        if ( index != -1 ) {
            size_t dof = index * d_DOFsPerElement + d_begin;
            for ( size_t j = 0, k = N0; j < d_DOFsPerElement && k < capacity; j++, k++, dof++ )
                dofs[k] = dof;
            return d_DOFsPerElement;
        }
    } else {
        // Search for the dof in the remote list
        auto index = find( d_remote_id, id );
        if ( index != -1 ) {
            // The id was found
            size_t dof = d_remote_dof[index] * d_DOFsPerElement;
            for ( size_t j = 0, k = N0; j < d_DOFsPerElement && k < capacity; j++, k++, dof++ )
                dofs[k] = dof;
            return d_DOFsPerElement;
        }
    }
    return 0;
}


/****************************************************************
 * Get the element ID give a dof                                 *
 ****************************************************************/
AMP::Mesh::MeshElementID simpleDOFManager::getElementID( size_t dof ) const
{
    if ( dof >= d_begin && dof < d_end ) {
        // We are searching for a local dof
        return d_local_id[( dof - d_begin ) / d_DOFsPerElement];
    }
    const size_t dof2 = dof / d_DOFsPerElement;
    for ( size_t i = 0; i < d_remote_id.size(); i++ ) {
        if ( d_remote_dof[i] == dof2 )
            return d_remote_id[i];
    }
    return AMP::Mesh::MeshElementID();
}
std::unique_ptr<AMP::Mesh::MeshElement> simpleDOFManager::getElement( size_t dof ) const
{
    auto id = simpleDOFManager::getElementID( dof );
    if ( id.isNull() )
        return {};
    return d_mesh->getElement( id );
}


/****************************************************************
 * Get the mesh / mesh iterator                                  *
 ****************************************************************/
std::shared_ptr<const AMP::Mesh::Mesh> simpleDOFManager::getMesh() const { return d_mesh; }
AMP::Mesh::MeshIterator simpleDOFManager::getIterator() const { return d_localIterator.begin(); }


/****************************************************************
 * Return the number of DOFs per element                         *
 ****************************************************************/
int simpleDOFManager::getDOFsPerPoint() const { return d_DOFsPerElement; }


/****************************************************************
 * Return the remote DOFs for a vector                           *
 ****************************************************************/
std::vector<size_t> simpleDOFManager::getRemoteDOFs() const
{
    // Create the list of remote DOFs
    size_t N = d_remote_id.size();
    std::vector<size_t> remote_DOFs( N * d_DOFsPerElement, (size_t) -1 );
    for ( size_t i = 0, k = 0; i < N; i++ ) {
        for ( int j = 0; j < d_DOFsPerElement; j++, k++ )
            remote_DOFs[k] = d_remote_dof[i] * d_DOFsPerElement + j;
    }
    AMP::Utilities::quicksort( remote_DOFs );
    return remote_DOFs;
}


/****************************************************************
 * Return the row DOFs                                           *
 ****************************************************************/
size_t simpleDOFManager::getRowDOFs( const AMP::Mesh::MeshElementID &id,
                                     size_t *dofs,
                                     size_t N_alloc,
                                     bool sort ) const
{
    // Check if the element is in the mesh
    bool found = false;
    for ( auto meshID : d_baseMeshIDs )
        found = found || id.meshID() == meshID;
    if ( !found )
        return 0;
    // Get a list of all element ids and corresponding DOFs that are part of the row
    size_t N      = 0;
    auto meshType = d_mesh->getGeomType();
    auto objType  = id.type();
    auto obj      = d_mesh->getElement( id );
    if ( objType == d_type && ( objType == AMP::Mesh::GeomType::Vertex || objType == meshType ) ) {
        // Use the getNeighbors function to get the neighbors of the current element
        N += appendDOFs( id, dofs, N, N_alloc );
        auto neighbors = obj->getNeighbors();
        for ( auto &elem : neighbors ) {
            if ( elem )
                N += appendDOFs( elem->globalID(), dofs, N, N_alloc );
        }
    } else if ( objType == d_type ) {
        // We need to use the mesh to get the connectivity of the elements of the same type
        auto parents = d_mesh->getElementParents( *obj, meshType );
        std::vector<AMP::Mesh::MeshElementID> ids;
        for ( auto &parent : parents ) {
            auto children = parent->getElements( objType );
            ids.reserve( ids.size() + children.size() );
            for ( auto &elem : children )
                ids.push_back( elem->globalID() );
        }
        AMP::Utilities::unique( ids );
        for ( auto &id2 : ids )
            N += appendDOFs( id2, dofs, N, N_alloc );
    } else if ( objType > d_type ) {
        // The desired element type is < the current element type, use getElements
        auto children = obj->getElements( d_type );
        for ( auto &elem : children )
            N += appendDOFs( elem->globalID(), dofs, N, N_alloc );
    } else if ( objType < d_type ) {
        // The desired element type is < the current element type, use getElementParents
        auto parents = d_mesh->getElementParents( *obj, meshType );
        std::vector<AMP::Mesh::MeshElementID> ids;
        for ( auto &parent : parents )
            N += appendDOFs( parent->globalID(), dofs, N, N_alloc );
    } else {
        AMP_ERROR( "Internal error" );
    }
    // Sort the row dofs
    if ( sort )
        AMP::Utilities::quicksort( std::min( N, N_alloc ), dofs );
    return N;
}


/****************************************************************
 * Find the remote DOF given a set of mesh element IDs           *
 * Note: for this function to work correctly, the remote ids     *
 * must be sorted, and d_local_id must be set                    *
 ****************************************************************/
std::vector<size_t>
simpleDOFManager::getRemoteDOF( const std::vector<AMP::Mesh::MeshElementID> &remote_ids ) const
{
    if ( d_comm.getSize() == 1 )
        return std::vector<size_t>(); // There are no remote DOFs
    // Get the rank that will own each MeshElement on the current communicator
    std::vector<int> owner_rank( remote_ids.size(), -1 );
    auto globalComm = AMP::AMPManager::getCommWorld();
    std::vector<int> rankMap( globalComm.getSize(), -1 );
    auto globalRanks = d_comm.globalRanks();
    for ( size_t i = 0; i < globalRanks.size(); i++ )
        rankMap[globalRanks[i]] = i;
    for ( auto meshID : d_mesh->getLocalBaseMeshIDs() ) {
        // Get the mesh with the given meshID
        auto submesh      = d_mesh->Subset( meshID );
        auto submeshComm  = submesh->getComm();
        auto submeshRanks = submeshComm.globalRanks();
        // Get the rank of the proccessor that will own each meshElement
        for ( size_t i = 0; i < remote_ids.size(); i++ ) {
            if ( remote_ids[i].meshID() == meshID ) {
                int rank      = remote_ids[i].owner_rank();
                owner_rank[i] = rankMap[submeshRanks[rank]];
                AMP_ASSERT( owner_rank[i] != -1 );
            }
        }
    }
    // Check that each element has a valid owner rank
    int commSize = d_comm.getSize();
    for ( size_t i = 0; i < remote_ids.size(); i++ )
        AMP_ASSERT( owner_rank[i] >= 0 && owner_rank[i] < commSize );
    // Resort the remote ids according the the owner rank
    auto remote_ids2 = remote_ids;
    AMP::Utilities::quicksort( owner_rank, remote_ids2 );
    // Determine the send/recv count and displacements for each processor
    std::vector<int> send_cnt( d_comm.getSize(), 0 );
    for ( int rank : owner_rank )
        send_cnt[rank]++;
    std::vector<int> send_disp, recv_disp, recv_cnt;
    size_t tot_size = d_comm.calcAllToAllDisp( send_cnt, send_disp, recv_cnt, recv_disp );
    // Perform an allToAll to send the remote ids for DOF identification
    std::vector<AMP::Mesh::MeshElementID> recv_id( tot_size + 1 );
    AMP::Mesh::MeshElementID *send_buffer = nullptr;
    if ( !remote_ids2.empty() )
        send_buffer = &remote_ids2[0];
    size_t N = d_comm.allToAll<AMP::Mesh::MeshElementID>(
        send_buffer, &send_cnt[0], &send_disp[0], &recv_id[0], &recv_cnt[0], &recv_disp[0], true );
    AMP_INSIST( N == tot_size, "Unexpected receive size" );
    recv_id.resize( tot_size );
    // Determine the DOF for each received id
    std::vector<size_t> received_DOF( tot_size );
    for ( size_t i = 0; i < tot_size; i++ ) {
        int j = AMP::Utilities::findfirst( d_local_id, recv_id[i] );
        AMP_ASSERT( d_local_id[j] == recv_id[i] );
        received_DOF[i] = d_begin / d_DOFsPerElement + j;
    }
    // Send the DOFs back to the original processor
    std::vector<size_t> remote_dof;
    remote_dof.resize( remote_ids2.size() + 1, static_cast<size_t>( -1 ) );
    size_t *send_buffer_DOFs = nullptr;
    if ( tot_size > 0 )
        send_buffer_DOFs = &received_DOF[0];
    N = d_comm.allToAll<size_t>( send_buffer_DOFs,
                                 &recv_cnt[0],
                                 &recv_disp[0],
                                 &remote_dof[0],
                                 &send_cnt[0],
                                 &send_disp[0],
                                 true );
    AMP_INSIST( N == remote_ids2.size(), "Unexpected receive size" );
    remote_dof.resize( remote_ids2.size() );
    // Sort the dofs back to the original order for the remote_ids
    AMP::Utilities::quicksort( remote_ids2, remote_dof );
    for ( size_t i = 0; i < remote_ids.size(); i++ )
        AMP_ASSERT( remote_ids[i] == remote_ids2[i] );
    return remote_dof;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void simpleDOFManager::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    DOFManager::registerChildObjects( manager );
    manager->registerObject( d_mesh );
    manager->registerObject( d_localIterator.shared_from_this() );
    manager->registerObject( d_ghostIterator.shared_from_this() );
}
void simpleDOFManager::writeRestart( int64_t fid ) const
{
    DOFManager::writeRestart( fid );
    IO::writeHDF5( fid, "isBaseMesh", d_isBaseMesh );
    IO::writeHDF5( fid, "geomType", d_type );
    IO::writeHDF5( fid, "DOFsPerElement", d_DOFsPerElement );
    IO::writeHDF5( fid, "meshID", d_meshID );
    IO::writeHDF5( fid, "baseMeshIDs", d_baseMeshIDs );
    IO::writeHDF5( fid, "localIterator", d_localIterator.getID() );
    IO::writeHDF5( fid, "ghostIterator", d_ghostIterator.getID() );
    IO::writeHDF5( fid, "local_id", d_local_id );
    IO::writeHDF5( fid, "remote_id", d_remote_id );
    IO::writeHDF5( fid, "remote_dof", d_remote_dof );
}
simpleDOFManager::simpleDOFManager( int64_t fid, AMP::IO::RestartManager *manager )
    : DOFManager( fid, manager )
{
    uint64_t localIteratorID, ghostIteratorID;
    IO::readHDF5( fid, "isBaseMesh", d_isBaseMesh );
    IO::readHDF5( fid, "geomType", d_type );
    IO::readHDF5( fid, "DOFsPerElement", d_DOFsPerElement );
    IO::readHDF5( fid, "meshID", d_meshID );
    IO::readHDF5( fid, "baseMeshIDs", d_baseMeshIDs );
    IO::readHDF5( fid, "localIterator", localIteratorID );
    IO::readHDF5( fid, "ghostIterator", ghostIteratorID );
    IO::readHDF5( fid, "local_id", d_local_id );
    IO::readHDF5( fid, "remote_id", d_remote_id );
    IO::readHDF5( fid, "remote_dof", d_remote_dof );
    d_mesh          = manager->getData<AMP::Mesh::Mesh>( d_meshID.getHash() );
    d_localIterator = *manager->getData<AMP::Mesh::MeshIterator>( localIteratorID );
    d_ghostIterator = *manager->getData<AMP::Mesh::MeshIterator>( ghostIteratorID );
}


} // namespace AMP::Discretization
