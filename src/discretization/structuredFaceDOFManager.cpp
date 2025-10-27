#include "AMP/discretization/structuredFaceDOFManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <set>
#include <vector>


namespace AMP::Discretization {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
structuredFaceDOFManager::structuredFaceDOFManager( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                                    int DOFsPerFace[3],
                                                    int gcw )
{
    AMP_ASSERT( mesh );
    AMP_INSIST( mesh->getDim() == 3, "The mesh must be a 3D mesh" );
    AMP_INSIST( mesh->getGeomType() == AMP::Mesh::GeomType::Cell,
                "The mesh must be a volume mesh" );
    d_comm = mesh->getComm();
    d_mesh = mesh;
    d_gcw  = gcw;
    for ( int i = 0; i < 3; i++ )
        d_DOFsPerFace[i] = DOFsPerFace[i];
    initialize();
}


/****************************************************************
 * Deconstructor                                                 *
 ****************************************************************/
structuredFaceDOFManager::~structuredFaceDOFManager() = default;


/****************************************************************
 * Initialize the data                                           *
 ****************************************************************/
void structuredFaceDOFManager::initialize()
{
    // Create a sorted list of the local and remote types
    for ( int d = 0; d < 3; d++ ) {
        d_local_ids[d].resize( 0 );
        d_remote_ids[d].resize( 0 );
        if ( d_DOFsPerFace[d] == 0 )
            continue;
        auto localIterator = AMP::Mesh::StructuredMeshHelper::getFaceIterator( d_mesh, 0, d );
        auto ghostIterator = AMP::Mesh::StructuredMeshHelper::getFaceIterator( d_mesh, d_gcw, d );
        d_local_ids[d].resize( localIterator.size() );
        d_local_dofs[d].resize( localIterator.size() );
        auto it = localIterator.begin();
        for ( size_t i = 0; i < localIterator.size(); ++i, ++it ) {
            d_local_ids[d][i]  = it->globalID();
            d_local_dofs[d][i] = i;
            AMP_ASSERT( d_local_ids[d][i].is_local() );
        }
        d_remote_ids[d].reserve( ghostIterator.size() - localIterator.size() );
        it = ghostIterator.begin();
        for ( size_t i = 0; i < ghostIterator.size(); ++i, ++it ) {
            AMP::Mesh::MeshElementID id = it->globalID();
            if ( !id.is_local() )
                d_remote_ids[d].push_back( id );
        }
        AMP_ASSERT( d_local_ids[d].size() == localIterator.size() );
        AMP_ASSERT( d_remote_ids[d].size() == ghostIterator.size() - localIterator.size() );
        // Sort the ids (keeping track of the original order)
        AMP::Utilities::quicksort( d_local_ids[d], d_local_dofs[d] );
        AMP::Utilities::quicksort( d_remote_ids[d] );
    }
    // Get the number of local elements per processor and the global number of DOFs
    size_t N_local = 0;
    for ( int d = 0; d < 3; d++ )
        N_local += d_local_ids[d].size() * d_DOFsPerFace[d];
    d_comm.sumScan<size_t>( &N_local, &d_end, 1 );
    d_begin  = d_end - N_local;
    d_global = d_comm.bcast( d_end, d_comm.getSize() - 1 );
    // Correct the local dof indicies
    size_t offset = d_begin;
    for ( int d = 0; d < 3; d++ ) {
        for ( size_t i = 0; i < d_local_ids[d].size(); ++i )
            d_local_dofs[d][i] = offset + d_local_dofs[d][i] * d_DOFsPerFace[d];
        offset += d_local_ids[d].size() * d_DOFsPerFace[d];
    }
    // Determine the remote DOFs
    // Note: this must be done after d_local_id is set, d_begin and d_global are set, and remote_ids
    // must be sorted.
    for ( int d = 0; d < 3; d++ )
        d_remote_dofs[d] = getRemoteDOF( d_remote_ids[d] );
}


/****************************************************************
 * Get the DOFs for the element                                  *
 ****************************************************************/
size_t structuredFaceDOFManager::appendDOFs( const AMP::Mesh::MeshElementID &id,
                                             size_t *dofs,
                                             size_t N,
                                             size_t capacity ) const
{
    if ( id.type() != AMP::Mesh::GeomType::Face )
        return 0;
    // Search for the dof locally
    for ( int d = 0; d < 3; d++ ) {
        if ( !d_local_ids[d].empty() ) {
            size_t index = AMP::Utilities::findfirst( d_local_ids[d], id );
            if ( index == d_local_ids[d].size() ) {
                index--;
            }
            if ( id == d_local_ids[d][index] ) {
                // The id was found
                for ( size_t j = 0, k = N; j < d_DOFsPerFace[d] && k < capacity; j++, k++ )
                    dofs[k] = d_local_dofs[d][index] + j;
                return d_DOFsPerFace[d];
            }
        }
    }
    // Search for the dof in the remote list
    for ( int d = 0; d < 3; d++ ) {
        if ( !d_remote_ids[d].empty() ) {
            size_t index = AMP::Utilities::findfirst( d_remote_ids[d], id );
            if ( index == d_remote_ids[d].size() ) {
                index--;
            }
            if ( id == d_remote_ids[d][index] ) {
                // The id was found
                for ( size_t j = 0, k = N; j < d_DOFsPerFace[d] && k < capacity; j++, k++ )
                    dofs[k] = d_remote_dofs[d][index] + j;
                return d_DOFsPerFace[d];
            }
        }
    }
    return 0;
}


/****************************************************************
 * Get the element ID give a dof                                 *
 ****************************************************************/
AMP::Mesh::MeshElementID structuredFaceDOFManager::getElementID( size_t dof ) const
{
    AMP::Mesh::MeshElementID id;
    if ( dof >= d_begin && dof < d_end ) {
        // We are searching for a local dof
        for ( int i = 0; i < 3; i++ ) {
            for ( size_t j = 0; j < d_local_ids[i].size(); j++ ) {
                if ( dof >= d_local_dofs[i][j] && dof < d_local_dofs[i][j] + d_DOFsPerFace[i] )
                    id = d_local_ids[i][j];
            }
        }
    } else {
        // We are searching for a remote dof
        for ( int i = 0; i < 3; i++ ) {
            for ( size_t j = 0; j < d_remote_ids[i].size(); j++ ) {
                if ( dof >= d_remote_dofs[i][j] && dof < d_remote_dofs[i][j] + d_DOFsPerFace[i] )
                    id = d_remote_ids[i][j];
            }
        }
    }
    return id;
}
std::unique_ptr<AMP::Mesh::MeshElement> structuredFaceDOFManager::getElement( size_t dof ) const
{
    auto id = getElementID( dof );
    if ( id.isNull() )
        return {};
    return d_mesh->getElement( id );
}


/****************************************************************
 * Return the mesh                                               *
 ****************************************************************/
std::shared_ptr<const AMP::Mesh::Mesh> structuredFaceDOFManager::getMesh() const { return d_mesh; }


/****************************************************************
 * Get an entry over the mesh elements associated with the DOFs  *
 ****************************************************************/
AMP::Mesh::MeshIterator structuredFaceDOFManager::getIterator() const
{
    std::vector<AMP::Mesh::MeshIterator> faces( 3 );
    for ( int i = 0; i < 3; i++ ) {
        if ( d_DOFsPerFace[i] > 0 )
            faces[i] = AMP::Mesh::StructuredMeshHelper::getFaceIterator( d_mesh, 0, i );
    }
    return AMP::Mesh::MultiIterator( faces, 0 );
}


/****************************************************************
 * Return the remote DOFs                                        *
 ****************************************************************/
std::vector<size_t> structuredFaceDOFManager::getRemoteDOFs() const
{
    // Create the list of remote DOFs
    size_t N_remote = 0;
    for ( size_t d = 0; d < 3; d++ )
        N_remote += d_remote_ids[d].size() * d_DOFsPerFace[d];
    std::vector<size_t> remote_DOFs( N_remote, (size_t) -1 );
    int i = 0;
    for ( size_t d = 0; d < 3; d++ ) {
        for ( size_t j = 0; j < d_remote_ids[d].size(); j++ ) {
            for ( int k = 0; k < d_DOFsPerFace[d]; k++ ) {
                remote_DOFs[i] = d_remote_dofs[d][j] + k;
                ++i;
            }
        }
    }
    AMP::Utilities::quicksort( remote_DOFs );
    return remote_DOFs;
}


/****************************************************************
 * Return the row DOFs                                           *
 ****************************************************************/
size_t structuredFaceDOFManager::getRowDOFs( const AMP::Mesh::MeshElementID &id,
                                             size_t *dofs_out,
                                             size_t N_alloc,
                                             bool sort ) const
{
    if ( id.type() != AMP::Mesh::GeomType::Face )
        return 0;
    // Get a list of all element ids that are part of the row
    // Only faces that share an element are part of the row
    auto obj     = d_mesh->getElement( id );
    auto parents = d_mesh->getElementParents( *obj, AMP::Mesh::GeomType::Cell );
    AMP_ASSERT( parents.size() == 1 || parents.size() == 2 );
    // Temporarily add neighbor elements
    size_t p_size = parents.size();
    for ( size_t i = 0; i < p_size; i++ ) {
        auto neighbors = parents[i]->getNeighbors();
        for ( auto &neighbor : neighbors ) {
            if ( neighbor != nullptr )
                parents.push_back( neighbor->clone() );
        }
    }
    AMP::Utilities::unique( parents );
    // Get the face ids of interest
    std::vector<AMP::Mesh::MeshElementID> ids;
    ids.reserve( 6 * parents.size() );
    for ( auto &parent : parents ) {
        auto children = parent->getElements( AMP::Mesh::GeomType::Face );
        AMP_ASSERT( children.size() == 6 );
        for ( auto &elem : children )
            ids.push_back( elem->globalID() );
    }
    AMP::Utilities::unique( ids );
    // AMP_ASSERT(ids.size()==6||ids.size()==11);
    // Get all dofs for each element id
    uint8_t maxDOFsPerFace = 0;
    for ( auto &elem : d_DOFsPerFace )
        maxDOFsPerFace = std::max( maxDOFsPerFace, elem );
    std::vector<size_t> dofs;
    dofs.reserve( ids.size() * maxDOFsPerFace );
    std::vector<size_t> dofs2( maxDOFsPerFace );
    for ( auto &id : ids ) {
        getDOFs( id, dofs2 );
        for ( auto &elem : dofs2 )
            dofs.push_back( elem );
    }
    // Sort the row dofs
    if ( sort )
        AMP::Utilities::quicksort( dofs );
    for ( size_t i = 0; i < std::min( dofs.size(), N_alloc ); i++ )
        dofs_out[i] = dofs[i];
    return dofs.size();
}


/****************************************************************
 * Find the remote DOF given a set of mesh element IDs           *
 * Note: for this function to work correctly, the remote ids     *
 * must be sorted, and d_local_id must be set                    *
 ****************************************************************/
std::vector<size_t> structuredFaceDOFManager::getRemoteDOF(
    const std::vector<AMP::Mesh::MeshElementID> &remote_ids ) const
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
    // Check that each element has a vaild owner rank
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
    std::vector<size_t> received_DOF( tot_size ), dofs;
    for ( size_t i = 0; i < tot_size; i++ ) {
        getDOFs( recv_id[i], dofs );
        AMP_ASSERT( dofs.size() > 0 );
        for ( size_t j = 1; j < dofs.size(); j++ )
            AMP_ASSERT( dofs[j] == dofs[j - 1] + 1 );
        received_DOF[i] = dofs[0];
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
} // namespace AMP::Discretization
