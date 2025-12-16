#include "AMP/discretization/subsetDOFManager.h"

#include "AMP/mesh/MultiIterator.h"
#include "AMP/utils/Utilities.h"
#include "ProfilerApp.h"


namespace AMP::Discretization {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
std::shared_ptr<DOFManager>
subsetDOFManager::create( std::shared_ptr<const DOFManager> parentDOFManager,
                          const std::vector<size_t> &dofs,
                          const AMP::Mesh::MeshIterator &iterator,
                          const AMP_MPI &comm_in )
{
    // Limit the new comm to be <= the parent comm
    if ( parentDOFManager.get() == nullptr || comm_in.isNull() )
        return std::shared_ptr<DOFManager>();
    PROFILE( "subsetDOFManager", 2 );
    AMP_MPI comm = AMP_MPI::intersect( parentDOFManager->getComm(), comm_in );
    // Set the basic info
    std::shared_ptr<subsetDOFManager> subsetDOF( new subsetDOFManager() );
    subsetDOF->d_comm             = comm;
    subsetDOF->d_iterator         = iterator;
    subsetDOF->d_parentDOFManager = parentDOFManager;
    // Get the parent DOFs
    subsetDOF->d_parentBegin  = parentDOFManager->beginDOF();
    subsetDOF->d_parentEnd    = parentDOFManager->endDOF();
    subsetDOF->d_parentGlobal = parentDOFManager->numGlobalDOF();
    // Copy the local list of DOFs
    subsetDOF->d_localDOFs.reserve( dofs.size() );
    size_t begin_dof = parentDOFManager->beginDOF();
    size_t end_dof   = parentDOFManager->endDOF();
    for ( auto &dof : dofs ) {
        if ( dof >= begin_dof && dof < end_dof )
            subsetDOF->d_localDOFs.push_back( dof );
    }
    AMP::Utilities::unique( subsetDOF->d_localDOFs );
    // Get the begin and global DOFs for the subset
    size_t N_local      = dofs.size();
    subsetDOF->d_end    = subsetDOF->d_comm.sumScan( N_local );
    subsetDOF->d_begin  = subsetDOF->d_end - N_local;
    subsetDOF->d_global = subsetDOF->d_comm.sumReduce( N_local );
    // Return if the subset DOF is empty
    if ( subsetDOF->d_global == 0 )
        return std::shared_ptr<DOFManager>();
    // Return if the subset DOF == parent DOF
    if ( subsetDOF->d_global == parentDOFManager->numGlobalDOF() )
        return std::const_pointer_cast<DOFManager>( parentDOFManager );
    // Determine which remote DOFs we will need to keep
    size_t *send_data = nullptr;
    if ( N_local > 0 )
        send_data = &( subsetDOF->d_localDOFs[0] );
    std::vector<int> N_remote( subsetDOF->d_comm.getSize(), 0 );
    std::vector<int> N_disp( subsetDOF->d_comm.getSize(), 0 );
    std::vector<size_t> recv_data( subsetDOF->d_global );
    subsetDOF->d_comm.allGather( (int) N_local, &N_remote[0] );
    N_disp[0] = 0;
    for ( int i = 1; i < subsetDOF->d_comm.getSize(); i++ )
        N_disp[i] = N_disp[i - 1] + N_remote[i - 1];
    subsetDOF->d_comm.allGather(
        send_data, (int) N_local, &recv_data[0], &N_remote[0], &N_disp[0], true );
    AMP::Utilities::quicksort( recv_data );
    std::vector<size_t> remoteDOFs = subsetDOF->d_parentDOFManager->getRemoteDOFs();
    subsetDOF->d_remoteParentDOFs  = std::vector<size_t>();
    subsetDOF->d_remoteSubsetDOFs  = std::vector<size_t>();
    subsetDOF->d_remoteParentDOFs.reserve( remoteDOFs.size() );
    subsetDOF->d_remoteSubsetDOFs.reserve( remoteDOFs.size() );
    for ( auto &remoteDOF : remoteDOFs ) {
        size_t index = AMP::Utilities::findfirst( recv_data, remoteDOF );
        if ( index == recv_data.size() ) {
            index--;
        }
        if ( recv_data[index] == remoteDOF ) {
            subsetDOF->d_remoteParentDOFs.push_back( remoteDOF );
            subsetDOF->d_remoteSubsetDOFs.push_back( index );
        }
    }
    if ( subsetDOF->numGlobalDOF() == 0 )
        return std::shared_ptr<DOFManager>();
    return subsetDOF;
}


/****************************************************************
 * Deconstructor                                                 *
 ****************************************************************/
subsetDOFManager::~subsetDOFManager() = default;


/****************************************************************
 * Return the number of DOFs per element                         *
 ****************************************************************/
int subsetDOFManager::getDOFsPerPoint() const
{
    if ( d_iterator.empty() )
        return d_parentDOFManager->getDOFsPerPoint();
    int N = 0;
    size_t dofs[16];
    for ( auto &elem : d_iterator ) {
        int N2 = appendDOFs( elem.globalID(), dofs, 0, 16 );
        if ( N == 0 )
            N = N2;
        if ( N2 != N )
            return -1;
    }
    if ( N == 0 )
        return -1;
    return N;
}


/****************************************************************
 * Get the dofs for the given element                            *
 ****************************************************************/
size_t subsetDOFManager::appendDOFs( const AMP::Mesh::MeshElementID &id,
                                     size_t *dofs,
                                     size_t index,
                                     size_t capacity ) const
{
    // Get the parent DOFs
    std::vector<size_t> parentDOFs;
    d_parentDOFManager->getDOFs( id, parentDOFs );
    if ( parentDOFs.empty() )
        return 0;
    // Get the subset DOFs
    auto subsetDOFs = getSubsetDOF( parentDOFs );
    // Remove any DOFs == -1
    size_t N = 0;
    for ( auto dof : subsetDOFs ) {
        if ( dof < d_global ) {
            N++;
            if ( index < capacity )
                dofs[index++] = dof;
        }
    }
    return N;
}


/****************************************************************
 * Get the element ID give a dof                                 *
 ****************************************************************/
AMP::Mesh::MeshElementID subsetDOFManager::getElementID( size_t dof ) const
{
    auto dof2 = getParentDOF( { dof } );
    return d_parentDOFManager->getElementID( dof2[0] );
}
std::unique_ptr<AMP::Mesh::MeshElement> subsetDOFManager::getElement( size_t dof ) const
{
    auto dof2 = getParentDOF( { dof } );
    return d_parentDOFManager->getElement( dof2[0] );
}


/****************************************************************
 * Get the mesh / mesh iterator                                  *
 ****************************************************************/
std::shared_ptr<const AMP::Mesh::Mesh> subsetDOFManager::getMesh() const
{
    return d_parentDOFManager->getMesh();
}
AMP::Mesh::MeshIterator subsetDOFManager::getIterator() const { return d_iterator; }


/****************************************************************
 * Return the remote DOFs for a vector                           *
 ****************************************************************/
std::vector<size_t> subsetDOFManager::getRemoteDOFs() const { return d_remoteSubsetDOFs; }


/****************************************************************
 * Return the global number of D.O.F.s                           *
 ****************************************************************/
size_t subsetDOFManager::getRowDOFs( const AMP::Mesh::MeshElementID &id,
                                     size_t *dofs,
                                     size_t N_alloc,
                                     bool sort ) const
{
    auto parentDOFs = d_parentDOFManager->getRowDOFs( id );
    auto subsetDOFs = getSubsetDOF( parentDOFs );
    size_t index    = 0;
    for ( size_t i = 0; i < subsetDOFs.size(); i++ ) {
        if ( subsetDOFs[i] < d_global ) {
            subsetDOFs[index] = subsetDOFs[i];
            index++;
        }
    }
    subsetDOFs.resize( index );
    for ( size_t i = 0; i < std::min( index, N_alloc ); i++ )
        dofs[i] = subsetDOFs[i];
    // Sort the row dofs
    if ( sort )
        AMP::Utilities::quicksort( std::min( index, N_alloc ), dofs );
    return index;
}


/****************************************************************
 * Function to convert DOFs                                      *
 ****************************************************************/
std::vector<size_t> subsetDOFManager::getParentDOF( const std::vector<size_t> &subsetDOFs ) const
{
    std::vector<size_t> parentDOFs( subsetDOFs.size() );
    for ( size_t i = 0; i < subsetDOFs.size(); i++ ) {
        size_t DOF = subsetDOFs[i];
        AMP_ASSERT( DOF < d_global );
        if ( DOF >= d_begin && DOF < d_end ) {
            // The DOF is local
            parentDOFs[i] = d_localDOFs[DOF - d_begin];
        } else {
            // The DOF is a remote DOF
            size_t index = AMP::Utilities::findfirst( d_remoteSubsetDOFs, DOF );
            AMP_ASSERT( d_remoteSubsetDOFs[index] == DOF );
            parentDOFs[i] = d_remoteParentDOFs[index];
        }
    }
    return parentDOFs;
}
std::vector<size_t> subsetDOFManager::getSubsetDOF( const std::vector<size_t> &parentDOFs ) const
{
    std::vector<size_t> subsetDOFs( parentDOFs.size(), (size_t) -1 );
    for ( size_t i = 0; i < parentDOFs.size(); i++ ) {
        size_t DOF = parentDOFs[i];
        AMP_ASSERT( DOF < d_parentGlobal );
        if ( DOF >= d_parentBegin && DOF < d_parentEnd ) {
            // The DOF is local
            size_t index = AMP::Utilities::findfirst( d_localDOFs, DOF );
            if ( index == d_localDOFs.size() ) {
                index--;
            }
            if ( d_localDOFs[index] == DOF )
                subsetDOFs[i] = index + d_begin;
        } else if ( !d_remoteParentDOFs.empty() ) {
            // The DOF is a remote DOF
            size_t index = AMP::Utilities::findfirst( d_remoteParentDOFs, DOF );
            if ( index == d_remoteParentDOFs.size() ) {
                index--;
            }
            if ( d_remoteParentDOFs[index] == DOF )
                subsetDOFs[i] = d_remoteSubsetDOFs[index];
        }
    }
    return subsetDOFs;
}
std::vector<size_t> subsetDOFManager::getLocalParentDOFs() const { return d_localDOFs; }


/****************************************************************
 * Function to return the DOFManagers                            *
 ****************************************************************/
std::shared_ptr<const DOFManager> subsetDOFManager::getDOFManager() const
{
    return d_parentDOFManager;
}
} // namespace AMP::Discretization
