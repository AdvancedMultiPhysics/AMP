#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/MeshListIterator.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Discretization {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
multiDOFManager::multiDOFManager( const AMP_MPI &globalComm,
                                  std::vector<std::shared_ptr<DOFManager>> managers,
                                  std::shared_ptr<const AMP::Mesh::Mesh> mesh )
    : DOFManager()
{
    d_comm = globalComm;
    AMP_ASSERT( !d_comm.isNull() );
    reset( std::move( managers ), mesh );
}
multiDOFManager::multiDOFManager( std::shared_ptr<DOFManager> dof ) : DOFManager()
{
    AMP_ASSERT( dof );
    d_comm       = dof->getComm();
    d_managers   = { dof };
    d_mesh       = dof->getMesh();
    d_dofMap     = multiDOFHelper( *dof );
    d_begin      = dof->beginDOF();
    d_end        = dof->endDOF();
    d_global     = dof->numGlobalDOF();
    d_localSize  = { d_end - d_begin };
    d_globalSize = { d_global };
}

void multiDOFManager::reset( std::vector<std::shared_ptr<DOFManager>> managers,
                             std::shared_ptr<const AMP::Mesh::Mesh> mesh )
{
    d_managers = managers;
    d_mesh     = mesh;
    d_dofMap   = multiDOFHelper( managers, d_comm );
    // Compute the total begin, end, and global size
    d_begin  = d_dofMap.begin();
    d_end    = d_dofMap.end();
    d_global = d_dofMap.numGlobal();
    d_localSize.resize( managers.size(), 0 );
    d_globalSize.resize( managers.size(), 0 );
    for ( size_t i = 0; i < d_managers.size(); i++ ) {
        d_globalSize[i] = d_managers[i]->numGlobalDOF();
        d_localSize[i]  = d_managers[i]->numLocalDOF();
    }
    // Check the multimesh if provided
    if ( d_mesh ) {
        auto meshIdList = d_mesh->getLocalMeshIDs();
        std::set<AMP::Mesh::MeshID> ids( meshIdList.begin(), meshIdList.end() );
        for ( auto &dof : d_managers ) {
            auto mesh = dof->getMesh();
            if ( mesh ) {
                for ( auto &id : mesh->getLocalMeshIDs() )
                    AMP_ASSERT( ids.find( id ) != ids.end() );
            }
        }
    }
}


/****************************************************************
 * Deconstructor                                                 *
 ****************************************************************/
multiDOFManager::~multiDOFManager() = default;


/****************************************************************
 * Get the dofs for the element                                  *
 ****************************************************************/
size_t multiDOFManager::appendDOFs( const AMP::Mesh::MeshElementID &id,
                                    size_t *dofs,
                                    size_t index,
                                    size_t capacity ) const
{
    if ( d_managers.empty() )
        return 0;
    if ( d_managers[0]->numGlobalDOF() == this->d_global ) {
        // We are dealing with a multiDOFManager with only 1 sub DOF
        //   this happens with multivectors
        return d_managers[0]->appendDOFs( id, dofs, index, capacity );
    } else {
        size_t N = 0;
        for ( size_t i = 0; i < d_managers.size(); i++ ) {
            size_t N2 = d_managers[i]->appendDOFs( id, dofs, index, capacity );
            for ( size_t j = index; j < std::min( index + N2, capacity ); j++ )
                dofs[j] = subToGlobal( i, dofs[j] );
            N += N2;
            index += N2;
        }
        return N;
    }
}


/****************************************************************
 * Return the number of DOFs per element                         *
 ****************************************************************/
int multiDOFManager::getDOFsPerPoint() const
{
    if ( d_managers.empty() )
        return -1;
    int N = d_managers[0]->getDOFsPerPoint();
    for ( size_t i = 1; i < d_managers.size(); i++ ) {
        if ( d_managers[i]->getDOFsPerPoint() != N )
            return -1;
    }
    return N;
}


/****************************************************************
 * Get the element ID give a dof                                 *
 ****************************************************************/
AMP::Mesh::MeshElementID multiDOFManager::getElementID( size_t dof ) const
{
    auto map = globalToSub( dof );
    AMP_ASSERT( map.second >= 0 );
    return d_managers[map.second]->getElementID( map.first );
}
std::unique_ptr<AMP::Mesh::MeshElement> multiDOFManager::getElement( size_t dof ) const
{
    auto map = globalToSub( dof );
    AMP_ASSERT( map.second >= 0 );
    return d_managers[map.second]->getElement( map.first );
}


/****************************************************************
 * Get the mesh                                                  *
 ****************************************************************/
std::shared_ptr<const AMP::Mesh::Mesh> multiDOFManager::getMesh() const { return d_mesh; }


/****************************************************************
 * Get an entry over the mesh elements associated with the DOFs  *
 * Note: if any sub-DOFManagers are the same, then this will     *
 * iterate over repeated elements.                               *
 ****************************************************************/
AMP::Mesh::MeshIterator multiDOFManager::getIterator() const
{
    if ( d_managers.size() == 1 ) {
        return d_managers[0]->getIterator();
    }
    // Get the iterators for all sub DOFmanagers
    std::vector<std::shared_ptr<AMP::Mesh::MeshIterator>> iterators( d_managers.size() );
    for ( size_t i = 0; i < d_managers.size(); i++ )
        iterators[i] = std::make_shared<AMP::Mesh::MeshIterator>( d_managers[i]->getIterator() );
    // Get the list of unique elements
    size_t N_tot = 0;
    for ( auto &iterator : iterators )
        N_tot += iterator->size();
    auto elements = std::make_shared<std::vector<std::unique_ptr<AMP::Mesh::MeshElement>>>( 0 );
    elements->reserve( N_tot );
    for ( auto &iterator : iterators ) {
        for ( const auto &elem : *iterator )
            elements->push_back( elem.clone() );
    }
    AMP::Utilities::unique( *elements );
    // Create an iterator over the elements
    return AMP::Mesh::MeshListIterator( elements );
}


/****************************************************************
 * Return the remote DOFs for a vector                           *
 ****************************************************************/
std::vector<size_t> multiDOFManager::getRemoteDOFs() const
{
    std::vector<size_t> global_dofs;
    for ( size_t i = 0; i < d_managers.size(); i++ ) {
        auto local_dofs = d_managers[i]->getRemoteDOFs();
        global_dofs.reserve( global_dofs.size() + local_dofs.size() );
        for ( size_t j = 0; j < local_dofs.size(); j++ )
            global_dofs.push_back( subToGlobal( i, local_dofs[j] ) );
    }
    AMP::Utilities::quicksort( global_dofs );
    return global_dofs;
}


/****************************************************************
 * Return the global number of D.O.F.s                           *
 ****************************************************************/
size_t multiDOFManager::getRowDOFs( const AMP::Mesh::MeshElementID &id,
                                    size_t *dofs,
                                    size_t N_alloc,
                                    bool sort ) const
{
    size_t N = 0;
    for ( size_t i = 0; i < d_managers.size(); i++ ) {
        size_t N_alloc2 = N >= N_alloc ? 0 : N_alloc - N;
        size_t N_local  = d_managers[i]->getRowDOFs( id, &dofs[N], N_alloc2, false );
        size_t N2       = std::min( N_local, N_alloc2 );
        for ( size_t j = 0; j < N2; j++ )
            dofs[N + j] = subToGlobal( i, dofs[N + j] );
        N += N_local;
    }
    if ( sort )
        AMP::Utilities::quicksort( std::min( N, N_alloc ), dofs );
    return N;
}


/****************************************************************
 * Subset the DOF manager                                        *
 ****************************************************************/
std::shared_ptr<DOFManager> multiDOFManager::subset( const AMP_MPI &comm_in )
{
    // Check if we are dealing with a compatible comm
    if ( comm_in.compare( d_comm ) != 0 )
        return shared_from_this();
    // Get the comm for the new DOFManager
    AMP_MPI comm = AMP_MPI::intersect( comm_in, d_comm );
    // Subset all of the DOFManagers within this DOFManager
    std::vector<std::shared_ptr<DOFManager>> sub_managers;
    for ( auto &elem : d_managers ) {
        auto subset = elem->subset( comm );
        if ( subset != nullptr )
            sub_managers.push_back( subset );
    }
    // Check that we have a valid DOF manager somewhere
    bool valid_DOF = !sub_managers.empty();
    valid_DOF      = comm.anyReduce( valid_DOF );
    if ( !valid_DOF )
        return std::shared_ptr<DOFManager>();
    // Create the new multiDOFManager
    return std::make_shared<multiDOFManager>( comm, sub_managers );
}
std::shared_ptr<DOFManager>
multiDOFManager::subset( const std::shared_ptr<const AMP::Mesh::Mesh> mesh, bool useMeshComm )
{
    // Get the comm for the new DOFManager
    AMP_MPI comm( AMP_COMM_NULL );
    if ( useMeshComm ) {
        if ( mesh )
            comm = mesh->getComm();
    } else {
        comm = d_comm;
    }
    if ( comm.isNull() )
        return std::shared_ptr<DOFManager>();
    // Subset all of the DOFManagers within this DOFManager
    bool changed = false;
    std::vector<std::shared_ptr<DOFManager>> sub_managers;
    for ( auto &elem : d_managers ) {
        auto subset = elem->subset( mesh, useMeshComm );
        if ( subset.get() != elem.get() )
            changed = true;
        if ( subset != nullptr )
            sub_managers.push_back( subset );
    }
    // Check if the DOF manager changed and the comms are compatible
    if ( comm.compare( d_comm ) != 0 ) {
        changed = comm.anyReduce( changed );
        if ( !changed )
            return shared_from_this();
    }
    // Check that we have a valid DOF manager somewhere
    bool valid_DOF = !sub_managers.empty();
    valid_DOF      = comm.anyReduce( valid_DOF );
    if ( !valid_DOF )
        return std::shared_ptr<DOFManager>();
    // Create the new multiDOFManager
    return std::make_shared<multiDOFManager>( comm, sub_managers );
}
std::shared_ptr<DOFManager> multiDOFManager::subset( const AMP::Mesh::MeshIterator &iterator,
                                                     const AMP_MPI &comm_in )
{
    // Get the comm for the new DOFManager
    AMP_MPI comm = AMP_MPI::intersect( comm_in, d_comm );
    // Subset all of the DOFManagers within this DOFManager
    bool changed = false;
    std::vector<std::shared_ptr<DOFManager>> sub_managers;
    for ( auto &elem : d_managers ) {
        auto subset = elem->subset( iterator, comm );
        if ( subset.get() != elem.get() )
            changed = true;
        if ( subset != nullptr )
            sub_managers.push_back( subset );
    }
    // Check if the DOF manager changed and the comms are compatible
    if ( comm.compare( d_comm ) != 0 ) {
        changed = comm.anyReduce( changed );
        if ( !changed )
            return shared_from_this();
    }
    // Check that we have a valid DOF manager somewhere
    bool valid_DOF = !sub_managers.empty();
    valid_DOF      = comm.anyReduce( valid_DOF );
    if ( !valid_DOF )
        return std::shared_ptr<DOFManager>();
    // Create the new multiDOFManager
    return std::make_shared<multiDOFManager>( comm, sub_managers );
}


/****************************************************************
 * Convert between local and global ids                          *
 ****************************************************************/
std::vector<size_t> multiDOFManager::getGlobalDOF( const int manager,
                                                   const std::vector<size_t> &dofs ) const
{
    return d_dofMap.getGlobalDOF( manager, dofs );
}
std::vector<size_t> multiDOFManager::getSubDOF( const int manager,
                                                const std::vector<size_t> &dofs ) const
{
    return d_dofMap.getSubDOF( manager, dofs );
}


/****************************************************************
 * Get the local sizes on each rank                              *
 ****************************************************************/
std::vector<size_t> multiDOFManager::getLocalSizes() const { return d_dofMap.getLocalSize(); }


} // namespace AMP::Discretization


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
#include "AMP/utils/AMP_MPI.I"
template std::vector<int> AMP::AMP_MPI::bcast<std::vector<int>>( std::vector<int> const &,
                                                                 int ) const;
template std::vector<std::array<double, 3ul>>
AMP::AMP_MPI::bcast<std::vector<std::array<double, 3ul>>>(
    std::vector<std::array<double, 3ul>> const &, int ) const;
