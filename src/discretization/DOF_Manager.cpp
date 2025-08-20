#include "AMP/discretization/DOF_Manager.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOFManagerFactory.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/subsetCommSelfDOFManager.h"
#include "AMP/discretization/subsetDOFManager.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/utils/Utilities.h"

#include <set>
#include <utility>


namespace AMP::Discretization {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
DOFManager::DOFManager( size_t N_local, const AMP_MPI &comm, std::vector<size_t> remoteDOFs )
    : d_comm( comm )
{
    d_comm.sumScan( &N_local, &d_end, 1 );
    d_begin      = d_end - N_local;
    d_global     = d_comm.bcast( d_end, d_comm.getSize() - 1 );
    d_remoteDOFs = std::move( remoteDOFs );
}


/****************************************************************
 * Destructor                                                    *
 ****************************************************************/
DOFManager::~DOFManager() = default;


/****************************************************************
 * Default class name                                            *
 ****************************************************************/
std::string DOFManager::className() const { return "DOFManager"; }


/****************************************************************
 * Get the DOFs for the element                                  *
 ****************************************************************/
void DOFManager::getDOFs( const AMP::Mesh::MeshElementID &id, std::vector<size_t> &dofs ) const
{
    dofs.resize( 8 );
    size_t N = appendDOFs( id, dofs.data(), 0, dofs.size() );
    if ( N > dofs.size() ) {
        dofs.resize( N );
        N = appendDOFs( id, dofs.data(), 0, dofs.size() );
    }
    dofs.resize( N );
#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    for ( size_t i = 0; i < N; i++ )
        AMP_ASSERT( dofs[i] < d_global );
#endif
}
void DOFManager::getDOFs( int N_ids,
                          const AMP::Mesh::MeshElementID *ids,
                          std::vector<size_t> &dofs ) const
{
    size_t N = 0;
    dofs.resize( N_ids );
    for ( int i = 0; i < N_ids; i++ ) {
        size_t N2 = appendDOFs( ids[i], dofs.data(), N, dofs.size() );
        if ( N + N2 >= dofs.size() ) {
            dofs.resize( std::max( N + N2, 2 * dofs.size() ) );
            N2 = appendDOFs( ids[i], dofs.data(), N, dofs.size() );
        }
        N += N2;
    }
    dofs.resize( N );
#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    for ( size_t i = 0; i < N; i++ )
        AMP_ASSERT( dofs[i] < d_global );
#endif
}
void DOFManager::getDOFs( const std::vector<AMP::Mesh::MeshElementID> &ids,
                          std::vector<size_t> &dofs ) const
{
    return getDOFs( ids.size(), ids.data(), dofs );
}
size_t DOFManager::appendDOFs( const AMP::Mesh::MeshElementID &, size_t *, size_t, size_t ) const
{
    AMP_ERROR( "getDOFs is not implemented for the base class" );
}


/****************************************************************
 * Get the element ID give a dof                                 *
 ****************************************************************/
AMP::Mesh::MeshElement DOFManager::getElement( size_t ) const
{
    AMP_ERROR( "getElement is not implemented for the base class" );
    return AMP::Mesh::MeshElement();
}
AMP::Mesh::MeshElementID DOFManager::getElementID( size_t ) const
{
    AMP_ERROR( "getElement is not implemented for the base class" );
    return AMP::Mesh::MeshElementID();
}


/****************************************************************
 * Get the mesh / mesh iterator                                  *
 ****************************************************************/
std::shared_ptr<const AMP::Mesh::Mesh> DOFManager::getMesh() const { return {}; }
AMP::Mesh::MeshIterator DOFManager::getIterator() const { return {}; }


/****************************************************************
 * Return the first D.O.F. on this core                          *
 ****************************************************************/
size_t DOFManager::beginDOF() const { return d_begin; }


/****************************************************************
 * Return the last D.O.F. on this core                           *
 ****************************************************************/
size_t DOFManager::endDOF() const { return d_end; }


/****************************************************************
 * Return the local number of D.O.F.s                           *
 ****************************************************************/
size_t DOFManager::numLocalDOF() const { return ( d_end - d_begin ); }
std::vector<size_t> DOFManager::getLocalSizes() const
{
    if ( d_localSize.empty() )
        d_localSize = d_comm.allGather( d_end - d_begin );
    return d_localSize;
}


/****************************************************************
 * Return the global number of D.O.F.s                           *
 ****************************************************************/
size_t DOFManager::numGlobalDOF() const { return d_global; }


/****************************************************************
 * Return the global number of D.O.F.s                           *
 ****************************************************************/
std::vector<size_t> DOFManager::getRemoteDOFs() const { return d_remoteDOFs; }


/****************************************************************
 * Return the row D.O.F.s                                        *
 ****************************************************************/
size_t DOFManager::getRowDOFs( const AMP::Mesh::MeshElementID &, size_t *, size_t, bool ) const
{
    AMP_ERROR( "getRowDOFs(element) is not implemented for the base class" );
    return 0;
}
std::vector<size_t> DOFManager::getRowDOFs( const AMP::Mesh::MeshElementID &id ) const
{
    std::vector<size_t> dofs( 32 );
    size_t N = getRowDOFs( id, dofs.data(), dofs.size() );
    if ( N >= dofs.size() ) {
        dofs.resize( N );
        N = getRowDOFs( id, dofs.data(), dofs.size() );
    }
    dofs.resize( N );
    return dofs;
}


/****************************************************************
 * Compare two DOFManagers                                       *
 ****************************************************************/
bool DOFManager::operator==( const DOFManager &rhs ) const
{
    if ( this == &rhs ) {
        // We are pointing to the same object, it must be ==
        return true;
    }
    if ( this->d_comm.compare( rhs.d_comm ) == 0 ) {
        // The comms do not match, the objects cannot be ==
        return false;
    }
    if ( this->numGlobalDOF() != rhs.numGlobalDOF() || this->beginDOF() != rhs.beginDOF() ||
         this->endDOF() != rhs.endDOF() ) {
        // The DOFs do not match, the objects cannot be ==
        return false;
    }
    if ( this->getIterator() != rhs.getIterator() ) {
        // The iterators do not match, the objects cannot be ==
        return false;
    }
    return true;
}
bool DOFManager::operator!=( const DOFManager &rhs ) const { return !( this->operator==( rhs ) ); }


/****************************************************************
 * Subset the DOF manager                                        *
 ****************************************************************/
std::shared_ptr<DOFManager> DOFManager::subset( const AMP_MPI &comm )
{
    if ( comm.compare( d_comm ) != 0 )
        return shared_from_this();
    if ( comm.getSize() == 1 )
        return std::make_shared<subsetCommSelfDOFManager>( shared_from_this() );
    std::vector<size_t> local_dofs( numLocalDOF(), beginDOF() );
    for ( size_t i = 0; i < numLocalDOF(); i++ )
        local_dofs[i] += i;
    return subsetDOFManager::create( shared_from_this(), local_dofs, getIterator(), comm );
}
std::shared_ptr<DOFManager> DOFManager::subset( const std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                                bool useMeshComm )
{
    if ( !mesh || !getMesh() )
        return std::shared_ptr<DOFManager>();
    // Get a list of the elements in the mesh
    auto subsetIterator = mesh->isMember( getIterator() );
    // Get the DOFs
    std::vector<AMP::Mesh::MeshElementID> id_list;
    id_list.reserve( subsetIterator.size() );
    for ( auto &elem : subsetIterator )
        id_list.push_back( elem.globalID() );
    std::vector<size_t> dofs;
    getDOFs( id_list, dofs );
    // Sort and check the DOFs for errors
    AMP::Utilities::quicksort( dofs );
    for ( auto &dof : dofs ) {
        if ( dof < d_begin || dof >= d_end )
            AMP_ERROR( "Internal error subsetting DOF manager (out of range)" );
    }
    for ( size_t i = 1; i < dofs.size(); i++ ) {
        if ( dofs[i] == dofs[i - 1] )
            AMP_ERROR( "Internal error subsetting DOF manager (duplicate)" );
    }
    // Create the subset DOF Manager
    AMP_MPI comm( AMP_COMM_NULL );
    if ( useMeshComm ) {
        if ( mesh )
            comm = mesh->getComm();
    } else {
        comm = d_comm;
    }
    if ( comm.isNull() )
        return std::shared_ptr<DOFManager>();
    return subsetDOFManager::create( shared_from_this(), dofs, subsetIterator, comm );
}
std::shared_ptr<DOFManager> DOFManager::subset( const AMP::Mesh::MeshIterator &iterator,
                                                const AMP_MPI &comm )
{
    // Get the intesection of the current iterator with the given iterator
    auto intersection =
        AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Intersection, iterator, getIterator() );
    if ( intersection.size() == getIterator().size() )
        intersection = getIterator();
    // Get the list of element we want
    std::vector<AMP::Mesh::MeshElementID> element_list( iterator.size() );
    for ( size_t i = 0; i < intersection.size(); i++ ) {
        element_list[i] = intersection->globalID();
        ++intersection;
    }
    // Get the DOFs
    std::vector<size_t> dofs;
    getDOFs( element_list, dofs );
    // Sort and check the DOFs for errors
    AMP::Utilities::quicksort( dofs );
    for ( auto &dof : dofs ) {
        if ( dof < d_begin || dof >= d_end )
            AMP_ERROR( "Internal error subsetting DOF manager (out of range)" );
    }
    for ( size_t i = 1; i < dofs.size(); i++ ) {
        if ( dofs[i] == dofs[i - 1] )
            AMP_ERROR( "Internal error subsetting DOF manager (duplicate)" );
    }
    // Create the subset DOF Manager
    if ( comm.isNull() )
        return std::shared_ptr<DOFManager>();
    return subsetDOFManager::create( shared_from_this(), dofs, intersection, comm );
}


/****************************************************************
 * Get an id                                                     *
 ****************************************************************/
uint64_t DOFManager::getID() const
{
    return getComm().bcast( reinterpret_cast<uint64_t>( this ), 0 );
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void DOFManager::writeRestart( int64_t fid ) const
{
    IO::writeHDF5( fid, "begin", d_begin );
    IO::writeHDF5( fid, "end", d_end );
    IO::writeHDF5( fid, "global", d_global );
    IO::writeHDF5( fid, "comm", d_comm.hash() );
}
DOFManager::DOFManager( int64_t fid, AMP::IO::RestartManager *manager )
{
    uint64_t commHash;
    IO::readHDF5( fid, "comm", commHash );
    IO::readHDF5( fid, "begin", d_begin );
    IO::readHDF5( fid, "end", d_end );
    IO::readHDF5( fid, "global", d_global );
    d_comm = manager->getComm( commHash );
}
void DOFManager::registerChildObjects( AMP::IO::RestartManager * ) const {}


} // namespace AMP::Discretization


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::Discretization::DOFManager>::DataStoreType(
    std::shared_ptr<const AMP::Discretization::DOFManager> data, RestartManager *manager )
    : d_data( data )
{
    d_hash = data->getID();
    // Register the comm
    manager->registerComm( data->getComm() );
    // Register child objects
    d_data->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::Discretization::DOFManager>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    d_data->writeRestart( gid );
    writeHDF5( gid, "ClassType", d_data->className() );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::Discretization::DOFManager>
AMP::IO::RestartManager::DataStoreType<AMP::Discretization::DOFManager>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid = openGroup( fid, name );
    auto dofs = AMP::Discretization::DOFManagerFactory::create( gid, manager );
    closeGroup( gid );
    return dofs;
}
