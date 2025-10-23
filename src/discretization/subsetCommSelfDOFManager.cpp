#include "AMP/discretization/subsetCommSelfDOFManager.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"


namespace AMP::Discretization {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
subsetCommSelfDOFManager::subsetCommSelfDOFManager() : d_parentBegin( 0 ), d_parentEnd( 0 )
{
    d_begin  = 0;
    d_end    = 0;
    d_global = 0;
}
subsetCommSelfDOFManager::subsetCommSelfDOFManager( std::shared_ptr<const DOFManager> DOF )
    : d_parentDOFManager( DOF ), d_parentBegin( DOF->beginDOF() ), d_parentEnd( DOF->endDOF() )
{
    d_comm   = AMP_COMM_SELF;
    d_begin  = 0;
    d_end    = d_parentEnd - d_parentBegin;
    d_global = d_end;
}


/****************************************************************
 * Convert DOF indicies                                          *
 ****************************************************************/
size_t subsetCommSelfDOFManager::getSubsetDOF( size_t N, size_t *dofs ) const
{
    size_t N2 = 0;
    for ( size_t i = 0; i < N; i++ ) {
        if ( dofs[i] >= d_parentBegin && dofs[i] < d_parentEnd )
            dofs[N2++] = dofs[i] - d_parentBegin;
    }
    return N2;
}


/****************************************************************
 * Get the dofs for the given element                            *
 ****************************************************************/
size_t subsetCommSelfDOFManager::appendDOFs( const AMP::Mesh::MeshElementID &id,
                                             size_t *dofs,
                                             size_t index,
                                             size_t capacity ) const
{
    // Get the dofs from the parent
    auto dofs2 = &dofs[index];
    size_t N   = d_parentDOFManager->appendDOFs( id, dofs2, 0, capacity - index );
    if ( N + index > capacity ) {
        dofs2 = new size_t[N];
        N     = d_parentDOFManager->appendDOFs( id, dofs2, 0, N );
    }
    // Convert to local removing remote values
    size_t N2 = getSubsetDOF( N, dofs2 );
    // Copy and free temporary memory if needed
    if ( dofs2 != &dofs[index] ) {
        for ( size_t i = 0; i < std::min( N2, capacity - index ); i++ )
            dofs[index + i] = dofs2[i];
        delete[] dofs2;
    }
    return N2;
}


/****************************************************************
 * Get the element ID give a dof                                 *
 ****************************************************************/
AMP::Mesh::MeshElementID subsetCommSelfDOFManager::getElementID( size_t dof ) const
{
    return d_parentDOFManager->getElementID( dof + d_parentBegin );
}
std::unique_ptr<AMP::Mesh::MeshElement> subsetCommSelfDOFManager::getElement( size_t dof ) const
{
    return d_parentDOFManager->getElement( dof + d_parentBegin );
}


/****************************************************************
 * Return the mesh                                               *
 ****************************************************************/
std::shared_ptr<const AMP::Mesh::Mesh> subsetCommSelfDOFManager::getMesh() const
{
    return d_parentDOFManager->getMesh();
}

/****************************************************************
 * Get an entry over the mesh elements associated with the DOFs  *
 * Note: if any sub-DOFManagers are the same, then this will     *
 * iterate over repeated elements.                               *
 ****************************************************************/
AMP::Mesh::MeshIterator subsetCommSelfDOFManager::getIterator() const
{
    return d_parentDOFManager->getIterator();
}


/****************************************************************
 * Return the remote DOFs for a vector                           *
 ****************************************************************/
std::vector<size_t> subsetCommSelfDOFManager::getRemoteDOFs() const { return {}; }


/****************************************************************
 * Return the global number of D.O.F.s                           *
 ****************************************************************/
size_t subsetCommSelfDOFManager::getRowDOFs( const AMP::Mesh::MeshElementID &id,
                                             size_t *dofs,
                                             size_t capacity,
                                             bool sort ) const
{
    // Get the dofs from the parent
    auto dofs2 = dofs;
    size_t N   = d_parentDOFManager->getRowDOFs( id, dofs, capacity, sort );
    if ( N > capacity ) {
        dofs2 = new size_t[N];
        N     = d_parentDOFManager->getRowDOFs( id, dofs, N, sort );
    }
    // Convert to local removing remote values
    size_t N2 = getSubsetDOF( N, dofs2 );
    // Copy and free temporary memory if needed
    if ( dofs2 != dofs ) {
        for ( size_t i = 0; i < std::min( N2, capacity ); i++ )
            dofs[i] = dofs2[i];
        delete[] dofs2;
    }
    return N;
}


/****************************************************************
 * Function to convert DOFs                                      *
 ****************************************************************/
std::vector<size_t>
subsetCommSelfDOFManager::getParentDOF( const std::vector<size_t> &subsetDOFs ) const
{
    auto parentDOFs = subsetDOFs;
    for ( auto &dof : parentDOFs )
        dof += d_parentBegin;
    return parentDOFs;
}
std::vector<size_t>
subsetCommSelfDOFManager::getSubsetDOF( const std::vector<size_t> &parentDOFs ) const
{
    auto subsetDOFs = parentDOFs;
    size_t N2       = getSubsetDOF( subsetDOFs.size(), subsetDOFs.data() );
    subsetDOFs.resize( N2 );
    return subsetDOFs;
}
std::vector<size_t> subsetCommSelfDOFManager::getLocalParentDOFs() const
{
    std::vector<size_t> dofs( d_parentEnd - d_parentBegin );
    for ( size_t i = 0; i < dofs.size(); i++ )
        dofs[i] = i + d_parentBegin;
    return dofs;
}


/****************************************************************
 * Function to return the DOFManagers                            *
 ****************************************************************/
std::shared_ptr<const DOFManager> subsetCommSelfDOFManager::getDOFManager() const
{
    return d_parentDOFManager;
}
} // namespace AMP::Discretization
