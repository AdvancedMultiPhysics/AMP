#include "discretization/DOF_Manager.h"
#include "discretization/subsetDOFManager.h"
#include "utils/Utilities.h"


namespace AMP {
namespace Discretization {


/****************************************************************
* Constructors                                                  *
****************************************************************/
DOFManager::DOFManager ( size_t N_local, AMP_MPI comm )
{
    d_comm = comm;
    d_comm.sumScan(&N_local,&d_end,1);
    d_begin = d_end - N_local;
    d_global = d_comm.bcast(d_end,d_comm.getSize()-1);
}


/****************************************************************
* Get the entry indices of DOFs given a mesh element            *
****************************************************************/
void DOFManager::getDOFs( const AMP::Mesh::MeshElement &obj, std::vector<size_t> &dofs, std::vector<size_t> ) const
{
    AMP_ERROR("getDOFs is not implimented for the base class");
}
void DOFManager::getDOFs( const AMP::Mesh::MeshElementID &id, std::vector<size_t> &dofs ) const
{
    AMP_ERROR("getDOFs is not implimented for the base class");
}
void DOFManager::getDOFs( const std::vector<AMP::Mesh::MeshElementID> &ids, std::vector<size_t> &dofs ) const
{
    // This is a simple loop to provide a vector interface.  Ideally this should be overwritten by the user
    dofs.resize(0);
    dofs.reserve(2);
    std::vector<size_t> local_dofs;
    for (size_t i=0; i<ids.size(); i++) {
        getDOFs( ids[i], local_dofs );
        if ( local_dofs.size()+dofs.size() > dofs.capacity() )
            dofs.reserve(2*dofs.capacity());
        for (size_t j=0; j<local_dofs.size(); j++)
            dofs.push_back( local_dofs[j] );
    }
}


/****************************************************************
* Get an entry over the mesh elements associated with the DOFs  *
****************************************************************/
AMP::Mesh::MeshIterator DOFManager::getIterator( ) const
{
    AMP_ERROR("getIterator is not implimented for the base class");
    return AMP::Mesh::MeshIterator();
}



/****************************************************************
* Return the first D.O.F. on this core                          *
****************************************************************/
size_t DOFManager::beginDOF( ) const
{
    return d_begin;
}


/****************************************************************
* Return the last D.O.F. on this core                           *
****************************************************************/
size_t DOFManager::endDOF( ) const
{
    return d_end;
}


/****************************************************************
* Return the local number of D.O.F.s                           *
****************************************************************/
size_t DOFManager::numLocalDOF( ) const
{
    return (d_end-d_begin);
}


/****************************************************************
* Return the global number of D.O.F.s                           *
****************************************************************/
size_t DOFManager::numGlobalDOF( ) const
{
    return d_global;
}


/****************************************************************
* Return the communicator                                       *
****************************************************************/
AMP_MPI DOFManager::getComm( ) const
{
    return d_comm;
}


/****************************************************************
* Return the global number of D.O.F.s                           *
****************************************************************/
std::vector<size_t> DOFManager::getRemoteDOFs( ) const
{
    return std::vector<size_t>();
}


/****************************************************************
* Return the global number of D.O.F.s                           *
****************************************************************/
std::vector<size_t> DOFManager::getRowDOFs( const AMP::Mesh::MeshElement &obj ) const
{
    AMP_ERROR("getRowDOFs is not implimented for the base class");
    return std::vector<size_t>();
}



/****************************************************************
* Subset the DOF manager                                        *
****************************************************************/
boost::shared_ptr<DOFManager>  DOFManager::subset( const AMP::Mesh::Mesh::shared_ptr mesh )
{
    // Get a list of the elements in the mesh
    AMP::Mesh::MeshIterator iterator = getIterator();
    std::vector<AMP::Mesh::MeshID> meshIDs;
    if ( mesh.get() != NULL )
        meshIDs = mesh->getBaseMeshIDs();
    std::vector<AMP::Mesh::MeshElementID> element_list;
    element_list.reserve(iterator.size());
    for (size_t i=0; i<iterator.size(); i++) {
        AMP::Mesh::MeshElementID id = iterator->globalID();
        AMP::Mesh::MeshID meshID = id.meshID();
        for (size_t j=0; j<meshIDs.size(); j++) {
            if ( meshID == meshIDs[j] ) {
                element_list.push_back(id);
                break;
            }
        }
        ++iterator;
    }
    // Get the DOFs
    std::vector<size_t> dofs;
    getDOFs( element_list, dofs );
    // Sort and check the DOFs for errors
    AMP::Utilities::quicksort(dofs);
    for (size_t i=0; i<dofs.size(); i++) {
        if ( dofs[i]<d_begin || dofs[i]>=d_end )
            AMP_ERROR("Internal error subsetting DOF manager (out of range)");
    }
    for (size_t i=1; i<dofs.size(); i++) {
        if ( dofs[i]==dofs[i-1] )
            AMP_ERROR("Internal error subsetting DOF manager (duplicate)");
    }
    // Create the subset DOF Manager
    /*size_t tot_size = d_comm.sumReduce(dofs.size());
    if ( tot_size == 0 )
        return DOFManager::shared_ptr();
    if ( tot_size == d_global )
        return shared_from_this();*/
    boost::shared_ptr<DOFManager> parentDOF = shared_from_this();
    return boost::shared_ptr<subsetDOFManager>( new subsetDOFManager( parentDOF, dofs ) );
}
boost::shared_ptr<DOFManager>  DOFManager::subset( const AMP::Mesh::MeshIterator &iterator )
{
    // Get the intesection of the current iterator with the given iterator
    AMP::Mesh::MeshIterator intersection = AMP::Mesh::Mesh::getIterator( AMP::Mesh::Intersection, iterator, getIterator() );
    // Get the list of element we want
    std::vector<AMP::Mesh::MeshElementID> element_list(iterator.size());
    for (size_t i=0; i<intersection.size(); i++) {
        element_list[i] = intersection->globalID();
        ++intersection;
    }
    // Get the DOFs
    std::vector<size_t> dofs;
    getDOFs( element_list, dofs );
    // Sort and check the DOFs for errors
    AMP::Utilities::quicksort(dofs);
    for (size_t i=0; i<dofs.size(); i++) {
        if ( dofs[i]<d_begin || dofs[i]>=d_end )
            AMP_ERROR("Internal error subsetting DOF manager (out of range)");
    }
    for (size_t i=1; i<dofs.size(); i++) {
        if ( dofs[i]==dofs[i-1] )
            AMP_ERROR("Internal error subsetting DOF manager (duplicate)");
    }
    // Create the subset DOF Manager    
    /*size_t tot_size = d_comm.sumReduce(dofs.size());
    if ( tot_size == 0 )
        return DOFManager::shared_ptr();
    if ( tot_size == d_global )
        return shared_from_this();*/
    return boost::shared_ptr<subsetDOFManager>( new subsetDOFManager( shared_from_this(), dofs ) );
}


}
}

