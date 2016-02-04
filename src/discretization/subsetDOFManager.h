#ifndef included_subsetDOF_Manager
#define included_subsetDOF_Manager

#include "ampmesh/MeshElement.h"
#include "discretization/DOF_Manager.h"
#include "utils/shared_ptr.h"


namespace AMP {
namespace Discretization {


/**
 * \class subsetDOFManager
 * \brief A derived class to subset a DOFManagers
 * \details  This derived class impliments a concrete DOF_Manager for maintaining
 *   a subset of a DOFManager.
 */
class subsetDOFManager : public DOFManager
{
public:
    using DOFManager::getDOFs;
    using DOFManager::subset;

    /** \brief Default constructor
     * \details  This is the default constructor for creating a subset DOF manager.
     * \param[in] parentDOFManager  The parent DOF manager
     * \param[in] dofs      The DOFs that will be part of the subset (may be a local list)
     * \param[in] iterator  The iterator over the subset of elements in the subsetDOFManager
     * \param[in] comm      The new comm for the subset DOF Manager
     */
    static DOFManager::shared_ptr create( AMP::shared_ptr<const DOFManager> parentDOFManager,
                                          const std::vector<size_t> &dofs,
                                          const AMP::Mesh::MeshIterator &iterator,
                                          const AMP_MPI &comm );


    /** \brief Get the entry indices of DOFs given a mesh element ID
     * \details  This will return a vector of pointers into a Vector that are associated with which.
     *  Note: this function only works if the element we are search for is a element on which a DOF
     * exists
     *  (the underlying mesh element type must match the geometric entity type specified at
     * construction).
     * \param[in]  id       The element ID to collect nodal objects for.  Note: the mesh element may
     * be any type
     * (include a vertex).
     * \param[out] dofs     The entries in the vector associated with D.O.F.s on the nodes
     */
    virtual void getDOFs( const AMP::Mesh::MeshElementID &id, std::vector<size_t> &dofs ) const;


    //! Deconstructor
    virtual ~subsetDOFManager();


    /** \brief   Get an entry over the mesh elements associated with the DOFs
     * \details  This will return an iterator over the mesh elements associated with the DOFs.
     * Note: if any sub-DOFManagers are the same, then this will iterate over repeated elements.
     */
    virtual AMP::Mesh::MeshIterator getIterator() const;


    //! Get the remote DOFs for a vector
    virtual std::vector<size_t> getRemoteDOFs() const;


    //! Get the row DOFs given a mesh element
    virtual std::vector<size_t> getRowDOFs( const AMP::Mesh::MeshElement &obj ) const;


    //! Function to return the local DOFs on the parent DOF manager
    virtual std::vector<size_t> getLocalParentDOFs() const;


    //! Function to convert DOFs from a subset DOFManager DOF to the parent DOF
    virtual std::vector<size_t> getParentDOF( const std::vector<size_t> & ) const;


    /**
      *  Function to convert DOFs from the parent DOF to a subset manager DOF.
      *  Note: if the parent DOF does not exist in the subset, then -1 will be
      *  returned in it's place
      */
    virtual std::vector<size_t> getSubsetDOF( const std::vector<size_t> & ) const;


    //! Get the parent DOFManager
    virtual AMP::shared_ptr<const DOFManager> getDOFManager() const;


protected:
    // The constructor is protected
    subsetDOFManager() : d_parentBegin( 0 ), d_parentEnd( 0 ), d_parentGlobal( 0 ) {}

private:
    //! The parent DOF Manager
    AMP::shared_ptr<const DOFManager> d_parentDOFManager;

    //! The parent begin, end, and global DOFs
    size_t d_parentBegin, d_parentEnd, d_parentGlobal;

    //! The list of local DOFs (sorted, using the parent DOF numbering)
    std::vector<size_t> d_localDOFs;

    //! The list of remote DOFs
    std::vector<size_t> d_remoteParentDOFs;
    std::vector<size_t> d_remoteSubsetDOFs;

    //! The iterator for the subset
    AMP::Mesh::MeshIterator d_iterator;
};
}
}

#endif
