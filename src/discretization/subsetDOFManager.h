#ifndef included_AMP_subsetDOF_Manager
#define included_AMP_subsetDOF_Manager

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/MeshElement.h"
#include <memory>


namespace AMP::Discretization {


/**
 * \class subsetDOFManager
 * \brief A derived class to subset a DOFManagers
 * \details  This derived class impliments a concrete DOF_Manager for maintaining
 *   a subset of a DOFManager.
 */
class subsetDOFManager : public DOFManager
{
public:
    using DOFManager::subset;

    /** \brief Default constructor
     * \details  This is the default constructor for creating a subset DOF manager.
     * \param[in] parentDOFManager  The parent DOF manager
     * \param[in] dofs      The DOFs that will be part of the subset (may be a local list)
     * \param[in] iterator  The iterator over the subset of elements in the subsetDOFManager
     * \param[in] comm      The new comm for the subset DOF Manager
     */
    static std::shared_ptr<DOFManager> create( std::shared_ptr<const DOFManager> parentDOFManager,
                                               const std::vector<size_t> &dofs,
                                               const AMP::Mesh::MeshIterator &iterator,
                                               const AMP_MPI &comm );


    /** \brief Get the mesh element ID for a DOF
     * \details  This will return the mesh element id associated with a given DOF.
     * \param[in] dof       The entry in the vector associated with DOF
     * @return              The element id for the given DOF.
     */
    AMP::Mesh::MeshElementID getElementID( size_t dof ) const override;


    /** \brief Get the mesh element for a DOF
     * \details  This will return the mesh element associated with a given DOF.
     * \param[in] dof       The entry in the vector associated with DOF
     * @return              The element for the given DOF.
     */
    AMP::Mesh::MeshElement getElement( size_t dof ) const override;


    //! Deconstructor
    virtual ~subsetDOFManager();


    /** \brief   Get the underlying mesh
     * \details  This will return the mesh(es) that underly the DOF manager (if they exist)
     */
    std::shared_ptr<const AMP::Mesh::Mesh> getMesh() const override;


    /** \brief   Get an entry over the mesh elements associated with the DOFs
     * \details  This will return an iterator over the mesh elements associated with the DOFs.
     * Note: if any sub-DOFManagers are the same, then this will iterate over repeated elements.
     */
    AMP::Mesh::MeshIterator getIterator() const override;


    //! Get the remote DOFs for a vector
    std::vector<size_t> getRemoteDOFs() const override;


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
    virtual std::shared_ptr<const DOFManager> getDOFManager() const;


public: // Advanced interfaces
    //! Get the row DOFs given a mesh element
    size_t getRowDOFs( const AMP::Mesh::MeshElementID &id,
                       size_t *dofs,
                       size_t N_alloc,
                       bool sort = true ) const override;
    using DOFManager::getRowDOFs;

    // Append DOFs to the list
    size_t appendDOFs( const AMP::Mesh::MeshElementID &id,
                       size_t *dofs,
                       size_t index,
                       size_t capacity ) const override;


protected:
    // The constructor is protected
    subsetDOFManager() : d_parentBegin( 0 ), d_parentEnd( 0 ), d_parentGlobal( 0 ) {}

private:
    //! The parent DOF Manager
    std::shared_ptr<const DOFManager> d_parentDOFManager;

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
} // namespace AMP::Discretization

#endif
