#ifndef included_AMP_subsetCommSelfDOFManager
#define included_AMP_subsetCommSelfDOFManager

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/MeshElement.h"
#include <memory>


namespace AMP::Discretization {


/**
 * \class subsetCommSelfDOFManager
 * \brief A derived class to subset a DOFManagers
 * \details  This derived class impliments a concrete DOF_Manager for maintaining
 *   a subset of a DOFManager.
 */
class subsetCommSelfDOFManager : public DOFManager
{
public:
    using DOFManager::subset;

    //! Empty constructor
    subsetCommSelfDOFManager();

    /** \brief Default constructor
     * \details  This is the default constructor for creating a subset DOF manager.
     * \param[in] parentDOFManager  The parent DOF manager
     */
    subsetCommSelfDOFManager( std::shared_ptr<const DOFManager> parentDOFManager );


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
    std::unique_ptr<AMP::Mesh::MeshElement> getElement( size_t dof ) const override;


    //! Deconstructor
    virtual ~subsetCommSelfDOFManager() = default;


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


    //! Get the local sizes on each rank
    std::vector<size_t> getLocalSizes() const override { return { d_end - d_begin }; }

    /** \brief Get the number of DOFs per element
     * \details  This will return the number of DOFs per mesh element.
     *    If some DOFs are not associated with a mesh element or if all elements
     *    do not contain the same number of DOFs than this routine will return -1.
     */
    int getDOFsPerPoint() const override;


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
    // Convert dof indicies
    size_t getSubsetDOF( size_t N, size_t *dofs ) const;

private:
    //! The parent DOF Manager
    std::shared_ptr<const DOFManager> d_parentDOFManager;

    //! The parent begin, end, and global DOFs
    size_t d_parentBegin, d_parentEnd;
};
} // namespace AMP::Discretization

#endif
