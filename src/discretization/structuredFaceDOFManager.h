#ifndef included_AMP_structuredFaceDOFManager
#define included_AMP_structuredFaceDOFManager

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include <memory>


namespace AMP::Discretization {


/**
 * \class structuredFaceDOFManager
 * \brief A derived class to create a DOFManager for faces
 * \details  This derived class impliments a concrete DOFManager for creating Vectors
 *    and matricies over a mesh on the faces of structured meshes.
 *    This is a specific implementation designed for rectangular 3d meshes,
 *    and will create the unknowns on the faces.  Two faces are neighbors if they
 *    share an element.
 */
class structuredFaceDOFManager final : public DOFManager
{
public:
    using DOFManager::subset;


    //! Empty Constructor
    structuredFaceDOFManager() = delete;


    /**
     * \brief Create a new DOF manager object
     * \details  This is the standard constructor for creating a new DOF manager object.
     * \param mesh          Mesh over which we want to construct the DOF map
     * \param DOFsPerFace   The desired number of DOFs per face (x,y,z)
     * \param gcw           The desired ghost width (based on the volumes)
     */
    structuredFaceDOFManager( std::shared_ptr<AMP::Mesh::Mesh> mesh, int DOFsPerFace[3], int gcw );


    //! Deconstructor
    virtual ~structuredFaceDOFManager();


    /** \brief   Get the underlying mesh
     * \details  This will return the mesh(es) that underly the DOF manager (if they exist)
     */
    std::shared_ptr<const AMP::Mesh::Mesh> getMesh() const override;


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


    /** \brief   Get an entry over the mesh elements associated with the DOFs
     * \details  This will return an iterator over the mesh elements associated
     *  with the DOFs.  Each element in the iterator will have 1 or more DOFs
     *  that are associated with that element.  For eaxample, a NodalVectorDOF
     *  would have 3 DOFs stored at each node, and would return an iterator over
     *  all the nodes.
     */
    AMP::Mesh::MeshIterator getIterator() const override;


    //! Get the remote DOFs for a vector
    std::vector<size_t> getRemoteDOFs() const override;

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


private:
    // Function to find the remote DOF given a set of mesh element IDs
    std::vector<size_t>
    getRemoteDOF( const std::vector<AMP::Mesh::MeshElementID> &remote_ids ) const;

    // Function to initialize the data
    void initialize();

    // Data members
    std::shared_ptr<AMP::Mesh::Mesh> d_mesh;
    uint8_t d_DOFsPerFace[3] = { 0, 0, 0 };
    uint8_t d_gcw            = 0;

    std::vector<AMP::Mesh::MeshElementID> d_local_ids[3];
    std::vector<AMP::Mesh::MeshElementID> d_remote_ids[3];
    std::vector<size_t> d_local_dofs[3];
    std::vector<size_t> d_remote_dofs[3];
};
} // namespace AMP::Discretization

#endif
