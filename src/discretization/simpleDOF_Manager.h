#ifndef included_simpleDOF_Manager
#define included_simpleDOF_Manager

#include "AMP/ampmesh/Mesh.h"
#include "AMP/ampmesh/MeshElement.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/shared_ptr.h"


namespace AMP {
namespace Discretization {


/**
 * \class simpleDOFManager
 * \brief A derived class to create a simple DOF_Manager
 * \details  This derived class impliments a concrete DOF_Manager for creating Vectors
 *    over a mesh on a particular mesh entity.  For example it can create a NodalVector
 *    over the entire Mesh.  Note: this class will be replaced by a more complete
 *    Discretization interface.
 */
class simpleDOFManager : public DOFManager
{
public:
    using DOFManager::subset;


    /**
     * \brief Create a new DOF manager object
     * \details  This is the standard constructor for creating a new DOF manager object.
     * \param mesh          Mesh over which we want to construct the DOF map
     * \param type          The geometric entity type for the DOF map
     * \param gcw           The desired ghost width
     * \param DOFsPerElement The desired number of DOFs pere element
     * \param split         Do we want to split the DOFManager by the meshes returning a
     * multiDOFManager
     */
    static DOFManager::shared_ptr create( AMP::shared_ptr<AMP::Mesh::Mesh> mesh,
                                          AMP::Mesh::GeomType type,
                                          int gcw,
                                          int DOFsPerElement,
                                          bool split = true );


    /**
     * \brief Create a new DOF manager object
     * \details  This is will create a new simpleDOFManager from a mesh iterator
     * \param mesh          Mesh over which the iterators are defined
     * \param it1           The iterator over the elements (including ghost cells)
     * \param it2           The iterator over the elements (excluding ghost cells)
     * \param DOFsPerElement The desired number of DOFs pere element
     */
    static DOFManager::shared_ptr create( AMP::shared_ptr<AMP::Mesh::Mesh> mesh,
                                          const AMP::Mesh::MeshIterator &it1,
                                          const AMP::Mesh::MeshIterator &it2,
                                          int DOFsPerElement );


    /**
     * \brief Create a new DOF manager object
     * \details  This is will create a new simpleDOFManager from a mesh iterator
     *   on the local processor only (no remote DOFs).
     * \param it             The iterator over the elements (no ghost cells)
     * \param DOFsPerElement The desired number of DOFs pere element
     */
    static DOFManager::shared_ptr create( const AMP::Mesh::MeshIterator &it, int DOFsPerElement );


    //! Destructor
    virtual ~simpleDOFManager();


    /** \brief Get the mesh element for a DOF
     * \details  This will return the mesh element associated with a given DOF.
     * \param[in] dof       The entry in the vector associated with DOF
     * @return              The element for the given DOF.
     */
    virtual AMP::Mesh::MeshElement getElement( size_t dof ) const override;


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
    virtual void getDOFs( const AMP::Mesh::MeshElementID &id,
                          std::vector<size_t> &dofs ) const override;


    /** \brief Get the entry indices of DOFs given a mesh element ID
     * \details  This will return a vector of pointers into a Vector that are associated with which.
     * \param[in]  ids      The element IDs to collect nodal objects for.
     *                      Note: the mesh element may be any type (include a vertex).
     * \param[out] dofs     The entries in the vector associated with D.O.F.s on the nodes
     */
    virtual void getDOFs( const std::vector<AMP::Mesh::MeshElementID> &ids,
                          std::vector<size_t> &dofs ) const override;


    /** \brief   Get an entry over the mesh elements associated with the DOFs
     * \details  This will return an iterator over the mesh elements associated
     *  with the DOFs.  Each element in the iterator will have 1 or more DOFs
     *  that are associated with that element.  For eaxample, a NodalVectorDOF
     *  would have 3 DOFs stored at each node, and would return an iterator over
     *  all the nodes.
     */
    virtual AMP::Mesh::MeshIterator getIterator() const override;


    //! Get the remote DOFs for a vector
    virtual std::vector<size_t> getRemoteDOFs() const override;


    //! Get the row DOFs given a mesh element
    virtual std::vector<size_t> getRowDOFs( const AMP::Mesh::MeshElement &obj ) const override;


    /** \brief Subset the DOF Manager for a mesh
     * \details  This will subset a DOF manager for a particular mesh.  The resulting DOFManager
     *    can exist on either the comm of the parent DOF manager, or the comm of the mesh (default).
     * \param[in]  mesh         The mesh to use to subset
     * \param[in]  useMeshComm  Do we want to use the mesh comm for the new DOFManager.
     *                          Note: if this is true, any processors that do not contain the mesh
     * will return NULL.
     */
    virtual DOFManager::shared_ptr subset( const AMP::Mesh::Mesh::shared_ptr mesh,
                                           bool useMeshComm = true ) override;


private:
    // Private constructor
    simpleDOFManager() : d_isBaseMesh( false ), DOFsPerElement( 0 ) {}

    // Function to find the remote DOF given a set of mesh element IDs
    std::vector<size_t> getRemoteDOF( std::vector<AMP::Mesh::MeshElementID> remote_ids ) const;

    // Function to initialize the data
    void initialize();

    // Append DOFs
    inline void appendDOFs( const AMP::Mesh::MeshElementID &id, std::vector<size_t> &dofs ) const;

    // Data members
    AMP::shared_ptr<AMP::Mesh::Mesh> d_mesh;
    bool d_isBaseMesh;
    AMP::Mesh::MeshID d_meshID;
    std::vector<AMP::Mesh::MeshID> d_baseMeshIDs; // Must be global list
    AMP::Mesh::GeomType d_type;
    AMP::Mesh::MeshIterator d_localIterator;
    AMP::Mesh::MeshIterator d_ghostIterator;
    int DOFsPerElement;
    std::vector<AMP::Mesh::MeshElementID> d_local_id;
    std::vector<AMP::Mesh::MeshElementID> d_remote_id;
    std::vector<size_t> d_remote_dof;
};
} // namespace Discretization
} // namespace AMP

#endif
