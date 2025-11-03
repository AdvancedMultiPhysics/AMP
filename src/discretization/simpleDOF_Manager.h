#ifndef included_AMP_simpleDOF_Manager
#define included_AMP_simpleDOF_Manager

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include <memory>


namespace AMP::Discretization {


/**
 * \class simpleDOFManager
 * \brief A derived class to create a simple DOF_Manager
 * \details  This derived class implements a concrete DOF_Manager for creating Vectors
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
     * \param DOFsPerElement The desired number of DOFs per element
     * \param split         Do we want to split the DOFManager by the meshes returning a
     * multiDOFManager
     */
    static std::shared_ptr<DOFManager> create( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                               AMP::Mesh::GeomType type,
                                               int gcw,
                                               int DOFsPerElement,
                                               bool split = true );

    /**
     * \brief Create a new DOF manager object
     * \details  This is the standard constructor for creating a new DOF manager object.
     * \param mesh          Mesh over which we want to construct the DOF map
     * \param local         Local iterator
     * \param ghost         Ghost iterator
     * \param type          The geometric entity type for the DOF map
     * \param DOFsPerElement The desired number of DOFs per element
     */
    simpleDOFManager( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                      const AMP::Mesh::MeshIterator &local,
                      const AMP::Mesh::MeshIterator &ghost,
                      AMP::Mesh::GeomType type,
                      int DOFsPerElement );


    /**
     * \brief Create a new DOF manager object
     * \details  This is will create a new simpleDOFManager from a mesh iterator
     * \param mesh          Mesh over which the iterators are defined
     * \param ghost         Ghost iterator
     * \param local         Local iterator
     * \param DOFsPerElement The desired number of DOFs per element
     */
    static std::shared_ptr<DOFManager> create( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                               const AMP::Mesh::MeshIterator &ghost,
                                               const AMP::Mesh::MeshIterator &local,
                                               int DOFsPerElement );


    /**
     * \brief Create a new DOF manager object
     * \details  This is will create a new simpleDOFManager from a mesh iterator
     *   on the local processor only (no remote DOFs).
     * \param it             The iterator over the elements (no ghost cells)
     * \param DOFsPerElement The desired number of DOFs pere element
     */
    static std::shared_ptr<DOFManager> create( const AMP::Mesh::MeshIterator &it,
                                               int DOFsPerElement );


    //! Destructor
    virtual ~simpleDOFManager();


    //! Return a string with the mesh class name
    std::string className() const override { return "simpleDOFManager"; }


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


    /** \brief   Get the underlying mesh
     * \details  This will return the mesh(es) that underly the DOF manager (if they exist)
     */
    std::shared_ptr<const AMP::Mesh::Mesh> getMesh() const override;


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


    /** \brief Subset the DOF Manager for a mesh
     * \details  This will subset a DOF manager for a particular mesh.  The resulting DOFManager
     *    can exist on either the comm of the parent DOF manager, or the comm of the mesh (default).
     * \param[in]  mesh         The mesh to use to subset
     * \param[in]  useMeshComm  Do we want to use the mesh comm for the new DOFManager.
     *                          Note: if this is true, any processors that do not contain the mesh
     * will return NULL.
     */
    std::shared_ptr<DOFManager> subset( const std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                        bool useMeshComm = true ) override;


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


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t ) const override;
    simpleDOFManager( int64_t, AMP::IO::RestartManager * );

protected:
    // Private constructor
    simpleDOFManager() = delete;

    // Function to find the remote DOF given a set of mesh element IDs
    std::vector<size_t> getRemoteDOF( const std::vector<AMP::Mesh::MeshElementID> &ids ) const;

    // Function to initialize the data
    void initialize();


protected:                                                     // Data members
    bool d_isBaseMesh          = false;                        // Is the mesh a base mesh
    AMP::Mesh::GeomType d_type = AMP::Mesh::GeomType::Nullity; // entity type
    uint8_t d_DOFsPerElement   = 0;                            // # Of DOFs per type
    std::shared_ptr<const AMP::Mesh::Mesh> d_mesh;             // Mesh
    AMP::Mesh::MeshID d_meshID;                                // MeshID
    std::vector<AMP::Mesh::MeshID> d_baseMeshIDs;              // Must be global list
    AMP::Mesh::MeshIterator d_localIterator;                   // Local iterator
    AMP::Mesh::MeshIterator d_ghostIterator;                   // global iterator
    std::vector<AMP::Mesh::MeshElementID> d_local_id;          // List of local ids
    std::vector<AMP::Mesh::MeshElementID> d_remote_id;         // List of remote ids
    std::vector<size_t> d_remote_dof;                          // remote dofs
};
} // namespace AMP::Discretization

#endif
