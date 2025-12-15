#ifndef included_AMP_MultiDOF_Manager
#define included_AMP_MultiDOF_Manager

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/MultiDOFHelper.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include <memory>


namespace AMP::Discretization {


/**
 * \class multiDOFManager
 * \brief A derived class to combine multiple DOFManagers
 * \details  This derived class impliments a concrete DOF_Manager for creating DOFs that
 *   consist of multiple DOFManagers.  This is useful to combine multiple DOFs over meshes
 *   on a multiVector, for combining multiple discretizations, and for combining vectors.
 *   A multivector will have a pointer to a multiDOFManager instead of a standard DOFManager.
 *   It is also possible that a standard vector can use a multiDOFManager.
 */
class multiDOFManager : public DOFManager
{
public:
    using DOFManager::subset;

    /**
     * \brief Create a new DOF manager object
     * \details  This is the standard constructor for creating a new multiDOFManager object.
     * \param comm  Comm over which the DOFManager will exist
     * \param managers  List of the DOFManagers on the current processor
     * \param mesh  Optional mesh over which dof managers are defined (usually a multimesh)
     */
    multiDOFManager( const AMP_MPI &comm,
                     std::vector<std::shared_ptr<DOFManager>> managers,
                     std::shared_ptr<const AMP::Mesh::Mesh> mesh = {} );

    /**
     * \brief Create a new DOF manager object
     * \details  Create a multiDOFManager that is a view of a single DOFManager
     * \param manager  Original DOF Manager
     */
    multiDOFManager( std::shared_ptr<DOFManager> manager );

    //! Deconstructor
    virtual ~multiDOFManager() override;


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
     * \details  This will return an iterator over the mesh elements associated with the DOFs.
     * Note: if any sub-DOFManagers are the same, then this will iterate over repeated elements.
     */
    AMP::Mesh::MeshIterator getIterator() const override;


    //! Get the remote DOFs for a vector
    std::vector<size_t> getRemoteDOFs() const override;


    /** \brief Subset the DOF Manager for a AMP_MPI communicator
     * \details  This will subset a DOF manager for a given communicator.
     * \param[in]  comm         The communicator to use to subset
     */
    std::shared_ptr<DOFManager> subset( const AMP_MPI &comm ) override;


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


    /** \brief Subset the DOF Manager for a mesh element iterator
     * \details  This will subset a DOF manager for a given mesh element iterator.
     *    The resulting DOFManager will exist on the privided comm.
     * \param[in]  iterator     The mesh iterator for the subset
     * \param[in]  comm         The desired comm
     */
    std::shared_ptr<DOFManager> subset( const AMP::Mesh::MeshIterator &iterator,
                                        const AMP_MPI &comm ) override;

    /** reset a dof manager based on component dof managers
     * \param managers  List of the DOFManagers on the current processor
     * \param mesh  Optional mesh over which dof managers are defined (usually a multimesh)
     **/
    void reset( std::vector<std::shared_ptr<DOFManager>> managers,
                std::shared_ptr<const AMP::Mesh::Mesh> mesh = {} );

    /** \brief Get the number of DOFs per element
     * \details  This will return the number of DOFs per mesh element.
     *    If some DOFs are not associated with a mesh element or if all elements
     *    do not contain the same number of DOFs than this routine will return -1.
     */
    int getDOFsPerPoint() const override;

public:
    //! Get the DOFManagers that compose the multiDOFManager
    inline const auto &getDOFManagers() const { return d_managers; }

    //! get the i-th dof manager
    inline const auto &getDOFManager( const size_t i ) const { return d_managers[i]; }


    /** \brief   Function to convert DOFs from a sub-manager DOF to the global DOF
     * \details  This function returns the global DOF given the local DOF.  Note that
     *      subDOFManager is specified by the index in the std::vector of DOFManagers
     *      (see getDOFManagers).  This is needed since the same DOFManager may be
     *      repeated many times so searching is not an option.  For example, consider
     *      a multiVector with multiple vectors of the same type.
     * \param[in]  DOFManager       The index to the desired DOFManager (see getDOFManagers)
     * \param[in]  localDOF         The local DOF to convert to global DOF
     */
    std::vector<size_t> getGlobalDOF( const int DOFManager,
                                      const std::vector<size_t> &localDOF ) const;


    /** Function to convert DOFs from the global DOF to a sub-manager DOF
     *  If a given global DOF is not in the given sub-manager, then -1 will
     *  be returned for its value.
     */
    /** \brief   Function to convert DOFs from the global DOF to a sub-manager DOF
     * \details  This function returns the DOF on a a sub-manager given the global DOF.
     *      If a given global DOF is not in the given sub-manager, then -1 will
     *      be returned for its value.  Note that subDOFManager is specified by the
     *      index in the std::vector of DOFManagers (see getDOFManagers).
     *      This is needed since the same DOFManager may be repeated many times so
     *      searching is not an option.  For example, consider a multiVector with
     *      multiple vectors of the same type.
     * \param[in]  DOFManager       The index to the desired DOFManager (see getDOFManagers)
     * \param[in]  globalDOF        The global DOF to convert to a DOF on the sub DOFManager
     */
    std::vector<size_t> getSubDOF( const int DOFManager,
                                   const std::vector<size_t> &globalDOF ) const;


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

    //! Get the local sizes on each rank
    std::vector<size_t> getLocalSizes() const override;

    // Get the map
    inline const multiDOFHelper &getMap() const { return d_dofMap; }

private:
    // Convert the local to global dof
    inline size_t subToGlobal( int manager, size_t dof ) const
    {
        return d_dofMap.subToGlobal( manager, dof );
    }

    // Convert the global to local dof
    inline std::pair<size_t, int> globalToSub( size_t dof ) const
    {
        return d_dofMap.globalToSub( dof );
    }


private:
    multiDOFManager() = delete;

    std::shared_ptr<const AMP::Mesh::Mesh> d_mesh;
    std::vector<std::shared_ptr<DOFManager>> d_managers;
    std::vector<size_t> d_globalSize;
    multiDOFHelper d_dofMap;
};
} // namespace AMP::Discretization

#endif
