#ifndef included_AMP_DOF_Manager
#define included_AMP_DOF_Manager

#include "AMP/discretization/DOF_ManagerParameters.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/utils/AMP_MPI.h"


namespace AMP::IO {
class RestartManager;
}


namespace AMP::Discretization {


/**
 * \class DOF_Manager
 * \brief A class used to provide DOF and vector creation routines
 *
 * \details  This class provides routines for calculating, accessing, and
 *    using the degrees of freedom (DOF) per object.  It is also responsible
 *    for creating vectors.
 */
class DOFManager : public AMP::enable_shared_from_this<AMP::Discretization::DOFManager>
{
public:
    /** \brief Basic constructor for DOFManager
     * \details  This will create a very simple DOFManager with the given number
     *    of DOFs on each processor.  It will not contain info to relate that to a mesh.
     *    A derived implementation should be used for more advanced features.
     *    For example see simpleDOFManager and multiDOFManager.
     * \param[in]  N_local      The local number of DOFs
     * \param[in]  comm         The comm over which the DOFManager exists
     * \param[in]  remoteDOFs   Optional list of remote DOFs
     */
    DOFManager( size_t N_local, const AMP_MPI &comm, std::vector<size_t> remoteDOFs = {} );


    //! Deconstructor
    virtual ~DOFManager();


    //! Return a string with the mesh class name
    virtual std::string className() const;


    /** \brief  Compares two DOFManager for equality.
     * \details This operation compares two DOF managers to see if they are equivalent
     * \param  rhs     DOFManager to compare
     */
    virtual bool operator==( const DOFManager &rhs ) const;


    /** \brief  Inverse of ==
     * \details This function performs an equality check and negates it.  Hence, it is not virtual
     * \param  rhs     DOFManager to compare
     */
    bool operator!=( const DOFManager &rhs ) const;


    /** \brief Get the mesh element for a DOF
     * \details  This will return the mesh element associated with a given DOF.
     * \param[in] dof       The entry in the vector associated with DOF
     * @return              The element for the given DOF.
     */
    virtual AMP::Mesh::MeshElementID getElementID( size_t dof ) const;


    /** \brief Get the mesh element for a DOF
     * \details  This will return the mesh element associated with a given DOF.
     * \param[in] dof       The entry in the vector associated with DOF
     * @return              The element for the given DOF.
     */
    virtual std::unique_ptr<AMP::Mesh::MeshElement> getElement( size_t dof ) const;


    /** \brief Get the number of DOFs per element
     * \details  This will return the number of DOFs per mesh element.
     *    If some DOFs are not associated with a mesh element or if all elements
     *    do not contain the same number of DOFs than this routine will return -1.
     */
    virtual int getDOFsPerPoint() const;


    /** \brief Get the entry indices of DOFs given a mesh element ID
     * \details  This will return a vector of pointers into a Vector that are associated with which.
     * \param[in]  id       The element ID to collect nodal objects for.
     *                      Note: the mesh element may be any type (include a vertex).
     * \param[out] dofs     The entries in the vector associated with D.O.F.s
     */
    void getDOFs( const AMP::Mesh::MeshElementID &id, std::vector<size_t> &dofs ) const;


    /** \brief Get the entry indices of DOFs given a mesh element ID
     * \details  This will return a vector of pointers into a Vector that are associated with which.
     * \param[in]  ids      The element IDs to collect nodal objects for.
     *                      Note: the mesh element may be any type (include a vertex).
     * \param[out] dofs     The entries in the vector associated with D.O.F.s on the nodes
     */
    void getDOFs( const std::vector<AMP::Mesh::MeshElementID> &ids,
                  std::vector<size_t> &dofs ) const;


    /** \brief Get the entry indices of DOFs given a mesh element ID
     * \details  This will return a vector of pointers into a Vector that are associated with which.
     * \param[in]  N_ids    The number of element ids
     * \param[in]  ids      The element IDs to collect nodal objects for.
     *                      Note: the mesh element may be any type (include a vertex).
     * \param[out] dofs     The entries in the vector associated with D.O.F.s on the nodes
     */
    void getDOFs( int N_ids, const AMP::Mesh::MeshElementID *ids, std::vector<size_t> &dofs ) const;


    /** \brief   Get the underlying mesh
     * \details  This will return the mesh(es) that underly the DOF manager (if they exist)
     */
    virtual std::shared_ptr<const AMP::Mesh::Mesh> getMesh() const;


    /** \brief   Get an entry over the mesh elements associated with the DOFs
     * \details  This will return an iterator over the mesh elements associated
     *     with the DOFs.  Each element in the iterator will have 1 or more DOFs
     *     that are associated with that element.  For example, a nodal vector with
     *     3 DOFs stored at each node would return an iterator over all the nodes
     *     with no element repeated.
     *  Note that this iterator does not contain ghost elements because there would
     *     be repeated elements between the different processors.  Calling this iterator
     *     ensures that each owned element is called once regardless of the number of
     *     DOFs on that element and the number of processors that share a ghost copy.
     */
    virtual AMP::Mesh::MeshIterator getIterator() const;


    /** \brief  The first D.O.F. on this core
     * \return The first D.O.F. on this core
     */
    virtual size_t beginDOF() const;


    /** \brief  One past the last D.O.F. on this core
     * \return One past the last D.O.F. on this core
     */
    virtual size_t endDOF() const;


    /** \brief  The local number of D.O.F
     * \return  The local number of D.O.F
     */
    virtual size_t numLocalDOF() const;


    /** \brief  The global number of D.O.F
     * \return  The global number of D.O.F
     */
    virtual size_t numGlobalDOF() const;


    /** \brief  The local number of D.O.F on each rank
     * \return  The local number of D.O.F on each rank
     */
    virtual std::vector<size_t> getLocalSizes() const;


    //! Get the comm for the DOFManger
    inline const AMP_MPI &getComm() const { return d_comm; }


    //! Get the remote DOFs for a vector
    virtual std::vector<size_t> getRemoteDOFs() const;

    virtual void replaceRemoteDOFs( std::vector<size_t> &newRemote ) { d_remoteDOFs = newRemote; }


    //! Get the row DOFs given a mesh element
    std::vector<size_t> getRowDOFs( const AMP::Mesh::MeshElementID &id ) const;


    /** \brief Subset the DOF Manager for a AMP_MPI communicator
     * \details  This will subset a DOF manager for a given communicator.
     * \param[in]  comm         The communicator to use to subset
     */
    virtual std::shared_ptr<DOFManager> subset( const AMP_MPI &comm );


    /** \brief Subset the DOF Manager for a mesh
     * \details  This will subset a DOF manager for a particular mesh.  The resulting DOFManager
     *    can exist on either the comm of the parent DOF manager, or the comm of the mesh (default).
     * \param[in]  mesh         The mesh to use to subset
     * \param[in]  useMeshComm  Do we want to use the mesh comm for the new DOFManager.
     *                          Note: if this is true, any processors that do not contain the mesh
     * will return NULL.
     */
    virtual std::shared_ptr<DOFManager> subset( const std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                                bool useMeshComm = true );


    /** \brief Subset the DOF Manager for a mesh element iterator
     * \details  This will subset a DOF manager for a given mesh element iterator.
     *    The resulting DOFManager will exist on the privided comm.
     * \param[in]  iterator     The mesh iterator for the subset
     * \param[in]  comm         The desired comm
     */
    virtual std::shared_ptr<DOFManager> subset( const AMP::Mesh::MeshIterator &iterator,
                                                const AMP_MPI &comm );


    //! Get a unique id hash
    uint64_t getID() const;


public: // Advanced interfaces
    //! Get the row DOFs given a mesh element
    virtual size_t getRowDOFs( const AMP::Mesh::MeshElementID &id,
                               size_t *dofs,
                               size_t N_alloc,
                               bool sort = true ) const;

    // Append DOFs to the list
    virtual size_t appendDOFs( const AMP::Mesh::MeshElementID &id,
                               size_t *dofs,
                               size_t index,
                               size_t capacity ) const;


public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    virtual void writeRestart( int64_t fid ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid       File identifier to read
     * \param manager   Restart manager
     */
    DOFManager( int64_t fid, AMP::IO::RestartManager *manager );


protected:
    //!  Empty constructor for a DOF manager object
    DOFManager(){};

    //! The DOF manager parameters
    std::shared_ptr<DOFManagerParameters> params;

    //! The begining DOF, ending DOF and number of local DOFs for this processor
    size_t d_begin = 0, d_end = 0, d_global = 0;
    mutable std::vector<size_t> d_localSize;

    //! The remote dofs (if cached)
    std::vector<size_t> d_remoteDOFs;

    //! The comm for this DOFManager
    AMP_MPI d_comm;
};


} // namespace AMP::Discretization

#endif
