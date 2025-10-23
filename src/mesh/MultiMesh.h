#ifndef included_AMP_MultiMesh
#define included_AMP_MultiMesh

#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/loadBalance/loadBalanceSimulator.h"

namespace AMP::Mesh {


/**
 * \class MultiMesh
 * \brief A concrete mesh class for a multi-mesh
 *
 * \details  This class provides routines for creating and accessing multimeshes.
 *   The concept of a multimesh is a mesh that is composed of multiple meshes.
 *   This takes care of the need for a mesh manager while allowing all operations
 *   on a given mesh to apply to multiple meshes.  Note: all meshes within a multimesh
 *   are stored in a flat array.  This applies when we have a multimesh that may contain
 *   other multimeshes that may (or may not) overlap.
 *
 * Valid database entries:
 *   LoadBalanceMethod - Method to use for load balancer
 *                       1 - Use independent processor sets for all meshes (default)
 *                       2 - Use all processors for all meshes
 */
class MultiMesh final : public Mesh
{
public:
    /**
     * \brief Default constructor
     * \details  This constructor works with the input parameters to create the mesh.
     * For trivial parallelism, this method reads in the meshes on each processor.  Each
     * processor contains a piece of each mesh.  For massive parallelism, each mesh is on its own
     * communicator.  As such, some math libraries must be initialized accordingly.
     * \param params Parameters for constructing a mesh from an input database
     */
    explicit MultiMesh( std::shared_ptr<const MeshParameters> params );


    /**
     * \brief Constructor to create a MultiMesh from existing meshes
     * \details  This constructor takes a list of meshes and a communicator
     *    and generates the appropriate multimesh
     * \param name      Name of the new mesh
     * \param comm      Desired communicator for the multimesh
     * \param meshes    Meshes to be used as part of the multimesh
     * \param checkComm Check that the comm spans all meshes (advanced option)
     */
    MultiMesh( const std::string &name,
               const AMP_MPI &comm,
               const std::vector<std::shared_ptr<Mesh>> &meshes,
               bool checkComm = true );


    //! Deconstructor
    virtual ~MultiMesh();


    //! Return a string with the mesh class name
    std::string meshClass() const override;


    //! Function to clone the mesh (allows use to properly copy the derived class)
    std::unique_ptr<Mesh> clone() const override;

    /**
     * \brief   Estimate the number of elements in the mesh
     * \details  This function will estimate the number of elements in the mesh.
     *   This is used so that we can properly balance the meshes across multiple processors.
     *   Ideally this should be both an accurate estimate and very fast.  It should not require
     *   any communication and should not have to actually load a mesh.
     * \param params Parameters for constructing a mesh from an input database
     */
    static size_t estimateMeshSize( std::shared_ptr<const MeshParameters> params );

    /**
     * \brief   Return the maximum number of processors that can be used with the mesh
     * \details  This function will return the maximum number of processors that can
     *   be used with the mesh.
     * \param params Parameters for constructing a mesh from an input database
     */
    static size_t maxProcs( std::shared_ptr<const MeshParameters> params );

    /* Return the number of local element of the given type
     * \param type   Geometric type
     */
    size_t numLocalElements( const GeomType type ) const override;


    /* Return the global number of elements of the given type.
     * Note: for a multimesh this will require global communication.
     * To avoid this in the future we would need to cache the value, and register
     *  some type of listener to check if the value changed on any sub meshes.
     * \param type   Geometric type
     */
    size_t numGlobalElements( const GeomType type ) const override;


    /* Return the number of ghost elements of the given type on the current processor
     * \param type   Geometric type
     */
    size_t numGhostElements( const GeomType type, const int gcw ) const override;


    /**
     * \brief    Subset a mesh given a MeshID
     * \details  This function will return the mesh with the given meshID.
     *    Note: for multmeshes, this will return the mesh with the given id.
     *    For a single mesh this will return a pointer to itself if the meshID
     *    matches the meshID of the mesh, and a null pointer otherwise.
     * \param meshID  MeshID of the desired mesh
     */
    std::shared_ptr<Mesh> Subset( MeshID meshID ) const override;


    /**
     * \brief    Subset a mesh given a MeshIterator
     * \details  This function will subset a mesh over a given iterator.
     *   This will return a new mesh object.
     * \param iterator  MeshIterator used to subset
     * \param isGlobal  Is the new subset mesh global over the entire mesh (true,default),
     *                  or do we only want to keep the local mesh (false)
     */
    virtual std::shared_ptr<Mesh> Subset( const MeshIterator &iterator,
                                          bool isGlobal = true ) const override;


    /**
     * \brief    Subset a mesh given a mesh name
     * \details  This function will return the mesh with the given name.
     *    For a single mesh this will return a pointer to itself if the mesh name
     *    matches the name of the mesh, and a null pointer otherwise.
     *    Note: The mesh name is not guaranteed to be unique.  If there are multiple
     *    meshes with the same name, all meshed with the given name will be returned
     *    within a new multimesh.
     *    It is strongly recommended to use the meshID when possible.
     * \param name  Name of the desired mesh
     */
    std::shared_ptr<Mesh> Subset( std::string name ) const override;


    /**
     * \brief    Return an MeshIterator over the given geometric objects
     * \details  Return an MeshIterator over the given geometric objects
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    MeshIterator getIterator( const GeomType type, const int gcw = 0 ) const override;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the surface
     * \details  Return an MeshIterator over the given geometric objects on the surface
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getSurfaceIterator( const GeomType type,
                                             const int gcw = 0 ) const override;


    /**
     * \brief    Return the list of all boundary ID sets in the mesh
     * \details  Return the list of all boundary ID sets in the mesh
     * Note: depending on the mesh this routine may require global communication across the mesh.
     */
    std::vector<int> getBoundaryIDs() const override;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the given boundary ID
     * set
     * \details  Return an MeshIterator over the given geometric objects on the given boundary ID
     * set
     * \param type   Geometric type to iterate over
     * \param id     Boundary id for the elements (example: sideset id)
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator
    getBoundaryIDIterator( const GeomType type, const int id, const int gcw = 0 ) const override;

    /**
     * \brief    Return the list of all boundary ID sets in the mesh
     * \details  Return the list of all boundary ID sets in the mesh
     * Note: depending on the mesh this routine may require global communication across the mesh.
     */
    std::vector<int> getBlockIDs() const override;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the given block ID set
     * \details  Return an MeshIterator over the given geometric objects on the given block ID set
     * \param type   Geometric type to iterate over
     * \param id     Block id for the elements (example: block id in cubit, subdomain in libmesh)
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator
    getBlockIDIterator( const GeomType type, const int id, const int gcw = 0 ) const override;


    /**
     * \brief    Check if an element is in the mesh
     * \details  This function queries the mesh to determine if the given element is a member of the
     * mesh
     * \param id    Mesh element id we are querying.
     */
    bool isMember( const MeshElementID &id ) const override;


    /**
     * \brief    Check if elements are in the mesh
     * \details  This function queries the mesh to determine if each of the given elements
     *           is a member of the mesh and returns an iterator over those elements
     * \param it    Mesh element id we are querying.
     */
    MeshIterator isMember( const MeshIterator &it ) const override;


    /**
     * \brief    Return a mesh element given it's id.
     * \details  This function queries the mesh to get an element given the mesh id.
     *    This function is only required to return an element if the id is local.
     *    Ideally, this should be done in O(1) time, but the implementation is up to
     *    the underlying mesh.  The base class provides a basic implementation, but
     *    uses mesh iterators and requires O(N) time on the number of elements in the mesh.
     * \param id    Mesh element id we are requesting.
     */
    MeshElementPtr getElement( const MeshElementID &id ) const override;


    /**
     * \brief    Return the parent elements of the given mesh element
     * \details  This function queries the mesh to get an element given the mesh id,
     *    then returns the parent elements that have the element as a child
     * \param elem  Mesh element of interest
     * \param type  Element type of the parents requested
     */
    virtual std::vector<MeshElementPtr> getElementParents( const MeshElement &elem,
                                                           const GeomType type ) const override;


    //! Is the current mesh a base mesh
    inline bool isBaseMesh() const override { return false; }


    //! Check if two meshes are equal
    bool operator==( const Mesh &mesh ) const override;


    /**
     *  Get the meshIDs of all meshes that compose the current mesh (including its self)
     *  Note: This function will require global communication
     */
    std::vector<MeshID> getAllMeshIDs() const override;


    /**
     *  Get the meshIDs of all the basic meshes that compose the current mesh (excluding multimeshes
     * and subset meshes)
     *  Note: This function will require global communication
     */
    std::vector<MeshID> getBaseMeshIDs() const override;


    /**
     *  Get the meshIDs of all meshes that compose the current mesh (including its self)
     *  on the current processor.
     */
    std::vector<MeshID> getLocalMeshIDs() const override;


    /**
     *  Get the meshIDs of all the basic meshes that compose the current mesh
     *  (excluding multimeshes and subset meshes) on the current processor.
     */
    std::vector<MeshID> getLocalBaseMeshIDs() const override;


    /**
     *  Get the meshes composing the multimesh
     */
    virtual std::vector<std::shared_ptr<AMP::Mesh::Mesh>> getMeshes();


    /**
     *  Get the meshes composing the multimesh
     */
    virtual std::vector<std::shared_ptr<const AMP::Mesh::Mesh>> getMeshes() const;


    /**
     * \brief    Is the mesh movable
     * \details  This function will check if the mesh can be displaced.
     *    It will return 0 if the mesh cannont be moved, 1 if it can be displaced,
     *    and 2 if the individual nodes can be moved.
     * @return  The if
     */
    Mesh::Movable isMeshMovable() const override;


    /**
     * \brief    Identify if the position has moved
     * \details  This function will return a hash that can be used to
     *    identify if the mesh has been moved.  Any time that displaceMesh
     *    is called, the hash value should change.  There is no requirement
     *    that dispacing a mesh and returning it back to the original position
     *    will return the original hash.
     * @return   hash value with current position id
     */
    uint64_t positionHash() const override;


    /**
     * \brief    Displace the entire mesh
     * \details  This function will displace the entire mesh by a scalar value.
     *   This function is a blocking call for the mesh communicator, and requires
     *   the same value on all processors.  The displacement vector should be the
     *   size of the physical dimension.
     * \param x  Displacement vector
     */
    void displaceMesh( const std::vector<double> &x ) override;

    /**
     * \brief    Displace the entire mesh
     * \details  This function will displace the entire mesh by displacing
     *   each node by the values provided in the vector.  This function is
     *   a blocking call for the mesh communicator
     * \param x  Displacement vector.  Must have N DOFs per node where N
     *           is the physical dimension of the mesh.
     */
    void displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> x ) override;


    // Needed to prevent problems with virtual functions
    using Mesh::Subset;


public: // Default constructors
    MultiMesh()                           = delete;
    explicit MultiMesh( MultiMesh &&rhs ) = default;
    explicit MultiMesh( const MultiMesh &rhs );
    MultiMesh &operator=( MultiMesh &&rhs ) = delete;
    MultiMesh &operator=( const MultiMesh &rhs ) = delete;

public: // Functions to help with load balancing
    /**
     * \brief    A function to compute the AMP_MPI comms for each mesh
     * \details  This function computes the AMP_MPI comms for each mesh given the comm groups
     * \param comm     Parent communicator
     * \param groups   List of ranks for each communication group
     */
    static std::vector<AMP_MPI> createComms( const AMP_MPI &comm,
                                             const std::vector<std::vector<int>> &groups );


public:
    //! Function to create the databases for the meshes within the multimesh
    static std::vector<std::shared_ptr<AMP::Database>>
    createDatabases( std::shared_ptr<const AMP::Database> database );


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t ) const override;
    MultiMesh( int64_t, AMP::IO::RestartManager * );


private:
    void initialize();

private:
    std::vector<std::shared_ptr<Mesh>> d_meshes;       //!< A list of all meshes in the multimesh
    Mesh::Movable d_isMovable = Mesh::Movable::Deform; //!< Are all meshes movable
};

} // namespace AMP::Mesh

#endif
