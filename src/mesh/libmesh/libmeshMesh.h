#ifndef included_AMP_LibMesh
#define included_AMP_LibMesh

#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/libmesh/initializeLibMesh.h"

// LibMesh include
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/mesh.h"
#include "libmesh/node.h"
ENABLE_WARNINGS


namespace AMP::Mesh {


class libmeshMeshElement;
class libmeshElemIterator;


/**
 * \class libmeshMesh
 * \brief A concrete mesh class for libMesh
 *
 * \details  This class provides routines for reading, accessing and writing libMesh meshes.
 * The generation of the mesh is controlled by the database passed in through the params object.
 * The database fields control the mesh and provide several options:
 * @code
 *    dim - required integer specifying the physical dimension
 *    FileName - If specified this will load the mesh from the given file
 *    Generator - If specified this will generate a new mesh using the optional parameters in the
 * database
 *       This field must be a string specifying the generator to use.  Valid gerators are:
 *          "cube" - Will generate a cube mesh
 *       Additional areguments:
 *          size - Integer array specifying the number of elements in each direction
 * @endcode
 * The parallel decomposition of the mesh is controlled by libmesh and occurs on the communicator
 * specified through the params object.  Note that libmesh does not support meshes on overlapping
 * communicators.  If multiple meshes are used, they must either share communicators or have unique
 * communicators.
 */
class libmeshMesh final : public Mesh
{
public:
    /**
     * \brief Read in mesh files, partition domain, and prepare environment for simulation
     * \details  For trivial parallelsim, this method reads in the meshes on each processor.  Each
     * processor contains a piece of each mesh.  For massive parallelism, each mesh is on its own
     * communicator.  As such, some math libraries must be initialized accordingly.
     * \param params Parameters for constructing a mesh from an input database
     */
    explicit libmeshMesh( std::shared_ptr<const MeshParameters> params );

    /**
     * \brief Constructor to create a libmeshMesh object from a libMesh mesh.
     * \details This constructor allows the user to construct a mesh directly in libmesh
     * and use it to create the new mesh object.  This function is intended for advanced
     * users only.  Note: the user is responsible for ensuring that libMesh is properly
     * initialized (using initializeLibMesh), and that the mesh is created properly.
     * \param mesh The mesh in libmesh we want to use to construct the new mesh object
     * \param name The name of the new mesh object
     * \param libMeshComm Optional libMesh communicator to use
     */
    explicit libmeshMesh( std::shared_ptr<libMesh::Mesh> mesh,
                          const std::string &name,
                          std::shared_ptr<libMesh::Parallel::Communicator> libMeshComm = nullptr );

    //! Deconstructor
    virtual ~libmeshMesh();


    //! Return a string with the mesh class name
    std::string meshClass() const override;


    //! Function to copy the mesh (allows use to proply copy the derived class)
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


    /* Return the global number of elements of the given type
     * \param type   Geometric type
     */
    size_t numGlobalElements( const GeomType type ) const override;


    /* Return the number of ghost elements of the given type on the current processor
     * \param type   Geometric type
     * \param gcw    Desired ghost cell width
     */
    size_t numGhostElements( const GeomType type, const int gcw ) const override;


    //! Check if two meshes are equal
    bool operator==( const Mesh &mesh ) const override;


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
     * \brief    Return a mesh element given it's id.
     * \details  This function queries the mesh to get an element given the mesh id.
     *    This function is only required to return an element if the id is local.
     *    This function uses libmesh for elements it owns, and uses an O(ln(N)) for
     *    elements that were constructed internally.
     * \param id    Mesh element id we are requesting.
     */
    MeshElement getElement( const MeshElementID &id ) const override;


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


    //! Return the underlying libMesh object
    inline std::shared_ptr<libMesh::Mesh> getlibMesh() const { return d_libMesh; }


    /**
     * \brief    Displace the entire mesh
     * \details  This function will displace the entire mesh by displacing
     *   each node by the values provided in the vector.  This function is
     *   a blocking call for the mesh communicator
     * \param x  Displacement vector.  Must have N DOFs per node where N
     *           is the physical dimension of the mesh.
     */
    void displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> x ) override;


    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    void writeRestart( int64_t fid ) const override;


protected:
    /**
     *  Function to return the neighbors for a node.  The neighbors are defined
     *  as those nodes that share an element with the given node.
     *  Note: the nodes returns are returned in unsorted order.
     */
    std::vector<libMesh::Node *> getNeighborNodes( const MeshElementID & ) const;


private:
    // Friend functions to access protected functions
    friend class libmeshMeshElement;

    typedef std::shared_ptr<std::vector<libmeshMeshElement>> ElemListPtr;


private: // Functions use for initialization
    libmeshMesh(){};
    void initialize();
    ElemListPtr generateGhosts() const;
    ElemListPtr generateLocalElements( GeomType ) const;
    ElemListPtr generateGhostElements( GeomType ) const;
    void fillBoundaryElements();
    libmeshElemIterator localElements() const;

private:
    // Index indicating number of times the position has changed
    uint64_t d_pos_hash;

    // libMesh objects
    std::shared_ptr<libMesh::Mesh> d_libMesh;
    std::shared_ptr<libMesh::Parallel::Communicator> d_libMeshComm;

    // Some basic internal data
    std::vector<size_t> n_local, n_global, n_ghost;
    std::shared_ptr<initializeLibMesh> libmeshInit;

    // Data used to store the node neighbor lists
    std::vector<unsigned int> neighborNodeIDs;
    std::vector<std::vector<libMesh::Node *>> neighborNodes;

    // Data used to store elements that libmesh doesn't create
    ElemListPtr d_localElements[4];
    ElemListPtr d_ghostElements[4];

    // Data used to store the boundary elements
    std::map<std::pair<int, GeomType>, ElemListPtr> d_boundarySets;

    // Data used to store the surface elements
    std::vector<ElemListPtr> d_localSurfaceElements;
    std::vector<ElemListPtr> d_ghostSurfaceElements;

    // Data used to store block info
    std::vector<int> d_block_ids;
};

} // namespace AMP::Mesh

#endif
