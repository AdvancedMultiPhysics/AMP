#ifndef included_AMP_TriangleMesh
#define included_AMP_TriangleMesh

#include "AMP/geometry/Geometry.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/mesh/MeshIterator.h"

#include <array>
#include <map>
#include <memory>
#include <vector>


namespace AMP::Mesh {


template<uint8_t NG, uint8_t NP, uint8_t TYPE>
class TriangleMeshIterator;
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
class TriangleMeshElement;


// Class to store parent data
template<class TYPE>
class StoreCompressedList
{
public:
    inline StoreCompressedList() {}
    inline explicit StoreCompressedList( const std::vector<std::vector<TYPE>> &data )
    {
        size_t Nt = 0;
        for ( size_t i = 0; i < data.size(); i++ )
            Nt += data[i].size();
        d_size.resize( data.size() );
        d_offset.resize( data.size() );
        d_data.resize( Nt );
        for ( size_t i = 0, k = 0; i < data.size(); i++ ) {
            d_size[i]   = data[i].size();
            d_offset[i] = k;
            for ( size_t j = 0; j < d_size[i]; j++, k++ )
                d_data[k] = data[i][j];
        }
    }
    inline const ElementID *begin( size_t i ) const
    {
        if ( i >= d_size.size() )
            return nullptr;
        return &d_data[d_offset[i]];
    }
    inline const ElementID *end( size_t i ) const
    {
        if ( i >= d_size.size() )
            return nullptr;
        return &d_data[d_offset[i]] + d_size[i];
    }

private:
    std::vector<size_t> d_size;
    std::vector<size_t> d_offset;
    std::vector<TYPE> d_data;
};


/**
 * \class TriangleMesh
 * \brief A class used to represent an unstructured mesh of Triangles/Tetrahedrals
 */
template<uint8_t NG, uint8_t NP = NG + 1>
class TriangleMesh : public AMP::Mesh::Mesh
{
public: // Convenience typedefs
    static_assert( NG <= 3, "Not programmed for higher dimensions yet" );
    typedef std::array<double, NP> Point;
    typedef std::array<ElementID, 2> Edge;
    typedef std::array<ElementID, 3> Triangle;
    typedef std::array<ElementID, 4> Tetrahedron;

public:
    /**
     * \brief Generate a triangle mesh from local triangle coordinates
     * \details  Create a triangle mesh from the local triangle coordinates.
     *    Note: Triangle list should be unique for each rank, load balance
     *    will be automatically adjusted.
     * \param triangles  List of triangles (each rank may contribute a unique list)
     * \param comm       Communicator to use
     *                   (load balance will be automatically generated on this comm)
     * \param tol        Relative tolerance (based on range of points) to use to determine
     *                   if two points are the same
     */
    static std::shared_ptr<TriangleMesh<NG, NP>>
    generate( const std::vector<std::array<Point, NG + 1>> &triangles,
              const AMP_MPI &comm,
              double tol = 1e-12 );

    /**
     * \brief Generate a triangle mesh from local triangle coordinates
     * \details  Create a triangle mesh from the local triangle coordinates.
     *    Note: Triangle list should be unique for each rank,
     *          load balance will be automatically adjusted.
     * \param vertices   List of vertices
     * \param triangles  List of triangles (each rank may contribute a unique list)
     * \param tri_nab    List of triangles neighbors
     * \param comm       Communicator to use
     *                   (load balance will be automatically generated on this comm)
     * \param geom       Optional geometry to associate with the mesh
     * \param blockID    Optional vector with the block id for each triangle
     */
    static std::shared_ptr<TriangleMesh<NG, NP>>
    generate( std::vector<std::array<double, NP>> vertices,
              std::vector<std::array<int, NG + 1>> triangles,
              std::vector<std::array<int, NG + 1>> tri_nab,
              const AMP_MPI &comm,
              std::shared_ptr<Geometry::Geometry> geom = nullptr,
              std::vector<int> blockID                 = std::vector<int>() );


    //! Return a string with the mesh class name
    std::string meshClass() const override;


    //! Virtual function to copy the mesh (allows use to proply copy the derived class)
    std::unique_ptr<Mesh> clone() const override final;


    //! Check if two meshes are equal
    bool operator==( const Mesh &mesh ) const override;


    // Copy/move constructors
    TriangleMesh( const TriangleMesh & );
    TriangleMesh( TriangleMesh && ) = default;

    TriangleMesh &operator=( const TriangleMesh & ) = delete;
    TriangleMesh &operator=( TriangleMesh && )      = default;

    //! Deconstructor
    virtual ~TriangleMesh();


    /* Return the number of local element of the given type
     * \param type   Geometric type
     */
    size_t numLocalElements( const GeomType type ) const override final;


    /* Return the global number of elements of the given type
     * Note: depending on the mesh this routine may require global communication across the mesh.
     * \param type   Geometric type
     */
    size_t numGlobalElements( const GeomType type ) const override final;


    /* Return the number of ghost elements of the given type on the current processor
     * \param type   Geometric type
     */
    size_t numGhostElements( const GeomType type, const int gcw ) const override final;


    /**
     * \brief    Return an MeshIterator over the given geometric objects
     * \details  Return an MeshIterator over the given geometric objects
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    MeshIterator getIterator( const GeomType type, const int gcw = 0 ) const override final;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the surface
     * \details  Return an MeshIterator over the given geometric objects on the surface
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getSurfaceIterator( const GeomType type,
                                             const int gcw = 0 ) const override final;


    /**
     * \brief    Return the list of all boundary ID sets in the mesh
     * \details  Return the list of all boundary ID sets in the mesh
     * Note: depending on the mesh this routine may require global communication across the mesh.
     */
    std::vector<int> getBoundaryIDs() const override final;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the given boundary ID
     * set
     * \details  Return an MeshIterator over the given geometric objects on the given boundary ID
     * set
     * \param type   Geometric type to iterate over
     * \param id     Boundary id for the elements (example: sideset id)
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getBoundaryIDIterator( const GeomType type,
                                                const int id,
                                                const int gcw = 0 ) const override final;

    /**
     * \brief    Return the list of all boundary ID sets in the mesh
     * \details  Return the list of all boundary ID sets in the mesh
     * Note: depending on the mesh this routine may require global communication across the mesh.
     */
    std::vector<int> getBlockIDs() const override final;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the given block ID set
     * \details  Return an MeshIterator over the given geometric objects on the given block ID set
     * \param type   Geometric type to iterate over
     * \param id     Block id for the elements (example: block id in cubit, subdomain in libmesh)
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator
    getBlockIDIterator( const GeomType type, const int id, const int gcw = 0 ) const override final;


    /**
     * \brief    Return a mesh element given it's id.
     * \details  This function queries the mesh to get an element given the mesh id.
     *    This function is only required to return an element if the id is local.
     *    Ideally, this should be done in O(1) time, but the implementation is up to
     *    the underlying mesh.  The base class provides a basic implementation, but
     *    uses mesh iterators and requires O(N) time on the number of elements in the mesh.
     * \param id    Mesh element id we are requesting.
     */
    MeshElement getElement( const MeshElementID &id ) const override final;


    /**
     * \brief    Return the parent elements of the given mesh element
     * \details  This function queries the mesh to get an element given the mesh id,
     *    then returns the parent elements that have the element as a child
     * \param elem  Mesh element of interest
     * \param type  Element type of the parents requested
     */
    virtual std::vector<MeshElement> getElementParents( const MeshElement &elem,
                                                        const GeomType type ) const override final;

    /**
     * \brief    Is the mesh movable
     * \details  This function will check if the mesh can be displaced.
     * @return   enum indicating the extent the mesh can be moved
     */
    Movable isMeshMovable() const override { return Movable::Deform; };


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


    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    void writeRestart( int64_t fid ) const override;


protected:
    // Constructors
    TriangleMesh() = default;
    explicit TriangleMesh( std::shared_ptr<const MeshParameters> );
    explicit TriangleMesh( std::vector<std::array<double, NP>> vertices,
                           std::vector<std::array<int, NG + 1>> triangles,
                           std::vector<std::array<int, NG + 1>> tri_nab,
                           const AMP_MPI &comm,
                           std::shared_ptr<Geometry::Geometry> geom,
                           std::vector<int> block );
    void initialize();
    void initializeIterators();
    void initializeBoundingBox();

protected:
    // Create an iterator from a list
    MeshIterator createIterator( std::shared_ptr<std::vector<ElementID>> ) const;

    // Return the IDs of the elements composing the current element
    void getElementsIDs( const ElementID &id, const GeomType type, ElementID *IDs ) const;
    void getVerticies( const ElementID &id, ElementID *IDs ) const;

    // Return the IDs of the neighboring elements
    void getNeighborIDs( const ElementID &id, std::vector<ElementID> &IDs ) const;

    // Return the IDs of the parent elements
    std::pair<const ElementID *, const ElementID *> getElementParents( const ElementID &id,
                                                                       const GeomType type ) const;

    // Return a new element (user must delete)
    MeshElement *getElement2( const MeshElementID &id ) const;

    // Return the coordinated of the given vertex
    // Note: no error checking is done to make sure it is a valid vertex
    TriangleMesh::Point getPos( const ElementID &id ) const;

    // Check if the element is on the given boundry, block, etc
    bool isOnSurface( const ElementID &elemID ) const;
    bool isOnBoundary( const ElementID &elemID, int id ) const;
    bool isInBlock( const ElementID &elemID, int id ) const;
    static bool inIterator( const ElementID &id, const MeshIterator *it );

    // Friends
    friend TriangleMeshIterator<NG, NP, 0>;
    friend TriangleMeshIterator<NG, NP, 1>;
    friend TriangleMeshIterator<NG, NP, 2>;
    friend TriangleMeshIterator<NG, NP, 3>;
    friend TriangleMeshElement<NG, NP, 0>;
    friend TriangleMeshElement<NG, NP, 1>;
    friend TriangleMeshElement<NG, NP, 2>;
    friend TriangleMeshElement<NG, NP, 3>;


private: // Internal data
    // Store the locat start indicies
    std::array<size_t, 4> d_N_global;

    // Store the local triangle data
    typedef std::array<ElementID, NG + 1> NeighborIDs;
    std::vector<Point> d_vert;
    std::vector<Edge> d_edge;
    std::vector<Triangle> d_tri;
    std::vector<Tetrahedron> d_tet;
    std::vector<NeighborIDs> d_neighbors;
    std::vector<int> d_blockID;

    // Store the ghost data
    std::map<ElementID, Point> d_remote_vert;
    std::map<ElementID, Edge> d_remote_edge;
    std::map<ElementID, Triangle> d_remote_tri;
    std::map<ElementID, Tetrahedron> d_remote_tet;
    std::map<ElementID, NeighborIDs> d_remote_neighbors;
    std::map<ElementID, int> d_remote_blockID;

    // Store the parent data
    StoreCompressedList<ElementID> d_parents[NG][NG + 1];

    // Store children data
    std::vector<std::array<ElementID, 3>> d_tri_edge;
    std::vector<std::array<ElementID, 4>> d_tet_tri;
    std::vector<std::array<ElementID, 6>> d_tet_edge;

    // Store common iterators
    std::vector<int> d_block_ids, d_boundary_ids;
    using IteratorSet = std::array<MeshIterator, NG + 1>;
    std::vector<IteratorSet> d_iterators;                       // [gcw][type]
    std::vector<IteratorSet> d_surface_iterators;               // [gcw][type]
    std::vector<std::vector<IteratorSet>> d_boundary_iterators; // [id][gcw][type]
    std::vector<std::vector<IteratorSet>> d_block_iterators;    // [id][gcw][type]

    // Index indicating number of times the position has changed
    uint64_t d_pos_hash;
};


} // namespace AMP::Mesh


#endif
