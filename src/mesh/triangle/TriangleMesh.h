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


template<uint8_t NG, uint8_t TYPE>
class TriangleMeshIterator;
template<uint8_t NG, uint8_t TYPE>
class TriangleMeshElement;


// Class to store vertex data
template<class TYPE, size_t N>
class StoreTriData final
{
public:
    StoreTriData() = default;
    StoreTriData( std::vector<std::array<TYPE, N>> x,
                  std::vector<int> offset,
                  int rank,
                  GeomType type );
    inline int size() const { return d_x.size(); }
    inline int start() const { return d_start; }
    inline int end() const { return d_end; }
    inline int start( int r ) const { return d_offset[r]; }
    inline int end( int r ) const { return d_offset[r + 1]; }
    int rank( int i ) const;
    inline auto data() { return d_x.data(); }
    inline const auto data() const { return d_x.data(); }
    inline auto &offset() const { return d_offset; }
    inline int index( const ElementID &id ) const
    {
        AMP_DEBUG_ASSERT( id.type() == d_type );
        int rank = id.owner_rank();
        return d_offset[rank] + id.local_id();
    }
    ElementID getID( int i ) const;
    int find( const std::array<TYPE, N> &x ) const;
    inline auto &operator[]( int i ) { return d_x[i]; }
    inline auto &operator[]( int i ) const { return d_x[i]; }
    inline auto &operator[]( const ElementID &id ) { return d_x[index( id )]; }
    inline auto &operator[]( const ElementID &id ) const { return d_x[index( id )]; }
    inline auto &operator()() { return d_x; }
    inline const auto &operator()() const { return d_x; }

private:
    GeomType d_type;
    int d_start;
    int d_end;
    int d_rank;
    std::vector<int> d_offset;
    std::vector<std::array<TYPE, N>> d_x;
};


// Class to store parent data
template<class TYPE>
class StoreCompressedList
{
public:
    inline StoreCompressedList() {}
    inline StoreCompressedList( size_t N )
        : StoreCompressedList( std::vector<std::vector<TYPE>>( N ) )
    {
    }
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
template<uint8_t NG>
class TriangleMesh : public AMP::Mesh::Mesh
{
public: // Convenience typedefs
    static_assert( NG <= 3, "Not programmed for higher dimensions yet" );
    typedef std::array<double, 3> Point;
    typedef std::array<int, 2> Edge;
    typedef std::array<int, 3> Face;
    typedef std::array<int, NG + 1> TRI;
    using IteratorSet = std::array<MeshIterator, NG + 1>;
    typedef StoreCompressedList<ElementID> ElementList;


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
    template<uint8_t NP>
    static std::shared_ptr<TriangleMesh<NG>>
    generate( const std::vector<std::array<std::array<double, NP>, NG + 1>> &triangles,
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
    template<uint8_t NP>
    static std::shared_ptr<TriangleMesh<NG>>
    generate( const std::vector<std::array<double, NP>> &vertices,
              const std::vector<TRI> &triangles,
              const std::vector<TRI> &tri_nab,
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
    explicit TriangleMesh( int NP,
                           std::vector<Point> vertices,
                           std::vector<TRI> triangles,
                           std::vector<TRI> tri_nab,
                           const AMP_MPI &comm,
                           std::shared_ptr<Geometry::Geometry> geom,
                           std::vector<int> block,
                           int max_gcw = 2 );
    void initialize();
    void initializeIterators();
    void initializeBoundingBox();
    std::vector<IteratorSet> createBlockIterators( int block );
    void createSurfaceIterators();


public:
    // Create an iterator from a list
    MeshIterator createIterator( std::shared_ptr<std::vector<ElementID>> ) const;
    MeshIterator createIterator( GeomType type, int gcw ) const;

    // Return the IDs of the elements composing the current element
    void getElementsIDs( const ElementID &id, const GeomType type, ElementID *IDs ) const;
    void getVertexCoord( const ElementID &id, std::array<double, 3> *x ) const;

    // Return the IDs of the neighboring elements
    void getNeighborIDs( const ElementID &id, std::vector<ElementID> &IDs ) const;

    // Return the IDs of the parent elements
    std::pair<const ElementID *, const ElementID *> getElementParents( const ElementID &id,
                                                                       const GeomType type ) const;

    // Return a new element (user must delete)
    MeshElement *getElement2( const MeshElementID &id ) const;


    // Check if the element is on the given boundry, block, etc
    bool isOnSurface( const ElementID &elemID ) const;
    bool isOnBoundary( const ElementID &elemID, int id ) const;
    bool isInBlock( const ElementID &elemID, int id ) const;
    static bool inIterator( const ElementID &id, const MeshIterator *it );

    template<uint8_t TYPE>
    std::array<int, TYPE + 1> getElem( const ElementID &id ) const;
    template<uint8_t TYPE>
    ElementID getID( const std::array<int, TYPE + 1> &id ) const;

    // Friends
    friend TriangleMeshElement<NG, 0>;
    friend TriangleMeshElement<NG, 1>;
    friend TriangleMeshElement<NG, 2>;
    friend TriangleMeshElement<NG, 3>;


private:
    void loadBalance( const std::vector<Point> &vertices,
                      const std::vector<TRI> &tri,
                      const std::vector<TRI> &tri_nab,
                      const std::vector<int> &block );
    void buildChildren();
    ElementList computeNodeParents( int parentType );
    ElementList getParents( int childType, int parentType );

private:
    std::array<size_t, 4> d_N_global;          //!< The number of global elements
    StoreTriData<double, 3> d_vertex;          //!< Store the global coordinates
    StoreTriData<int, NG + 1> d_globalTri;     //!< Store the global triangles
    std::vector<TRI> d_globalNab;              //!< Store the global triangle neighbors
    std::vector<int> d_blockID;                //!< The block id index for each triangle
    std::vector<std::vector<int>> d_remoteTri; //!< The unique ghost triangles for each gcw
    StoreTriData<int, 2> d_childEdge;          //!< The list of local children edges
    StoreTriData<int, 3> d_childFace;          //!< The list of local children faces
    ElementList d_parents[NG][NG + 1];         //!< Parent data
    std::vector<int> d_block_ids;              //!< The global list of block ids
    std::vector<int> d_boundary_ids;           //!< The global list of boundary ids
    std::vector<bool> d_isSurface[NG];         //!< Global list of surface elements
    std::vector<IteratorSet> d_iterators;      //!< [gcw][type]
    std::vector<IteratorSet> d_surface_it;     //!< [gcw][type]
    std::vector<std::vector<IteratorSet>> d_boundary_it; //!< [id][gcw][type]
    std::vector<std::vector<IteratorSet>> d_block_it;    //!< [id][gcw][type]
    uint64_t d_pos_hash; //!< Index indicating number of times the position has changed
};


} // namespace AMP::Mesh


#endif
