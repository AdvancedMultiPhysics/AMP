#ifndef included_AMP_BoxMesh
#define included_AMP_BoxMesh

#include "AMP/geometry/Geometry.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/mesh/MeshIterator.h"
#include "AMP/utils/ArraySize.h"
#include "AMP/utils/Utilities.h"

namespace AMP::LinearAlgebra {
class Vector;
}

#include <array>
#include <map>
#include <memory>
#include <vector>


namespace AMP::Mesh {

class structuredMeshElement;
class structuredMeshIterator;


/**
 * \class BoxMesh
 * \brief A class used to represent a logically rectangular box mesh
 * \details  This class provides routines for creating and managing a logically
 *    rectangular mesh domain.  The mesh is described by the number of elements
 *    in each direction and may be periodic along any given direction.
 *    The database may specify some simple options to generate meshes:
\verbatim
   MeshName - Name of the mesh
   dim - Dimension of the mesh
   Generator - "cube", "circle, "cylinder", "tube", "sphere", "shell"
   Size - ndim array with the number of intervals in each direction.
          nx, ny, nz are the number of intervals in each direction, nr is the number of intervals
          in the radius, and nphi is the number of intervals in the asmuthal direction.
          cube (2d) - [ nx, ny      ]
          cube (3d) - [ nx, ny, nz  ]
          circle    - [ nr          ]
          cylinder  - [ nr, nz      ]
          tube      - [ nr, nphi, nz]
          sphere    - [ nr          ]
          shell     - [ nr, nphi    ]
   Range - Array specifying the physical size of the mesh.
          cube (2d) - [ x-min, x-max, y-min, y-max, z-min, z-max ]
          cube (3d) - [ x-min, x-max, y-min, y-max               ]
          circle    - [ r                                        ]
          cylinder  - [ r,     z_min, z_max                      ]
          tube      - [ r_min, r_max, z_min, z_max               ]
          sphere    - [ r                                        ]
          shell     - [ r_min, r_max                             ]
   Periodic - Are any dimensions periodic (optional)
          cube (2d) - [ x_dir, y_dir ]
          cube (3d) - [ x_dir, y_dir, z_dir ]
          circle    - Not supported
          cylinder  - [ z_dir ]
          tube      - [ z_dir ]
          sphere    - Not supported
          shell     - Not supported
   GCW -  The maximum ghost cell width to support (optional, default is 1)
   x_offset - Offset in x-direction (optional)
   y_offset - Offset in y-direction (optional)
   z_offset - Offset in z-direction (optional)
\endverbatim
 */
class BoxMesh : public AMP::Mesh::Mesh
{
public:
    /**
     * \class Box
     * \brief Structure to identify a logical box
     * \details  This class contains a logical box
     */
    class Box
    {
    public:
        /**
         * \brief   Default constructor
         * \details  The default constructor
         * \param ifirst  First x-coordinate
         * \param ilast   Last x-coordinate
         * \param jfirst  First y-coordinate
         * \param jlast   Last y-coordinate
         * \param kfirst  First z-coordinate
         * \param klast   Last z-coordinate
         */
        constexpr explicit Box(
            int ifirst, int ilast, int jfirst = 0, int jlast = 0, int kfirst = 0, int klast = 0 );
        constexpr Box();                  //!< Empty constructor
        constexpr ArraySize size() const; //!< Return the size of the box
        int first[3];                     //!< Starting element
        int last[3];                      //!< Ending element
        std::string print() const;

    private:
    };

    /**
     * \class MeshElementIndex
     * \brief Structure to uniquely identify an element
     * \details  This class help convert between logical coordinates and the mesh element of
     * interest
     */
    class MeshElementIndex
    {
    public:
        //! Empty constructor
        constexpr MeshElementIndex();
        /**
         * \brief   Default constructor
         * \details  The default constructor
         * \param type  Element type
         * \param side  Side of the parent element (ignored if it is the parent or vertex)
         * \param x     Logical coordinate of the element
         * \param y     Logical coordinate of the element
         * \param z     Logical coordinate of the element
         */
        constexpr explicit MeshElementIndex(
            GeomType type, uint8_t side, int x, int y = 0, int z = 0 );
        constexpr void reset();
        constexpr void reset( GeomType type, uint8_t side, int x, int y = 0, int z = 0 );
        constexpr bool isNull() const { return d_side == 255; }
        constexpr bool operator==( const MeshElementIndex &rhs ) const; //!< Operator ==
        constexpr bool operator!=( const MeshElementIndex &rhs ) const; //!< Operator !=
        constexpr bool operator>( const MeshElementIndex &rhs ) const;  //!< Operator >
        constexpr bool operator>=( const MeshElementIndex &rhs ) const; //!< Operator >=
        constexpr bool operator<( const MeshElementIndex &rhs ) const;  //!< Operator <
        constexpr bool operator<=( const MeshElementIndex &rhs ) const; //!< Operator <=
        constexpr auto index() const { return d_index; }
        constexpr int index( int d ) const { return d_index[d]; }
        constexpr int &index( int d ) { return d_index[d]; }
        constexpr GeomType type() const { return static_cast<GeomType>( d_type ); }
        constexpr uint8_t side() const { return d_side; }
        static constexpr size_t numElements( const MeshElementIndex &first,
                                             const MeshElementIndex &last );
        std::string print() const;

    private:
        uint8_t d_type;             //!<  Mesh element type
        uint8_t d_side;             //!<  Are we dealing with x, y, or z faces/edges
        std::array<int, 3> d_index; //!<  Global x, y, z index (may be negative if periodic)
        friend class BoxMesh;
        friend class structuredMeshElement;
    };

    /**
     * \class MeshElementIndexIterator
     * \brief Iterator over a box
     * \details  This class will iterator over a MeshElementIndex box
     */
    class MeshElementIndexIterator final
    {
    public: // iterator_traits
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = MeshElementIndex;
        using difference_type   = ptrdiff_t;
        using pointer           = const MeshElementIndex *;
        using reference         = const MeshElementIndex &;

    public:
        MeshElementIndexIterator() = default;
        MeshElementIndexIterator( const MeshElementIndex &first,
                                  const MeshElementIndex &last,
                                  const AMP::Mesh::BoxMesh *mesh,
                                  size_t pos = 0 );
        MeshElementIndexIterator &operator++();
        MeshElementIndexIterator &operator--();
        MeshElementIndexIterator &operator+=( int N );
        MeshElementIndexIterator &operator+=( const MeshElementIndexIterator &it );
        MeshElementIndexIterator &operator[]( int );
        MeshElementIndexIterator begin() const;
        MeshElementIndexIterator end() const;
        MeshElementIndex operator*() const;
        bool operator==( const MeshElementIndexIterator &rhs ) const;
        bool operator!=( const MeshElementIndexIterator &rhs ) const;
        bool empty() const { return d_size == 0; }
        void set( uint32_t i ) { d_pos = i; }
        size_t size() const { return d_size; }
        size_t position() const { return d_pos; }
        inline auto first() const { return d_first; }
        inline auto last() const { return d_last; }

    private:
        // Data members
        bool d_checkBoundary             = false;
        std::array<bool, 3> d_isPeriodic = { false, false, false };
        std::array<int, 3> d_globalSize  = { 0, 0, 0 };
        uint32_t d_pos                   = { 0 };
        uint32_t d_size                  = { 0 };
        MeshElementIndex d_first;
        MeshElementIndex d_last;
    };


public:
    /**
     * \brief Read in mesh files, partition domain, and prepare environment for simulation
     * \details  For trivial parallelism, this method reads in the meshes on each processor.  Each
     * processor contains a piece of each mesh.  For massive parallelism, each mesh is on its own
     * communicator.  As such, some math libraries must be initialized accordingly.
     * \param params  Parameters for constructing a mesh from an input database
     */
    static std::shared_ptr<BoxMesh> generate( std::shared_ptr<const MeshParameters> params );


    //! Virtual function to copy the mesh (allows use to proply copy the derived class)
    std::unique_ptr<Mesh> clone() const override = 0;


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
     * \brief   Estimate the number of elements in the mesh
     * \details  This function will estimate the number of elements in the mesh.
     *   This is used so that we can properly balance the meshes across multiple processors.
     *   Ideally this should be both an accurate estimate and very fast.  It should not require
     *   any communication and should not have to actually load a mesh.
     * \param params Parameters for constructing a mesh from an input database
     */
    static ArraySize estimateLogicalMeshSize( std::shared_ptr<const MeshParameters> params );


    /**
     * \brief   Return the maximum number of processors that can be used with the mesh
     * \details  This function will return the maximum number of processors that can
     *   be used with the mesh.
     * \param params Parameters for constructing a mesh from an input database
     */
    static size_t maxProcs( std::shared_ptr<const MeshParameters> params );


    //! Destructor
    virtual ~BoxMesh();


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
     * \brief    Return an MeshIterator constructed through a set operation of two other
     * MeshIterators.
     * \details  Return an MeshIterator constructed through a set operation of two other
     * MeshIterators.
     * \param OP Set operation to perform.
     *           SetOP::Union - Perform a union of the iterators ( A U B )
     *           SetOP::Intersection - Perform an intersection of the iterators ( A n B )
     *           SetOP::Complement - Perform a compliment of the iterators ( A - B )
     * \param A  Pointer to MeshIterator A
     * \param B  Pointer to MeshIterator B
     */
    static MeshIterator getIterator( SetOP OP, const MeshIterator &A, const MeshIterator &B );


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


    //! Return the global logical box
    inline Box getGlobalBox( int gcw = 0 ) const;


    //! Return the local logical box
    inline Box getLocalBox( int gcw = 0 ) const;


    //! Return the bool vector indicating which directions are periodic
    inline std::vector<bool> periodic() const;

    //! Return the size of the mesh
    inline std::vector<size_t> size() const;

    //! Return the number of blocks
    inline std::vector<size_t> numBlocks() const;


    //! Check if two meshes are equal
    bool operator==( const Mesh &mesh ) const override;


    /**
     * \brief    Return a mesh element given it's id.
     * \details  This function queries the mesh to get an element given the mesh id.
     *    This function is only required to return an element if the id is local.
     *    Ideally, this should be done in O(1) time, but the implementation is up to
     *    the underlying mesh.  The base class provides a basic implementation, but
     *    uses mesh iterators and requires O(N) time on the number of elements in the mesh.
     * \param index    Mesh element index we are requesting.
     */
    structuredMeshElement getElement( const MeshElementIndex &index ) const;

    /**
     * \brief    Return a mesh element's coordinates given it's id.
     * \details  This function queries the mesh to get an element's coordinates given the mesh id.
     *    Ideally, this should be done in O(1) time, but the implementation is up to
     *    the underlying mesh.
     * \param[in] index     Mesh element index we are requesting.
     * \param[out] pos      Mesh element coordinates
     */
    virtual void coord( const MeshElementIndex &index, double *pos ) const = 0;

    //! Check if the element is on the surface
    virtual bool isOnSurface( const MeshElementIndex &index ) const;

    //! Check if the element is on the given boundary id
    virtual bool isOnBoundary( const MeshElementIndex &index, int id ) const;


public: // BoxMesh specific functionality
    /**
     * \brief    Return the logical coordinates
     * \details  This function queries the mesh to get the logical coordinates in [0,1]
     *     from the physical coordinates.  Not all meshes support this functionallity.
     * \param[in] x         Physical coordinates
     * @return              Returns the logical coordinates
     */
    virtual AMP::Geometry::Point physicalToLogical( const AMP::Geometry::Point &x ) const = 0;

    /**
     * \brief    Return the element containing the point
     * \details  This function queries the mesh to get the element index given a logical point
     * \param[in] x         Logical coordinates in [0,1]
     * \param type          Geometric type to get
     * @return              Returns the element containing the logical point.
     *                      Note: it will return a null index (isNull) if no element of
     *                      the given type contains the point.
     */
    MeshElementIndex getElementFromLogical( const AMP::Geometry::Point &x, GeomType type ) const;

    /**
     * \brief    Return the element containing the point
     * \details  This function queries the mesh to get the element index given a physical point.
     *    This functionallity requires physicalToLogical which may not be supported by all meshes.
     * \param[in] x         Physical coordinates
     * \param type          Geometric type to get
     * @return              Returns the element containing the physical point.
     *                      Note: it will return a null index (isNull) if no element of
     *                      the given type contains the point.
     */
    MeshElementIndex getElementFromPhysical( const AMP::Geometry::Point &x, GeomType type ) const;

    //! Get the rank that owns the element
    inline int getRank( const MeshElementIndex &id ) const;

    //! Convert the MeshElementIndex to the MeshElementID
    inline MeshElementID convert( const MeshElementIndex &id ) const;

    //! Convert the MeshElementID to the MeshElementIndex
    inline MeshElementIndex convert( const MeshElementID &id ) const;


    //! Create an ArrayVector over the mesh
    std::shared_ptr<AMP::LinearAlgebra::Vector> createVector( const std::string &name,
                                                              int gcw = 0 );


public: // Convenience typedef
    typedef AMP::Utilities::stackVector<std::pair<MeshElementIndex, MeshElementIndex>, 32>
        ElementBlocks;


public: // Advanced functions
    // Get the surface id for a given surface
    int getSurfaceID( int surface ) const;

    // Get the surface set for a given surface/type
    ElementBlocks getSurface( int surface, GeomType type ) const;

    // Helper function to return the indices of the local block owned by the given processor
    inline std::array<int, 6> getLocalBlock( int rank ) const;

    // Helper functions to identify the iterator blocks
    ElementBlocks
    getIteratorRange( std::array<int, 6> range, const GeomType type, const int gcw ) const;
    ElementBlocks intersect( const ElementBlocks &v1, const ElementBlocks &v2 ) const;

    // Helper function to create an iterator from an ElementBlocks list
    MeshIterator createIterator( const ElementBlocks &list ) const;


protected:
    // Constructor
    explicit BoxMesh( std::shared_ptr<const MeshParameters> );
    explicit BoxMesh( const BoxMesh & );
    BoxMesh &operator=( const BoxMesh & ) = delete;

    // Function to create the load balancing
    static void loadBalance( std::array<int, 3> size,
                             int N_procs,
                             std::vector<int> *startIndex,
                             std::vector<int> minSize = {} );

    // Function to initialize the mesh data once the logical mesh info has been created
    void initialize( const std::array<int, 3> &size,
                     const std::array<int, 6> &ids,
                     const std::vector<int> &minSize = {} );

    // Function to finalize the mesh data once the coordinates have been set
    void finalize( const std::string &name, const std::vector<double> &displacement );

    // Function to finalize the mesh data once the coordinates have been set
    std::vector<double> getDisplacement( std::shared_ptr<const AMP::Database> db );

    // Function to finalize the mesh data once the coordinates have been set
    virtual void createBoundingBox();

    // Helper function to fill the node data for a uniform cartesian mesh
    static void fillCartesianNodes( int dim,
                                    const int *globalSize,
                                    const double *range,
                                    const std::vector<MeshElementIndex> &index,
                                    std::vector<double> *coord );

    // Helper function to check if an element is on a given side
    bool onSide( const MeshElementIndex &index, int d, int s ) const;

    // Get the surface (including mapped surfaces)
    ElementBlocks getSurface2( int surface, GeomType type ) const;

    // Helper function to map points on the boundary
    template<uint8_t NDIM>
    std::vector<MeshElementIndex> createMap( const std::vector<MeshElementIndex> ) const;
    void createMaps();

protected: // Write/read restart data
    void writeRestart( int64_t ) const override;
    BoxMesh( int64_t, AMP::IO::RestartManager * );


protected: // Friend functions to access protected functions
    friend class structuredMeshElement;
    friend class structuredMeshIterator;
    typedef std::vector<MeshElementIndex> IndexList;
    typedef std::shared_ptr<IndexList> ListPtr;
    typedef std::tuple<IndexList, IndexList> SurfaceMapStruct;


protected:                                     // Internal data
    const int d_rank, d_size;                  // Cached values for the rank and size
    int d_blockID;                             // Block id for the mesh
    const std::array<int, 3> d_globalSize;     // The size of the logical domain in each direction
    const std::array<int, 3> d_numBlocks;      // The number of local box in each direction
    const std::vector<int> d_startIndex[3];    // The first index for each block
    const std::vector<int> d_endIndex[3];      // The end index (last=1) for each block
    const std::array<int, 6> d_localIndex;     // Local index range (cached for performance)
    const std::array<int, 3> d_indexSize;      // Local index size (local box + 2) (cached)
    const std::array<int, 6> d_surfaceId;      // ID of each surface (if any, -1 if not)
    mutable std::vector<ListPtr> d_surface[4]; // List of surface elements
    mutable std::vector<ListPtr> d_bnd[4][6];  // List of boundary elements
    SurfaceMapStruct d_surfaceMaps[3];         // Surface maps

protected:
    BoxMesh();
};


// Function to write an index to std::ostream
std::ostream &operator<<( std::ostream &out, const BoxMesh::MeshElementIndex &x );


} // namespace AMP::Mesh

#include "AMP/mesh/structured/BoxMesh.inline.h"


#endif
