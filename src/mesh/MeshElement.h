#ifndef included_AMP_MeshElement
#define included_AMP_MeshElement

#include "AMP/mesh/MeshID.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/typeid.h"

#include <memory>
#include <vector>


namespace AMP::Mesh {


class Mesh;


/**
 * \class Mesh
 * \brief A class used to define a mesh element
 * \details  This class provides routines for accessing and using a mesh element.
 * A mesh element can be thought of as the smallest unit of a mesh.  It is of a type
 * of GeomType.  It contains the composing pieces of the element
 */
class MeshElement
{
public:
    using ElementList = std::vector<std::unique_ptr<MeshElement>>;

public: // non-virtual functions
    //! Empty constructor for a MeshElement
    MeshElement() = default;

    //! Is the mesh element null
    bool isNull() const;

    //! Return the element type
    inline GeomType elementType() const { return globalID().type(); }

    // Overload operators
    inline bool operator==( const MeshElement &rhs ) const { return globalID() == rhs.globalID(); }
    inline bool operator!=( const MeshElement &rhs ) const { return globalID() != rhs.globalID(); }
    inline bool operator<( const MeshElement &rhs ) const { return globalID() < rhs.globalID(); }
    inline bool operator>( const MeshElement &rhs ) const { return globalID() > rhs.globalID(); }
    inline bool operator<=( const MeshElement &rhs ) const { return globalID() <= rhs.globalID(); }
    inline bool operator>=( const MeshElement &rhs ) const { return globalID() >= rhs.globalID(); }
    inline bool operator==( const MeshElementID &rhs ) const { return globalID() == rhs; }
    inline bool operator!=( const MeshElementID &rhs ) const { return globalID() != rhs; }
    inline bool operator<( const MeshElementID &rhs ) const { return globalID() < rhs; }
    inline bool operator>( const MeshElementID &rhs ) const { return globalID() > rhs; }
    inline bool operator<=( const MeshElementID &rhs ) const { return globalID() <= rhs; }
    inline bool operator>=( const MeshElementID &rhs ) const { return globalID() >= rhs; }

    //! Return the elements composing the current element
    ElementList getElements( const GeomType type ) const;

    /**
     *  Return the elements neighboring the current element.
     *  One neighbor is returned for each side of the element.
     *  If the side is on the surface, then it's neighbor is null.
     *  For Verticies, a list of all vertices that share an element is returned.
     *  This list is in unsorted order.
     */
    ElementList getNeighbors() const;

    /**
     * \brief     Return the coordinate of the vertex
     * \details   This function returns the coordinates of the vertex
     *   in the given direction (only applies to vertices).
     * \param i     The direction requested.  Equivalent to coord()[i]
     */
    inline double coord( int i ) const { return coord()[i]; }


    /**
     * \brief     Return true if the element contains the point
     * \details   This function checks if the given point is inside or
     *   within TOL of the given element.  If the current element is a vertex,
     *   this function checks if the point is with TOL of the vertex.
     * \param pos   The coordinates of the point to check.
     * \param TOL   The tolerance to use for the computation.
     */
    bool containsPoint( const std::vector<double> &pos, double TOL = 1e-12 ) const;


    /**
     * \brief     Print info about the element
     * \details   This function returns debug info about the element
     * \param indent    The number of spaces to indent new lines
     */
    std::string print( uint8_t indent = 0 ) const;


public: // Virtual functions
    //! Return the typeID of the underlying element
    virtual typeID getTypeID() const;

    //! Return the unique global ID of the element
    virtual MeshElementID globalID() const;

    //! Return the owner rank according to AMP_COMM_WORLD
    unsigned int globalOwnerRank( const Mesh &mesh ) const;

    //! Return the element class
    virtual std::string elementClass() const;

    //! Return the coordinates of the vertex (only applies to vertices)
    virtual Point coord() const;

    /**
     * \brief     Return the centroid of the element
     * \details   This function returns the centroid of the element.  The
     *   centroid is defined as the average of the coordinates of the vertices.
     *   The centroid of a vertex is the vertex and will return the same result as coord().
     */
    virtual Point centroid() const;

    //! Return the volume of the current element (does not apply to vertices)
    virtual double volume() const;

    //! Return the normal to the current element (does not apply to all elements)
    virtual Point norm() const;

    /**
     * \brief     Return true if the element contains the point
     * \details   This function checks if the given point is inside or
     *   within TOL of the given element.  If the current element is a vertex,
     *   this function checks if the point is with TOL of the vertex.
     * \param pos   The coordinates of the point to check.
     * \param TOL   The tolerance to use for the computation.
     */
    virtual bool containsPoint( const Point &pos, double TOL = 1e-12 ) const;

    /**
     * \brief    Calculate the nearest point on the element
     * \details  This function computes nearest point on/in the element to the given point
     * \param[in] pos   Current position of the point
     */
    virtual Point nearest( const Point &pos ) const;

    /**
     * \brief    Calculate the distance to the element given a ray
     * \details  This function computes the distance to the element given a ray.
     *     If the ray will never intersect the element, this distance is inf.
     * \param[in] pos   Current position of ray
     * \param[in] dir   Direction of ray (should be normalized for most uses)
     * @return          Returns the distance to the element surface
     */
    virtual double distance( const Point &pos, const Point &dir ) const;

    //! Check if the element is on the surface
    virtual bool isOnSurface() const;

    /**
     * \brief     Check if the current element is on the given boundary
     * \details   Check if the current element is on the boundary specified by the given id
     * \param id  The boundary id to check
     */
    virtual bool isOnBoundary( int id ) const;

    /**
     * \brief     Check if the current element is in the given block
     * \details   Check if the current element is in the block specified by the given id
     * \param id  The block id to check
     */
    virtual bool isInBlock( int id ) const;


public: // Advanced functions
    //! Return the elements composing the current element
    virtual void getElements( const GeomType type, ElementList &elements ) const;

    //! Return the IDs of the elements composing the current element
    virtual int getElementsID( const GeomType type, MeshElementID *ID ) const;

    /**
     *  Return the elements neighboring the current element.
     *  One neighbor is returned for each side of the element.
     *  If the side is on the surface, then it's neighbor is null.
     *  For Verticies, a list of all vertices that share an element is returned.
     *  This list is in unsorted order.
     */
    virtual void getNeighbors( ElementList &neighbors ) const;

    /**
     *  Return the vertex coordinates composing the current element.
     *  For a node, this will return the equivalent to coord().
     *  For an element, this will return the equivalent to getElements(Vertex)->coord().
     *  Note the default implementation will call getElements followed by coord,
     *  derived classes may implement a more efficient alternative
     */
    virtual void getVertices( std::vector<Point> &vertices ) const;

    /**
     *  Return the vertex coordinates for all neighboring elements excluding
     *  verticies composing the current element.
     *  Note the default implementation will call getNeighbors followed by getVertices,
     *  derived classes may implement a more efficient alternative
     */
    virtual void getNeighborVertices( std::vector<Point> &vertices ) const;

    //! Clone the element
    virtual std::unique_ptr<MeshElement> clone() const;

    //! Destructor for a MeshElement
    virtual ~MeshElement() = default;


protected: // Constructors
    MeshElement( const MeshElement & )            = default;
    MeshElement( MeshElement && )                 = default;
    MeshElement &operator=( const MeshElement & ) = default;
    MeshElement &operator=( MeshElement && )      = default;
};


} // namespace AMP::Mesh


#endif
