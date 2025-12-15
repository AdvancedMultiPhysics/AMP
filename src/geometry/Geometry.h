#ifndef included_AMP_Geometry
#define included_AMP_Geometry

#include "AMP/mesh/MeshID.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/MeshPoint.h"

#include <memory>
#include <vector>


namespace AMP::IO {
class RestartManager;
}


namespace AMP::Geometry {


using Point = AMP::Mesh::MeshPoint<double>;


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
class Geometry
{
public:
    //! Destructor
    virtual ~Geometry() = default;

    //! Get the name of the geometry
    virtual std::string getName() const = 0;

    /**
     * \brief    Get the number of dimensions for the object
     * \details  This function returns the number of physical dimensions for the geometry
     * @return      Returns the number of physical dimensions
     */
    inline uint8_t getDim() const { return d_physicalDim; }


    /**
     * \brief    Get the geometric type for the geometry
     * \details  This function returns the largest geometric type supported by the geometry
     * @return      Returns the geometric dimensions
     */
    virtual AMP::Mesh::GeomType getGeomType() const;


    /**
     * \brief    Is the object convex
     * \details  Check if the geometric object is convex
     * @return      Returns true if the object is convex
     */
    virtual bool isConvex() const = 0;

    /**
     * \brief    Calculate the nearest point on the surface
     * \details  This function computes the nearest point on the surface
     * \param[in] pos   Current position of ray
     * @return          Returns the nearest surface point
     */
    virtual Point nearest( const Point &pos ) const = 0;

    /**
     * \brief    Calculate the distance to the object given a ray
     * \details  This function computes the distance to the object given a ray.
     *     If the ray is inside the object, this distance is negitive.  If the
     *     ray will never intersect the object, this distance is inf.
     * \param[in] pos   Current position of ray
     * \param[in] dir   Direction of ray (should be normalized for most uses)
     * @return          Returns the distance to the nearest surface
     *                  (intersection = pos + dir*distance)
     */
    virtual double distance( const Point &pos, const Point &dir ) const = 0;

    /**
     * \brief    Is the point in the geometry
     * \details  This function checks if the ray is in the geometry.  If it is on the surface,
     *     it will return true.
     * \param[in] pos   Current position
     * @return          Returns true if the point is inside the geometry (or on the surface)
     */
    virtual bool inside( const Point &pos ) const = 0;

    /**
     * \brief    Get the number of surfaces
     * \details     This function will return the number of unique surfaces
     * @return          Returns the number of unique surfaces
     */
    virtual int NSurface() const = 0;

    /**
     * \brief    Get the surface id
     * \details     This function will return the surface id closest to the point
     * \param[in] x     Current position
     * @return          Returns the surface id (0:NSurface-1)
     */
    virtual int surface( const Point &x ) const = 0;

    /**
     * \brief    Return the outward normal to a surface
     * \details  This function will return the surface id and outward normal to the surface at the
     given point
     * \param[in] x     Current position
     * @return          Returns the surface normal

     */
    virtual Point surfaceNorm( const Point &x ) const = 0;

    /**
     * \brief    Return the centroid
     * \details  This function will return the centroid of the object.
     *   The centroid is equivalent to the center of mass of object of uniform density.
     * @return          Returns the physical coordinates
     */
    virtual Point centroid() const = 0;

    /**
     * \brief    Return the bounding box
     * \details  This function will return the bounding box of the object
     * @return          Returns the bounding box [lb,ub]
     */
    virtual std::pair<Point, Point> box() const = 0;

    /**
     * \brief    Return the volume
     * \details  This function will return the interior volume of the object
     * @return          Returns the volume
     */
    virtual double volume() const = 0;

    /**
     * \brief    Displace the entire geometry
     * \details  This function will displace the entire geometry by a scalar value.
     *   The displacement vector should be the size of the physical dimension.
     * \param[in] x    Displacement vector
     */
    virtual void displace( const double *x ) = 0;

    //! Clone the object
    virtual std::unique_ptr<AMP::Geometry::Geometry> clone() const = 0;

    //! Check if two geometries are equal
    virtual bool operator==( const Geometry &rhs ) const = 0;

    //! Check if two geometries are not equal
    inline bool operator!=( const Geometry &rhs ) const { return !operator==( rhs ); }


public: // Write/read restart data
    //! Return a unique hash id
    virtual uint64_t getID() const;

    /**
     * \brief    Register child objects
     * \details  This function register child objects if necessary
     * \param manager    Restart manager
     */
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    virtual void writeRestart( int64_t fid ) const;

protected:
    //! Initialize the base class from file
    Geometry( int64_t fid );


public:
    /**
     * \brief   Create a geometry
     * \details  This function will create a geometry based on
     *   the input database.
     * \param[in] db    Parameters for constructing a geometry from an input database
     */
    static std::shared_ptr<AMP::Geometry::Geometry>
    buildGeometry( std::shared_ptr<const AMP::Database> db );


protected:
    //!  Empty constructor for the base class
    Geometry( int ndim ) : d_physicalDim( ndim ) {}

    // Delete copy constructors
    Geometry( Geometry && )      = delete;
    Geometry( const Geometry & ) = default;
    Geometry &operator=( Geometry && ) = delete;
    Geometry &operator=( const Geometry & ) = delete;


protected: // Internal data
    const uint8_t d_physicalDim;
};


} // namespace AMP::Geometry

#endif
