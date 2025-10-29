#ifndef included_AMP_LogicalGeometry
#define included_AMP_LogicalGeometry

#include "AMP/geometry/Geometry.h"
#include "AMP/utils/ArraySize.h"


namespace AMP::Geometry {


/**
 * \class LogicalGeometry
 * \brief A class used to abstract away logical geometry based operations
 * \details  This class provides routines for reading, accessing and writing logical geometries.
 */
class LogicalGeometry : public Geometry
{
public:
    //! Destructor
    virtual ~LogicalGeometry() = default;

    /**
     * \brief    Return the logical position
     * \details  This function will return logical coordinate given the physical coordinates
     * \param[in] x     Physical coordinate of the point
     * @return          Returns the logical coordinates
     */
    virtual Point logical( const Point &x ) const = 0;

    /**
     * \brief    Return the physical position
     * \details  This function will return physical coordinate given the logical coordinates
     * \param[in] x     Logical coordinate of the point
     * @return          Returns the physical coordinates
     */
    virtual Point physical( const Point &x ) const = 0;

    /**
     * \brief    Return the number of logical dimensions
     * \details  This function will return the number of logical dimensions
     *    of the underlying geometry.  If the geometry is not logically rectangular
     *    this function should return 0.
     */
    inline uint8_t getLogicalDim() const { return d_logicalDim; }

    /**
     * \brief    Return the logical grid size
     * \details  This function will return the dimensions of a logical grid
     *    given a size that makes sense for the object.
     * \param[in] x     Input size
     * @return          Return the logical grid size
     */
    virtual ArraySize getLogicalGridSize( const ArraySize &x ) const = 0;

    /**
     * \brief    Return the logical grid size
     * \details  This function will return the dimensions of a logical grid
     *    given a desired resolution
     * \param[in] res   Desired resolution
     * @return          Return the logical grid size
     */
    virtual ArraySize getLogicalGridSize( const std::vector<double> &res ) const = 0;

    /**
     * \brief    Return the logical grid periodic dimensions
     * \details  This function will return a vector indicating which logical grid
     *    dimensions are periodic.
     *    Note: The returned array may be larger than the number of dimensions
     * @return          Return the periodic dimensions
     */
    std::array<bool, 3> getPeriodicDim() const;

    /**
     * \brief    Return the surface ids for the logical boundaries
     * \details  This function will return the surface ids for each logical boundary.
     *    If a logical boundary is periodic, it will return -1.
     *    If a logical boundary maps to another point on the boundary, it will return -2.
     *    If a logical boundary is not periodic, is not physical, and does not map to
     *        another point on the boundary (i.e. unused dimensions), it will return -3.
     *    Note: The returned array may be larger than the number of dimensions
     * @return          Return the logical boundary ids (2*logicalDim)
     */
    inline const auto &getLogicalSurfaceIds() const { return d_ids; }

    /**
     * \brief    Get the geometric type for the geometry
     * \details  This function returns the largest geometric type supported by the geometry
     * @return      Returns the geometric dimensions
     */
    AMP::Mesh::GeomType getGeomType() const override final;


public: // Restart functions
    void writeRestart( int64_t fid ) const override;
    LogicalGeometry( int64_t fid );


protected:
    //!  Default constructor for the base class
    LogicalGeometry( int physical, int logical, std::array<int, 6> ids = { 1, 2, 3, 4, 5, 6 } );

    // Delete copy constructors
    LogicalGeometry( LogicalGeometry && )                 = delete;
    LogicalGeometry( const LogicalGeometry & )            = default;
    LogicalGeometry &operator=( LogicalGeometry && )      = delete;
    LogicalGeometry &operator=( const LogicalGeometry & ) = delete;


protected:                          // Internal data
    const uint8_t d_logicalDim;     // Logical dimension
    const std::array<int, 6> d_ids; // Logical surface ids
};


} // namespace AMP::Geometry

#endif
