#ifndef included_AMP_LogicalGeometry
#define included_AMP_LogicalGeometry

#include "AMP/ampmesh/Geometry.h"


namespace AMP {
namespace Geometry {


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
     *    If the coordinates cannot map to a logical grid, this function should throw
     *    a runtime exception.
     * \param[int] x    Input size
     * @return          Return the logical boundary ids (2*logicalDim)
     */
    virtual std::vector<int> getLogicalGridSize( const std::vector<int> &x ) const = 0;

    /**
     * \brief    Return the logical grid periodic dimensions
     * \details  This function will return a vector indicating which logical grid
     *    dimensions are periodic.  If the coordinates cannot map to a logical grid,
     *    this function should throw a runtime exception.
     * @return          Return the periodic dimensions
     */
    virtual std::vector<bool> getPeriodicDim() const = 0;

    /**
     * \brief    Return the surface ids for the logical boundaries
     * \details  This function will return the surface ids for each logical boundary.
     *    If a logical boundary does not map to a surface, it will return -1.
     *    If the coordinates cannot map to a logical grid, this function should
     *    throw a runtime exception.
     * @return          Return the logical boundary ids (2*logicalDim)
     */
    virtual std::vector<int> getLogicalSurfaceIds() const = 0;


protected:
    //!  Empty constructor for the base class
    LogicalGeometry() : Geometry(), d_logicalDim( 0 ) {}

    // Delete copy constructors
    LogicalGeometry( LogicalGeometry && )      = delete;
    LogicalGeometry( const LogicalGeometry & ) = default;
    LogicalGeometry &operator=( LogicalGeometry && ) = delete;
    LogicalGeometry &operator=( const LogicalGeometry & ) = delete;


protected: // Internal data
    uint8_t d_logicalDim;
};


} // namespace Geometry
} // namespace AMP

#endif