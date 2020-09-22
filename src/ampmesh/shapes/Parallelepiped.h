#ifndef included_AMP_Geometry_Parallelepiped
#define included_AMP_Geometry_Parallelepiped

#include "AMP/ampmesh/LogicalGeometry.h"

#include <vector>


namespace AMP {
namespace Geometry {


/**
 * \class Parallelepiped
 * \brief A class used to describe a Parallelepiped
 * \details  This class provides routines for reading, accessing and writing Parallelepiped.
 *     See https://en.wikipedia.org/wiki/Parallelepiped
 */
class Parallelepiped : public LogicalGeometry
{
public:
    /**
     * \brief Construct a Parallelepiped geometry
     * \param db        Input database
     */
    explicit Parallelepiped( std::shared_ptr<AMP::Database> db );

    // Functions inherited from Geometry
    virtual std::string getName() const override final { return "Parallelepiped"; }
    virtual bool isConvex() const override final { return true; }
    virtual Point nearest( const Point &pos ) const override final;
    virtual double distance( const Point &pos, const Point &dir ) const override final;
    virtual bool inside( const Point &pos ) const override final;
    virtual int NSurface() const override final { return 6; }
    virtual int surface( const Point & ) const override final;
    virtual Point surfaceNorm( const Point & ) const override final;
    virtual Point logical( const Point &x ) const override final;
    virtual Point physical( const Point &x ) const override final;
    virtual Point centroid() const override final;
    virtual std::pair<Point, Point> box() const override final;
    virtual double volume() const override final;
    virtual void displace( const double *x ) override final;
    virtual std::vector<int> getLogicalGridSize( const std::vector<int> &x ) const override final;
    virtual std::vector<int>
    getLogicalGridSize( const std::vector<double> &res ) const override final;
    virtual std::vector<bool> getPeriodicDim() const override final;
    virtual std::vector<int> getLogicalSurfaceIds() const override final;
    virtual std::unique_ptr<AMP::Geometry::Geometry> clone() const override final;

protected:
    // Internal data
    double d_a[3];           // First edge vector
    double d_b[3];           // Second edge vector
    double d_c[3];           // Third edge vector
    double d_offset[3];      // Offset
    double d_M_inv[9];       // Inverse matrix used to compute logical coordinates
    double d_V;              // Cached volume
    AMP::Mesh::Point d_n_ab; // Normal to the plane defined by a-b
    AMP::Mesh::Point d_n_ac; // Normal to the plane defined by a-c
    AMP::Mesh::Point d_n_bc; // Normal to the plane defined by b-c

private:
    // Private constuctor
    Parallelepiped();
};


} // namespace Geometry
} // namespace AMP

#endif