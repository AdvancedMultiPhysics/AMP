#ifndef included_AMP_Geometry_Cylinder
#define included_AMP_Geometry_Cylinder

#include "AMP/ampmesh/Geometry.h"

#include <vector>


namespace AMP {
namespace Geometry {


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
class Cylinder : public Geometry
{
public:
    /**
     * \brief Construct a Cylinder
     * \param range     The range of the Cylinder [xmin, xmax, ymin, ymax, zmin, zmax, ...]
     */
    explicit Cylinder( double r, double z_min, double z_max );

    // Functions inherited from Geometry
    virtual std::string getName() const override final { return "Cylinder"; }
    virtual double distance( const Point &pos, const Point &dir ) const override final;
    virtual bool inside( const Point &pos ) const override final;
    virtual int surface( const Point &x ) const override final;
    virtual Point surfaceNorm( const Point &x ) const override final;
    virtual Point logical( const Point &x ) const override final;
    virtual Point physical( const Point &x ) const override final;
    virtual Point centroid() const override final;
    virtual std::pair<Point, Point> box() const override final;
    virtual void displaceMesh( const double *x ) override final;

protected:
    // Internal data
    double d_r;
    double d_z_min;
    double d_z_max;
    double d_offset[3];

private:
    // Private constuctor
    Cylinder();
};


} // namespace Geometry
} // namespace AMP

#endif
