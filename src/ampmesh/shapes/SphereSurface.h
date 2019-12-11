#ifndef included_AMP_Geometry_SphereSurface
#define included_AMP_Geometry_SphereSurface

#include "AMP/ampmesh/Geometry.h"

#include <vector>


namespace AMP {
namespace Geometry {


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
class SphereSurface : public Geometry
{
public:
    /**
     * \brief Construct a SphereSurface geometry
     * \param db        Input database
     */
    explicit SphereSurface( std::shared_ptr<AMP::Database> db );

    /**
     * \brief Construct a SphereSurface geometry
     * \param range     The range of the SphereSurface [xmin, xmax, ymin, ymax, zmin, zmax, ...]
     */
    explicit SphereSurface( double R );

    // Functions inherited from Geometry
    virtual std::string getName() const override final { return "SphereSurface"; }
    virtual double distance( const Point &pos, const Point &dir ) const override final;
    virtual bool inside( const Point &pos ) const override final;
    virtual int NSurface() const override final { return 1; }
    virtual int surface( const Point & ) const override final { return 0; }
    virtual Point surfaceNorm( const Point &x ) const override final;
    virtual Point logical( const Point &x ) const override final;
    virtual Point physical( const Point &x ) const override final;
    virtual Point centroid() const override final;
    virtual std::pair<Point, Point> box() const override final;
    virtual void displaceMesh( const double *x ) override final;
    virtual std::vector<int> getLogicalGridSize( const std::vector<int> &x ) const override final;
    virtual std::vector<bool> getPeriodicDim() const override final;
    virtual std::vector<int> getLogicalSurfaceIds() const override final;
    virtual std::shared_ptr<AMP::Geometry::Geometry> clone() const override final;

protected:
    // Internal data
    double d_r;
    double d_offset[3];

private:
    // Private constuctor
    SphereSurface();
};


} // namespace Geometry
} // namespace AMP

#endif
