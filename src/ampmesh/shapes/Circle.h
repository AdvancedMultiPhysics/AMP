#ifndef included_AMP_Geometry_Circle
#define included_AMP_Geometry_Circle

#include "AMP/ampmesh/Geometry.h"

#include <vector>


namespace AMP {
namespace Geometry {


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
class Circle : public Geometry
{
public:
    /**
     * \brief Construct a Circle
     * \param range     The range of the Circle [xmin, xmax, ymin, ymax, zmin, zmax, ...]
     */
    explicit Circle( double R );

    // Functions inherited from Geometry
    virtual uint8_t getDim() const override final { return 2; }
    virtual double distance( const Point<double> &pos,
                             const Point<double> &dir ) const override final;
    virtual bool inside( const Point<double> &pos ) const override final;
    virtual int surface( const Point<double> &x ) const override final;
    virtual Point<double> surfaceNorm( const Point<double> &x ) const override final;
    virtual Point<double> logical( const Point<double> &x ) const override final;
    virtual Point<double> physical( const Point<double> &x ) const override final;
    virtual void displaceMesh( const double *x ) override final;

protected:
    // Internal data
    double d_R;
    std::array<double, 2> d_offset;

private:
    // Private constuctor
    Circle();
};

} // namespace Geometry
} // namespace AMP

#endif
