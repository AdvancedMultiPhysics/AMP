#ifndef included_AMP_Geometry_SphereSurface
#define included_AMP_Geometry_SphereSurface

#include "AMP/geometry/LogicalGeometry.h"

#include <array>
#include <vector>


namespace AMP::Geometry {


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
class SphereSurface final : public LogicalGeometry
{
public:
    /**
     * \brief Construct a SphereSurface geometry
     * \param db        Input database
     */
    explicit SphereSurface( std::shared_ptr<const AMP::Database> db );

    /**
     * \brief Construct a SphereSurface geometry
     * \param R     The radius
     */
    explicit SphereSurface( double R );

    //! Construct from restart
    SphereSurface( int64_t );

public: // Functions inherited from Geometry
    std::string getName() const override final { return "SphereSurface"; }
    bool isConvex() const override final { return true; }
    Point nearest( const Point &pos ) const override final;
    double distance( const Point &pos, const Point &dir ) const override final;
    bool inside( const Point &pos ) const override final;
    int NSurface() const override final { return 1; }
    int surface( const Point & ) const override final { return 0; }
    Point surfaceNorm( const Point &x ) const override final;
    Point logical( const Point &x ) const override final;
    Point physical( const Point &x ) const override final;
    Point centroid() const override final;
    std::pair<Point, Point> box() const override final;
    double volume() const override final;
    void displace( const double *x ) override final;
    ArraySize getLogicalGridSize( const ArraySize &x ) const override final;
    ArraySize getLogicalGridSize( const std::vector<double> &res ) const override final;
    std::unique_ptr<AMP::Geometry::Geometry> clone() const override final;
    bool operator==( const Geometry &rhs ) const override final;
    void writeRestart( int64_t ) const override;

protected:
    // Internal data
    double d_r;
    std::array<double, 3> d_offset;

private:
    // Private constructor
    SphereSurface();
};


} // namespace AMP::Geometry

#endif
