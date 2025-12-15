#ifndef included_AMP_Geometry_Tube
#define included_AMP_Geometry_Tube

#include "AMP/geometry/LogicalGeometry.h"

#include <array>
#include <vector>


namespace AMP::Geometry {


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
class Tube final : public LogicalGeometry
{
public:
    /**
     * \brief Construct a Tube geometry
     * \param db        Input database
     */
    explicit Tube( std::shared_ptr<const AMP::Database> db );

    /**
     * \brief Construct a Tube geometry
     * \param r_min     The minimum radius
     * \param r_max     The maximum radius
     * \param z_min     The minimum z coordinate
     * \param z_max     The maximum z coordinate
     */
    explicit Tube( double r_min, double r_max, double z_min, double z_max );

    //! Construct from restart
    Tube( int64_t );

public: // Functions inherited from Geometry
    std::string getName() const override final { return "Tube"; }
    bool isConvex() const override final { return false; }
    Point nearest( const Point &pos ) const override final;
    double distance( const Point &pos, const Point &dir ) const override final;
    bool inside( const Point &pos ) const override final;
    int surface( const Point &x ) const override final;
    int NSurface() const override final { return 4; }
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
    double d_r_min, d_r_max, d_z_min, d_z_max;
    std::array<double, 3> d_offset;

private:
    // Private constructor
    Tube();
};


} // namespace AMP::Geometry

#endif
