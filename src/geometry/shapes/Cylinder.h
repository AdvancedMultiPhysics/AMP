#ifndef included_AMP_Geometry_Cylinder
#define included_AMP_Geometry_Cylinder

#include "AMP/geometry/LogicalGeometry.h"

#include <array>
#include <vector>


namespace AMP::Geometry {


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
class Cylinder final : public LogicalGeometry
{
public:
    /**
     * \brief Construct a Cylinder geometry
     * \param db        Input database
     */
    explicit Cylinder( std::shared_ptr<const AMP::Database> db );

    /**
     * \brief Construct a Cylinder geometry
     * \param r         The radius of the cylinder
     * \param z_min     The lower z-coordinate
     * \param z_max     The upper z-coordinate
     */
    explicit Cylinder( double r, double z_min, double z_max );

    //! Construct from restart
    Cylinder( int64_t );

public: // Functions inherited from Geometry
    std::string getName() const override final { return "Cylinder"; }
    bool isConvex() const override final { return true; }
    Point nearest( const Point &pos ) const override final;
    double distance( const Point &pos, const Point &dir ) const override final;
    bool inside( const Point &pos ) const override final;
    int NSurface() const override final { return 3; }
    int surface( const Point &x ) const override final;
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

protected: // Internal data
    int d_method                    = 2;
    double d_r                      = 0;
    double d_z_min                  = 0;
    double d_z_max                  = 0;
    std::array<double, 3> d_offset  = { 0, 0, 0 };
    std::array<double, 2> d_chamfer = { 0, 0 };

private: // Private functions
    Cylinder();
    double getR( double z ) const;
};


} // namespace AMP::Geometry

#endif
