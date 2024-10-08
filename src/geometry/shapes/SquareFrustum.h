#ifndef included_AMP_Geometry_SquareFrustum
#define included_AMP_Geometry_SquareFrustum

#include "AMP/geometry/LogicalGeometry.h"

#include <vector>


namespace AMP::Geometry {


/**
 * \class SquareFrustum
 * \brief A geometry for a square frustum
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
class SquareFrustum final : public LogicalGeometry
{
public:
    /**
     * \brief Construct a SquareFrustum geometry
     * \param db        Input database
     */
    explicit SquareFrustum( std::shared_ptr<const AMP::Database> db );

    /**
     * \brief Construct a SquareFrustum geometry
     * \param range     The range of the SquareFrustum [xmin, xmax, ymin, ymax, zmin, zmax, ...]
     * \param dir       The direction of the pyramid { -x, x, -y, y, -z, z }
     * \param height    The height of the pyramid
     */
    explicit SquareFrustum( const std::vector<double> &range, int dir, double height );

    //! Construct from restart
    SquareFrustum( int64_t );

public: // Functions inherited from Geometry
    std::string getName() const override final { return "SquareFrustum"; }
    bool isConvex() const override final { return true; }
    Point nearest( const Point &pos ) const override final;
    double distance( const Point &pos, const Point &dir ) const override final;
    bool inside( const Point &pos ) const override final;
    int NSurface() const override final { return 6; }
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

protected:
    // Internal data
    using FaceNormal = std::array<Point, 6>;
    using FacePoints = std::array<std::array<Point, 4>, 6>;
    uint8_t d_dir;
    std::array<double, 6> d_range;        // The bounding box size
    std::array<double, 3> d_pyramid_size; // The underlying rotated pyramid size
    double d_scale_height;                // Ratio of frustum to pyramid height
    double d_volume;                      // Volume
    Point d_centroid;                     // Centroid
    FacePoints d_face;                    // Points forming each face
    FaceNormal d_normal;                  // Normal to each face

private:
    // Private constructor
    SquareFrustum();
    // Initialize the data
    void initialize( const std::vector<double> &range, int dir, double height );
};


} // namespace AMP::Geometry

#endif
