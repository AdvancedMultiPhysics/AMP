#ifndef included_AMP_Geometry_Box
#define included_AMP_Geometry_Box

#include "AMP/geometry/LogicalGeometry.h"

#include <array>
#include <vector>


namespace AMP::Geometry {


std::unique_ptr<AMP::Geometry::Geometry> buildBox( std::shared_ptr<const AMP::Database> db );


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
template<std::size_t NDIM>
class Box : public LogicalGeometry
{
public:
    /**
     * \brief Construct a Box geometry
     * \param db        Input database
     */
    explicit Box( std::shared_ptr<const AMP::Database> db );

    /**
     * \brief Construct a box
     * \param range     The range of the box [xmin, xmax, ymin, ymax, zmin, zmax, ...]
     */
    explicit Box( const std::vector<double> &range );

    //! ConstructGrid from restart
    Box( int64_t );

public: // Default constructors
    explicit Box( Box<NDIM> &&range )      = default;
    explicit Box( const Box<NDIM> &range ) = default;
    Box<NDIM> &operator=( Box<NDIM> &&range ) = default;
    Box<NDIM> &operator=( const Box<NDIM> &range ) = default;

public: // Functions inherited from Geometry
    std::string getName() const override;
    bool isConvex() const override final { return true; }
    Point nearest( const Point &pos ) const override final;
    double distance( const Point &pos, const Point &dir ) const override final;
    bool inside( const Point &pos ) const override final;
    int NSurface() const override final { return 2 * NDIM; }
    int surface( const Point &x ) const override final;
    Point surfaceNorm( const Point &x ) const override final;
    Point logical( const Point &x ) const override;
    Point physical( const Point &x ) const override;
    Point centroid() const override final;
    std::pair<Point, Point> box() const override final;
    double volume() const override final;
    void displace( const double *x ) override;
    ArraySize getLogicalGridSize( const ArraySize &x ) const override final;
    ArraySize getLogicalGridSize( const std::vector<double> &res ) const override final;
    std::unique_ptr<AMP::Geometry::Geometry> clone() const override;
    bool operator==( const Geometry &rhs ) const override;
    void writeRestart( int64_t ) const override;

protected:
    Box();

protected:
    // Internal data
    std::array<double, 6> d_range;
};


/**
 * \class Geometry
 * \brief A class used to abstract away geometry information from an application or mesh.
 * \details  This class provides routines for reading, accessing and writing geometries.
 */
template<std::size_t NDIM>
class Grid final : public Box<NDIM>
{
public:
    /**
     * \brief Construct a Grid geometry
     * \param db        Input database
     */
    explicit Grid( std::shared_ptr<const AMP::Database> db );

    /**
     * \brief Construct a grid geometry
     * \param coord     Optional list of coordinates to adjust the logical - physical mapping
     */
    explicit Grid( const std::vector<std::vector<double>> &coord );

    //! ConstructGrid from restart
    Grid( int64_t );

public: // Default constructors
    explicit Grid( Grid<NDIM> &&rhs )      = default;
    explicit Grid( const Grid<NDIM> &rhs ) = default;
    Grid<NDIM> &operator=( Grid<NDIM> &&rhs ) = default;
    Grid<NDIM> &operator=( const Grid<NDIM> &rhs ) = default;

public: // Functions inherited from Geometry
    std::string getName() const override;
    Point logical( const Point &x ) const override final;
    Point physical( const Point &x ) const override final;
    void displace( const double *x ) override final;
    std::unique_ptr<AMP::Geometry::Geometry> clone() const override final;
    bool operator==( const Geometry &rhs ) const override final;
    void writeRestart( int64_t ) const override;

protected:
    std::array<std::vector<double>, NDIM> d_coord; // Coordinates

private:
    // Private constructor
    Grid();
};


} // namespace AMP::Geometry

#endif
