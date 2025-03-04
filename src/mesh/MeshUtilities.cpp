#include "AMP/mesh/MeshUtilities.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/NearestPairSearch.h"
#include "AMP/utils/kdtree.h"


namespace AMP::Mesh {


// Helper function to check that no two points are too close
static inline double nearest( const std::vector<Point> &x )
{
    if ( x.empty() )
        return 0;
    auto index = find_min_dist( x );
    auto dx    = x[index.first] - x[index.second];
    return dx.abs();
}


static inline double dist( const kdtree &tree, const Point &p )
{
    double d;
    tree.find_nearest( p.data(), &d );
    return d;
}


static inline double norm2( const std::array<double, 3> &a, const std::array<double, 3> &b )
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


/********************************************************
 *  Get points in the mesh                               *
 ********************************************************/
template<class CONTAINER>
std::vector<AMP::Mesh::MeshElement>
getElements( const Mesh &mesh, const CONTAINER &elements, AMP::Mesh::GeomType type )
{
    // Function to get the composition elements
    std::set<AMP::Mesh::MeshElementID> ids;
    std::vector<AMP::Mesh::MeshElementID> children;
    for ( const auto &elem : elements ) {
        elem.getElementsID( type, children );
        ids.insert( children.begin(), children.end() );
    }
    std::vector<AMP::Mesh::MeshElement> elements2;
    elements2.reserve( ids.size() );
    for ( const auto &id : ids )
        elements2.emplace_back( mesh.getElement( id ) );
    return elements2;
}
template<class CONTAINER>
std::tuple<std::vector<Point>, std::vector<MeshElementID>>
sample( const Mesh &mesh, const CONTAINER &elements, double dx )
{
    std::vector<Point> points;
    std::vector<MeshElementID> ids;
    // Get the type of the elements
    const auto type = elements.begin()->elementType();
    // Start by adding all the nodes to preserve the geometry
    auto node_list = getElements( mesh, elements, AMP::Mesh::GeomType::Vertex );
    for ( auto &node : node_list ) {
        points.push_back( node.coord() );
        ids.push_back( node.globalID() );
    }
    double d0 = nearest( points );
    // Check if we are auto determining the resolution
    if ( dx == 0.0 )
        dx = d0;
    // Create a kdtree to quickly calculate distances
    kdtree tree( points );
    // Function to add points
    auto addPoints = [&points, &ids, &tree, dx]( const auto &it ) {
        for ( const auto &elem : it ) {
            auto points2 = sample( elem, dx );
            for ( const auto &p : points2 ) {
                if ( dist( tree, p ) > 0.8 * dx ) {
                    points.push_back( p );
                    ids.push_back( elem.globalID() );
                    tree.add( p.data() );
                }
            }
        }
    };
    // Add points for each of the composite elements
    for ( int t = 1; t < static_cast<int>( type ); t++ ) {
        auto elements2 = getElements( mesh, elements, static_cast<AMP::Mesh::GeomType>( t ) );
        addPoints( elements2 );
    }
    // Add the elements
    addPoints( elements );
    return std::tie( points, ids );
}
std::tuple<std::vector<Point>, std::vector<MeshElementID>> sample( const Mesh &mesh, double dx )
{
    return sample( mesh, mesh.getIterator( AMP::Mesh::GeomType::Vertex ), dx );
}


/********************************************************
 * Return points in the element at the given resolution  *
 ********************************************************/
std::vector<Point> sample( const MeshElement &elem, double dx )
{
    auto type = elem.globalID().type();
    if ( type == AMP::Mesh::GeomType::Vertex )
        return std::vector<Point>( 1, elem.coord() );
    // Get the nodes
    auto nodes = elem.getElements( AMP::Mesh::GeomType::Vertex );
    std::vector<Point> x( nodes.size() );
    for ( size_t i = 0; i < nodes.size(); i++ )
        x[i] = nodes[i].coord();
    // Check if we are dealing with a volume (in the coordinate space)
    if ( static_cast<int>( type ) == x[0].ndim() ) {
        // Create a uniform grid
        AMP_ERROR( "Not finished" );
    }
    // Code for the different object types
    std::vector<Point> p;
    if ( type == AMP::Mesh::GeomType::Edge ) {
        // We are dealing with an edge (easy)
        AMP_ASSERT( x.size() == 2u );
        double d = ( x[1] - x[0] ).abs();
        double N = d / dx;
        int n    = N + 1;
        auto dx2 = 1.0 / n * ( x[1] - x[0] );
        for ( int i = 0; i <= n; i++ )
            p.push_back( x[0] + static_cast<double>( i ) * dx2 );
    } else if ( x.size() == 3u ) {
        // We are dealing with a triangle
        p = AMP::Geometry::GeometryHelpers::subdivide( { x[0], x[1], x[2] }, dx );
    } else if ( type == AMP::Mesh::GeomType::Face ) {
        // Get the normal
        auto n      = elem.norm();
        auto center = x[0];
        for ( size_t i = 1; i < nodes.size(); i++ )
            center += x[i];
        center *= 1.0 / nodes.size();
        // Get two perpendicular unit vectors in the plane
        // Choose the furthest point for the first vector
        Point v1;
        double tmp = 0;
        for ( auto &y : x ) {
            auto v = y - center;
            auto d = v.norm();
            if ( d > tmp ) {
                v1  = normalize( v );
                tmp = d;
            }
        }
        // Compute the second vector using the norm and the first
        auto v2 = normalize( cross( n, v1 ) );
        // Get the range in the unit vectors
        double range[4] = { 0, 0, 0, 0 };
        for ( auto &y : x ) {
            double d1 = dot( y - center, v1 );
            double d2 = dot( y - center, v2 );
            range[0]  = std::min( range[0], d1 );
            range[1]  = std::max( range[1], d1 );
            range[2]  = std::min( range[2], d2 );
            range[3]  = std::max( range[3], d2 );
        }
        // Create the points
        auto N1 = ( range[1] - range[0] ) / dx;
        auto N2 = ( range[3] - range[2] ) / dx;
        auto x1 = range[0] + 0.5 * ( N1 - floor( N1 ) );
        auto x2 = range[2] + 0.5 * ( N2 - floor( N2 ) );
        for ( double d1 = x1; d1 <= range[1]; d1 += dx ) {
            for ( double d2 = x2; d2 <= range[3]; d2 += dx ) {
                Point p1 = center + d1 * v1 + d2 * v2;
                if ( elem.containsPoint( p1 ) )
                    p.push_back( p1 );
            }
        }
    } else {
        AMP_ERROR( "Not finished" );
    }
    return p;
}


/********************************************************
 * Compute the volume overlap with a geometry            *
 ********************************************************/
inline double
cellVolume1D( const AMP::Geometry::Geometry &geom, double x0, double dx, bool in[2], double tol )
{
    if ( in[0] && in[1] ) {
        return dx; // The entire cell is inside
    } else if ( !in[0] && !in[1] ) {
        return 0; // The entire cell is outside
    } else if ( dx < 2 * tol ) {
        return 0.5 * dx; // The error is less than the tolerance
    } else {
        // Subdivide the cell to estimate the volume
        bool in2[3] = { in[0], geom.inside( { x0 + 0.5 * dx } ), in[1] };
        return cellVolume1D( geom, x0, 0.5 * dx, in2, tol ) +
               cellVolume1D( geom, x0 + 0.5 * dx, 0.5 * dx, in2, tol );
    }
}
inline double cellVolume2D( const AMP::Geometry::Geometry &geom,
                            const double x0[2],
                            const double dx[2],
                            const bool in[4],
                            double tol )
{
    bool all_inside  = in[0] && in[1] && in[2] && in[3];
    bool all_outside = !in[0] && !in[1] && !in[2] && !in[3];
    double vol       = dx[0] * dx[1];
    if ( all_inside ) {
        // The entire cell is inside
        return vol;
    } else if ( all_outside ) {
        // The entire cell is outside
        return 0;
    } else if ( vol < 4 * tol ) {
        // The error is less than the tolerance
        int count = 0;
        for ( int i = 0; i < 4; i++ ) {
            if ( in[i] )
                count++;
        }
        return count * vol / 4.0;
    } else {
        // Subdivide the cell to estimate the volume
        bool in2[3][3];
        in2[0][0] = in[0];
        in2[2][0] = in[1];
        in2[0][2] = in[2];
        in2[2][2] = in[3];
        for ( int i = 0; i < 3; i++ ) {
            double x = x0[0] + 0.5 * i * dx[0];
            for ( int j = 0; j < 3; j++ ) {
                double y = x0[1] + 0.5 * j * dx[1];
                if ( i % 2 == 0 && j % 2 == 0 )
                    continue;
                in2[i][j] = geom.inside( { x, y } );
            }
        }
        // Subdivide the cell to recursively estimate the volume
        double volume = 0;
        double dx2[2] = { 0.5 * dx[0], 0.5 * dx[1] };
        for ( int i = 0; i < 2; i++ ) {
            double x = x0[0] + 0.5 * i * dx[0];
            for ( int j = 0; j < 2; j++ ) {
                double y     = x0[1] + 0.5 * j * dx[1];
                double xy[2] = { x, y };
                bool in3[8]  = { in2[i][j], in2[i + 1][j], in2[i][j + 1], in2[i + 1][j + 1] };
                volume += cellVolume2D( geom, xy, dx2, in3, tol );
            }
        }
        return volume;
    }
}
inline double cellVolume3D( const AMP::Geometry::Geometry &geom,
                            const double x0[3],
                            const double dx[3],
                            const bool in[8],
                            double tol )
{
    bool all_inside  = in[0] && in[1] && in[2] && in[3] && in[4] && in[5] && in[6] && in[7];
    bool all_outside = !in[0] && !in[1] && !in[2] && !in[3] && !in[4] && !in[5] && !in[6] && !in[7];
    double vol       = dx[0] * dx[1] * dx[2];
    if ( all_inside ) {
        // The entire cell is inside
        return vol;
    } else if ( all_outside ) {
        // The entire cell is outside
        return 0;
    } else if ( vol < 8 * tol ) {
        // The error is less than the tolerance
        int count = 0;
        for ( int i = 0; i < 8; i++ ) {
            if ( in[i] )
                count++;
        }
        return count * vol / 8.0;
    } else {
        // Subdivide the cell to estimate the volume
        bool in2[3][3][3];
        in2[0][0][0] = in[0];
        in2[2][0][0] = in[1];
        in2[0][2][0] = in[2];
        in2[2][2][0] = in[3];
        in2[0][0][2] = in[4];
        in2[2][0][2] = in[5];
        in2[0][2][2] = in[6];
        in2[2][2][2] = in[7];
        for ( int i = 0; i < 3; i++ ) {
            double x = x0[0] + 0.5 * i * dx[0];
            for ( int j = 0; j < 3; j++ ) {
                double y = x0[1] + 0.5 * j * dx[1];
                for ( int k = 0; k < 3; k++ ) {
                    double z = x0[2] + 0.5 * k * dx[2];
                    if ( i % 2 == 0 && j % 2 == 0 && k % 2 == 0 )
                        continue;
                    in2[i][j][k] = geom.inside( { x, y, z } );
                }
            }
        }
        // Subdivide the cell to recursively estimate the volume
        double volume = 0;
        double dx2[3] = { 0.5 * dx[0], 0.5 * dx[1], 0.5 * dx[2] };
        for ( int i = 0; i < 2; i++ ) {
            double x = x0[0] + 0.5 * i * dx[0];
            for ( int j = 0; j < 2; j++ ) {
                double y = x0[1] + 0.5 * j * dx[1];
                for ( int k = 0; k < 2; k++ ) {
                    double z      = x0[2] + 0.5 * k * dx[2];
                    double xyz[3] = { x, y, z };
                    bool in3[8]   = { in2[i][j][k],         in2[i + 1][j][k],
                                    in2[i][j + 1][k],     in2[i + 1][j + 1][k],
                                    in2[i][j][k + 1],     in2[i + 1][j][k + 1],
                                    in2[i][j + 1][k + 1], in2[i + 1][j + 1][k + 1] };
                    volume += cellVolume3D( geom, xyz, dx2, in3, tol );
                }
            }
        }
        return volume;
    }
}
Array<double> volumeOverlap( const AMP::Geometry::Geometry &geom, const std::vector<int> &N )
{
    AMP_ASSERT( N.size() == geom.getDim() );
    Array<double> volume;
    if ( geom.getDim() == 1 ) {
        // Get the bounding box
        auto [lb, ub] = geom.box();
        double dx     = ( ub[0] - lb[0] ) / N[0];
        // Get the volume for each cell
        volume.resize( N[0] );
        volume.fill( 0 );
        Array<bool> inside( N[0] + 1 );
        inside.fill( false );
        for ( int i = 0; i <= N[0]; i++ ) {
            double x    = lb[0] + i * dx;
            inside( i ) = geom.inside( { x } );
        }
        double tol = 0.001 * dx;
        for ( int i = 0; i < N[0]; i++ ) {
            double x    = lb[0] + i * dx;
            bool in[2]  = { inside( i ), inside( i + 1 ) };
            volume( i ) = cellVolume1D( geom, x, dx, in, tol );
        }
    } else if ( geom.getDim() == 2 ) {
        // Get the bounding box
        auto [lb, ub] = geom.box();
        double dx[2]  = { ( ub[0] - lb[0] ) / N[0], ( ub[1] - lb[1] ) / N[1] };
        // Get the volume for each cell
        volume.resize( N[0], N[1] );
        volume.fill( 0 );
        Array<bool> inside( N[0] + 1, N[1] + 1 );
        inside.fill( false );
        for ( int j = 0; j <= N[1]; j++ ) {
            double y = lb[1] + j * dx[1];
            for ( int i = 0; i <= N[0]; i++ ) {
                double x       = lb[0] + i * dx[0];
                inside( i, j ) = geom.inside( { x, y } );
            }
        }
        double tol = 0.01 * dx[0] * dx[1];
        for ( int j = 0; j < N[1]; j++ ) {
            double y = lb[1] + j * dx[1];
            for ( int i = 0; i < N[0]; i++ ) {
                double x     = lb[0] + i * dx[0];
                double xy[2] = { x, y };
                bool in[4]   = {
                    inside( i, j ), inside( i + 1, j ), inside( i, j + 1 ), inside( i + 1, j + 1 )
                };
                volume( i, j ) = cellVolume2D( geom, xy, dx, in, tol );
            }
        }
    } else if ( geom.getDim() == 3 ) {
        // Get the bounding box
        auto [lb, ub] = geom.box();
        double dx[3]  = { ( ub[0] - lb[0] ) / N[0],
                         ( ub[1] - lb[1] ) / N[1],
                         ( ub[2] - lb[2] ) / N[2] };
        // Get the volume for each cell
        volume.resize( N[0], N[1], N[2] );
        volume.fill( 0 );
        Array<bool> inside( N[0] + 1, N[1] + 1, N[2] + 1 );
        inside.fill( false );
        for ( int k = 0; k <= N[2]; k++ ) {
            double z = lb[2] + k * dx[2];
            for ( int j = 0; j <= N[1]; j++ ) {
                double y = lb[1] + j * dx[1];
                for ( int i = 0; i <= N[0]; i++ ) {
                    double x          = lb[0] + i * dx[0];
                    inside( i, j, k ) = geom.inside( { x, y, z } );
                }
            }
        }
        double tol = 0.01 * dx[0] * dx[1] * dx[2];
        for ( int k = 0; k < N[2]; k++ ) {
            double z = lb[2] + k * dx[2];
            for ( int j = 0; j < N[1]; j++ ) {
                double y = lb[1] + j * dx[1];
                for ( int i = 0; i < N[0]; i++ ) {
                    double x          = lb[0] + i * dx[0];
                    double xyz[3]     = { x, y, z };
                    bool in[8]        = { inside( i, j, k ),         inside( i + 1, j, k ),
                                   inside( i, j + 1, k ),     inside( i + 1, j + 1, k ),
                                   inside( i, j, k + 1 ),     inside( i + 1, j, k + 1 ),
                                   inside( i, j + 1, k + 1 ), inside( i + 1, j + 1, k + 1 ) };
                    volume( i, j, k ) = cellVolume3D( geom, xyz, dx, in, tol );
                }
            }
        }
    } else {
        AMP_ERROR( "Not implemented for this dimension" );
    }
    return volume;
}


/********************************************************
 * ElementFinder                                         *
 ********************************************************/
ElementFinder::ElementFinder( std::shared_ptr<AMP::Mesh::Mesh> mesh )
    : d_mesh( mesh ),
      d_pos_hash( static_cast<size_t>( -1 ) ),
      d_elements( mesh->getIterator( mesh->getGeomType() ) ),
      d_dist( std::numeric_limits<double>::quiet_NaN() )
{
    initialize();
}
ElementFinder::ElementFinder( std::shared_ptr<AMP::Mesh::Mesh> mesh, AMP::Mesh::MeshIterator it )
    : d_mesh( mesh ),
      d_pos_hash( static_cast<size_t>( -1 ) ),
      d_elements( std::move( it ) ),
      d_dist( std::numeric_limits<double>::quiet_NaN() )
{
    initialize();
}
void ElementFinder::initialize() const
{
    PROFILE( "initialize" );
    AMP_ASSERT( d_mesh->getDim() == 3u );
    // Choose the approximate spacing between points
    double volume = 0.0;
    auto type     = d_elements->globalID().type();
    for ( auto elem : d_elements ) {
        AMP_ASSERT( elem.globalID().type() == type );
        volume += elem.volume();
    }
    volume /= d_elements.size();
    d_dist = 0.5 * std::pow( volume, 1.0 / static_cast<int>( type ) );
    // Create a list of points in each element and the mesh ids
    std::vector<AMP::Mesh::MeshElementID> ids;
    std::vector<std::array<double, 3>> points;
    std::vector<MeshElement> children;
    std::vector<std::array<double, 3>> nodes;
    for ( const auto &elem : d_elements ) {
        auto id = elem.globalID();
        // Get the nodes for the current element
        elem.getElements( AMP::Mesh::GeomType::Vertex, children );
        nodes.resize( children.size() );
        for ( size_t i = 0; i < children.size(); i++ ) {
            auto p      = children[i].coord();
            nodes[i][0] = p.x();
            nodes[i][1] = p.y();
            nodes[i][2] = p.z();
        }
        // Choose points within the element such that they completely overlap the element
        using namespace AMP::Geometry::GeometryHelpers;
        std::vector<std::array<double, 3>> data;
        if ( nodes.size() == 2u ) {
            data = sampleLine( { nodes[0], nodes[1] }, d_dist, true );
        } else if ( nodes.size() == 3u ) {
            data = sampleTri( { nodes[0], nodes[1], nodes[2] }, d_dist, true );
        } else if ( type == AMP::Mesh::GeomType::Face && nodes.size() == 4u ) {
            data = sampleQuad( { nodes[0], nodes[1], nodes[2], nodes[3] }, d_dist, true );
        } else if ( type == AMP::Mesh::GeomType::Cell && nodes.size() == 4u ) {
            data = sampleTet( { nodes[0], nodes[1], nodes[2], nodes[3] }, d_dist, true );
        } else {
            printf( "type = %i\n", static_cast<int>( type ) );
            printf( "nodes = %i\n", static_cast<int>( nodes.size() ) );
            AMP_ERROR( "Not finished" );
        }
        for ( auto p : data ) {
            points.push_back( p );
            ids.push_back( id );
        }
    }
    // Update the internal hash
    d_pos_hash = d_mesh->positionHash();
    // Create a kdtree with the points
    d_tree = AMP::kdtree2<3, AMP::Mesh::MeshElementID>( points, ids );
}
void ElementFinder::update() const
{
    // Update cached data if position moved
    if ( d_pos_hash != d_mesh->positionHash() )
        initialize();
}
std::pair<AMP::Mesh::MeshElement, Point> ElementFinder::nearest( const Point &x ) const
{
    // Update cached data if position moved
    if ( d_pos_hash != d_mesh->positionHash() )
        initialize();
    // Get the nearest element
    std::array<double, 3> p0 = { x.x(), x.y(), x.z() };
    auto [p1, id]            = d_tree.findNearest( p0 );
    auto elem                = d_mesh->getElement( id );
    // Get the nearest point in the element
    Point p = x;
    if ( norm2( p1, p0 ) > 1e-12 )
        elem.nearest( x );
    return std::pair<AMP::Mesh::MeshElement, Point>( std::move( elem ), p );
}
double ElementFinder::distance( const Point &pos, const Point &dir ) const
{
    // Update cached data if position moved
    if ( d_pos_hash != d_mesh->positionHash() )
        initialize();
    // Get a list of ids to check
    std::set<AMP::Mesh::MeshElementID> ids;
    for ( const auto &tmp : d_tree.findNearestRay( pos, dir, d_dist ) )
        ids.insert( std::get<1>( tmp ) );
    // Check each element for the closest intersection
    int count = 0;
    double d  = std::numeric_limits<double>::infinity();
    for ( const auto &id : ids ) {
        auto elem = d_mesh->getElement( id );
        auto d2   = elem.distance( pos, dir );
        d2        = std::max( d2, 0.0 );
        if ( d2 < std::numeric_limits<double>::infinity() ) {
            d = std::min( d, d2 );
            count++;
        }
    }
    if ( count % 2 == 1 )
        d = -d;
    return d;
}

} // namespace AMP::Mesh


/********************************************************
 *  Explicit instantiations of kdtree2                   *
 ********************************************************/
#include "AMP/utils/kdtree2.hpp"
template class AMP::kdtree2<3, AMP::Mesh::MeshElementID>;
