#include "AMP/utils/kdtree.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/kdtree2.h"

#include "ProfilerApp.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

#define ERROR_MSG AMP_ERROR


namespace AMP {


// Helper function to calculate distance between two points
template<uint8_t NDIM>
static inline double calcDist( const std::array<double, NDIM> &x,
                               const std::array<double, NDIM> &y )
{
    double dist = 0;
    for ( int d = 0; d < NDIM; d++ )
        dist += ( x[d] - y[d] ) * ( x[d] - y[d] );
    return std::sqrt( dist );
}


/********************************************************
 * Constructor                                           *
 ********************************************************/
template<uint8_t NDIM>
void *createTree( const size_t N, const double *const *x )
{
    auto name = AMP::Utilities::stringf( "kdtree<%i>", NDIM );
    PROFILE2( name );
    std::vector<int> index( N );
    std::vector<std::array<double, NDIM>> x2( N );
    for ( size_t i = 0; i < N; i++ ) {
        index[i] = i;
        for ( int d = 0; d < NDIM; d++ )
            x2[i][d] = x[d][i];
    }
    return new kdtree2<NDIM, int>( x2, index );
}
kdtree::kdtree( const int N_dim, const size_t N, const double *const *x )
    : d_dim( N_dim ), d_N( N ), d_tree( nullptr )
{
    if ( d_dim == 1 )
        d_tree = createTree<1>( N, x );
    else if ( d_dim == 2 )
        d_tree = createTree<2>( N, x );
    else if ( d_dim == 3 )
        d_tree = createTree<3>( N, x );
    else if ( d_dim == 4 )
        d_tree = createTree<4>( N, x );
    else if ( d_dim == 5 )
        d_tree = createTree<5>( N, x );
    else
        AMP_ERROR( "Not finished" );
}
kdtree::kdtree( const std::vector<AMP::Mesh::MeshPoint<double>> &x )
    : d_dim( 0 ), d_N( x.size() ), d_tree( nullptr )
{
    if ( x.empty() )
        return;
    d_dim = x[0].ndim();
    double *x2[5];
    for ( int d = 0; d < d_dim; d++ )
        x2[d] = new double[d_N];
    for ( size_t i = 0; i < d_N; i++ ) {
        for ( int d = 0; d < d_dim; d++ )
            x2[d][i] = x[i][d];
    }
    if ( d_dim == 1 )
        d_tree = createTree<1>( d_N, x2 );
    else if ( d_dim == 2 )
        d_tree = createTree<2>( d_N, x2 );
    else if ( d_dim == 3 )
        d_tree = createTree<3>( d_N, x2 );
    else if ( d_dim == 4 )
        d_tree = createTree<4>( d_N, x2 );
    else if ( d_dim == 5 )
        d_tree = createTree<5>( d_N, x2 );
    else
        AMP_ERROR( "Not finished" );
    for ( int d = 0; d < d_dim; d++ )
        delete[] x2[d];
}
kdtree::kdtree( kdtree &&rhs )
{
    d_dim      = rhs.d_dim;
    d_N        = rhs.d_N;
    d_tree     = rhs.d_tree;
    rhs.d_tree = nullptr;
}
kdtree &kdtree::operator=( kdtree &&rhs )
{
    if ( this == &rhs )
        return *this;
    d_dim      = rhs.d_dim;
    d_N        = rhs.d_N;
    d_tree     = rhs.d_tree;
    rhs.d_tree = nullptr;
    return *this;
}


/********************************************************
 * Destructor                                            *
 ********************************************************/
kdtree::~kdtree()
{
    if ( d_dim == 1 )
        delete reinterpret_cast<kdtree2<1, int> *>( d_tree );
    else if ( d_dim == 2 )
        delete reinterpret_cast<kdtree2<2, int> *>( d_tree );
    else if ( d_dim == 3 )
        delete reinterpret_cast<kdtree2<3, int> *>( d_tree );
    else if ( d_dim == 4 )
        delete reinterpret_cast<kdtree2<4, int> *>( d_tree );
    else if ( d_dim == 5 )
        delete reinterpret_cast<kdtree2<5, int> *>( d_tree );
    else
        AMP_ERROR( "Not finished" );
    d_tree = nullptr;
}


/********************************************************
 * Specialized constructors                              *
 ********************************************************/
std::shared_ptr<kdtree> kdtree::create2d( const size_t N, const double *x, const double *y )
{
    const double *x2[2] = { x, y };
    return std::make_shared<kdtree>( 2, N, x2 );
}


/********************************************************
 * Specialized constructor (3D)                          *
 ********************************************************/
std::shared_ptr<kdtree>
kdtree::create3d( const size_t N, const double *x, const double *y, const double *z )
{
    const double *x2[3] = { x, y, z };
    return std::make_shared<kdtree>( 3, N, x2 );
}


/********************************************************
 * Return the bounding box of the tree                   *
 ********************************************************/
std::vector<double> kdtree::box() const
{
    if ( d_dim == 1 ) {
        auto ptr = reinterpret_cast<kdtree2<1, int> *>( d_tree );
        auto box = ptr->box();
        return std::vector<double>( box.begin(), box.end() );
    } else if ( d_dim == 2 ) {
        auto ptr = reinterpret_cast<kdtree2<2, int> *>( d_tree );
        auto box = ptr->box();
        return std::vector<double>( box.begin(), box.end() );
    } else if ( d_dim == 3 ) {
        auto ptr = reinterpret_cast<kdtree2<3, int> *>( d_tree );
        auto box = ptr->box();
        return std::vector<double>( box.begin(), box.end() );
    } else if ( d_dim == 4 ) {
        auto ptr = reinterpret_cast<kdtree2<4, int> *>( d_tree );
        auto box = ptr->box();
        return std::vector<double>( box.begin(), box.end() );
    } else if ( d_dim == 5 ) {
        auto ptr = reinterpret_cast<kdtree2<5, int> *>( d_tree );
        auto box = ptr->box();
        return std::vector<double>( box.begin(), box.end() );
    } else {
        AMP_ERROR( "Not finished" );
    }
    return std::vector<double>();
}


/********************************************************
 * Add a point                                           *
 ********************************************************/
void kdtree::add( const double *x )
{
    if ( d_dim == 1 ) {
        reinterpret_cast<kdtree2<1, int> *>( d_tree )->add( { x[0] }, d_N );
    } else if ( d_dim == 2 ) {
        reinterpret_cast<kdtree2<2, int> *>( d_tree )->add( { x[0], x[1] }, d_N );
    } else if ( d_dim == 3 ) {
        reinterpret_cast<kdtree2<3, int> *>( d_tree )->add( { x[0], x[1], x[2] }, d_N );
    } else if ( d_dim == 4 ) {
        reinterpret_cast<kdtree2<4, int> *>( d_tree )->add( { x[0], x[1], x[2], x[3] }, d_N );
    } else if ( d_dim == 5 ) {
        reinterpret_cast<kdtree2<5, int> *>( d_tree )->add( { x[0], x[1], x[2], x[3], x[4] }, d_N );
    } else {
        AMP_ERROR( "Not finished" );
    }
    d_N++;
}


/********************************************************
 * Nearest neighbor search                               *
 ********************************************************/
AMP::Mesh::MeshPoint<double> kdtree::find_nearest( const AMP::Mesh::MeshPoint<double> &p ) const
{
    auto p2      = p;
    double dist2 = 1e100;
    find_nearest2( p.data(), dist2, p2.data() );
    return p2;
}
size_t kdtree::find_nearest( const double *x, double *dist, double *pos ) const
{
    PROFILE( "find_nearest single", 5 );
    double dist2 = 1e100;
    double pos2[64];
    auto index = find_nearest2( x, dist2, pos2 );
    if ( dist )
        *dist = dist2;
    if ( pos )
        memcpy( pos, pos2, d_dim * sizeof( double ) );
    return index;
}
void kdtree::find_nearest( int N, const double *x, size_t *index, double *dist, double *pos ) const
{
    if ( N == 0 )
        return;
    if ( N < 0 )
        ERROR_MSG( "N must be >= 0" );
    if ( index == nullptr )
        ERROR_MSG( "index may not be null" );
    PROFILE( "find_nearest multiple", 3 );
    for ( int i = 0; i < N; i++ ) {
        double dist2 = 1e100;
        double pos2[64];
        index[i] = find_nearest2( &x[d_dim * i], dist2, pos2 );
        if ( dist != nullptr )
            dist[i] = dist2;
        if ( pos != nullptr ) {
            for ( unsigned int d = 0; d < d_dim; d++ )
                pos[d_dim * i + d] = pos2[d];
        }
    }
}
size_t kdtree::find_nearest2( const double *x, double &dist, double *pos ) const
{
    size_t index = 0;
    if ( d_dim == 1 ) {
        std::array<double, 1> x0 = { x[0] };
        auto [p, i]              = reinterpret_cast<kdtree2<1, int> *>( d_tree )->findNearest( x0 );
        dist                     = calcDist<1>( p, x0 );
        memcpy( pos, p.data(), sizeof( p ) );
        index = i;
    } else if ( d_dim == 2 ) {
        std::array<double, 2> x0 = { x[0], x[1] };
        auto [p, i]              = reinterpret_cast<kdtree2<2, int> *>( d_tree )->findNearest( x0 );
        dist                     = calcDist<2>( p, x0 );
        memcpy( pos, p.data(), sizeof( p ) );
        index = i;
    } else if ( d_dim == 3 ) {
        std::array<double, 3> x0 = { x[0], x[1], x[2] };
        auto [p, i]              = reinterpret_cast<kdtree2<3, int> *>( d_tree )->findNearest( x0 );
        dist                     = calcDist<3>( p, x0 );
        memcpy( pos, p.data(), sizeof( p ) );
        index = i;
    } else if ( d_dim == 4 ) {
        std::array<double, 4> x0 = { x[0], x[1], x[2], x[3] };
        auto [p, i]              = reinterpret_cast<kdtree2<4, int> *>( d_tree )->findNearest( x0 );
        dist                     = calcDist<4>( p, x0 );
        memcpy( pos, p.data(), sizeof( p ) );
        index = i;
    } else if ( d_dim == 5 ) {
        std::array<double, 5> x0 = { x[0], x[1], x[2], x[3], x[4] };
        auto [p, i]              = reinterpret_cast<kdtree2<5, int> *>( d_tree )->findNearest( x0 );
        dist                     = calcDist<5>( p, x0 );
        memcpy( pos, p.data(), sizeof( p ) );
        index = i;
    } else {
        AMP_ERROR( "Not finished" );
    }
    return index;
}
size_t kdtree::find_nearest2d( const double x, const double y ) const
{
    AMP_ASSERT( d_dim == 2 );
    PROFILE( "find_nearest 2d", 5 );
    double xy[2] = { x, y }, dist2, pos2[2] = { 0 };
    size_t index = find_nearest2( xy, dist2, pos2 );
    return index;
}
size_t kdtree::find_nearest3d( const double x, const double y, const double z ) const
{
    AMP_ASSERT( d_dim == 3 );
    PROFILE( "find_nearest 3d", 5 );
    double xyz[3] = { x, y, z }, dist2, pos2[3] = { 0 };
    size_t index = find_nearest2( xyz, dist2, pos2 );
    return index;
}


} // namespace AMP
