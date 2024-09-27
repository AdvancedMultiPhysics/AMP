#include "AMP/mesh/triangle/TriangleMeshElement.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/mesh/MeshPoint.h"
#include "AMP/mesh/triangle/TriangleMesh.h"
#include "AMP/mesh/triangle/TriangleMeshIterator.h"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/UtilityMacros.h"

#include <limits>


namespace AMP::Mesh {


/****************************************************************
 * Overload basic operations                                     *
 ****************************************************************/
template<size_t N>
static inline std::array<double, N> operator*( double x, const std::array<double, N> &y )
{
    if constexpr ( N == 1 )
        return { x * y[0] };
    else if constexpr ( N == 2 )
        return { x * y[0], x * y[1] };
    else if constexpr ( N == 3 )
        return { x * y[0], x * y[1], x * y[2] };
    return {};
}
template<size_t N>
static inline std::array<double, N> operator-( const std::array<double, N> &x,
                                               const std::array<double, N> &y )
{
    if constexpr ( N == 1 )
        return { x[0] - y[0] };
    else if constexpr ( N == 2 )
        return { x[0] - y[0], x[1] - y[1] };
    else if constexpr ( N == 3 )
        return { x[0] - y[0], x[1] - y[1], x[2] - y[2] };
    return {};
}
template<size_t N>
static inline double abs( const std::array<double, N> &x )
{
    if constexpr ( N == 1 )
        return std::abs( x[0] );
    else if constexpr ( N == 2 )
        return sqrt( x[0] * x[0] + x[1] * x[1] );
    else if constexpr ( N == 3 )
        return sqrt( x[0] * x[0] + x[1] * x[1] + x[2] * x[2] );
    return {};
}
template<size_t N>
static inline double dot( const std::array<double, N> &x, const std::array<double, N> &y )
{
    if constexpr ( N == 1 )
        return x[0] * y[0];
    else if constexpr ( N == 2 )
        return x[0] * y[0] + x[1] * y[1];
    else if constexpr ( N == 3 )
        return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    return {};
}
template<size_t N>
static inline std::array<double, N> normalize( const std::array<double, N> &x )
{
    if constexpr ( N == 1 ) {
        return { 1.0 };
    } else if constexpr ( N == 2 ) {
        double tmp = 1.0 / sqrt( x[0] * x[0] + x[1] * x[1] );
        return { tmp * x[0], tmp * x[1] };
    } else if constexpr ( N == 3 ) {
        double tmp = 1.0 / sqrt( x[0] * x[0] + x[1] * x[1] + x[2] * x[2] );
        return { tmp * x[0], tmp * x[1], tmp * x[2] };
    }
    return {};
}


/****************************************************************
 * Get the number of n-Simplex elements of each type             *
 ****************************************************************/
// clang-format off
static constexpr uint8_t n_Simplex_elements[4][4] = {
    {  1, 0, 0, 0 },
    {  2, 1, 0, 0 },
    {  3, 3, 1, 0 },
    {  4, 6, 4, 1 },
};
// clang-format on


/********************************************************
 * Create a unique id for each class                     *
 ********************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
std::string TriangleMeshElement<NG, NP, TYPE>::elementClass() const
{
    return AMP::Utilities::stringf( "TriangleMeshElement<%u,%u,%u>", NG, NP, TYPE );
}


/********************************************************
 * Constructors                                          *
 ********************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshElement<NG, NP, TYPE>::TriangleMeshElement()
{
    static constexpr auto hash = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( hash != 0 );
    d_typeHash = hash;
    d_element  = nullptr;
    d_mesh     = nullptr;
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshElement<NG, NP, TYPE>::TriangleMeshElement( const MeshElementID &id,
                                                        const TriangleMesh<NG, NP> *mesh )
{
    static constexpr auto hash = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( hash != 0 );
    d_typeHash = hash;
    d_element  = nullptr;
    d_globalID = id;
    d_mesh     = mesh;
#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    auto type = static_cast<uint8_t>( id.type() );
    if ( type != TYPE && type != 255 )
        printf( "%i %i %i %i\n", NG, NP, TYPE, type );
    AMP_ASSERT( type == TYPE || type == 255 );
#endif
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshElement<NG, NP, TYPE>::TriangleMeshElement( const TriangleMeshElement &rhs )
    : MeshElement(), d_mesh( rhs.d_mesh ), d_globalID( rhs.d_globalID )
{
    static constexpr auto hash = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( hash != 0 );
    d_typeHash = hash;
    d_element  = rhs.d_element;
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshElement<NG, NP, TYPE>::TriangleMeshElement( TriangleMeshElement &&rhs )
    : MeshElement(), d_mesh( rhs.d_mesh ), d_globalID{ rhs.d_globalID }
{
    d_typeHash = rhs.d_typeHash;
    d_element  = nullptr;
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshElement<NG, NP, TYPE> &
TriangleMeshElement<NG, NP, TYPE>::operator=( const TriangleMeshElement &rhs )
{
    if ( &rhs == this )
        return *this;
    d_typeHash = rhs.d_typeHash;
    d_element  = nullptr;
    d_globalID = rhs.d_globalID;
    d_mesh     = rhs.d_mesh;
    return *this;
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshElement<NG, NP, TYPE> &
TriangleMeshElement<NG, NP, TYPE>::operator=( TriangleMeshElement &&rhs )
{
    if ( &rhs == this )
        return *this;
    d_typeHash = rhs.d_typeHash;
    d_element  = nullptr;
    d_globalID = rhs.d_globalID;
    d_mesh     = rhs.d_mesh;
    return *this;
}


/****************************************************************
 * Function to clone the element                                 *
 ****************************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshElement *TriangleMeshElement<NG, NP, TYPE>::clone() const
{
    return new TriangleMeshElement<NG, NP, TYPE>( *this );
}


/****************************************************************
 * Return the global rank of the owner rank                      *
 ****************************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
unsigned int TriangleMeshElement<NG, NP, TYPE>::globalOwnerRank() const
{
    return d_mesh->getComm().globalRanks()[d_globalID.owner_rank()];
}


/****************************************************************
 * Function to get the elements composing the current element    *
 ****************************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
void TriangleMeshElement<NG, NP, TYPE>::getElementsID( const GeomType type,
                                                       std::vector<MeshElementID> &ID ) const
{
    // Number of elements composing a given type
    int N = n_Simplex_elements[TYPE][static_cast<uint8_t>( type )];
    // Get the element ids
    ElementID tmp[6];
    d_mesh->getElementsIDs( d_globalID.elemID(), type, tmp );
    ID.resize( N );
    for ( int i = 0; i < N; i++ )
        ID[i] = MeshElementID( d_globalID.meshID(), tmp[i] );
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
void TriangleMeshElement<NG, NP, TYPE>::getElements( const GeomType type,
                                                     std::vector<MeshElement> &children ) const
{
    // Number of elements composing a given type
    int N = n_Simplex_elements[TYPE][static_cast<uint8_t>( type )];
    // Get the element ids
    ElementID tmp[6];
    d_mesh->getElementsIDs( d_globalID.elemID(), type, tmp );
    // Create the mesh elements
    auto meshID = d_globalID.meshID();
    children.resize( N );
    for ( int i = 0; i < N; i++ )
        children[i] = d_mesh->getElement( MeshElementID( meshID, tmp[i] ) );
}


/****************************************************************
 * Function to get the neighboring elements                      *
 ****************************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
void TriangleMeshElement<NG, NP, TYPE>::getNeighbors(
    std::vector<std::unique_ptr<MeshElement>> &neighbors ) const
{
    std::vector<ElementID> neighborIDs;
    d_mesh->getNeighborIDs( d_globalID.elemID(), neighborIDs );
    neighbors.resize( neighborIDs.size() );
    auto meshID = d_globalID.meshID();
    for ( size_t i = 0; i < neighborIDs.size(); i++ )
        neighbors[i].reset( d_mesh->getElement2( MeshElementID( meshID, neighborIDs[i] ) ) );
}


/****************************************************************
 * Get the coordinates of the vertices                          *
 ****************************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
inline std::array<std::array<double, NP>, TYPE + 1>
TriangleMeshElement<NG, NP, TYPE>::getVertexCoord() const
{
    if constexpr ( TYPE == 0 ) {
        return { d_mesh->getPos( d_globalID.elemID() ) };
    } else {
        std::array<std::array<double, NP>, TYPE + 1> x;
        ElementID ids[TYPE + 1];
        d_mesh->getVerticies( d_globalID.elemID(), ids );
        for ( size_t i = 0; i <= TYPE; i++ )
            x[i] = d_mesh->getPos( ids[i] );
        return x;
    }
}


/****************************************************************
 * Functions to get basic element properties                     *
 ****************************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
double TriangleMeshElement<NG, NP, TYPE>::volume() const
{
    if constexpr ( TYPE == 0 ) {
        return 0;
    } else if constexpr ( TYPE == 1 ) {
        auto x = getVertexCoord();
        return abs( x[1] - x[0] );
    } else if constexpr ( TYPE == 2 ) {
        auto x   = getVertexCoord();
        auto AB  = x[1] - x[0];
        auto AC  = x[2] - x[0];
        double t = dot( AB, AC );
        double V = 0.5 * sqrt( dot( AB, AB ) * dot( AC, AC ) - t * t );
        AMP_ASSERT( V == V );
        return V;
    } else if constexpr ( TYPE == NP ) {
        // Calculate the volume of a N-dimensional simplex
        auto x = getVertexCoord();
        auto V = DelaunayHelpers::calcVolume<TYPE, double, double>( x.data() );
        AMP_ASSERT( V > 0.0 );
        return V;
    } else {
        AMP_ERROR( elementClass() + "volume - Not finished" );
        return 0;
    }
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshPoint<double> TriangleMeshElement<NG, NP, TYPE>::norm() const
{
    if constexpr ( TYPE == 2 && NP == 3 ) {
        auto x = getVertexCoord();
        auto n = AMP::Geometry::GeometryHelpers::normal( x[0], x[1], x[2] );
        return { n[0], n[1], n[2] };
    } else {
        AMP_ERROR( elementClass() + "norm - Not finished" );
    }
    return MeshPoint<double>();
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshPoint<double> TriangleMeshElement<NG, NP, TYPE>::coord() const
{
    if constexpr ( TYPE == 0 ) {
        auto x = d_mesh->getPos( d_globalID.elemID() );
        return MeshPoint<double>( NP, x.data() );
    } else {
        AMP_ERROR( "coord is only valid for vertices: " + std::to_string( (int) TYPE ) );
        return MeshPoint<double>();
    }
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshPoint<double> TriangleMeshElement<NG, NP, TYPE>::centroid() const
{
    if constexpr ( TYPE == 0 )
        return MeshPoint<double>( d_mesh->getPos( d_globalID.elemID() ) );
    auto x = getVertexCoord();
    for ( size_t i = 1; i <= TYPE; i++ ) {
        for ( size_t d = 0; d < NP; d++ )
            x[0][d] += x[i][d];
    }
    for ( size_t d = 0; d < NP; d++ )
        x[0][d] /= ( TYPE + 1 );
    return MeshPoint<double>( (size_t) NP, x[0].data() );
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
bool TriangleMeshElement<NG, NP, TYPE>::containsPoint( const MeshPoint<double> &pos,
                                                       double TOL ) const
{
    // Check if the point is in the triangle
    if constexpr ( TYPE == 2 && NP == 3 ) {
        // Compute barycentric coordinates
        auto x = getVertexCoord();
        auto L =
            AMP::Geometry::GeometryHelpers::barycentric<3, 3>( x, { pos.x(), pos.y(), pos.z() } );
        return ( L[0] >= -TOL ) && ( L[1] >= -TOL ) && ( L[2] >= -TOL );
    } else if constexpr ( TYPE == 3 && NP == 3 ) {
        // Compute barycentric coordinates
        auto x = getVertexCoord();
        auto L =
            AMP::Geometry::GeometryHelpers::barycentric<4, 3>( x, { pos.x(), pos.y(), pos.z() } );
        return ( L[0] >= -TOL ) && ( L[1] >= -TOL ) && ( L[2] >= -TOL ) && ( L[3] >= -TOL );
    } else {
        AMP_ERROR( elementClass() + "containsPoint - Not finished" );
    }
    return false;
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
bool TriangleMeshElement<NG, NP, TYPE>::isOnSurface() const
{
    return d_mesh->isOnSurface( d_globalID.elemID() );
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
bool TriangleMeshElement<NG, NP, TYPE>::isOnBoundary( int id ) const
{
    return d_mesh->isOnBoundary( d_globalID.elemID(), id );
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
bool TriangleMeshElement<NG, NP, TYPE>::isInBlock( int id ) const
{
    return d_mesh->isInBlock( d_globalID.elemID(), id );
}


/****************************************************************
 * Calculate the nearest point on the element                    *
 ****************************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshPoint<double> TriangleMeshElement<NG, NP, TYPE>::nearest( const MeshPoint<double> &pos ) const
{
    // Get the vertex coordinates
    if constexpr ( TYPE == 0 ) {
        // Nearest point to a vertex is the vertex
        auto x = getVertexCoord();
        return MeshPoint( NP, x[0].data() );
    } else if constexpr ( TYPE == 1 && NP == 3 ) {
        // Nearest point to a line in 3D
        auto x = getVertexCoord();
        auto p =
            AMP::Geometry::GeometryHelpers::nearest( x[0], x[1], { pos.x(), pos.y(), pos.z() } );
        return { p[0], p[1], p[2] };
    } else if constexpr ( TYPE == 2 && NP == 3 ) {
        // Nearest point to a triangle in 3D
        auto x = getVertexCoord();
        auto p = AMP::Geometry::GeometryHelpers::nearest( x, { pos.x(), pos.y(), pos.z() } );
        return { p[0], p[1], p[2] };
    } else if constexpr ( TYPE == 3 && NP == 3 ) {
        // Nearest point to a tet in 3D
        if ( containsPoint( pos ) )
            return pos;
        auto x                   = getVertexCoord();
        std::array<double, 3> p0 = { pos.x(), pos.y(), pos.z() };
        MeshPoint<double> p      = { 1e100, 1e100, 1e100 };
        for ( int i = 0; i < 4; i++ ) {
            std::array<std::array<double, 3>, 4> x2;
            for ( int j = 0, k = 0; j < 4; j++ ) {
                if ( i != j ) {
                    x2[k] = x[i];
                    k++;
                }
            }
            MeshPoint<double> p2 = AMP::Geometry::GeometryHelpers::nearest( x2, p0 );
            if ( ( p2 - pos ).norm() < ( p - pos ).norm() )
                p = p2;
        }
        return p;
    } else {
        AMP_ERROR( elementClass() + "nearest - Not finished" );
    }
    return MeshPoint<double>();
}


/****************************************************************
 * Calculate the distance to the element                         *
 ****************************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
double TriangleMeshElement<NG, NP, TYPE>::distance( const MeshPoint<double> &pos,
                                                    const MeshPoint<double> &dir ) const
{
    // Get the vertex coordinates
    if constexpr ( TYPE == 2 && NP == 3 ) {
        auto x = getVertexCoord();
        return AMP::Geometry::GeometryHelpers::distanceToTriangle( x, pos, dir );
    } else if constexpr ( TYPE == 3 && NP == 3 ) {
        auto x = getVertexCoord();
        return AMP::Geometry::GeometryHelpers::distanceToTetrahedron( x, pos, dir );
    } else {
        AMP_ERROR( elementClass() + "::distance - Not finished" );
    }
    return 0;
}


/********************************************************
 *  Explicit instantiations of TriangleMeshElement       *
 ********************************************************/
DISABLE_WARNINGS
template class TriangleMeshElement<1, 1, 0>;
template class TriangleMeshElement<1, 1, 1>;
template class TriangleMeshElement<1, 2, 0>;
template class TriangleMeshElement<1, 2, 1>;
template class TriangleMeshElement<1, 3, 0>;
template class TriangleMeshElement<1, 3, 1>;
template class TriangleMeshElement<2, 2, 0>;
template class TriangleMeshElement<2, 2, 1>;
template class TriangleMeshElement<2, 2, 2>;
template class TriangleMeshElement<2, 3, 0>;
template class TriangleMeshElement<2, 3, 1>;
template class TriangleMeshElement<2, 3, 2>;
template class TriangleMeshElement<3, 3, 0>;
template class TriangleMeshElement<3, 3, 1>;
template class TriangleMeshElement<3, 3, 2>;
template class TriangleMeshElement<3, 3, 3>;
ENABLE_WARNINGS


} // namespace AMP::Mesh
