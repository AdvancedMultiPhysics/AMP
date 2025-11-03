#include "AMP/mesh/MeshElement.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"

#include <algorithm>
#include <cstring>
#include <numeric>


namespace AMP::Mesh {


static_assert( sizeof( std::unique_ptr<MeshElement> ) == 8 );


/********************************************************
 * Get basic info                                        *
 ********************************************************/
typeID MeshElement::getTypeID() const { return AMP::getTypeID<MeshElement>(); }
bool MeshElement::isNull() const { return getTypeID() == AMP::getTypeID<MeshElement>(); }
std::string MeshElement::elementClass() const { return "MeshElement"; }
std::unique_ptr<MeshElement> MeshElement::clone() const { return std::make_unique<MeshElement>(); }


/********************************************************
 * Function to return the centroid of an d_element       *
 ********************************************************/
Point MeshElement::centroid() const
{
    if ( globalID().type() == GeomType::Vertex )
        return coord();
    ElementList nodes;
    getElements( GeomType::Vertex, nodes );
    AMP_ASSERT( !nodes.empty() );
    auto center = nodes[0]->coord();
    for ( size_t i = 1; i < nodes.size(); i++ ) {
        auto pos = nodes[i]->coord();
        for ( size_t j = 0; j < center.size(); j++ )
            center[j] += pos[j];
    }
    for ( auto &x : center )
        x /= nodes.size();
    return center;
}


/********************************************************
 * Return the neighbors/elements                         *
 ********************************************************/
MeshElement::ElementList MeshElement::getElements( const GeomType type ) const
{
    ElementList list;
    getElements( type, list );
    return list;
}
void MeshElement::getElements( const GeomType, ElementList & ) const
{
    AMP_ERROR( "getElements is not implemented for " + elementClass() );
}
int MeshElement::getElementsID( const GeomType type, MeshElementID *ID ) const
{
    ElementList elements;
    this->getElements( type, elements );
    for ( size_t i = 0; i < elements.size(); i++ )
        ID[i] = elements[i]->globalID();
    return elements.size();
}
MeshElement::ElementList MeshElement::getNeighbors() const
{
    ElementList list;
    getNeighbors( list );
    return list;
}
void MeshElement::getNeighbors( ElementList & ) const
{
    AMP_ERROR( "getNeighbors is not implemented for " + elementClass() );
}
void MeshElement::getNeighborVertices( std::vector<Point> &vertices ) const
{
    std::vector<Point> V0;
    getVertices( V0 );
    ElementList neighbors;
    getNeighbors( neighbors );
    vertices.resize( 0 );
    vertices.reserve( 24 );
    std::vector<Point> V1;
    for ( auto &elem : neighbors ) {
        if ( !elem )
            continue;
        elem->getVertices( V1 );
        for ( auto &p : V1 ) {
            bool found = false;
            for ( auto &p0 : V0 )
                found = found || p == p0;
            if ( !found )
                vertices.push_back( p );
        }
    }
}


/********************************************************
 * Return the vertices                                  *
 ********************************************************/
void MeshElement::getVertices( std::vector<Point> &vertices ) const
{
    if ( globalID().type() == GeomType::Vertex ) {
        vertices.resize( 1 );
        vertices[0] = coord();
    } else {
        auto elems = getElements( GeomType::Vertex );
        vertices.resize( elems.size() );
        for ( size_t i = 0; i < elems.size(); i++ )
            vertices[i] = elems[i]->coord();
    }
}


/********************************************************
 * Function to check if a point is within an d_element     *
 ********************************************************/
bool MeshElement::containsPoint( const Point &pos, double TOL ) const
{
    if ( globalID().type() == GeomType::Vertex ) {
        // double dist = 0.0;
        auto point   = this->coord();
        double dist2 = 0.0;
        for ( size_t i = 0; i < point.size(); i++ )
            dist2 += ( point[i] - pos[i] ) * ( point[i] - pos[i] );
        return dist2 <= TOL * TOL;
    }
    AMP_ERROR( "containsPoint is not finished for default d_elements yet" );
    return false;
}


/********************************************************
 * Function to print debug info                          *
 ********************************************************/
std::string MeshElement::print( uint8_t indent_N ) const
{
    using AMP::Utilities::stringf;
    char prefix[256] = { 0 };
    memset( prefix, 0x20, indent_N );
    int type        = static_cast<int>( elementType() );
    std::string out = prefix + elementClass() + "\n";
    out += stringf( "%s   ID = (%i,%i,%u,%u,%lu)\n",
                    prefix,
                    globalID().is_local() ? 1 : 0,
                    static_cast<int>( globalID().type() ),
                    globalID().local_id(),
                    globalID().owner_rank(),
                    globalID().meshID().getData() );
    out += stringf( "%s   Type: %i\n", prefix, type );
    out += stringf( "%s   Centroid: %s\n", prefix, centroid().print().data() );
    if ( type != 0 ) {
        auto nodes = getElements( AMP::Mesh::GeomType::Vertex );
        out += std::string( prefix ) + "   Nodes:";
        for ( const auto &node : nodes )
            out += " " + node->coord().print();
        out += stringf( "\n%s   Volume: %f\n", prefix, volume() );
    }
    if ( out.back() == '\n' )
        out.resize( out.size() - 1 );
    return out;
}

/********************************************************
 * Functions that aren't implemented for the base class  *
 ********************************************************/
Point MeshElement::coord() const { AMP_ERROR( "coord is not implemented for " + elementClass() ); }
double MeshElement::volume() const
{
    AMP_ERROR( "volume is not implemented for " + elementClass() );
}
Point MeshElement::norm() const { AMP_ERROR( "norm is not implemented for " + elementClass() ); }
Point MeshElement::nearest( const Point & ) const
{
    AMP_ERROR( "nearest is not implemented for " + elementClass() );
}
double MeshElement::distance( const Point &, const Point & ) const
{
    AMP_ERROR( "distance is not implemented for " + elementClass() );
}
bool MeshElement::isOnSurface() const
{
    AMP_ERROR( "isOnSurface is not implemented for " + elementClass() );
}
bool MeshElement::isOnBoundary( int ) const
{
    AMP_ERROR( "isOnBoundary is not implemented for " + elementClass() );
}
bool MeshElement::isInBlock( int ) const
{
    AMP_ERROR( "isInBlock is not implemented for " + elementClass() );
}
unsigned int MeshElement::globalOwnerRank( const Mesh &mesh ) const
{
    auto id = globalID();
    if ( id.meshID() == mesh.meshID() )
        return mesh.getComm().globalRanks()[id.owner_rank()];
    auto mesh2 = mesh.Subset( id.meshID() );
    if ( mesh2 )
        return mesh2->getComm().globalRanks()[id.owner_rank()];
    AMP_ERROR( "globalOwnerRank is not able to find mesh element" );
}
MeshElementID MeshElement::globalID() const { return MeshElementID(); }


// Stream operator
std::ostream &operator<<( std::ostream &out, const AMP::Mesh::MeshElement &x )
{
    out << x.print();
    return out;
}


} // namespace AMP::Mesh


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
using ElementPtr = std::unique_ptr<AMP::Mesh::MeshElement>;
static size_t find( size_t n, const ElementPtr *x, const AMP::Mesh::MeshElementID &id )
{
    if ( n == 0 )
        return 0;
    // Check if value is within the range of x
    if ( id <= x[0]->globalID() )
        return 0;
    else if ( id > x[n - 1]->globalID() )
        return n;
    // Perform the search
    size_t lower = 0;
    size_t upper = n - 1;
    size_t index;
    while ( ( upper - lower ) != 1 ) {
        index = ( upper + lower ) / 2;
        if ( x[index]->globalID() >= id )
            upper = index;
        else
            lower = index;
    }
    index = upper;
    return index;
}
template<>
size_t
AMP::Utilities::findfirst<ElementPtr>( size_t N, const ElementPtr *x, const ElementPtr &value )
{
    return find( N, x, value->globalID() );
}
template<>
void AMP::Utilities::quicksort<ElementPtr>( size_t N, ElementPtr *x )
{
    auto compare = []( const ElementPtr &a, const ElementPtr &b ) { return *a < *b; };
    std::sort( x, x + N, compare );
}
template<>
void AMP::Utilities::unique<ElementPtr>( std::vector<ElementPtr> &x )
{
    if ( x.size() <= 1 )
        return;
    // First perform a quicksort
    auto compare = []( const ElementPtr &a, const ElementPtr &b ) { return *a < *b; };
    std::sort( x.begin(), x.end(), compare );
    // Next remove duplicate entries
    size_t pos = 1;
    for ( size_t i = 1; i < x.size(); i++ ) {
        if ( *x[i] != *x[pos - 1] ) {
            x[pos] = std::move( x[i] );
            pos++;
        }
    }
    if ( pos < x.size() )
        x.resize( pos );
}
