#include "AMP/mesh/MeshElement.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"

#include <cstring>


namespace AMP::Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
constexpr auto MeshElementHash = AMP::getTypeID<MeshElement>().hash;
static_assert( MeshElementHash != 0 );
MeshElement::MeshElement() : d_typeHash( MeshElementHash ), d_element( nullptr ) {}
MeshElement::MeshElement( const MeshElement &rhs )
    : d_typeHash( MeshElementHash ), d_element( nullptr )
{
    if ( rhs.d_element == nullptr && rhs.d_typeHash == MeshElementHash ) {
        d_element = nullptr;
    } else if ( rhs.d_typeHash != MeshElementHash ) {
        d_element = rhs.clone();
    } else {
        d_element = rhs.d_element->clone();
    }
}
MeshElement::MeshElement( MeshElement &&rhs )
    : d_typeHash( MeshElementHash ), d_element( rhs.d_element )
{
    if ( rhs.d_typeHash != MeshElementHash )
        d_element = rhs.clone();
    rhs.d_element = nullptr;
}
MeshElement &MeshElement::operator=( const MeshElement &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    if ( d_element != nullptr ) {
        // Delete the existing d_element
        delete d_element;
        d_element = nullptr;
    }
    d_typeHash = MeshElementHash;
    if ( rhs.d_element == nullptr && rhs.d_typeHash == MeshElementHash ) {
        d_element = nullptr;
    } else if ( rhs.d_typeHash != MeshElementHash ) {
        d_element = rhs.clone();
    } else {
        d_element = rhs.d_element->clone();
    }
    return *this;
}
MeshElement &MeshElement::operator=( MeshElement &&rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    if ( d_element != nullptr ) {
        // Delete the existing d_element
        delete d_element;
        d_element = nullptr;
    }
    d_typeHash = MeshElementHash;
    std::swap( d_element, rhs.d_element );
    if ( rhs.d_typeHash != MeshElementHash )
        d_element = rhs.clone();
    return *this;
}
MeshElement::MeshElement( MeshElement *rhs ) : d_typeHash( MeshElementHash ), d_element( nullptr )
{
    if ( rhs->d_element ) {
        std::swap( d_element, rhs->d_element );
        delete rhs;
    } else {
        d_element = rhs;
    }
}


/********************************************************
 * Destructor                                            *
 ********************************************************/
MeshElement::~MeshElement()
{
    if ( d_element != nullptr )
        delete d_element;
    d_element = nullptr;
}


/********************************************************
 * Is the d_element null                                   *
 ********************************************************/
bool MeshElement::isNull() const { return d_typeHash == MeshElementHash && d_element == nullptr; }


/********************************************************
 * Function to return basic info                         *
 ********************************************************/
std::string MeshElement::elementClass() const
{
    return d_element == nullptr ? std::string( "MeshElement" ) : d_element->elementClass();
}


/********************************************************
 * Function to return the centroid of an d_element         *
 ********************************************************/
Point MeshElement::centroid() const
{
    if ( d_element != nullptr )
        return d_element->centroid();
    if ( globalID().type() == GeomType::Vertex )
        return coord();
    std::vector<MeshElement> nodes;
    ( d_element != nullptr ? d_element : this )->getElements( GeomType::Vertex, nodes );
    AMP_ASSERT( !nodes.empty() );
    auto center = nodes[0].coord();
    for ( size_t i = 1; i < nodes.size(); i++ ) {
        auto pos = nodes[i].coord();
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
void MeshElement::getElements( const GeomType type, std::vector<MeshElement> &elements ) const
{
    if ( d_element == nullptr )
        AMP_ERROR( "getElements is not implemented for the base class (" + elementClass() + ")" );
    d_element->getElements( type, elements );
}
int MeshElement::getElementsID( const GeomType type, MeshElementID *ID ) const
{
    if ( d_element != nullptr )
        return d_element->getElementsID( type, ID );
    std::vector<MeshElement> d_elements;
    this->getElements( type, d_elements );
    for ( size_t i = 0; i < d_elements.size(); i++ )
        ID[i] = d_elements[i].globalID();
    return d_elements.size();
}
void MeshElement::getNeighbors( std::vector<std::unique_ptr<MeshElement>> &neighbors ) const
{
    if ( d_element == nullptr )
        AMP_ERROR( "getNeighbors is not implemented for the base class (" + elementClass() + ")" );
    d_element->getNeighbors( neighbors );
}
void MeshElement::getNeighborVertices( std::vector<Point> &vertices ) const
{
    std::vector<Point> V0;
    getVertices( V0 );
    std::vector<std::unique_ptr<MeshElement>> neighbors;
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
            for ( auto p0 : V0 )
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
    if ( d_element )
        AMP_ERROR( "getNeighbors is not implemented for the base class (" + elementClass() + ")" );
    if ( globalID().type() == GeomType::Vertex ) {
        vertices.resize( 1 );
        vertices[0] = coord();
    } else {
        auto elems = getElements( GeomType::Vertex );
        vertices.resize( elems.size() );
        for ( size_t i = 0; i < elems.size(); i++ )
            vertices[i] = elems[i].coord();
    }
}


/********************************************************
 * Function to check if a point is within an d_element     *
 ********************************************************/
bool MeshElement::containsPoint( const Point &pos, double TOL ) const
{
    if ( d_element != nullptr )
        return d_element->containsPoint( pos, TOL );
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
    if ( d_element == nullptr && d_typeHash == MeshElementHash )
        return stringf( "%sMeshElement: null d_element", prefix );
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
            out += " " + node.coord().print();
        out += stringf( "\n%s   Volume: %f\n", prefix, volume() );
    }
    if ( out.back() == '\n' )
        out.resize( out.size() - 1 );
    return out;
}

/********************************************************
 * Functions that aren't implemented for the base class  *
 ********************************************************/
Point MeshElement::coord() const
{
    if ( d_element == nullptr )
        AMP_ERROR( "coord is not implemented for the base class (" + elementClass() + ")" );
    return d_element->coord();
}
double MeshElement::volume() const
{
    if ( d_element == nullptr )
        AMP_ERROR( "volume is not implemented for the base class (" + elementClass() + ")" );
    return d_element->volume();
}
Point MeshElement::norm() const
{
    if ( d_element == nullptr )
        AMP_ERROR( "norm is not implemented for the base class (" + elementClass() + ")" );
    return d_element->norm();
}
Point MeshElement::nearest( const Point &pos ) const
{
    if ( d_element == nullptr )
        AMP_ERROR( "nearest is not implemented for the base class (" + elementClass() + ")" );
    return d_element->nearest( pos );
}
double MeshElement::distance( const Point &pos, const Point &dir ) const
{
    if ( d_element == nullptr )
        AMP_ERROR( "distance is not implemented for the base class (" + elementClass() + ")" );
    return d_element->distance( pos, dir );
}
bool MeshElement::isOnSurface() const
{
    if ( d_element == nullptr )
        AMP_ERROR( "isOnSurface is not implemented for the base class (" + elementClass() + ")" );
    return d_element->isOnSurface();
}
bool MeshElement::isOnBoundary( int id ) const
{
    if ( d_element == nullptr )
        AMP_ERROR( "isOnBoundary is not implemented for the base class (" + elementClass() + ")" );
    return d_element->isOnBoundary( id );
}
bool MeshElement::isInBlock( int id ) const
{
    if ( d_element == nullptr )
        AMP_ERROR( "isInBlock is not implemented for the base class (" + elementClass() + ")" );
    return d_element->isInBlock( id );
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
MeshElementID MeshElement::globalID() const
{
    if ( d_element == nullptr )
        return MeshElementID();
    return d_element->globalID();
}


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
#include "AMP/mesh/MeshElementVectorIterator.hpp"
#include "AMP/utils/Utilities.hpp"
using intTuple3 = std::tuple<int, int, int>;
AMP_INSTANTIATE_SORT( intTuple3 );
AMP_INSTANTIATE_SORT( AMP::Mesh::MeshElement );
template void AMP::Utilities::quicksort<intTuple3, AMP::Mesh::MeshElement>(
    size_t, intTuple3 *, AMP::Mesh::MeshElement * );
template void AMP::Utilities::quicksort<intTuple3, AMP::Mesh::MeshElement>(
    std::vector<intTuple3> &, std::vector<AMP::Mesh::MeshElement> & );
template void AMP::Utilities::quicksort<AMP::Mesh::MeshElementID, size_t>(
    size_t, AMP::Mesh::MeshElementID *, size_t * );
template class AMP::Mesh::MeshElementVectorIterator<AMP::Mesh::MeshElement>;
