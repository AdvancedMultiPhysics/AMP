#include "AMP/mesh/libmesh/libmeshMeshElement.h"
#include "AMP/utils/Utilities.h"

// libMesh includes
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/boundary_info.h"
#include "libmesh/elem.h"

namespace AMP::Mesh {


// Functions to create new ids by mixing existing ids
static unsigned int generate_id( const std::vector<unsigned int> &ids );


/********************************************************
 * Constructors                                          *
 ********************************************************/
static constexpr auto elementTypeID = AMP::getTypeID<libmeshMeshElement>().hash;
static_assert( elementTypeID != 0 );
libmeshMeshElement::libmeshMeshElement()
    : d_dim( -1 ),
      d_rank( 0 ),
      ptr_element( nullptr ),
      d_mesh( nullptr ),
      d_delete_elem( false ),
      d_globalID( MeshElementID() )
{
    d_typeHash = elementTypeID;
    d_element  = nullptr;
}
libmeshMeshElement::libmeshMeshElement( int dim,
                                        GeomType type,
                                        void *libmesh_element,
                                        unsigned int rank,
                                        MeshID meshID,
                                        const libmeshMesh *mesh )
{
    AMP_ASSERT( libmesh_element );
    d_typeHash      = elementTypeID;
    d_element       = nullptr;
    d_dim           = dim;
    d_rank          = rank;
    d_mesh          = mesh;
    d_meshID        = meshID;
    ptr_element     = libmesh_element;
    auto local_id   = (unsigned int) -1;
    auto owner_rank = (unsigned int) -1;
    bool is_local   = false;
    if ( type == GeomType::Vertex ) {
        auto *node = (libMesh::Node *) ptr_element;
        local_id   = node->id();
        owner_rank = node->processor_id();
        is_local   = owner_rank == d_rank;
    } else if ( type == (GeomType) dim ) {
        auto *elem = (libMesh::Elem *) ptr_element;
        AMP_DEBUG_ASSERT( elem->n_neighbors() < 100 );
        local_id   = elem->id();
        owner_rank = elem->processor_id();
        is_local   = owner_rank == d_rank;
    } else {
        AMP_ERROR( "Unreconized element" );
    }
    d_globalID = MeshElementID( is_local, type, local_id, owner_rank, meshID );
}
libmeshMeshElement::libmeshMeshElement( int dim,
                                        GeomType type,
                                        std::shared_ptr<libMesh::Elem> libmesh_element,
                                        unsigned int rank,
                                        MeshID meshID,
                                        const libmeshMesh *mesh )
    : d_delete_elem( false )
{
    AMP_ASSERT( libmesh_element );
    d_typeHash      = elementTypeID;
    d_element       = nullptr;
    d_dim           = dim;
    d_rank          = rank;
    d_mesh          = mesh;
    d_meshID        = meshID;
    ptr2            = libmesh_element;
    ptr_element     = libmesh_element.get();
    auto local_id   = (unsigned int) -1;
    auto owner_rank = (unsigned int) -1;
    bool is_local   = false;
    if ( type == GeomType::Vertex ) {
        auto *node = (libMesh::Node *) ptr_element;
        local_id   = node->id();
        owner_rank = node->processor_id();
        is_local   = owner_rank == d_rank;
    } else {
        auto *elem = (libMesh::Elem *) ptr_element;
        local_id   = elem->id();
        owner_rank = elem->processor_id();
        is_local   = owner_rank == d_rank;
    }
    d_globalID = MeshElementID( is_local, type, local_id, owner_rank, meshID );
}
libmeshMeshElement::libmeshMeshElement( const libmeshMeshElement &rhs )
    : MeshElement(), // Note: we never want to call the base copy constructor
      ptr2( rhs.ptr2 ),
      d_meshID( rhs.d_meshID ),
      d_delete_elem( false ),
      d_globalID( rhs.d_globalID )
{
    d_typeHash  = elementTypeID;
    d_element   = nullptr;
    d_dim       = rhs.d_dim;
    ptr_element = rhs.ptr_element;
    d_rank      = rhs.d_rank;
    d_mesh      = rhs.d_mesh;
}
libmeshMeshElement &libmeshMeshElement::operator=( const libmeshMeshElement &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    this->d_typeHash    = elementTypeID;
    this->d_element     = nullptr;
    this->d_globalID    = rhs.d_globalID;
    this->d_dim         = rhs.d_dim;
    this->ptr_element   = rhs.ptr_element;
    this->ptr2          = rhs.ptr2;
    this->d_rank        = rhs.d_rank;
    this->d_mesh        = rhs.d_mesh;
    this->d_meshID      = rhs.d_meshID;
    this->d_delete_elem = false;
    return *this;
}


/****************************************************************
 * De-constructor                                                *
 ****************************************************************/
libmeshMeshElement::~libmeshMeshElement() { d_element = nullptr; }


/****************************************************************
 * Function to clone the element                                 *
 ****************************************************************/
MeshElement *libmeshMeshElement::clone() const { return new libmeshMeshElement( *this ); }


/****************************************************************
 * Function to get the elements composing the current element    *
 ****************************************************************/
void libmeshMeshElement::getElements( const GeomType type,
                                      std::vector<MeshElement> &children ) const
{
    AMP_INSIST( type <= d_globalID.type(), "sub-elements must be of a smaller or equivalent type" );
    children.clear();
    auto *elem = (libMesh::Elem *) ptr_element;
    if ( d_globalID.type() == GeomType::Vertex ) {
        // A vertex does not have children, return itself
        if ( type != GeomType::Vertex )
            AMP_ERROR( "A vertex is the base element and cannot have and sub-elements" );
        children.resize( 1 );
        children[0] = *this;
    } else if ( type == d_globalID.type() ) {
        // Return the children of the current element
        if ( elem->has_children() ) {
            children.resize( elem->n_children() );
            for ( unsigned int i = 0; i < children.size(); i++ )
                children[i] = libmeshMeshElement(
                    d_dim, type, (void *) elem->child_ptr( i ), d_rank, d_meshID, d_mesh );
        } else {
            children.resize( 1 );
            children[0] = *this;
        }
    } else if ( type == GeomType::Vertex ) {
        // Return the nodes of the current element
        children.resize( elem->n_nodes() );
        for ( unsigned int i = 0; i < children.size(); i++ )
            children[i] = libmeshMeshElement(
                d_dim, type, (void *) elem->node_ptr( i ), d_rank, d_meshID, d_mesh );
    } else {
        // Return the children
        if ( type == GeomType::Edge )
            children.resize( elem->n_edges() );
        else if ( type == GeomType::Face )
            children.resize( elem->n_faces() );
        else
            AMP_ERROR( "Internal error" );
        for ( unsigned int i = 0; i < children.size(); i++ ) {
            // We need to build a valid element
            std::unique_ptr<libMesh::Elem> tmp;
            if ( type == GeomType::Edge )
                tmp = elem->build_edge_ptr( i );
            else if ( type == GeomType::Face )
                tmp = elem->build_side_ptr( i, false );
            else
                AMP_ERROR( "Internal error" );
            std::shared_ptr<libMesh::Elem> element( tmp.release() );
            // We need to generate a vaild id and owning processor
            unsigned int n_node_min = ( (unsigned int) type ) + 1;
            AMP_ASSERT( element->n_nodes() >= n_node_min );
            std::vector<libMesh::Node *> nodes( element->n_nodes() );
            std::vector<unsigned int> node_ids( element->n_nodes() );
            for ( size_t j = 0; j < nodes.size(); j++ ) {
                nodes[j]    = element->node_ptr( j );
                node_ids[j] = nodes[j]->id();
            }
            AMP::Utilities::quicksort( node_ids, nodes );
            element->processor_id() = nodes[0]->processor_id();
            unsigned int id         = generate_id( node_ids );
            element->set_id()       = id;
            // Create the libmeshMeshElement
            children[i] = libmeshMeshElement( d_dim, type, element, d_rank, d_meshID, d_mesh );
        }
    }
}


/****************************************************************
 * Function to get the neighboring elements                      *
 ****************************************************************/
void libmeshMeshElement::getNeighbors( std::vector<std::unique_ptr<MeshElement>> &neighbors ) const
{
    neighbors.clear();
    if ( d_globalID.type() == GeomType::Vertex ) {
        // Return the neighbors of the current node
        auto neighbor_nodes = d_mesh->getNeighborNodes( d_globalID );
        int n_neighbors     = neighbor_nodes.size();
        neighbors.reserve( n_neighbors );
        for ( int i = 0; i < n_neighbors; i++ ) {
            neighbors.emplace_back( new libmeshMeshElement(
                d_dim, GeomType::Vertex, (void *) neighbor_nodes[i], d_rank, d_meshID, d_mesh ) );
        }
    } else if ( (int) d_globalID.type() == d_dim ) {
        // Return the neighbors of the current element
        auto *elem      = (libMesh::Elem *) ptr_element;
        int n_neighbors = elem->n_neighbors();
        neighbors.resize( n_neighbors );
        for ( int i = 0; i < n_neighbors; i++ ) {
            auto *neighbor_elem = (void *) elem->neighbor_ptr( i );
            if ( neighbor_elem == nullptr )
                continue;
            neighbors[i] = std::unique_ptr<MeshElement>( new libmeshMeshElement(
                d_dim, d_globalID.type(), neighbor_elem, d_rank, d_meshID, d_mesh ) );
        }
    } else {
        // We constructed a temporary libmesh object and do not have access to the neighbor info
    }
}


/****************************************************************
 * Functions to get basic element properties                     *
 ****************************************************************/
double libmeshMeshElement::volume() const
{
    if ( d_globalID.type() == GeomType::Vertex )
        AMP_ERROR( "volume is is not defined for nodes" );
    auto *elem = (libMesh::Elem *) ptr_element;
    return elem->volume();
}
Point libmeshMeshElement::norm() const
{
    AMP_ERROR( "norm not implemented yet" );
    return Point();
}
Point libmeshMeshElement::coord() const
{
    if ( d_globalID.type() != GeomType::Vertex )
        AMP_ERROR( "coord is only defined for nodes" );
    auto *node = (libMesh::Node *) ptr_element;
    Point x( (size_t) d_dim );
    for ( int i = 0; i < d_dim; i++ )
        x[i] = ( *node )( i );
    return x;
}
Point libmeshMeshElement::centroid() const
{
    if ( d_globalID.type() == GeomType::Vertex )
        return coord();
    auto *elem            = (libMesh::Elem *) ptr_element;
    libMesh::Point center = elem->vertex_average();
    AMP::Mesh::Point x( (size_t) d_dim );
    for ( int i = 0; i < d_dim; i++ )
        x[i] = center( i );
    return x;
}
bool libmeshMeshElement::containsPoint( const Point &pos, double TOL ) const
{
    if ( d_globalID.type() == GeomType::Vertex ) {
        // double dist = 0.0;
        auto point   = this->coord();
        double dist2 = 0.0;
        for ( size_t i = 0; i < point.size(); i++ )
            dist2 += ( point[i] - pos[i] ) * ( point[i] - pos[i] );
        return dist2 <= TOL * TOL;
    }
    auto *elem = (libMesh::Elem *) ptr_element;
    libMesh::Point point( pos[0], pos[1], pos[2] );
    return elem->contains_point( point, TOL );
}
bool libmeshMeshElement::isOnSurface() const
{
    auto type = static_cast<int>( d_globalID.type() );
    if ( d_globalID.is_local() ) {
        const auto &data = *( d_mesh->d_localSurfaceElements[type] );
        if ( data.empty() )
            return false; // There are no elements on the surface for this processor
        size_t index = AMP::Utilities::findfirst( data, *this );
        if ( index < data.size() ) {
            if ( d_mesh->d_localSurfaceElements[type]->operator[]( index ).globalID() ==
                 d_globalID )
                return true;
        }
    } else {
        const auto &data = *( d_mesh->d_ghostSurfaceElements[type] );
        if ( data.empty() )
            return false; // There are no elements on the surface for this processor
        size_t index = AMP::Utilities::findfirst( data, *this );
        if ( index < data.size() ) {
            if ( d_mesh->d_ghostSurfaceElements[type]->operator[]( index ).globalID() ==
                 d_globalID )
                return true;
        }
    }
    return false;
}
bool libmeshMeshElement::isOnBoundary( int id ) const
{
    GeomType type    = d_globalID.type();
    bool on_boundary = false;
    auto d_libMesh   = d_mesh->getlibMesh();
    if ( type == GeomType::Vertex ) {
        // Entity is a libmesh node
        auto *node = (libMesh::Node *) ptr_element;
        std::vector<libMesh::boundary_id_type> bids;
        d_libMesh->boundary_info->boundary_ids( node, bids );
        for ( auto &bid : bids ) {
            if ( bid == id )
                on_boundary = true;
        }
    } else if ( (int) type == d_dim ) {
        // Entity is a libmesh node
        auto *elem        = (libMesh::Elem *) ptr_element;
        unsigned int side = d_libMesh->boundary_info->side_with_boundary_id( elem, id );
        if ( side != static_cast<unsigned int>( -1 ) )
            on_boundary = true;
    } else {
        // All other entities are on the boundary iff all of their vertices are on the surface
        std::vector<MeshElement> nodes;
        this->getElements( GeomType::Vertex, nodes );
        on_boundary = true;
        for ( auto &node : nodes )
            on_boundary = on_boundary && node.isOnBoundary( id );
    }
    return on_boundary;
}
bool libmeshMeshElement::isInBlock( int id ) const
{
    GeomType type = d_globalID.type();
    bool in_block = false;
    if ( type == GeomType::Vertex ) {
        // Entity is a libmesh node
        AMP_ERROR( "isInBlock is not currently implemented for anything but elements" );
    } else if ( (int) type == d_dim ) {
        // Entity is a libmesh node
        auto *elem = (libMesh::Elem *) ptr_element;
        in_block   = (int) elem->subdomain_id() == id;
    } else {
        // All other entities are on the boundary iff all of their vertices are on the surface
        AMP_ERROR( "isInBlock is not currently implemented for anything but elements" );
    }
    return in_block;
}


/****************************************************************
 * Functions to generate a new id based on the nodes             *
 * Note: this function requires the node ids to be sorted        *
 ****************************************************************/
static unsigned int fliplr( unsigned int x )
{
    unsigned int y     = 0;
    unsigned int mask1 = 1;
    unsigned int mask2 = mask1 << ( sizeof( unsigned int ) * 8 - 1 );
    for ( size_t i = 0; i < sizeof( unsigned int ) * 8; i++ ) {
        y += ( x & mask1 ) ? mask2 : 0;
        mask1 <<= 1;
        mask2 >>= 1;
    }
    return y;
}
unsigned int generate_id( const std::vector<unsigned int> &ids )
{
    unsigned int id0 = ids[0];
    unsigned int id_diff[100];
    for ( size_t i = 1; i < ids.size(); i++ )
        id_diff[i - 1] = ids[i] - ids[i - 1];
    unsigned int tmp = 0;
    for ( size_t i = 0; i < ids.size() - 1; i++ ) {
        unsigned int shift = ( 7 * i ) % 13;
        tmp                = tmp ^ ( id_diff[i] << shift );
    }
    unsigned int id = id0 ^ ( fliplr( tmp ) >> 1 );
    return id;
}


} // namespace AMP::Mesh


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
#include "AMP/utils/Utilities.hpp"
template void
AMP::Utilities::quicksort<unsigned int, libMesh::Node *>( std::vector<unsigned int> &,
                                                          std::vector<libMesh::Node *> & );
