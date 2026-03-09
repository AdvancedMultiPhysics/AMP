#include "AMP/mesh/libmesh/libmeshMeshElement.h"
#include "AMP/utils/Utilities.h"

// libMesh includes
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/boundary_info.h"
#include "libmesh/elem.h"

namespace AMP::Mesh {


// Functions to create new ids by mixing existing ids
static unsigned int generate_id( int N, const unsigned int *ids );


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
}
libmeshMeshElement::libmeshMeshElement( int dim,
                                        GeomType type,
                                        void *libmesh_element,
                                        unsigned int rank,
                                        MeshID meshID,
                                        const libmeshMesh *mesh )
    : d_delete_elem( true )
{
    AMP_ASSERT( libmesh_element );
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
    d_dim       = rhs.d_dim;
    ptr_element = rhs.ptr_element;
    d_rank      = rhs.d_rank;
    d_mesh      = rhs.d_mesh;
}
libmeshMeshElement &libmeshMeshElement::operator=( const libmeshMeshElement &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
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
 * Destructor                                                    *
 ****************************************************************/
libmeshMeshElement::~libmeshMeshElement() {}


/********************************************************
 * Get basic info                                        *
 ********************************************************/
static auto TypeID = AMP::getTypeID<AMP::Mesh::libmeshMeshElement>();
const typeID &libmeshMeshElement::getTypeID() const { return TypeID; }


/****************************************************************
 * Function to clone the element                                 *
 ****************************************************************/
std::unique_ptr<MeshElement> libmeshMeshElement::clone() const
{
    return std::make_unique<libmeshMeshElement>( *this );
}


/****************************************************************
 * Function to get the elements composing the current element    *
 ****************************************************************/
MeshElement::ElementListPtr libmeshMeshElement::getElements( const GeomType type ) const
{
    AMP_INSIST( type <= d_globalID.type(), "sub-elements must be of a smaller or equivalent type" );
    auto *elem = (libMesh::Elem *) ptr_element;
    std::unique_ptr<MeshElementVector<libmeshMeshElement>> children;
    if ( d_globalID.type() == GeomType::Vertex ) {
        // A vertex does not have children, return itself
        if ( type != GeomType::Vertex )
            AMP_ERROR( "A vertex is the base element and cannot have and sub-elements" );
        children = std::make_unique<MeshElementVector<libmeshMeshElement>>( *this );
    } else if ( type == d_globalID.type() ) {
        // Return the children of the current element
        if ( elem->has_children() ) {
            children =
                std::make_unique<MeshElementVector<libmeshMeshElement>>( elem->n_children() );
            for ( unsigned int i = 0; i < children->size(); i++ )
                ( *children )[i] = libmeshMeshElement(
                    d_dim, type, (void *) elem->child_ptr( i ), d_rank, d_meshID, d_mesh );
        } else {
            children = std::make_unique<MeshElementVector<libmeshMeshElement>>( *this );
        }
    } else if ( type == GeomType::Vertex ) {
        // Return the nodes of the current element
        children = std::make_unique<MeshElementVector<libmeshMeshElement>>( elem->n_nodes() );
        for ( unsigned int i = 0; i < children->size(); i++ )
            ( *children )[i] = libmeshMeshElement(
                d_dim, type, (void *) elem->node_ptr( i ), d_rank, d_meshID, d_mesh );
    } else {
        // Return the children
        size_t N_children = 0;
        if ( type == GeomType::Edge )
            N_children = elem->n_edges();
        else if ( type == GeomType::Face )
            N_children = elem->n_faces();
        else
            AMP_ERROR( "Internal error" );
        children = std::make_unique<MeshElementVector<libmeshMeshElement>>( N_children );
        for ( unsigned int i = 0; i < N_children; i++ ) {
            // We need to build a valid element
            std::shared_ptr<libMesh::Elem> element;
            if ( type == GeomType::Edge )
                element = elem->build_edge_ptr( i );
            else if ( type == GeomType::Face )
                element = elem->build_side_ptr( i, false );
            else
                AMP_ERROR( "Internal error" );
            // We need to generate a valid id and owning processor
            int N = element->n_nodes();
            AMP_ASSERT( N > ( (int) type ) && N <= 32 );
            uint32_t node_ids[32];
            int proc[32];
            for ( int j = 0; j < N; j++ ) {
                auto node   = element->node_ptr( j );
                node_ids[j] = node->id();
                proc[j]     = node->processor_id();
            }
            AMP::Utilities::quicksort( N, node_ids, proc );
            element->processor_id() = proc[0];
            unsigned int id         = generate_id( N, node_ids );
            element->set_id()       = id;
            // Create the libmeshMeshElement
            ( *children )[i] =
                libmeshMeshElement( d_dim, type, std::move( element ), d_rank, d_meshID, d_mesh );
        }
    }
    AMP_ASSERT( children );
    return children;
}
int libmeshMeshElement::getElementsID( const GeomType type, MeshElementID *ID ) const
{
    AMP_INSIST( type <= d_globalID.type(), "sub-elements must be of a smaller or equivalent type" );
    auto *elem = (libMesh::Elem *) ptr_element;
    if ( d_globalID.type() == GeomType::Vertex ) {
        // A vertex does not have children, return itself
        if ( type != GeomType::Vertex )
            AMP_ERROR( "A vertex is the base element and cannot have and sub-elements" );
        ID[0] = globalID();
        return 1;
    } else if ( type == d_globalID.type() ) {
        // Return the children of the current element
        if ( elem->has_children() ) {
            for ( unsigned int i = 0; i < elem->n_children(); i++ ) {
                auto child     = elem->child_ptr( i );
                int local_id   = child->id();
                int owner_rank = child->processor_id();
                bool is_local  = owner_rank == (int) d_rank;
                ID[i]          = MeshElementID( is_local, type, local_id, owner_rank, d_meshID );
            }
            return elem->n_children();
        } else {
            ID[0] = globalID();
            return 1;
        }
    } else if ( type == GeomType::Vertex ) {
        // Return the nodes of the current element
        for ( unsigned int i = 0; i < elem->n_nodes(); i++ ) {
            auto node      = elem->node_ptr( i );
            int local_id   = node->id();
            int owner_rank = node->processor_id();
            bool is_local  = owner_rank == (int) d_rank;
            ID[i]          = MeshElementID( is_local, type, local_id, owner_rank, d_meshID );
        }
        return elem->n_nodes();
    } else {
        // Get the elements then the ids
        auto elements = this->getElements( type );
        for ( size_t i = 0; i < elements.size(); i++ )
            ID[i] = elements[i].globalID();
        return elements.size();
    }
}


/****************************************************************
 * Function to get the neighboring elements                      *
 ****************************************************************/
MeshElement::ElementListPtr libmeshMeshElement::getNeighbors() const
{
    if ( d_globalID.type() == GeomType::Vertex ) {
        // Return the neighbors of the current node
        auto neighbor_nodes = d_mesh->getNeighborNodes( d_globalID );
        int n_neighbors     = neighbor_nodes.size();
        auto neighbors = std::make_unique<MeshElementVector<libmeshMeshElement>>( n_neighbors );
        for ( int i = 0; i < n_neighbors; i++ ) {
            ( *neighbors )[i] = libmeshMeshElement(
                d_dim, GeomType::Vertex, (void *) neighbor_nodes[i], d_rank, d_meshID, d_mesh );
        }
        return neighbors;
    } else if ( (int) d_globalID.type() == d_dim ) {
        // Return the neighbors of the current element
        auto *elem      = (libMesh::Elem *) ptr_element;
        int n_neighbors = elem->n_neighbors();
        auto neighbors  = std::make_unique<MeshElementVector<libmeshMeshElement>>( n_neighbors );
        for ( int i = 0; i < n_neighbors; i++ ) {
            auto *neighbor_elem = (void *) elem->neighbor_ptr( i );
            if ( neighbor_elem == nullptr )
                continue;
            ( *neighbors )[i] = libmeshMeshElement(
                d_dim, d_globalID.type(), neighbor_elem, d_rank, d_meshID, d_mesh );
            return neighbors;
        }
    } else {
        // We constructed a temporary libmesh object and do not have access to the neighbor info
    }
    return std::make_unique<MeshElementVector<libmeshMeshElement>>();
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
    GeomType type       = d_globalID.type();
    auto d_libMesh      = d_mesh->getlibMesh();
    auto &boundary_info = d_libMesh->get_boundary_info();
    if ( type == GeomType::Vertex ) {
        // Entity is a libmesh node
        auto *node = (libMesh::Node *) ptr_element;
        return boundary_info.has_boundary_id( node, id );
    } else if ( (int) type == d_dim ) {
        // Entity is a libmesh node
        auto *elem        = (libMesh::Elem *) ptr_element;
        unsigned int side = boundary_info.side_with_boundary_id( elem, id );
        return side != static_cast<unsigned int>( -1 );
    } else {
        // All other entities are on the boundary iff all of their vertices are on the surface
        bool on_boundary = true;
        auto *elem       = (libMesh::Elem *) ptr_element;
        for ( unsigned int i = 0; i < elem->n_nodes(); i++ ) {
            auto node   = elem->node_ptr( i );
            on_boundary = on_boundary && boundary_info.has_boundary_id( node, id );
        }
        return on_boundary;
    }
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
static inline uint32_t reverseBits( uint32_t x )
{
    uint32_t y = x;
    y = ( ( y >> 1 ) & 0x55555555 ) | ( ( y & 0x55555555 ) << 1 ); // Swap adjacent 1-bit groups
    y = ( ( y >> 2 ) & 0x33333333 ) | ( ( y & 0x33333333 ) << 2 ); // Swap 2-bit groups
    y = ( ( y >> 4 ) & 0x0F0F0F0F ) | ( ( y & 0x0F0F0F0F ) << 4 ); // Swap 4-bit groups
    y = ( ( y >> 8 ) & 0x00FF00FF ) | ( ( y & 0x00FF00FF ) << 8 ); // Swap 8-bit groups
    y = ( y >> 16 ) | ( y << 16 );                                 // Swap 16-bit groups
    return y;
}
unsigned int generate_id( int N, const unsigned int *ids )
{
    unsigned int id0 = ids[0];
    unsigned int id_diff[100];
    for ( int i = 1; i < N; i++ )
        id_diff[i - 1] = ids[i] - ids[i - 1];
    unsigned int tmp = 0;
    for ( int i = 0; i < N - 1; i++ ) {
        unsigned int shift = ( 7 * i ) % 13;
        tmp                = tmp ^ ( id_diff[i] << shift );
    }
    unsigned int id = id0 ^ ( reverseBits( tmp ) >> 1 );
    return id;
}


} // namespace AMP::Mesh


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
#include "AMP/utils/Utilities.hpp"
template void AMP::Utilities::quicksort<unsigned int, int>( size_t, unsigned int *, int * );
