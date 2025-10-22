#include "AMP/mesh/structured/structuredMeshElement.h"

#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Mesh {


// Function to evaluate the magnitude of a cross product in 3d
double cross3magnitude( const double a[3], const double b[3] )
{
    double v[3];
    v[0] = a[1] * b[2] - a[2] * b[1];
    v[1] = a[2] * b[0] - a[0] * b[2];
    v[2] = a[0] * b[1] - a[1] * b[0];
    return std::sqrt( v[0] * v[0] + v[1] * v[1] + v[2] * v[2] );
}
// Function to evaluate the dot produce of a vector and a cross product in 3d ( a . ( b X c ) )
double dot3cross( const double a[3], const double b[3], const double c[3] )
{
    double v[3];
    v[0] = b[1] * c[2] - b[2] * c[1];
    v[1] = b[2] * c[0] - b[0] * c[2];
    v[2] = b[0] * c[1] - b[1] * c[0];
    return a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
}


/********************************************************
 * Constructors                                          *
 ********************************************************/
static constexpr auto elementTypeID = AMP::getTypeID<structuredMeshElement>().hash;
static_assert( elementTypeID != 0 );
structuredMeshElement::structuredMeshElement() { reset(); }
void structuredMeshElement::reset()
{
    d_index       = BoxMesh::MeshElementIndex();
    d_meshType    = GeomType::Nullity;
    d_physicalDim = 0;
}
structuredMeshElement::structuredMeshElement( const BoxMesh::MeshElementIndex &index,
                                              const AMP::Mesh::BoxMesh *mesh )
{
    reset( index, mesh );
}
void structuredMeshElement::reset( const BoxMesh::MeshElementIndex &index,
                                   const AMP::Mesh::BoxMesh *mesh )
{
    d_mesh        = mesh;
    d_meshType    = d_mesh->getGeomType();
    d_physicalDim = d_mesh->getDim();
    AMP_ASSERT( static_cast<int>( d_meshType ) > 0 && static_cast<int>( d_meshType ) <= 3 );
    d_index = index;
}
structuredMeshElement::structuredMeshElement( const structuredMeshElement &rhs )
    : MeshElement(), // Note: we never want to call the base copy constructor
      d_meshType( rhs.d_meshType ),
      d_physicalDim( rhs.d_physicalDim ),
      d_index( rhs.d_index ),
      d_mesh( rhs.d_mesh )
{
}
structuredMeshElement &structuredMeshElement::operator=( const structuredMeshElement &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    this->d_meshType    = rhs.d_meshType;
    this->d_physicalDim = rhs.d_physicalDim;
    this->d_index       = rhs.d_index;
    this->d_mesh        = rhs.d_mesh;
    return *this;
}


/****************************************************************
 * Destructor                                                    *
 ****************************************************************/
structuredMeshElement::~structuredMeshElement() = default;


/****************************************************************
 * Function to clone the element                                 *
 ****************************************************************/
std::unique_ptr<MeshElement> structuredMeshElement::clone() const
{
    return std::make_unique<structuredMeshElement>( *this );
}


/********************************************************
 * Get the general boundary conditions                   *
 *    0 - Physical boundary                              *
 *    1 - Periodic boundary                              *
 *    2 - Mapped boundary                                *
 ********************************************************/
std::array<int8_t, 3> structuredMeshElement::getBC() const
{
    std::array<int8_t, 3> BC = { 0, 0, 0 };
    for ( int d = 0; d < 3; d++ ) {
        if ( d_mesh->d_surfaceId[2 * d] == -1 )
            BC[d] = 1;
        else if ( d_mesh->d_surfaceId[2 * d] == -2 || d_mesh->d_surfaceId[2 * d + 1] == -2 )
            BC[d] = 2;
    }
    return BC;
}


/********************************************************
 * Return the vertices                                   *
 ********************************************************/
void structuredMeshElement::getVertices( std::vector<Point> &vertices ) const
{
    int N = 0;
    BoxMesh::MeshElementIndex nodes[8];
    getElementIndex( GeomType::Vertex, N, nodes );
    vertices.resize( N, Point( d_physicalDim, { 0, 0, 0 } ) );
    for ( int i = 0; i < N; i++ )
        d_mesh->coord( nodes[i], vertices[i].data() );
}


/****************************************************************
 * Function to get the elements composing the current element    *
 * We use a Canonical numbering system                           *
 ****************************************************************/
void structuredMeshElement::getElements( const GeomType type, ElementList &elements ) const
{
    int N = 0;
    BoxMesh::MeshElementIndex index[12];
    getElementIndex( type, N, index );
    // Create the elements
    elements.clear();
    elements.reserve( N );
    for ( int i = 0; i < N; i++ )
        elements.emplace_back( std::make_unique<structuredMeshElement>( index[i], d_mesh ) );
}
int structuredMeshElement::getElementsID( const GeomType type, MeshElementID *ID ) const
{
    int N = 0;
    BoxMesh::MeshElementIndex index[12];
    getElementIndex( type, N, index );
    for ( int i = 0; i < N; i++ )
        ID[i] = d_mesh->convert( index[i] );
    return N;
}
void structuredMeshElement::getElementIndex( const GeomType type,
                                             int &N,
                                             BoxMesh::MeshElementIndex *index ) const
{
    AMP_ASSERT( type <= d_meshType );
    N = 0;
    if ( type == d_index.type() ) {
        N        = 1;
        index[0] = d_index;
        return;
    }
    const int *ijk = d_index.d_index.data();
    if ( type == GeomType::Vertex ) {
        // We want to get the vertices composing the elements
        if ( d_index.type() == d_meshType ) {
            // We are dealing with a entity of type dim and want the vertices
            if ( d_meshType == GeomType::Edge ) {
                N = 2;
                index[0].reset( GeomType::Vertex, 0, ijk[0] );
                index[1].reset( GeomType::Vertex, 0, ijk[0] + 1 );
            } else if ( d_meshType == GeomType::Face ) {
                N = 4;
                index[0].reset( GeomType::Vertex, 0, ijk[0], ijk[1] );
                index[1].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1] );
                index[2].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1] + 1 );
                index[3].reset( GeomType::Vertex, 0, ijk[0], ijk[1] + 1 );
            } else if ( d_meshType == GeomType::Cell ) {
                N = 8;
                index[0].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] );
                index[1].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1], ijk[2] );
                index[2].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1] + 1, ijk[2] );
                index[3].reset( GeomType::Vertex, 0, ijk[0], ijk[1] + 1, ijk[2] );
                index[4].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] + 1 );
                index[5].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1], ijk[2] + 1 );
                index[6].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1] + 1, ijk[2] + 1 );
                index[7].reset( GeomType::Vertex, 0, ijk[0], ijk[1] + 1, ijk[2] + 1 );
            } else {
                AMP_ERROR( "Dimension not supported yet" );
            }
        } else if ( d_index.type() == GeomType::Edge ) {
            N = 2;
            index[0].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] );
            index[1].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] );
            index[1].d_index[d_index.d_side]++;
        } else if ( d_index.type() == GeomType::Face ) {
            N = 4;
            if ( d_index.d_side == 0 ) {
                index[0].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] );
                index[1].reset( GeomType::Vertex, 0, ijk[0], ijk[1] + 1, ijk[2] );
                index[2].reset( GeomType::Vertex, 0, ijk[0], ijk[1] + 1, ijk[2] + 1 );
                index[3].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] + 1 );
            } else if ( d_index.d_side == 1 ) {
                index[0].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] );
                index[1].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1], ijk[2] );
                index[2].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1], ijk[2] + 1 );
                index[3].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] + 1 );
            } else if ( d_index.d_side == 2 ) {
                index[0].reset( GeomType::Vertex, 0, ijk[0], ijk[1], ijk[2] );
                index[1].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1], ijk[2] );
                index[2].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1] + 1, ijk[2] );
                index[3].reset( GeomType::Vertex, 0, ijk[0], ijk[1] + 1, ijk[2] );
            } else {
                AMP_ERROR( "Internal error" );
            }
        } else if ( d_index.type() == GeomType::Cell ) {
            AMP_ERROR( "Not ready for dimensions > 3" );
        } else {
            AMP_ERROR( "Not finished" );
        }
    } else if ( type == GeomType::Edge ) {
        if ( d_meshType == GeomType::Face ) {
            N = 4;
            index[0].reset( GeomType::Edge, 0, ijk[0], ijk[1], 0 );
            index[1].reset( GeomType::Edge, 1, ijk[0] + 1, ijk[1], 0 );
            index[2].reset( GeomType::Edge, 0, ijk[0], ijk[1] + 1, 0 );
            index[3].reset( GeomType::Edge, 1, ijk[0], ijk[1], 0 );
        } else if ( d_meshType == GeomType::Cell ) {
            if ( d_index.type() == GeomType::Face ) {
                N = 4;
                if ( d_index.d_side == 0 ) {
                    // We are dealing with an x-face
                    index[0].reset( GeomType::Edge, 1, ijk[0], ijk[1], ijk[2] );
                    index[1].reset( GeomType::Edge, 2, ijk[0], ijk[1] + 1, ijk[2] );
                    index[2].reset( GeomType::Edge, 1, ijk[0], ijk[1], ijk[2] + 1 );
                    index[3].reset( GeomType::Edge, 2, ijk[0], ijk[1], ijk[2] );
                } else if ( d_index.d_side == 1 ) {
                    // We are dealing with an y-face
                    index[0].reset( GeomType::Edge, 0, ijk[0], ijk[1], ijk[2] );
                    index[1].reset( GeomType::Edge, 2, ijk[0] + 1, ijk[1], ijk[2] );
                    index[2].reset( GeomType::Edge, 0, ijk[0], ijk[1], ijk[2] + 1 );
                    index[3].reset( GeomType::Edge, 2, ijk[0], ijk[1], ijk[2] );
                } else if ( d_index.d_side == 2 ) {
                    // We are dealing with an z-face
                    index[0].reset( GeomType::Edge, 0, ijk[0], ijk[1], ijk[2] );
                    index[1].reset( GeomType::Edge, 1, ijk[0] + 1, ijk[1], ijk[2] );
                    index[2].reset( GeomType::Edge, 0, ijk[0], ijk[1] + 1, ijk[2] );
                    index[3].reset( GeomType::Edge, 1, ijk[0], ijk[1], ijk[2] );
                } else {
                    AMP_ERROR( "Internal error" );
                }
            } else if ( d_index.type() == GeomType::Cell ) {
                AMP_ASSERT( d_index.d_side == 0 );
                N = 12;
                index[0].reset( GeomType::Edge, 0, ijk[0], ijk[1], ijk[2] );
                index[1].reset( GeomType::Edge, 1, ijk[0] + 1, ijk[1], ijk[2] );
                index[2].reset( GeomType::Edge, 0, ijk[0], ijk[1] + 1, ijk[2] );
                index[3].reset( GeomType::Edge, 1, ijk[0], ijk[1], ijk[2] );
                index[4].reset( GeomType::Edge, 2, ijk[0], ijk[1], ijk[2] );
                index[5].reset( GeomType::Edge, 2, ijk[0] + 1, ijk[1], ijk[2] );
                index[6].reset( GeomType::Edge, 2, ijk[0] + 1, ijk[1] + 1, ijk[2] );
                index[7].reset( GeomType::Edge, 2, ijk[0], ijk[1] + 1, ijk[2] );
                index[8].reset( GeomType::Edge, 0, ijk[0], ijk[1], ijk[2] + 1 );
                index[9].reset( GeomType::Edge, 1, ijk[0] + 1, ijk[1], ijk[2] + 1 );
                index[10].reset( GeomType::Edge, 0, ijk[0], ijk[1] + 1, ijk[2] + 1 );
                index[11].reset( GeomType::Edge, 1, ijk[0], ijk[1], ijk[2] + 1 );
            } else {
                AMP_ERROR( "Dimensions > 3 are not supported yet" );
            }
        } else {
            AMP_ERROR( "Dimensions > 3 are not supported yet" );
        }
    } else if ( type == GeomType::Face ) {
        if ( d_index.type() == GeomType::Cell ) {
            N = 6;
            index[0].reset( GeomType::Face, 1, ijk[0], ijk[1], ijk[2] );
            index[1].reset( GeomType::Face, 0, ijk[0] + 1, ijk[1], ijk[2] );
            index[2].reset( GeomType::Face, 1, ijk[0], ijk[1] + 1, ijk[2] );
            index[3].reset( GeomType::Face, 0, ijk[0], ijk[1], ijk[2] );
            index[4].reset( GeomType::Face, 2, ijk[0], ijk[1], ijk[2] );
            index[5].reset( GeomType::Face, 2, ijk[0], ijk[1], ijk[2] + 1 );
        } else {
            AMP_ERROR( "Dimensions > 3 are not supported yet" );
        }
    } else if ( type == GeomType::Cell ) {
        AMP_ERROR( "Dimensions > 3 are not supported yet" );
    } else {
        AMP_ERROR( "Not finished" );
    }
    // Fix any elements that are beyond a periodic boundary
    for ( int i = 0; i < N; i++ )
        d_mesh->fixPeriodic( index[i] );
}


/****************************************************************
 * Function to get the neighboring elements                      *
 ****************************************************************/
void structuredMeshElement::getNeighbors( ElementList &neighbors ) const
{
    BoxMesh::MeshElementIndex index[27];
    int N = getNeighborIndex( index );
    neighbors.resize( N );
    for ( int i = 0; i < N; i++ ) {
        if ( !index[i].isNull() )
            neighbors[i] = std::make_unique<structuredMeshElement>( index[i], d_mesh );
    }
}
int structuredMeshElement::getNeighborIndex( BoxMesh::MeshElementIndex *index ) const
{
    int N = 0;
    // Get the neighbor indicies
    const int *ijk = d_index.d_index.data();
    if ( d_index.type() == GeomType::Vertex ) {
        // Get the list of neighbor nodex (there are no null neighbors)
        // The node neighbors are the list of nodes that share any element
        if ( d_meshType == GeomType::Edge ) {
            N = 2;
            index[0].reset( GeomType::Vertex, 0, ijk[0] - 1 );
            index[1].reset( GeomType::Vertex, 0, ijk[0] + 1 );
        } else if ( d_meshType == GeomType::Face ) {
            N = 8;
            index[0].reset( GeomType::Vertex, 0, ijk[0] - 1, ijk[1] - 1 );
            index[1].reset( GeomType::Vertex, 0, ijk[0], ijk[1] - 1 );
            index[2].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1] - 1 );
            index[3].reset( GeomType::Vertex, 0, ijk[0] - 1, ijk[1] );
            index[4].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1] );
            index[5].reset( GeomType::Vertex, 0, ijk[0] - 1, ijk[1] + 1 );
            index[6].reset( GeomType::Vertex, 0, ijk[0], ijk[1] + 1 );
            index[7].reset( GeomType::Vertex, 0, ijk[0] + 1, ijk[1] + 1 );
        } else if ( d_meshType == GeomType::Cell ) {
            N = 0;
            for ( int k = -1; k <= 1; k++ ) {
                for ( int j = -1; j <= 1; j++ ) {
                    for ( int i = -1; i <= 1; i++ ) {
                        if ( i == 0 && j == 0 && k == 0 )
                            continue;
                        index[N].reset( GeomType::Vertex, 0, ijk[0] + i, ijk[1] + j, ijk[2] + k );
                        N++;
                    }
                }
            }
        } else {
            AMP_ERROR( "Dimension not supported yet" );
        }
    } else if ( d_index.type() == GeomType::Edge ) {
        if ( d_meshType == GeomType::Edge ) {
            N = 2;
            index[0].reset( GeomType::Edge, 0, ijk[0] - 1 );
            index[1].reset( GeomType::Edge, 0, ijk[0] + 1 );
        } else {
            // GeomType::Edge neighbors in dimensions > 1 are not supported yet
        }
    } else if ( d_index.type() == GeomType::Face ) {
        if ( d_meshType == GeomType::Face ) {
            N = 4;
            index[0].reset( GeomType::Face, 0, ijk[0], ijk[1] - 1 );
            index[1].reset( GeomType::Face, 0, ijk[0] + 1, ijk[1] );
            index[2].reset( GeomType::Face, 0, ijk[0], ijk[1] + 1 );
            index[3].reset( GeomType::Face, 0, ijk[0] - 1, ijk[1] );
        } else {
            // GeomType::Face neighbors in dimensions > 2 are not supported yet
        }
    } else if ( d_index.type() == GeomType::Cell ) {
        if ( d_meshType == GeomType::Cell ) {
            N = 6;
            index[0].reset( GeomType::Cell, 0, ijk[0], ijk[1] - 1, ijk[2] );
            index[1].reset( GeomType::Cell, 0, ijk[0] + 1, ijk[1], ijk[2] );
            index[2].reset( GeomType::Cell, 0, ijk[0], ijk[1] + 1, ijk[2] );
            index[3].reset( GeomType::Cell, 0, ijk[0] - 1, ijk[1], ijk[2] );
            index[4].reset( GeomType::Cell, 0, ijk[0], ijk[1], ijk[2] - 1 );
            index[5].reset( GeomType::Cell, 0, ijk[0], ijk[1], ijk[2] + 1 );
        } else {
            // GeomType::Cell neighbors in dimensions > 3 are not supported yet
        }
    } else {
        AMP_ERROR( "Unknown entity type" );
    }
    // Apply boundary conditions
    auto BC   = getBC();
    auto size = d_mesh->d_globalSize;
    for ( int i = 0; i < N; i++ ) {
        bool in_mesh = true;
        auto &elem   = index[i];
        for ( int d = 0; d < static_cast<int>( d_meshType ); d++ ) {
            if ( BC[d] == 1 ) {
                if ( elem.d_index[d] < 0 )
                    elem.d_index[d] += size[d];
                if ( elem.d_index[d] >= size[d] )
                    elem.d_index[d] -= size[d];
            } else if ( BC[d] == 2 ) {
                AMP_WARN_ONCE( "Not finished" );
                if ( elem.d_index[d] < 0 )
                    in_mesh = false;
                if ( d_index.type() == d_meshType ) {
                    if ( elem.d_index[d] >= size[d] )
                        in_mesh = false;
                } else {
                    if ( elem.d_index[d] > size[d] )
                        in_mesh = false;
                }
            } else {
                if ( elem.d_index[d] < 0 )
                    in_mesh = false;
                if ( d_index.type() == d_meshType ) {
                    if ( elem.d_index[d] >= size[d] )
                        in_mesh = false;
                } else {
                    if ( elem.d_index[d] > size[d] )
                        in_mesh = false;
                }
            }
        }
        if ( !in_mesh )
            index[i].reset();
    }
    return N;
}


/****************************************************************
 * Function to get the parent elements                           *
 ****************************************************************/
MeshElement::ElementList structuredMeshElement::getParents( GeomType type ) const
{
    AMP_INSIST( static_cast<int>( type ) >= d_index.d_type,
                "We can't return a parent of geometric type < current type" );
    auto getErrMsg = [this, type]( const std::string &message ) {
        return AMP::Utilities::stringf(
            "%s: dim=%i, elem_type=%i, type=%i, meshClass = %s, meshName = %s",
            message.data(),
            static_cast<int>( d_meshType ),
            static_cast<int>( d_index.d_type ),
            static_cast<int>( type ),
            d_mesh->meshClass().data(),
            d_mesh->getName().data() );
    };
    // Get the indicies of the parent elements (ignore boundaries for now)
    std::vector<BoxMesh::MeshElementIndex> index_list;
    const int *ijk = d_index.d_index.data();
    if ( d_index.d_type == static_cast<int>( type ) ) {
        // We are looking for the current element
        ElementList tmp;
        tmp.push_back( std::make_unique<structuredMeshElement>( *this ) );
        return tmp;
    } else if ( static_cast<int>( type ) == d_index.d_type + 1 && type == d_meshType ) {
        // We have an entity that is the geometric type-1 and we want to get the
        // parents of the geometric type of the mesh
        BoxMesh::MeshElementIndex index( type, 0, ijk[0], ijk[1], ijk[2] );
        index_list.emplace_back( index );
        index.d_index[d_index.d_side]--;
        index_list.emplace_back( index );
    } else if ( d_index.d_type == static_cast<int>( GeomType::Vertex ) ) {
        // We want to get the parents of a vertex
        AMP_ASSERT( static_cast<int>( d_meshType ) <= 3 );
        if ( type == d_meshType ) {
            for ( int i = ijk[0] - 1; i <= ijk[0]; i++ ) {
                for ( int j = ijk[1] - 1; j <= ijk[1]; j++ ) {
                    for ( int k = ijk[2] - 1; k <= ijk[2]; k++ ) {
                        index_list.emplace_back( BoxMesh::MeshElementIndex( type, 0, i, j, k ) );
                    }
                }
            }
        } else if ( type == GeomType::Edge ) {
            for ( int d = 0; d < static_cast<int>( d_meshType ); d++ ) {
                BoxMesh::MeshElementIndex index( type, d, ijk[0], ijk[1], ijk[2] );
                index_list.emplace_back( index );
                index.d_index[d]--;
                index_list.emplace_back( index );
            }
        } else if ( type == GeomType::Face && d_meshType == GeomType::Cell ) {
            index_list.resize( 12 );
            index_list[0].reset( type, 0, ijk[0], ijk[1] - 1, ijk[2] - 1 );
            index_list[1].reset( type, 0, ijk[0], ijk[1] - 1, ijk[2] );
            index_list[2].reset( type, 0, ijk[0], ijk[1], ijk[2] - 1 );
            index_list[3].reset( type, 0, ijk[0], ijk[1], ijk[2] );
            index_list[4].reset( type, 1, ijk[0] - 1, ijk[1], ijk[2] - 1 );
            index_list[5].reset( type, 1, ijk[0] - 1, ijk[1], ijk[2] );
            index_list[6].reset( type, 1, ijk[0], ijk[1], ijk[2] - 1 );
            index_list[7].reset( type, 1, ijk[0], ijk[1], ijk[2] );
            index_list[8].reset( type, 2, ijk[0] - 1, ijk[1] - 1, ijk[2] );
            index_list[9].reset( type, 2, ijk[0] - 1, ijk[1], ijk[2] );
            index_list[10].reset( type, 2, ijk[0], ijk[1] - 1, ijk[2] );
            index_list[11].reset( type, 2, ijk[0], ijk[1], ijk[2] );
        } else {
            AMP_ERROR( getErrMsg( "Unknown type" ) );
        }
    } else if ( d_index.d_type == static_cast<int>( GeomType::Edge ) ) {
        // We want to get the parents of an edge
        AMP_ASSERT( static_cast<int>( d_meshType ) <= 3 );
        int i = ijk[0];
        int j = ijk[1];
        int k = ijk[2];
        if ( type == GeomType::Face && d_meshType == GeomType::Cell ) {
            if ( d_index.d_side == 0 ) {
                index_list.emplace_back( type, 2, i, j - 1, k );
                index_list.emplace_back( type, 2, i, j, k );
                index_list.emplace_back( type, 1, i, j, k - 1 );
                index_list.emplace_back( type, 1, i, j, k );
            } else if ( d_index.d_side == 1 ) {
                index_list.emplace_back( type, 2, i - 1, j, k );
                index_list.emplace_back( type, 2, i, j, k );
                index_list.emplace_back( type, 0, i, j, k - 1 );
                index_list.emplace_back( type, 0, i, j, k );
            } else if ( d_index.d_side == 2 ) {
                index_list.emplace_back( type, 1, i - 1, j, k );
                index_list.emplace_back( type, 1, i, j, k );
                index_list.emplace_back( type, 0, i, j - 1, k );
                index_list.emplace_back( type, 0, i, j, k );
            } else {
                AMP_ERROR( "Internal error" );
            }
        } else if ( type == GeomType::Cell && d_meshType == GeomType::Cell ) {
            if ( d_index.d_side == 0 ) {
                index_list.emplace_back( type, 0, i, j - 1, k - 1 );
                index_list.emplace_back( type, 0, i, j, k - 1 );
                index_list.emplace_back( type, 0, i, j - 1, k );
                index_list.emplace_back( type, 0, i, j, k );
            } else if ( d_index.d_side == 1 ) {
                index_list.emplace_back( type, 0, i - 1, j, k - 1 );
                index_list.emplace_back( type, 0, i, j, k - 1 );
                index_list.emplace_back( type, 0, i - 1, j, k );
                index_list.emplace_back( type, 0, i, j, k );
            } else if ( d_index.d_side == 2 ) {
                index_list.emplace_back( type, 0, i - 1, j - 1, k );
                index_list.emplace_back( type, 0, i, j - 1, k );
                index_list.emplace_back( type, 0, i - 1, j, k );
                index_list.emplace_back( type, 0, i, j, k );
            } else {
                AMP_ERROR( "Internal error" );
            }
        } else {
            AMP_ERROR( getErrMsg( "Unknown type" ) );
        }
    } else {
        AMP_ERROR( getErrMsg( "Case not programmed yet" ) );
    }
    // Get some basic properties from the mesh
    auto meshGeomDim = (int) d_meshType;
    int size[3]      = { 1, 1, 1 };
    for ( int d = 0; d < meshGeomDim; d++ )
        size[d] = d_mesh->d_globalSize[d];
    auto BC = getBC();
    // Remove any elements that are outside the physical domain
    size_t k = 0;
    for ( size_t i = 0; i < index_list.size(); i++ ) {
        bool erase = false;
        if ( static_cast<int>( d_meshType ) < 2 && index_list[i].d_index[1] != 0 )
            erase = true;
        if ( static_cast<int>( d_meshType ) < 3 && index_list[i].d_index[2] != 0 )
            erase = true;
        for ( int d = 0; d < meshGeomDim; d++ ) {
            if ( BC[d] == 1 )
                continue;
            if ( BC[d] == 2 )
                AMP_WARN_ONCE( "Not finished" );
            int i_max = size[d];
            if ( (int) type == meshGeomDim ) {
                // The geometric type of the mesh must be within the domain
            } else if ( type == GeomType::Vertex ) {
                // Verticies can exist on all faces
                i_max++;
            } else if ( (int) type == meshGeomDim - 1 ) {
                if ( index_list[i].d_side == d )
                    i_max++;
            } else {
                if ( index_list[i].d_side != d )
                    i_max++;
            }
            if ( index_list[i].d_index[d] < 0 || index_list[i].d_index[d] >= i_max )
                erase = true;
        }
        if ( !erase ) {
            index_list[k] = index_list[i];
            k++;
        }
    }
    index_list.resize( k );
    // Fix any elements that are beyond a periodic boundary
    for ( auto &index : index_list )
        d_mesh->fixPeriodic( index );
    // Create the elements
    AMP::Utilities::quicksort( index_list );
    std::vector<std::unique_ptr<MeshElement>> elements( index_list.size() );
    for ( size_t i = 0; i < index_list.size(); i++ )
        elements[i] = std::make_unique<structuredMeshElement>( index_list[i], d_mesh );
    return elements;
}


/****************************************************************
 * Functions to get the centroid                                 *
 ****************************************************************/
Point structuredMeshElement::centroid() const
{
    auto type = d_index.type();
    if ( type == GeomType::Vertex )
        return this->coord();
    // Get the number of vertices and their indicies
    int N = 0;
    BoxMesh::MeshElementIndex nodes[8];
    getElementIndex( GeomType::Vertex, N, nodes );
    // Calculate the centroid
    Point p( 0, 0, 0 );
    for ( int i = 0; i < N; i++ ) {
        double x[3] = { 0, 0, 0 };
        d_mesh->coord( nodes[i], x );
        p.x() += x[0];
        p.y() += x[1];
        p.z() += x[2];
    }
    p.x() /= N;
    p.y() /= N;
    p.z() /= N;
    p.setNdim( d_physicalDim );
    return p;
}


/****************************************************************
 * Functions to get the element volume                           *
 ****************************************************************/
double structuredMeshElement::volume() const
{
    auto type = d_index.type();
    if ( type == GeomType::Vertex )
        AMP_ERROR( "volume is is not defined Nodes" );
    int N = 0;
    BoxMesh::MeshElementIndex nodes[8];
    getElementIndex( GeomType::Vertex, N, nodes );
    if ( type == GeomType::Edge ) {
        AMP_ASSERT( N == 2 );
        double x[2][3];
        d_mesh->coord( nodes[0], x[0] );
        d_mesh->coord( nodes[1], x[1] );
        double dist2 = 0.0;
        for ( int i = 0; i < d_physicalDim; i++ )
            dist2 += ( x[0][i] - x[1][i] ) * ( x[0][i] - x[1][i] );
        return std::sqrt( dist2 );
    } else if ( type == GeomType::Face ) {
        // Use 2x2 quadrature to approximate the surface area. See for example,
        // Y. Zhang, C. Bajaj, G. Xu. Surface Smoothing and Quality Improvement
        // of Quadrilateral/Hexahedral Meshes with Geometric Flow. The special
        // issue of the Journal Communications in Numerical Methods in
        // Engineering (CNME), submitted as an invited paper, 2006.
        // http://www.ices.utexas.edu/~jessica/paper/quadhexgf/quadhex_geomflow_CNM.pdf
        AMP_ASSERT( N == 4 );
        double x[4][3];
        d_mesh->coord( nodes[0], x[0] );
        d_mesh->coord( nodes[1], x[1] );
        d_mesh->coord( nodes[2], x[2] );
        d_mesh->coord( nodes[3], x[3] );
        double AB[3] = { 0, 0, 0 }, AC[3] = { 0, 0, 0 }, AD[3] = { 0, 0, 0 },
               AC_AB_AD[3] = { 0, 0, 0 };
        for ( int i = 0; i < d_physicalDim; i++ ) {
            AC[i]       = x[2][i] - x[0][i];     // Vector pointing from A to C
            AB[i]       = x[1][i] - x[0][i];     // Vector pointing from A to B
            AD[i]       = x[3][i] - x[0][i];     // Vector pointing from A to D
            AC_AB_AD[i] = AC[i] - AB[i] - AD[i]; // The diagonal vector minus the side vectors
        }
        if ( AC_AB_AD[0] == 0 && AC_AB_AD[1] == 0 && AC_AB_AD[2] == 0 ) {
            // The points are co-planar
            return cross3magnitude( AB, AD );
        } else {
            const double q[2] = { 0.5 - std::sqrt( 3.0 ) / 6.0, 0.5 + std::sqrt( 3.0 ) / 6.0 };
            double vol        = 0.0;
            double v1[3], v2[3];
            for ( auto &elem : q ) {
                for ( auto &q_j : q ) {
                    v1[0] = AB[0] + elem * AC_AB_AD[0];
                    v1[1] = AB[1] + elem * AC_AB_AD[1];
                    v1[2] = AB[2] + elem * AC_AB_AD[2];
                    v2[0] = AD[0] + q_j * AC_AB_AD[0];
                    v2[1] = AD[1] + q_j * AC_AB_AD[1];
                    v2[2] = AD[2] + q_j * AC_AB_AD[2];
                    vol += cross3magnitude( v1, v2 );
                }
            }
            return 0.25 * vol;
        }
    } else if ( type == GeomType::Cell ) {
        // Compute the volume of the tri-linear hex by splitting it
        // into 6 sub-pyramids and applying the formula in:
        //   "Calculation of the Volume of a General Hexahedron for Flow Predictions",
        //    IAA Journal v.23, no.6, 1984, p.954-
        constexpr uint8_t sub_pyr[6][4] = { { 0, 3, 2, 1 }, { 6, 7, 4, 5 }, { 0, 1, 5, 4 },
                                            { 3, 7, 6, 2 }, { 0, 4, 7, 3 }, { 1, 2, 6, 5 } };
        // Get the vertices
        AMP_ASSERT( N == 8 );
        double x[8][3];
        for ( int i = 0; i < 8; i++ )
            d_mesh->coord( nodes[i], x[i] );
        AMP_ASSERT( N == 8 );
        // The centroid is a convenient point to use
        // for the apex of all the pyramids
        double R[3] = { 0, 0, 0 };
        for ( auto &y : x ) {
            R[0] += y[0];
            R[1] += y[1];
            R[2] += y[2];
        }
        R[0] *= 0.125;
        R[1] *= 0.125;
        R[2] *= 0.125;
        // Compute the volume using 6 sub-pyramids
        int pyr_base[4];
        double vol = 0.0;
        for ( auto elem : sub_pyr ) {
            // Set the nodes of the pyramid base
            for ( unsigned int i = 0; i < 4; ++i )
                pyr_base[i] = elem[i];
            // Compute diff vectors
            double a[3], b[3], c[3], d[3], e[3];
            for ( int i = 0; i < 3; i++ ) {
                a[i] = x[pyr_base[0]][i] - R[i];
                b[i] = x[pyr_base[1]][i] - x[pyr_base[3]][i];
                c[i] = x[pyr_base[2]][i] - x[pyr_base[0]][i];
                d[i] = x[pyr_base[3]][i] - x[pyr_base[0]][i];
                e[i] = x[pyr_base[1]][i] - x[pyr_base[0]][i];
            }
            // Compute pyramid volume
            double sub_vol =
                ( 1.0 / 6.0 ) * dot3cross( a, b, c ) + ( 1.0 / 12.0 ) * dot3cross( c, d, e );
            AMP_ASSERT( sub_vol > 0.0 );
            vol += sub_vol;
        }
        return vol;
    }
    AMP_ERROR( "Internal error" );
    return 0.0;
}


/****************************************************************
 * Function to get the element normal                            *
 ****************************************************************/
Point structuredMeshElement::norm() const
{
    auto type = d_index.type();
    if ( type == GeomType::Vertex )
        AMP_ERROR( "norm is is not defined Nodes" );
    if ( static_cast<int>( type ) + 1 != static_cast<int>( d_physicalDim ) )
        AMP_ERROR( "norm is not valid" );
    int N = 0;
    BoxMesh::MeshElementIndex nodes[8];
    getElementIndex( GeomType::Vertex, N, nodes );
    if ( type == GeomType::Edge ) {
        AMP_ASSERT( N == 2 );
        Point p1 = { 0, 0 }, p2 = { 0, 0 };
        d_mesh->coord( nodes[0], p1.data() );
        d_mesh->coord( nodes[1], p2.data() );
        Point n = { -( p2.y() - p1.y() ), p2.x() - p1.x() };
        return normalize( n );
    } else if ( d_index.type() == GeomType::Face ) {
        AMP_ASSERT( N == 4 );
        std::array<std::array<double, 3>, 4> p = { std::array<double, 3>{ 0, 0, 0 } };
        d_mesh->coord( nodes[0], p[0].data() );
        d_mesh->coord( nodes[1], p[1].data() );
        d_mesh->coord( nodes[2], p[2].data() );
        d_mesh->coord( nodes[3], p[3].data() );
        return AMP::Geometry::GeometryHelpers::normalToQuadrilateral( p );
    } else if ( d_index.type() == GeomType::Cell ) {
        AMP_ERROR( "Not finished (dimension>3)" );
    }
    AMP_ERROR( "Internal error" );
    return {};
}


/****************************************************************
 * Misc functions                                                *
 ****************************************************************/
bool structuredMeshElement::containsPoint( const Point &, double ) const
{
    AMP_ERROR( "Not finsihed" );
    return false;
}
bool structuredMeshElement::isOnSurface() const { return d_mesh->isOnSurface( d_index ); }
bool structuredMeshElement::isOnBoundary( int id ) const
{
    return d_mesh->isOnBoundary( d_index, id );
}
bool structuredMeshElement::isInBlock( int id ) const
{
    if ( id == d_mesh->d_blockID )
        return true;
    return false;
}


/****************************************************************
 * Calculate the nearest point on the element                    *
 ****************************************************************/
MeshPoint<double> structuredMeshElement::nearest( const MeshPoint<double> &pos0 ) const
{
    using AMP::Geometry::GeometryHelpers::nearest;
    using AMP::Geometry::GeometryHelpers::Point2D;
    using AMP::Geometry::GeometryHelpers::Point3D;
    // Get the vertex coordinates
    int N                      = 0;
    std::array<double, 3> x[8] = { { 0, 0, 0 } };
    BoxMesh::MeshElementIndex nodes[8];
    getElementIndex( GeomType::Vertex, N, nodes );
    std::array<double, 3> centroid = { { 0, 0, 0 } };
    for ( int i = 0; i < N; i++ ) {
        d_mesh->coord( nodes[i], x[i].data() );
        centroid[0] += x[i][0];
        centroid[1] += x[i][1];
        centroid[2] += x[i][2];
    }
    centroid[0] /= N;
    centroid[1] /= N;
    centroid[2] /= N;
    // Get the nearest point
    std::array<double, 3> pos = { pos0.x(), pos0.y(), pos0.z() };
    std::array<double, 3> y   = { 0, 0, 0 };
    if ( d_index.type() == GeomType::Vertex ) {
        AMP_ASSERT( N == 1 );
        y = x[0];
    } else if ( d_index.type() == GeomType::Edge ) {
        AMP_ASSERT( N == 2 );
        y = nearest( x[0], x[1], pos );
    } else if ( d_index.type() == GeomType::Face ) {
        AMP_ASSERT( N == 4 );
        using TRI = std::array<Point3D, 3>;
        auto y1   = nearest( TRI( { x[0], x[1], centroid } ), pos );
        auto y2   = nearest( TRI( { x[1], x[2], centroid } ), pos );
        auto y3   = nearest( TRI( { x[2], x[3], centroid } ), pos );
        auto y4   = nearest( TRI( { x[3], x[0], centroid } ), pos );
        double d1 = distance( pos, y1 );
        double d2 = distance( pos, y2 );
        double d3 = distance( pos, y3 );
        double d4 = distance( pos, y4 );
        if ( d1 <= std::min( { d2, d3, d4 } ) )
            y = y1;
        else if ( d2 <= std::min( d3, d4 ) )
            y = y2;
        else if ( d3 <= d4 )
            y = y3;
        else
            y = y4;
    } else if ( d_index.type() == GeomType::Cell ) {
        AMP_ASSERT( N == 8 );
        AMP_ERROR( "structuredMeshElement::nearest is not implemented for Volume" );
    } else {
        AMP_ERROR( "Internal error in structuredMeshElement::nearest" );
    }
    return MeshPoint<double>( pos0.ndim(), y.data() );
}


/****************************************************************
 * Calculate the distance to the element                         *
 ****************************************************************/
template<std::size_t N>
static inline std::array<double, N> point( const double *x )
{
    std::array<double, N> y;
    for ( size_t i = 0; i < N; i++ )
        y[i] = x[i];
    return y;
}
template<std::size_t N>
static inline std::array<double, N> point( const MeshPoint<double> &x )
{
    return point<N>( x.data() );
}
double structuredMeshElement::distance( const MeshPoint<double> &pos,
                                        const MeshPoint<double> &dir ) const
{
    using AMP::Geometry::GeometryHelpers::distanceToLine;
    using AMP::Geometry::GeometryHelpers::distanceToQuadrilateral;
    using AMP::Geometry::GeometryHelpers::Point2D;
    using AMP::Geometry::GeometryHelpers::Point3D;
    // Get the vertex coordinates
    int N          = 0;
    double x[8][3] = { { 0, 0, 0 } };
    BoxMesh::MeshElementIndex nodes[8];
    getElementIndex( GeomType::Vertex, N, nodes );
    for ( int i = 0; i < N; i++ )
        d_mesh->coord( nodes[i], x[i] );
    // Compute the distance
    if ( d_index.type() == GeomType::Vertex ) {
        AMP_ASSERT( N == 1 );
        AMP_ERROR( "structuredMeshElement::distance is not implemented for node" );
    } else if ( d_index.type() == GeomType::Edge ) {
        AMP_ASSERT( N == 2 );
        if ( d_physicalDim == 2 ) {
            return distanceToLine(
                point<2>( pos ), point<2>( dir ), point<2>( x[0] ), point<2>( x[0] ) );
        } else if ( d_physicalDim == 3 ) {
            return distanceToLine(
                point<3>( pos ), point<3>( dir ), point<3>( x[0] ), point<3>( x[0] ) );
        } else {
            AMP_ERROR( "Not finished" );
        }
    } else if ( d_index.type() == GeomType::Face ) {
        AMP_ASSERT( N == 4 );
        if ( d_physicalDim == 2 ) {
            std::array<Point2D, 4> quad = {
                point<2>( x[0] ), point<2>( x[1] ), point<2>( x[2] ), point<2>( x[3] )
            };
            return distanceToQuadrilateral( quad, point<2>( pos ), point<2>( dir ) );
        } else if ( d_physicalDim == 3 ) {
            std::array<Point3D, 4> quad = {
                point<3>( x[0] ), point<3>( x[1] ), point<3>( x[2] ), point<3>( x[3] )
            };
            return distanceToQuadrilateral( quad, point<3>( pos ), point<3>( dir ) );
        } else {
            AMP_ERROR( "Not finished" );
        }
    } else if ( d_index.type() == GeomType::Cell ) {
        AMP_ASSERT( N == 8 );
        AMP_ERROR( "Not finished" );
    } else {
        AMP_ERROR( "Internal error in structuredMeshElement::distance" );
    }
    return 0;
}


} // namespace AMP::Mesh
