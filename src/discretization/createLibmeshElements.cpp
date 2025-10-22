#include "AMP/discretization/createLibmeshElements.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"


#ifdef AMP_USE_LIBMESH

DISABLE_WARNINGS
    #include "libmesh/auto_ptr.h"
    #include "libmesh/cell_hex27.h"
    #include "libmesh/cell_hex8.h"
    #include "libmesh/elem.h"
    #include "libmesh/enum_fe_family.h"
    #include "libmesh/enum_order.h"
    #include "libmesh/enum_quadrature_type.h"
    #include "libmesh/face_quad4.h"
    #include "libmesh/face_quad9.h"
    #include "libmesh/fe_base.h"
    #include "libmesh/fe_type.h"
    #include "libmesh/node.h"
    #include "libmesh/quadrature.h"
    #include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


namespace AMP::Discretization {


// Default constructor
createLibmeshElements::createLibmeshElements() = default;


// De-constructor
createLibmeshElements::~createLibmeshElements()
{
    reinit( AMP::Mesh::MeshIterator(),
            libMeshEnums::INVALID_Q_RULE,
            libMeshEnums::INVALID_ORDER,
            std::shared_ptr<const libMesh::FEType>() );
}


// Re-initialize the class
void createLibmeshElements::reinit( const AMP::Mesh::MeshIterator &iterator_in )
{
    reinit( iterator_in,
            libMeshEnums::INVALID_Q_RULE,
            libMeshEnums::INVALID_ORDER,
            std::shared_ptr<const libMesh::FEType>() );
}


// Re-initialize the class
void createLibmeshElements::reinit( const AMP::Mesh::MeshIterator &iterator_in,
                                    libMeshEnums::QuadratureType qtype,
                                    libMeshEnums::Order qorder,
                                    std::shared_ptr<const libMesh::FEType> type,
                                    bool cache_fe )
{
    PROFILE( "reinit" );
    // Destroy the existing libmesh elements
    for ( auto &elem : d_base_element ) {
        delete elem;
        elem = nullptr;
    }
    d_base_element.resize( 0 );
    for ( auto &elem : d_rule_element ) {
        delete elem;
        elem = nullptr;
    }
    d_rule_element.resize( 0 );
    for ( auto &elem : d_elements ) {
        for ( size_t j = 0; j < elem->n_nodes(); j++ ) {
            delete elem->node_ptr( j );
            elem->set_node( j ) = nullptr;
        }
        delete elem;
        elem = nullptr;
    }
    d_ids.resize( 0 );
    d_index.resize( 0 );
    d_elements.resize( 0 );
    d_base_element.resize( 0 );
    d_rule_element.resize( 0 );
    d_base.reset();
    d_rule.reset();
    // Create the new libmesh elements
    d_qtype                          = qtype;
    d_qorder                         = qorder;
    d_type                           = type;
    AMP::Mesh::MeshIterator iterator = iterator_in.begin();
    if ( d_type != nullptr && iterator.size() > 0 ) {
        auto dim = (int) iterator->elementType();
        d_rule.reset( libMesh::QBase::build( d_qtype, dim, d_qorder ).release() );
        d_base.reset( libMesh::FEBase::build( dim, *d_type ).release() );
        d_base->attach_quadrature_rule( d_rule.get() );
    }
    const size_t N = iterator.size();
    d_ids.resize( N );
    std::vector<libMesh::Elem *> elements( N, nullptr );
    std::vector<libMesh::FEBase *> base( N, nullptr );
    std::vector<libMesh::QBase *> rule( N, nullptr );
    for ( size_t i = 0; i < N; ++i, ++iterator ) {
        d_ids[i] = iterator->globalID();
        // Create the libmesh element
        elements[i] = createLibmeshElements::createElement( *iterator );
        // Create the libmesh QBase and FEBase for the element
        if ( d_type != nullptr && cache_fe ) {
            auto dim = (int) iterator->elementType();
            rule[i]  = libMesh::QBase::build( d_qtype, dim, d_qorder ).release();
            base[i]  = libMesh::FEBase::build( dim, *d_type ).release();
            base[i]->attach_quadrature_rule( rule[i] );
            base[i]->reinit( elements[i] );
        }
    }
    // Sort the ids and elements for fast search
    d_index.resize( N, 0 );
    for ( size_t i = 0; i < N; i++ )
        d_index[i] = i;
    AMP::Utilities::quicksort( d_ids, d_index );
    d_elements.resize( N, nullptr );
    for ( size_t i = 0; i < N; i++ )
        d_elements[i] = elements[d_index[i]];
    if ( !d_base_element.empty() ) {
        d_base_element.resize( N, nullptr );
        d_rule_element.resize( N );
        for ( size_t i = 0; i < N; i++ )
            d_base_element[i] = base[d_index[i]];
        for ( size_t i = 0; i < N; i++ )
            d_rule_element[i] = rule[d_index[i]];
    }
}


// Create a libmesh element
libMesh::Elem *createLibmeshElements::createElement( const AMP::Mesh::MeshElement &elem )
{
    auto dim               = (int) elem.elementType();
    auto nodes             = elem.getElements( AMP::Mesh::GeomType::Vertex );
    libMesh::Elem *element = nullptr;
    // Create the libmesh element
    if ( dim == 3 && nodes.size() == 8 ) {
        // We are dealing with a hex8 element
        element = new libMesh::Hex8;
    } else if ( dim == 3 && nodes.size() == 27 ) {
        // We are dealing with a hex27 element
        element = new libMesh::Hex27;
    } else if ( dim == 2 && nodes.size() == 4 ) {
        // We are dealing with a quad4 element
        element = new libMesh::Quad4;
    } else if ( dim == 2 && nodes.size() == 9 ) {
        // We are dealing with a quad9 element
        element = new libMesh::Quad9;
    } else {
        AMP_ERROR( "Unknown element type" );
    }
    for ( size_t j = 0; j < nodes.size(); j++ ) {
        auto pt = nodes[j]->coord();
        if ( pt.size() == 3 )
            element->set_node( j ) = new libMesh::Node( pt[0], pt[1], pt[2], j );
        else if ( pt.size() == 2 )
            element->set_node( j ) = new libMesh::Node( pt[0], pt[1], 0, j );
        else
            AMP_ERROR( "Unsupported physical dimension" );
    }
    return element;
}


// Search and return the desired libmesh element
const libMesh::Elem *createLibmeshElements::getElement( const AMP::Mesh::MeshElementID &id ) const
{
    PROFILE( "getElement", 2 );
    size_t index = AMP::Utilities::findfirst( d_ids, id );
    index        = std::min<size_t>( index, d_ids.size() - 1 );
    AMP_INSIST( d_ids[index] == id, "Desired element was not found" );
    return d_elements[index];
}


// Search and return the desired libmesh FEBase
const libMesh::FEBase *createLibmeshElements::getFEBase( const AMP::Mesh::MeshElementID &id ) const
{
    PROFILE( "getFEBase", 2 );
    const libMesh::FEBase *result = nullptr;
    if ( d_base != nullptr ) {
        size_t index = AMP::Utilities::findfirst( d_ids, id );
        index        = std::min<size_t>( index, d_ids.size() - 1 );
        AMP_INSIST( d_ids[index] == id, "Desired element was not found" );
        if ( !d_base_element.empty() ) {
            result = d_base_element[index];
        } else {
            if ( d_last_id != id ) {
                const libMesh::Elem *elem = d_elements[index];
                d_base->reinit( elem );
                d_last_id = id;
            }
            result = d_base.get();
        }
    }
    return result;
}


// Search and return the desired libmesh FEBase
const libMesh::QBase *createLibmeshElements::getQBase( const AMP::Mesh::MeshElementID &id ) const
{
    PROFILE( "getQBase", 2 );
    const libMesh::QBase *result = nullptr;
    if ( d_rule != nullptr ) {
        size_t index = AMP::Utilities::findfirst( d_ids, id );
        index        = std::min<size_t>( index, d_ids.size() - 1 );
        AMP_INSIST( d_ids[index] == id, "Desired element was not found" );
        if ( !d_rule_element.empty() ) {
            result = d_rule_element[index];
        } else {
            if ( d_last_id != id ) {
                const libMesh::Elem *elem = d_elements[index];
                d_base->reinit( elem );
                d_last_id = id;
            }
            result = d_rule.get();
        }
    }
    return result;
}
} // namespace AMP::Discretization


#endif
