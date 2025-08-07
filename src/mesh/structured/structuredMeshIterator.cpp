#include "AMP/mesh/structured/structuredMeshIterator.h"
#include "AMP/IO/HDF.hpp"
#include "AMP/IO/RestartManager.h"
#include "AMP/mesh/structured/structuredMeshElement.h"

#include <utility>


namespace AMP::Mesh {


// unused global variable to prevent compiler warning
static MeshElement nullElement;


/********************************************************
 * Constructors                                          *
 ********************************************************/
inline BoxMesh::MeshElementIndex structuredMeshIterator::getCurrentIndex() const
{
    if ( d_pos >= d_size )
        return {};
    else if ( d_elements )
        return d_elements->operator[]( d_pos );
    else
        return *d_it;
}


/********************************************************
 * Constructors                                          *
 ********************************************************/
static constexpr auto MeshIteratorType = AMP::getTypeID<structuredMeshIterator>().hash;
static_assert( MeshIteratorType != 0 );
structuredMeshIterator::structuredMeshIterator()
{
    d_typeHash    = MeshIteratorType;
    d_iterator    = nullptr;
    d_pos         = 0;
    d_size        = 0;
    d_mesh        = nullptr;
    d_element     = &d_cur_element;
    d_cur_element = structuredMeshElement();
}
structuredMeshIterator::structuredMeshIterator( const BoxMesh::MeshElementIndexIterator &it,
                                                const AMP::Mesh::BoxMesh *mesh,
                                                size_t pos )
    : d_it( it ), d_mesh( mesh )
{
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_pos      = pos;
    d_size     = d_it.size();
    d_element  = &d_cur_element;
    d_it.set( d_pos );
    d_cur_element = structuredMeshElement( getCurrentIndex(), d_mesh );
}
structuredMeshIterator::structuredMeshIterator( const BoxMesh::MeshElementIndex &first,
                                                const BoxMesh::MeshElementIndex &last,
                                                const AMP::Mesh::BoxMesh *mesh,
                                                size_t pos )
    : structuredMeshIterator(
          BoxMesh::MeshElementIndexIterator( first, last, mesh, pos ), mesh, pos )
{
}
structuredMeshIterator::structuredMeshIterator(
    std::shared_ptr<const std::vector<BoxMesh::MeshElementIndex>> elements,
    const AMP::Mesh::BoxMesh *mesh,
    size_t pos )
    : d_elements( std::move( elements ) ), d_mesh( mesh )
{
    d_typeHash    = MeshIteratorType;
    d_iterator    = nullptr;
    d_pos         = pos;
    d_size        = d_elements->size();
    d_element     = &d_cur_element;
    d_cur_element = structuredMeshElement( getCurrentIndex(), d_mesh );
}
structuredMeshIterator::structuredMeshIterator( const structuredMeshIterator &rhs )
    : MeshIterator(), d_it( rhs.d_it ), d_elements( rhs.d_elements ), d_mesh( rhs.d_mesh )
{
    d_pos         = rhs.d_pos;
    d_size        = rhs.d_size;
    d_typeHash    = MeshIteratorType;
    d_iterator    = nullptr;
    d_element     = &d_cur_element;
    d_cur_element = structuredMeshElement( getCurrentIndex(), d_mesh );
}
structuredMeshIterator &structuredMeshIterator::operator=( const structuredMeshIterator &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    d_typeHash    = MeshIteratorType;
    d_iterator    = nullptr;
    d_pos         = rhs.d_pos;
    d_size        = rhs.d_size;
    d_it          = rhs.d_it;
    d_mesh        = rhs.d_mesh;
    d_elements    = rhs.d_elements;
    d_element     = &d_cur_element;
    d_cur_element = structuredMeshElement( getCurrentIndex(), d_mesh );
    return *this;
}


/********************************************************
 * Function to clone the iterator                        *
 ********************************************************/
MeshIterator *structuredMeshIterator::clone() const { return new structuredMeshIterator( *this ); }


/********************************************************
 * De-constructor                                        *
 ********************************************************/
structuredMeshIterator::~structuredMeshIterator() = default;


/********************************************************
 * Return an iterator to the beginning or end            *
 ********************************************************/
MeshIterator structuredMeshIterator::begin() const
{
    if ( d_elements )
        return structuredMeshIterator( d_elements, d_mesh, 0 );
    else
        return structuredMeshIterator( d_it, d_mesh, 0 );
}
MeshIterator structuredMeshIterator::end() const
{
    if ( d_elements )
        return structuredMeshIterator( d_elements, d_mesh, d_size );
    else
        return structuredMeshIterator( d_it, d_mesh, d_size );
}


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
MeshIterator &structuredMeshIterator::operator++()
{
    // Prefix increment (increment and return this)
    d_pos++;
    if ( d_pos >= d_size )
        d_pos = d_size;
    d_it.set( d_pos );
    d_cur_element.reset( getCurrentIndex() );
    return *this;
}
MeshIterator &structuredMeshIterator::operator--()
{
    // Prefix decrement (increment and return this)
    if ( d_pos != 0 )
        d_pos--;
    d_it.set( d_pos );
    d_cur_element.reset( getCurrentIndex() );
    return *this;
}


/********************************************************
 * Random access iterators                               *
 ********************************************************/
MeshIterator &structuredMeshIterator::operator+=( int n )
{
    if ( n >= 0 ) { // increment *this
        auto n2 = static_cast<size_t>( n );
        if ( d_pos + n2 > d_size )
            AMP_ERROR( "Iterated past end of iterator" );
        d_pos += n2;
    } else { // decrement *this
        auto n2 = static_cast<size_t>( -n );
        if ( n2 > d_pos )
            AMP_ERROR( "Iterated past beginning of iterator" );
        d_pos -= n2;
    }
    d_it.set( d_pos );
    d_cur_element.reset( getCurrentIndex() );
    return *this;
}


/********************************************************
 * Compare two iterators                                 *
 ********************************************************/
bool structuredMeshIterator::operator==( const MeshIterator &rhs ) const
{
    if ( size() != rhs.size() )
        return false;
    const structuredMeshIterator *rhs2 = nullptr;
    // Convert rhs to a structuredMeshIterator* so we can access the base class members
    auto *tmp = reinterpret_cast<const structuredMeshIterator *>( &rhs );
    if ( tmp->d_typeHash == MeshIteratorType ) {
        rhs2 = tmp; // We can safely cast rhs to a structuredMeshIterator
    } else if ( tmp->d_iterator ) {
        tmp = reinterpret_cast<const structuredMeshIterator *>( tmp->d_iterator );
        if ( tmp->d_typeHash == MeshIteratorType )
            rhs2 = tmp; // We can safely cast rhs.iterator to a structuredMeshIterator
    }
    // Perform direct comparisions if we are dealing with two structuredMeshIterators
    if ( rhs2 ) {
        if ( d_mesh != rhs2->d_mesh || d_pos != rhs2->d_pos || d_size != rhs2->d_size )
            return false;
        if ( d_elements || rhs2->d_elements ) {
            auto set1 = this->getElements();
            auto set2 = rhs2->getElements();
            if ( set1.get() != set2.get() ) {
                for ( size_t i = 0; i < d_size; i++ ) {
                    if ( set1->operator[]( i ) != set2->operator[]( i ) )
                        return false;
                }
            }
        } else {
            if ( d_it != rhs2->d_it )
                return false;
        }
        return true;
    }
    /* We are comparing a structuredMeshIterator to an arbitrary iterator
     * The iterators are the same if they point to the same position and iterate
     * over the same elements in the same order
     */
    // Check the size
    if ( this->size() != rhs.size() )
        return false;
    // Check the current position
    if ( this->position() != rhs.position() )
        return false;
    // Check that the elements match
    auto iterator = rhs.begin();
    auto set1     = getElements();
    for ( size_t i = 0; i < d_size; i++ ) {
        auto *elem2 = dynamic_cast<structuredMeshElement *>( iterator->getRawElement() );
        if ( elem2 == nullptr )
            return false;
        const auto &index1 = set1->operator[]( i );
        const auto &index2 = elem2->d_index;
        if ( index1 != index2 )
            return false;
        ++iterator;
    }
    return true;
}
bool structuredMeshIterator::operator!=( const MeshIterator &rhs ) const
{
    return !( ( *this ) == rhs );
}


/********************************************************
 * Get all elements in the iterator                      *
 ********************************************************/
std::shared_ptr<const std::vector<BoxMesh::MeshElementIndex>>
structuredMeshIterator::getElements() const
{
    if ( d_elements )
        return d_elements;
    auto elements = std::make_shared<std::vector<BoxMesh::MeshElementIndex>>();
    elements->reserve( d_size );
    for ( auto index : d_it )
        elements->emplace_back( index );
    return elements;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void structuredMeshIterator::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    manager->registerObject( d_mesh->shared_from_this() );
}
void structuredMeshIterator::writeRestart( int64_t fid ) const
{
    MeshIterator::writeRestart( fid );
    auto elements = d_elements;
    IO::writeHDF5( fid, "meshID", d_mesh->meshID() );
    if ( d_elements ) {
        IO::writeHDF5( fid, "elements", *d_elements );
    } else {
        IO::writeHDF5( fid, "first", d_it.first() );
        IO::writeHDF5( fid, "last", d_it.last() );
    }
}
structuredMeshIterator::structuredMeshIterator( int64_t fid, AMP::IO::RestartManager *manager )
    : MeshIterator( fid )
{
    MeshID meshID;
    IO::readHDF5( fid, "meshID", meshID );
    auto mesh = manager->getData<AMP::Mesh::Mesh>( meshID.getHash() ).get();
    d_mesh    = dynamic_cast<BoxMesh *>( mesh );
    AMP_ASSERT( d_mesh );
    if ( IO::H5Gexists( fid, "elements" ) || IO::H5Dexists( fid, "elements" ) ) {
        std::vector<BoxMesh::MeshElementIndex> elements;
        IO::readHDF5( fid, "elements", elements );
        d_elements =
            std::make_shared<std::vector<BoxMesh::MeshElementIndex>>( std::move( elements ) );
    } else {
        BoxMesh::MeshElementIndex first, last;
        IO::readHDF5( fid, "first", first );
        IO::readHDF5( fid, "last", last );
        d_it = BoxMesh::MeshElementIndexIterator( first, last, d_mesh, d_pos );
        d_it.set( d_pos );
    }
    d_element     = &d_cur_element;
    d_cur_element = structuredMeshElement( getCurrentIndex(), d_mesh );
}


} // namespace AMP::Mesh
