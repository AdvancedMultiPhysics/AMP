#include "AMP/mesh/structured/structuredMeshIterator.h"
#include "AMP/IO/HDF.hpp"
#include "AMP/IO/RestartManager.h"
#include "AMP/mesh/structured/structuredMeshElement.h"

#include <utility>


namespace AMP::Mesh {


// unused global variable to prevent compiler warning
static MeshElement nullElement;


/********************************************************
 * Get the index                                         *
 ********************************************************/
inline BoxMesh::MeshElementIndex structuredMeshIterator::getIndex( int i ) const
{
    if ( i < 0 || i >= (int) d_size ) {
        return BoxMesh::MeshElementIndex();
    } else if ( d_elements ) {
        return d_elements->operator[]( i );
    } else {
        int s1 = d_last.index( 0 ) - d_first.index( 0 ) + 1;
        int s2 = s1 * ( d_last.index( 1 ) - d_first.index( 1 ) + 1 );
        int k  = i / s2;
        i -= k * s2;
        int j = i / s1;
        i -= j * s1;
        i += d_first.index( 0 );
        j += d_first.index( 1 );
        k += d_first.index( 2 );
        if ( d_checkBoundary ) {
            if ( d_isPeriodic[0] && ( i < 0 || i >= d_globalSize[0] ) )
                i = ( i + d_globalSize[0] ) % d_globalSize[0];
            if ( d_isPeriodic[1] && ( j < 0 || j >= d_globalSize[1] ) )
                j = ( j + d_globalSize[1] ) % d_globalSize[1];
            if ( d_isPeriodic[2] && ( k < 0 || k >= d_globalSize[2] ) )
                k = ( k + d_globalSize[2] ) % d_globalSize[2];
        }
        return BoxMesh::MeshElementIndex( d_first.type(), d_first.side(), i, j, k );
    }
}


/********************************************************
 * Constructors                                          *
 ********************************************************/
static constexpr auto MeshIteratorType = AMP::getTypeID<structuredMeshIterator>().hash;
static_assert( MeshIteratorType != 0 );
structuredMeshIterator::structuredMeshIterator()
{
    d_typeHash      = MeshIteratorType;
    d_iterator      = nullptr;
    d_pos           = 0;
    d_size          = 0;
    d_checkBoundary = false;
    d_mesh          = nullptr;
    d_isPeriodic.fill( false );
    d_globalSize.fill( 0 );
    d_element = &d_cur_element;
}
structuredMeshIterator::structuredMeshIterator( const BoxMesh::MeshElementIndex &first,
                                                const BoxMesh::MeshElementIndex &last,
                                                const AMP::Mesh::BoxMesh *mesh,
                                                size_t pos )
    : d_checkBoundary( false ),
      d_isPeriodic(
          { mesh->d_surfaceId[1] == -1, mesh->d_surfaceId[3] == -1, mesh->d_surfaceId[5] == -1 } ),
      d_globalSize( mesh->d_globalSize ),
      d_first( first ),
      d_last( last ),
      d_mesh( mesh )
{
    AMP_ASSERT( first.side() == last.side() && first.type() == last.type() );
    d_typeHash      = MeshIteratorType;
    d_iterator      = nullptr;
    d_pos           = pos;
    d_size          = BoxMesh::MeshElementIndex::numElements( d_first, d_last );
    d_element       = &d_cur_element;
    d_cur_element   = structuredMeshElement( getIndex( d_pos ), d_mesh );
    d_checkBoundary = d_first.index( 0 ) < 0 || d_last.index( 0 ) >= d_globalSize[0] ||
                      d_first.index( 1 ) < 0 || d_last.index( 1 ) >= d_globalSize[1] ||
                      d_first.index( 2 ) < 0 || d_last.index( 2 ) >= d_globalSize[2];
}
structuredMeshIterator::structuredMeshIterator(
    std::shared_ptr<const std::vector<BoxMesh::MeshElementIndex>> elements,
    const AMP::Mesh::BoxMesh *mesh,
    size_t pos )
    : d_checkBoundary( false ),
      d_isPeriodic(
          { mesh->d_surfaceId[1] == -1, mesh->d_surfaceId[3] == -1, mesh->d_surfaceId[5] == -1 } ),
      d_globalSize( mesh->d_globalSize ),
      d_elements( std::move( elements ) ),
      d_mesh( mesh )
{
    d_typeHash    = MeshIteratorType;
    d_iterator    = nullptr;
    d_pos         = pos;
    d_size        = d_elements->size();
    d_element     = &d_cur_element;
    d_cur_element = structuredMeshElement( getIndex( d_pos ), d_mesh );
}
structuredMeshIterator::structuredMeshIterator( const structuredMeshIterator &rhs )
    : MeshIterator(),
      d_checkBoundary( rhs.d_checkBoundary ),
      d_isPeriodic( rhs.d_isPeriodic ),
      d_globalSize( rhs.d_globalSize ),
      d_first( rhs.d_first ),
      d_last( rhs.d_last ),
      d_elements( rhs.d_elements ),
      d_mesh( rhs.d_mesh )
{
    d_pos         = rhs.d_pos;
    d_size        = rhs.d_size;
    d_typeHash    = MeshIteratorType;
    d_iterator    = nullptr;
    d_element     = &d_cur_element;
    d_cur_element = structuredMeshElement( getIndex( d_pos ), d_mesh );
}
structuredMeshIterator &structuredMeshIterator::operator=( const structuredMeshIterator &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    d_typeHash      = MeshIteratorType;
    d_iterator      = nullptr;
    d_pos           = rhs.d_pos;
    d_size          = rhs.d_size;
    d_checkBoundary = rhs.d_checkBoundary;
    d_isPeriodic    = rhs.d_isPeriodic;
    d_globalSize    = rhs.d_globalSize;
    d_mesh          = rhs.d_mesh;
    d_first         = rhs.d_first;
    d_last          = rhs.d_last;
    d_elements      = rhs.d_elements;
    d_element       = &d_cur_element;
    d_cur_element   = structuredMeshElement( getIndex( d_pos ), d_mesh );
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
        return structuredMeshIterator( d_first, d_last, d_mesh, 0 );
}
MeshIterator structuredMeshIterator::end() const
{
    if ( d_elements )
        return structuredMeshIterator( d_elements, d_mesh, d_size );
    else
        return structuredMeshIterator( d_first, d_last, d_mesh, d_size );
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
    d_cur_element.reset( getIndex( d_pos ) );
    return *this;
}
MeshIterator &structuredMeshIterator::operator--()
{
    // Prefix decrement (increment and return this)
    if ( d_pos != 0 )
        d_pos--;
    d_cur_element.reset( getIndex( d_pos ) );
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
    d_cur_element.reset( getIndex( d_pos ) );
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
    } else if ( tmp->d_iterator != nullptr ) {
        tmp = reinterpret_cast<const structuredMeshIterator *>( tmp->d_iterator );
        if ( tmp->d_typeHash == MeshIteratorType )
            rhs2 = tmp; // We can safely cast rhs.iterator to a structuredMeshIterator
    }
    // Perform direct comparisions if we are dealing with two structuredMeshIterators
    if ( rhs2 != nullptr ) {
        if ( d_pos != rhs2->d_pos )
            return false;
        if ( d_size != rhs2->d_size )
            return false;
        if ( d_elements != nullptr || rhs2->d_elements != nullptr ) {
            auto set1 = this->getElements();
            auto set2 = rhs2->getElements();
            if ( set1.get() != set2.get() ) {
                for ( size_t i = 0; i < d_size; i++ ) {
                    if ( set1->operator[]( i ) != set2->operator[]( i ) )
                        return false;
                }
            }
        } else {
            if ( d_first != rhs2->d_first || d_last != rhs2->d_last )
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
    if ( d_elements != nullptr )
        return d_elements;
    auto elements = std::make_shared<std::vector<BoxMesh::MeshElementIndex>>();
    elements->reserve( d_size );
    for ( size_t pos = 0; pos < d_size; pos++ )
        elements->emplace_back( getIndex( pos ) );
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
    if ( !elements )
        elements = std::make_shared<std::vector<BoxMesh::MeshElementIndex>>();
    IO::writeHDF5( fid, "checkBoundary", d_checkBoundary );
    IO::writeHDF5( fid, "isPeriodic", d_isPeriodic );
    IO::writeHDF5( fid, "globalSize", d_globalSize );
    IO::writeHDF5( fid, "first", d_first );
    IO::writeHDF5( fid, "last", d_last );
    IO::writeHDF5( fid, "elements", *elements );
    IO::writeHDF5( fid, "meshID", d_mesh->meshID() );
}
structuredMeshIterator::structuredMeshIterator( int64_t fid, AMP::IO::RestartManager *manager )
    : MeshIterator( fid )
{
    MeshID meshID;
    std::vector<BoxMesh::MeshElementIndex> elements;
    IO::readHDF5( fid, "checkBoundary", d_checkBoundary );
    IO::readHDF5( fid, "isPeriodic", d_isPeriodic );
    IO::readHDF5( fid, "globalSize", d_globalSize );
    IO::readHDF5( fid, "first", d_first );
    IO::readHDF5( fid, "last", d_last );
    IO::readHDF5( fid, "elements", elements );
    IO::readHDF5( fid, "meshID", meshID );
    d_elements.reset();
    if ( !elements.empty() )
        d_elements = std::make_shared<decltype( elements )>( std::move( elements ) );
    auto mesh = manager->getData<AMP::Mesh::Mesh>( meshID.getHash() ).get();
    d_mesh    = dynamic_cast<BoxMesh *>( mesh );
    AMP_ASSERT( d_mesh );
    d_element     = &d_cur_element;
    d_cur_element = structuredMeshElement( getIndex( d_pos ), d_mesh );
}


} // namespace AMP::Mesh
