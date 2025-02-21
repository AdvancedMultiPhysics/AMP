#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/utils/typeid.h"


namespace AMP::Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
static constexpr auto MeshIteratorType = AMP::getTypeID<MultiVectorIterator>().hash;
static_assert( MeshIteratorType != 0 );
MultiVectorIterator::MultiVectorIterator()
{
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_pos      = 0;
    d_size     = 0;
    d_element  = nullptr;
}
MultiVectorIterator::MultiVectorIterator( std::shared_ptr<std::vector<MeshElement>> elements,
                                          size_t pos )
    : d_elements( elements )
{
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_pos      = pos;
    d_size     = d_elements->size();
    d_element  = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
}
MultiVectorIterator::MultiVectorIterator( const std::vector<MeshElement> &elements, size_t pos )
    : d_elements( new std::vector<MeshElement>( elements ) )
{
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_pos      = pos;
    d_size     = d_elements->size();
    d_element  = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
}
MultiVectorIterator::MultiVectorIterator( const MultiVectorIterator &rhs )
    : MeshIterator(), // Note: we never want to call the base copy constructor
      d_elements( rhs.d_elements )
{
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_pos      = rhs.d_pos;
    d_size     = rhs.d_size;
    d_element  = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
}
MultiVectorIterator &MultiVectorIterator::operator=( const MultiVectorIterator &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_elements = rhs.d_elements;
    d_pos      = rhs.d_pos;
    d_size     = rhs.d_size;
    d_element  = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
    return *this;
}


/********************************************************
 * Function to clone the iterator                        *
 ********************************************************/
MeshIterator *MultiVectorIterator::clone() const { return new MultiVectorIterator( *this ); }


/********************************************************
 * De-constructor                                        *
 ********************************************************/
MultiVectorIterator::~MultiVectorIterator() = default;


/********************************************************
 * Return an iterator to the beginning or end            *
 ********************************************************/
MeshIterator MultiVectorIterator::begin() const { return MultiVectorIterator( d_elements, 0 ); }
MeshIterator MultiVectorIterator::end() const
{
    return MultiVectorIterator( d_elements, d_elements->size() );
}


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
MeshIterator &MultiVectorIterator::operator++()
{
    // Prefix increment (increment and return this)
    d_pos++;
    d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
    return *this;
}
MeshIterator &MultiVectorIterator::operator--()
{
    // Prefix decrement (increment and return this)
    d_pos--;
    d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
    return *this;
}


/********************************************************
 * Random access iterators                               *
 ********************************************************/
MeshIterator &MultiVectorIterator::operator+=( int n )
{
    if ( n >= 0 ) { // increment *this
        auto n2 = static_cast<size_t>( n );
        if ( d_pos + n2 > d_elements->size() )
            AMP_ERROR( "Iterated past end of iterator" );
        d_pos += n2;
    } else { // decrement *this
        auto n2 = static_cast<size_t>( -n );
        if ( n2 > d_pos )
            AMP_ERROR( "Iterated past beginning of iterator" );
        d_pos -= n2;
    }
    d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
    return *this;
}


/********************************************************
 * Compare two iterators                                 *
 ********************************************************/
bool MultiVectorIterator::operator==( const MeshIterator &rhs ) const
{
    const MultiVectorIterator *rhs2 = nullptr;
    // Convert rhs to a MultiVectorIterator* so we can access the base class members
    const auto *tmp = reinterpret_cast<const MultiVectorIterator *>( &rhs );
    if ( tmp->d_typeHash == MeshIteratorType ) {
        rhs2 = tmp; // We can safely cast rhs.iterator to a MultiVectorIterator
    } else if ( tmp->d_iterator != nullptr ) {
        tmp = reinterpret_cast<const MultiVectorIterator *>( tmp->d_iterator );
        if ( tmp->d_typeHash == MeshIteratorType )
            rhs2 = tmp; // We can safely cast rhs.iterator to a MultiVectorIterator
    }
    // Perform direct comparisions if we are dealing with two MultiVectorIterators
    if ( rhs2 != nullptr ) {
        // Check that we are at the same position
        if ( d_pos != rhs2->d_pos )
            return false;
        // Check if we both arrays are the same memory address
        if ( d_elements.get() == rhs2->d_elements.get() )
            return true;
        // If we are dealing with different arrays, check that the are the same size and values
        if ( d_elements->size() != rhs2->d_elements->size() )
            return false;
        bool elements_match = true;
        for ( size_t i = 0; i < d_elements->size(); i++ ) {
            if ( d_elements->operator[]( i ) != rhs2->d_elements->operator[]( i ) )
                elements_match = false;
        }
        return elements_match;
    }
    /* We are comparing a MultiVectorIterator to an arbitrary iterator
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
    MeshIterator iterator = rhs.begin();
    bool elements_match   = true;
    for ( size_t i = 0; i < d_elements->size(); i++ ) {
        if ( iterator->globalID() != ( d_elements->operator[]( i ) ).globalID() )
            elements_match = false;
        ++iterator;
    }
    return elements_match;
}
bool MultiVectorIterator::operator!=( const MeshIterator &rhs ) const
{
    return !( ( *this ) == rhs );
}


} // namespace AMP::Mesh
