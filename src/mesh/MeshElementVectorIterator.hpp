#ifndef included_AMP_MeshElementVectorIterator_hpp
#define included_AMP_MeshElementVectorIterator_hpp

#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/utils/TypeTraits.h"
#include "AMP/utils/typeid.h"


namespace AMP::Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
template<class TYPE>
MeshElementVectorIterator<TYPE>::MeshElementVectorIterator()
{
    constexpr auto hash = AMP::getTypeID<MeshElementVectorIterator<TYPE>>().hash;
    static_assert( hash != 0 );
    d_typeHash = hash;
    d_iterator = nullptr;
    d_pos      = 0;
    d_size     = 0;
    d_element  = nullptr;
}
template<class TYPE>
MeshElementVectorIterator<TYPE>::MeshElementVectorIterator(
    std::shared_ptr<std::vector<TYPE>> elements, size_t pos )
    : d_elements( elements )
{
    constexpr auto hash = AMP::getTypeID<MeshElementVectorIterator<TYPE>>().hash;
    d_typeHash          = hash;
    d_iterator          = nullptr;
    d_pos               = pos;
    d_size              = d_elements->size();
    if constexpr ( is_unique_ptr_v<TYPE> )
        d_element = d_pos < d_size ? d_elements->operator[]( d_pos ).get() : nullptr;
    else
        d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
}
template<class TYPE>
MeshElementVectorIterator<TYPE>::MeshElementVectorIterator( const MeshElementVectorIterator &rhs )
    : MeshIterator(), // Note: we never want to call the base copy constructor
      d_elements( rhs.d_elements )
{
    constexpr auto hash = AMP::getTypeID<MeshElementVectorIterator<TYPE>>().hash;
    d_typeHash          = hash;
    d_iterator          = nullptr;
    d_pos               = rhs.d_pos;
    d_size              = rhs.d_size;
    if constexpr ( is_unique_ptr_v<TYPE> )
        d_element = d_pos < d_size ? d_elements->operator[]( d_pos ).get() : nullptr;
    else
        d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
}
template<class TYPE>
MeshElementVectorIterator<TYPE> &
MeshElementVectorIterator<TYPE>::operator=( const MeshElementVectorIterator &rhs )
{
    constexpr auto hash = AMP::getTypeID<MeshElementVectorIterator<TYPE>>().hash;
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    d_typeHash = hash;
    d_iterator = nullptr;
    d_elements = rhs.d_elements;
    d_pos      = rhs.d_pos;
    d_size     = rhs.d_size;
    if constexpr ( is_unique_ptr_v<TYPE> )
        d_element = d_pos < d_size ? d_elements->operator[]( d_pos ).get() : nullptr;
    else
        d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
    return *this;
}


/********************************************************
 * Function to clone the iterator                        *
 ********************************************************/
template<class TYPE>
MeshIterator *MeshElementVectorIterator<TYPE>::clone() const
{
    return new MeshElementVectorIterator( *this );
}


/********************************************************
 * Return an iterator to the beginning or end            *
 ********************************************************/
template<class TYPE>
MeshIterator MeshElementVectorIterator<TYPE>::begin() const
{
    return MeshElementVectorIterator( d_elements, 0 );
}
template<class TYPE>
MeshIterator MeshElementVectorIterator<TYPE>::end() const
{
    return MeshElementVectorIterator( d_elements, d_elements->size() );
}


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
template<class TYPE>
MeshIterator &MeshElementVectorIterator<TYPE>::operator++()
{
    // Prefix increment (increment and return this)
    d_pos++;
    if constexpr ( is_unique_ptr_v<TYPE> )
        d_element = d_pos < d_size ? d_elements->operator[]( d_pos ).get() : nullptr;
    else
        d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
    return *this;
}
template<class TYPE>
MeshIterator &MeshElementVectorIterator<TYPE>::operator--()
{
    // Prefix decrement (increment and return this)
    d_pos--;
    if constexpr ( is_unique_ptr_v<TYPE> )
        d_element = d_pos < d_size ? d_elements->operator[]( d_pos ).get() : nullptr;
    else
        d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
    return *this;
}


/********************************************************
 * Random access iterators                               *
 ********************************************************/
template<class TYPE>
MeshIterator &MeshElementVectorIterator<TYPE>::operator+=( int n )
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
    if constexpr ( is_unique_ptr_v<TYPE> )
        d_element = d_pos < d_size ? d_elements->operator[]( d_pos ).get() : nullptr;
    else
        d_element = d_pos < d_size ? &d_elements->operator[]( d_pos ) : nullptr;
    return *this;
}


/********************************************************
 * Compare two iterators                                 *
 ********************************************************/
template<class TYPE>
bool MeshElementVectorIterator<TYPE>::operator==( const MeshIterator &rhs ) const
{
    const MeshElementVectorIterator *rhs2 = nullptr;
    // Convert rhs to a MeshElementVectorIterator* so we can access the base class members
    constexpr auto hash = AMP::getTypeID<MeshElementVectorIterator<TYPE>>().hash;
    const auto *tmp     = reinterpret_cast<const MeshElementVectorIterator *>( &rhs );
    if ( tmp->d_typeHash == hash ) {
        rhs2 = tmp; // We can safely cast rhs.iterator to a MeshElementVectorIterator
    } else if ( tmp->d_iterator != nullptr ) {
        tmp = reinterpret_cast<const MeshElementVectorIterator *>( tmp->d_iterator );
        if ( tmp->d_typeHash == hash )
            rhs2 = tmp; // We can safely cast rhs.iterator to a MeshElementVectorIterator
    }
    // Perform direct comparisions if we are dealing with two MeshElementVectorIterators
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
        size_t N   = d_elements->size();
        auto p1    = d_elements->data();
        auto p2    = d_elements->data();
        bool match = true;
        if constexpr ( is_unique_ptr_v<TYPE> ) {
            for ( size_t i = 0; i < N; i++ )
                match = match && *p1[i] == *p2[i];
        } else {
            for ( size_t i = 0; i < N; i++ )
                match = match && p1[i] == p2[i];
        }
        return match;
    }
    /* We are comparing a MeshElementVectorIterator to an arbitrary iterator
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
    size_t N              = d_elements->size();
    auto p1               = d_elements->data();
    bool match            = true;
    if constexpr ( is_unique_ptr_v<TYPE> ) {
        for ( size_t i = 0; i < N; i++, ++iterator )
            match = match && *p1[i] == iterator->globalID();
    } else {
        for ( size_t i = 0; i < N; i++, ++iterator )
            match = match && p1[i] == iterator->globalID();
    }
    return match;
}
template<class TYPE>
bool MeshElementVectorIterator<TYPE>::operator!=( const MeshIterator &rhs ) const
{
    return !( *this == rhs );
}


} // namespace AMP::Mesh

#endif
