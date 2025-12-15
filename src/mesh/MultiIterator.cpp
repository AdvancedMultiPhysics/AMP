#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/utils/typeid.h"


namespace AMP::Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
static constexpr auto MeshIteratorType = AMP::getTypeID<MultiIterator>().hash;
static_assert( MeshIteratorType != 0 );
MultiIterator::MultiIterator() : d_localPos( 0 ), d_iteratorNum( 0 )
{
    d_typeHash = MeshIteratorType;
}
MultiIterator::MultiIterator( const std::vector<MeshIterator> &iterators, size_t global_pos )
{
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_iterators.resize( 0 );
    d_iteratorType = MeshIterator::Type::RandomAccess;
    for ( auto &iterator : iterators ) {
        d_iteratorType = std::min( d_iteratorType, iterator.type() );
        if ( iterator.size() > 0 )
            d_iterators.push_back( iterator );
    }
    d_iteratorSize = std::vector<size_t>( d_iterators.size(), 0 );
    d_size         = 0;
    for ( size_t i = 0; i < d_iterators.size(); i++ ) {
        d_iteratorSize[i] = d_iterators[i].size();
        d_size += d_iteratorSize[i];
    }
    d_pos        = global_pos;
    cur_iterator = MeshIterator();
    // Set the local position and the local iterator
    if ( d_pos > d_size ) {
        // The position is more than one past the last element
        AMP_ERROR( "Cannot create a MultiIterator with a current index that is more than 1 past "
                   "the last point" );
    } else if ( d_pos == d_size ) {
        // The position is one past the last element
        d_localPos    = 0;
        d_iteratorNum = d_iterators.size();
    } else {
        // We are inside the iterator, we need to point to the current element
        d_iteratorNum = 0;
        d_localPos    = global_pos;
        while ( d_localPos >= d_iteratorSize[d_iteratorNum] ) {
            d_iteratorNum++;
            d_localPos -= d_iteratorSize[d_iteratorNum];
        }
        cur_iterator = d_iterators[d_iteratorNum].begin();
        for ( size_t i = 0; i < d_localPos; i++ )
            ++cur_iterator;
    }
    d_element = cur_iterator.operator->();
}
MultiIterator::MultiIterator( const MultiIterator &rhs )
    : MeshIterator(), // Note: we never want to call the base copy constructor
      d_localPos( rhs.d_localPos ),
      d_iteratorNum( rhs.d_iteratorNum ),
      d_iteratorSize( rhs.d_iteratorSize ),
      d_iterators( rhs.d_iterators ),
      cur_iterator( rhs.cur_iterator )
{
    d_iterator     = nullptr;
    d_typeHash     = MeshIteratorType;
    d_iteratorType = rhs.d_iteratorType;
    d_size         = rhs.d_size;
    d_pos          = rhs.d_pos;
    d_element      = cur_iterator.operator->();
}
MultiIterator &MultiIterator::operator=( const MultiIterator &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    this->d_typeHash     = MeshIteratorType;
    this->d_iteratorType = rhs.d_iteratorType;
    this->d_iterator     = nullptr;
    this->d_iterators    = rhs.d_iterators;
    this->d_iteratorSize = rhs.d_iteratorSize;
    this->d_size         = rhs.d_size;
    this->d_localPos     = rhs.d_localPos;
    this->d_pos          = rhs.d_pos;
    this->d_iteratorNum  = rhs.d_iteratorNum;
    this->cur_iterator   = rhs.cur_iterator;
    d_element            = cur_iterator.operator->();
    return *this;
}


/********************************************************
 * Function to clone the iterator                        *
 ********************************************************/
MeshIterator *MultiIterator::clone() const { return new MultiIterator( *this ); }


/********************************************************
 * De-constructor                                        *
 ********************************************************/
MultiIterator::~MultiIterator() = default;


/********************************************************
 * Return an iterator to the beginning or end            *
 ********************************************************/
MeshIterator MultiIterator::begin() const { return MultiIterator( d_iterators, 0 ); }
MeshIterator MultiIterator::end() const { return MultiIterator( d_iterators, d_size ); }


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
MeshIterator &MultiIterator::operator++()
{
    // Prefix increment (increment and return this)
    if ( d_pos == d_size )
        AMP_ERROR( "Iterating more than one past the last element" );
    if ( d_pos + 1 == d_size ) {
        // We have moved to one past the last element
        d_pos         = d_size;
        d_localPos    = 0;
        d_iteratorNum = d_iterators.size();
        cur_iterator  = MeshIterator();
    } else if ( d_localPos + 1 == d_iteratorSize[d_iteratorNum] ) {
        // We need to change the internal iterator
        d_pos++;
        d_localPos = 0;
        d_iteratorNum++;
        cur_iterator = d_iterators[d_iteratorNum].begin();
    } else {
        // We are within the same iterator
        d_localPos++;
        d_pos++;
        ++cur_iterator; // preincrement for consistency and speed
    }
    d_element = cur_iterator.operator->();
    return *this;
}
MeshIterator &MultiIterator::operator--()
{
    // Prefix decrement (increment and return this)
    if ( d_pos == 0 )
        AMP_ERROR( "Iterating before the first element" );
    if ( d_pos == d_size ) {
        // We are starting at the end
        d_pos         = d_size + 1;
        d_iteratorNum = d_iterators.size() - 1;
        d_localPos    = d_iteratorSize[d_iteratorNum] - 1;
        cur_iterator  = d_iterators[d_iteratorNum].end();
        --cur_iterator;
    } else if ( d_localPos == 0 ) {
        // We need to change the internal iterator
        d_pos--;
        d_iteratorNum--;
        d_localPos   = d_iteratorSize[d_iteratorNum] - 1;
        cur_iterator = d_iterators[d_iteratorNum].end();
        --cur_iterator;
    } else {
        // We are within the same iterator
        d_localPos--;
        d_pos--;
        --cur_iterator; // predecrement for consistency and speed
    }
    d_element = cur_iterator.operator->();
    return *this;
}


/********************************************************
 * Random access iterators                               *
 ********************************************************/
MeshIterator &MultiIterator::operator+=( int n )
{
    if ( n >= 0 ) { // increment *this
        auto n2 = static_cast<size_t>( n );
        if ( d_pos + n2 > d_size )
            AMP_ERROR( "Iterated past end of iterator" );
        if ( d_pos + n2 == d_size ) {
            // We reached the end of the iterator
            d_pos         = d_size;
            d_localPos    = 0;
            d_iteratorNum = d_iterators.size();
            cur_iterator  = MeshIterator();
            return *this;
        }
        // Move to the correct iterator
        if ( d_localPos + n2 >= d_iteratorSize[d_iteratorNum] ) {
            size_t i = d_iteratorSize[d_iteratorNum] - d_localPos;
            n2 -= i;
            d_iteratorNum++;
            while ( d_iteratorSize[d_iteratorNum] < n2 ) {
                n2 -= d_iteratorSize[d_iteratorNum];
                d_iteratorNum++;
            }
            cur_iterator = d_iterators[d_iteratorNum].begin();
            d_localPos   = 0;
        }
        // Increment local iterator
        cur_iterator.operator+=( n2 );
        d_localPos += n2;
        d_pos += n;
    } else { // decrement *this
        auto n2 = static_cast<size_t>( -n );
        if ( d_pos < n2 )
            AMP_ERROR( "Iterated past beginning of iterator" );
        if ( d_pos == n2 ) {
            // We reached the beginning of the iterator
            d_pos         = 0;
            d_iteratorNum = 0;
            d_localPos    = 0;
            cur_iterator  = d_iterators[0].begin();
            return *this;
        }
        // Move to the correct iterator
        if ( n2 > d_localPos ) {
            // size_t i = d_iteratorSize[d_iteratorNum] - d_localPos;
            n2 -= d_localPos;
            d_iteratorNum--;
            while ( d_iteratorSize[d_iteratorNum] < n2 ) {
                n2 -= d_iteratorSize[d_iteratorNum];
                d_iteratorNum--;
            }
            cur_iterator = d_iterators[d_iteratorNum].end();
            d_localPos   = d_iteratorSize[d_iteratorNum];
        }
        // Increment local iterator
        cur_iterator.operator-=( n2 );
        d_localPos -= n2;
        d_pos += n;
    }
    d_element = cur_iterator.operator->();
    return *this;
}


/********************************************************
 * Compare two iterators                                 *
 * Two MultiIterators are the same if both the list of   *
 * iterators and the current position are the same.      *
 ********************************************************/
bool MultiIterator::operator==( const MeshIterator &rhs ) const
{
    const MultiIterator *rhs2 = nullptr;
    // Convert rhs to a MultiIterator* so we can access the base class members
    const auto *tmp = reinterpret_cast<const MultiIterator *>( &rhs );
    if ( tmp->d_typeHash == MeshIteratorType ) {
        rhs2 = tmp; // We can safely cast rhs to a MultiIterator
    } else if ( tmp->d_iterator != nullptr ) {
        tmp = reinterpret_cast<const MultiIterator *>( tmp->d_iterator );
        if ( tmp->d_typeHash == MeshIteratorType )
            rhs2 = tmp; // We can safely cast rhs.iterator to a MultiIterator
    }
    // Perform direct comparisions if we are dealing with two MultiIterator
    if ( rhs2 != nullptr ) {
        bool equal = true;
        equal      = equal && d_size == rhs2->d_size;
        equal      = equal && d_pos == rhs2->d_pos;
        equal      = equal && d_iterators.size() == rhs2->d_iterators.size();
        if ( equal ) {
            for ( size_t i = 0; i < d_iterators.size(); i++ )
                equal = equal && ( *d_iterators[i] ) == ( *rhs2->d_iterators[i] );
        }
        return equal;
    }
    /* We are comparing a MultiIterator to an arbitrary iterator
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
    MeshIterator it1    = this->begin();
    MeshIterator it2    = rhs.begin();
    bool elements_match = true;
    for ( size_t i = 0; i < it1.size(); ++i, ++it1, ++it2 ) {
        if ( it1->globalID() != it2->globalID() )
            elements_match = false;
    }
    return elements_match;
}
bool MultiIterator::operator!=( const MeshIterator &rhs ) const { return !operator==( rhs ); }


} // namespace AMP::Mesh
