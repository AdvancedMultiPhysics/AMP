#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/utils/typeid.h"


namespace AMP::Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
static constexpr auto MeshIteratorType = AMP::getTypeID<MultiIterator>().hash;
static_assert( MeshIteratorType != 0 );
MultiIterator::MultiIterator() { d_typeHash = MeshIteratorType; }
MultiIterator::MultiIterator( std::vector<MeshIterator> iterators, size_t global_pos )
{
    d_typeHash = MeshIteratorType;
    d_iterators.reserve( iterators.size() );
    for ( auto &iterator : iterators ) {
        auto multi = dynamic_cast<MultiIterator *>(
            const_cast<MeshIteratorBase *>( iterator.rawIterator() ) );
        if ( multi ) {
            for ( auto &it : multi->d_iterators )
                d_iterators.push_back( it );
            multi->d_iterators.clear();
        } else {
            if ( !iterator.empty() )
                d_iterators.push_back( iterator.release() );
        }
    }
    d_size         = 0;
    d_iteratorType = MeshIterator::Type::RandomAccess;
    for ( auto &iterator : d_iterators ) {
        d_size += iterator->size();
        d_iteratorType = std::min( d_iteratorType, iterator->type() );
    }
    setPos( global_pos );
}
MultiIterator::MultiIterator( std::vector<MeshIteratorBase *> &&iterators, size_t global_pos )
    : d_iterators( std::move( iterators ) )
{
    d_typeHash     = MeshIteratorType;
    d_size         = 0;
    d_iteratorType = MeshIterator::Type::RandomAccess;
    for ( auto &iterator : d_iterators ) {
        size_t N = iterator->size();
        AMP_DEBUG_ASSERT( N != 0 );
        d_size += N;
        d_iteratorType = std::min( d_iteratorType, iterator->type() );
    }
    setPos( global_pos );
}
MultiIterator::MultiIterator( MultiIterator &&rhs )
    : MeshIteratorBase(), // Note: we never want to call the base copy constructor
      d_localPos( rhs.d_localPos ),
      d_iteratorNum( rhs.d_iteratorNum ),
      d_iterators( std::move( rhs.d_iterators ) )
{
    d_typeHash     = MeshIteratorType;
    d_iteratorType = rhs.d_iteratorType;
    d_size         = rhs.d_size;
    d_pos          = rhs.d_pos;
    d_iterators[d_iteratorNum]->setPos( rhs.d_pos );
    d_element = nullptr;
    if ( d_pos < d_size )
        d_element = d_iterators[d_iteratorNum]->operator->();
}
MultiIterator &MultiIterator::operator=( MultiIterator &&rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    d_typeHash     = MeshIteratorType;
    d_iteratorType = rhs.d_iteratorType;
    d_size         = rhs.d_size;
    d_localPos     = rhs.d_localPos;
    d_pos          = rhs.d_pos;
    d_iteratorNum  = rhs.d_iteratorNum;
    d_iterators    = std::move( rhs.d_iterators );
    d_element      = nullptr;
    if ( d_pos < d_size )
        d_element = d_iterators[d_iteratorNum]->operator->();
    rhs.d_iterators.clear();
    rhs.d_element = nullptr;
    return *this;
}


/********************************************************
 * Function to clone the iterator                        *
 ********************************************************/
std::unique_ptr<MeshIteratorBase> MultiIterator::clone() const
{
    if ( d_iterators.empty() )
        return std::make_unique<MultiIterator>();
    auto it            = std::make_unique<MultiIterator>();
    it->d_typeHash     = MeshIteratorType;
    it->d_iteratorType = d_iteratorType;
    it->d_iterators.resize( d_iterators.size() );
    for ( size_t i = 0; i < d_iterators.size(); i++ )
        it->d_iterators[i] = d_iterators[i]->clone().release();
    it->d_size        = d_size;
    it->d_localPos    = d_localPos;
    it->d_pos         = d_pos;
    it->d_iteratorNum = d_iteratorNum;
    if ( d_pos < d_size ) {
        it->d_iterators[d_iteratorNum]->setPos( d_pos );
        it->d_element = d_iterators[d_iteratorNum]->operator->();
    }
    return it;
}


/********************************************************
 * Destructor                                            *
 ********************************************************/
MultiIterator::~MultiIterator()
{
    for ( size_t i = 0; i < d_iterators.size(); i++ )
        delete d_iterators[i];
    d_iterators.clear();
}


/********************************************************
 * Set the position                                      *
 ********************************************************/
void MultiIterator::setPos( size_t pos )
{
    AMP_ASSERT( pos <= d_size );
    d_pos = pos;
    if ( d_pos == d_size ) {
        d_localPos    = 0;
        d_iteratorNum = d_iterators.size();
        d_element     = nullptr;
    } else {
        d_iteratorNum = 0;
        d_localPos    = pos;
        while ( d_localPos >= d_iterators[d_iteratorNum]->size() ) {
            d_iteratorNum++;
            d_localPos -= d_iterators[d_iteratorNum]->size();
        }
        d_iterators[d_iteratorNum]->setPos( d_localPos );
        d_element = d_iterators[d_iteratorNum]->operator->();
    }
    for ( size_t i = 0; i < d_iteratorNum; i++ )
        d_iterators[i]->setPos( d_iterators[i]->size() );
    for ( size_t i = d_iteratorNum + 1; i < d_iterators.size(); i++ )
        d_iterators[i]->setPos( 0 );
}


/********************************************************
 * Return an iterator to the beginning or end            *
 ********************************************************/
MeshIterator MultiIterator::begin() const
{
    auto it = clone();
    it->setPos( 0 );
    AMP_DEBUG_ASSERT( d_element != nullptr || d_size == 0 );
    return MeshIterator( std::move( it ) );
}


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
MeshIteratorBase &MultiIterator::operator++()
{
    // Prefix increment (increment and return this)
    if ( d_pos == d_size )
        AMP_ERROR( "Iterating past the end of the iterator" );
    d_pos++;
    d_localPos++;
    d_iterators[d_iteratorNum]->operator++();
    if ( d_pos == d_size ) {
        // We have moved to one past the last element
        d_localPos    = 0;
        d_iteratorNum = d_iterators.size();
        d_element     = nullptr;
    } else if ( d_localPos == d_iterators[d_iteratorNum]->size() ) {
        // We need to change the internal iterator
        d_localPos = 0;
        d_iteratorNum++;
        AMP_DEBUG_ASSERT( d_iterators[d_iteratorNum]->pos() == 0 );
        d_element = d_iterators[d_iteratorNum]->operator->();
        AMP_DEBUG_ASSERT( d_element );
    } else {
        // We are within the same iterator
        d_element = d_iterators[d_iteratorNum]->operator->();
        AMP_DEBUG_ASSERT( d_element );
    }
    return *this;
}
MeshIteratorBase &MultiIterator::operator--()
{
    // Prefix decrement (increment and return this)
    if ( d_pos == 0 )
        AMP_ERROR( "Iterating before the first element" );
    if ( d_pos == d_size ) {
        // We are starting at the end
        d_pos         = d_size - 1;
        d_iteratorNum = d_iterators.size() - 1;
        d_localPos    = d_iterators[d_iteratorNum]->size() - 1;
    } else if ( d_localPos == 0 ) {
        // We need to change the internal iterator
        d_pos--;
        d_iteratorNum--;
        d_localPos = d_iterators[d_iteratorNum]->size() - 1;
    } else {
        // We are within the same iterator
        d_localPos--;
        d_pos--;
    }
    d_iterators[d_iteratorNum]->operator--();
    d_element = d_iterators[d_iteratorNum]->operator->();
    return *this;
}


/********************************************************
 * Random access iterators                               *
 ********************************************************/
MeshIteratorBase &MultiIterator::operator+=( int n )
{
    auto pos = static_cast<int64_t>( d_pos ) + n;
    AMP_ASSERT( pos >= 0 );
    setPos( pos );
    return *this;
}


/********************************************************
 * Compare two iterators                                 *
 * Two MultiIterators are the same if both the list of   *
 * iterators and the current position are the same.      *
 ********************************************************/
bool MultiIterator::operator==( const MeshIteratorBase &rhs ) const
{
    const MultiIterator *rhs2 = nullptr;
    // Convert rhs to a MultiIterator* so we can access the base class members
    const auto *tmp = reinterpret_cast<const MultiIterator *>( &rhs );
    if ( tmp->d_typeHash == MeshIteratorType ) {
        rhs2 = tmp; // We can safely cast rhs to a MultiIterator
    }
    // Perform direct comparisons if we are dealing with two MultiIterator
    if ( rhs2 != nullptr ) {
        bool equal = true;
        equal      = equal && d_size == rhs2->d_size;
        equal      = equal && d_pos == rhs2->d_pos;
        equal      = equal && d_iterators.size() == rhs2->d_iterators.size();
        if ( equal ) {
            for ( size_t i = 0; i < d_iterators.size(); i++ )
                equal = equal && d_iterators[i]->operator==( *rhs2->d_iterators[i] );
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
    if ( this->pos() != rhs.pos() )
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
bool MultiIterator::operator!=( const MeshIteratorBase &rhs ) const { return !operator==( rhs ); }


/********************************************************
 *  Write/read restart data (MeshIteratorBase)           *
 ********************************************************/
void MultiIterator::registerChildObjects( AMP::IO::RestartManager * ) const
{
    AMP_ERROR( "Not finished" );
}
void MultiIterator::writeRestart( int64_t ) const { AMP_ERROR( "Not finished" ); }
MultiIterator::MultiIterator( int64_t, AMP::IO::RestartManager * ) { AMP_ERROR( "Not finished" ); }


} // namespace AMP::Mesh
