#include "AMP/mesh/MeshElementVector.h"
#include "AMP/mesh/MeshElement.h"


namespace AMP::Mesh {


/****************************************************************
 * MeshElementVectorPtr                                             *
 ****************************************************************/
MeshElementVectorPtr::MeshElementVectorPtr()
    : ptr( std::make_unique<MeshElementVector<MeshElement>>( 0 ) )
{
}
MeshElementVectorIterator MeshElementVectorPtr::begin() const
{
    AMP_DEBUG_ASSERT( ptr );
    return MeshElementVectorIterator( ptr.get(), 0 );
}
MeshElementVectorIterator MeshElementVectorPtr::end() const
{
    AMP_DEBUG_ASSERT( ptr );
    return MeshElementVectorIterator( ptr.get(), ptr->size() );
}


/****************************************************************
 * MeshElementVectorIterator                                     *
 ****************************************************************/
MeshElementVectorIterator::MeshElementVectorIterator( const MeshElementVectorBase *ptr, size_t pos )
    : d_pos( pos ), d_size( ptr->size() ), d_data( ptr )
{
}
MeshElementVectorIterator &MeshElementVectorIterator::operator++()
{
    // Prefix increment (increment and return this)
    d_pos++;
    if ( d_pos >= d_size )
        d_pos = d_size;
    return *this;
}
MeshElementVectorIterator &MeshElementVectorIterator::operator--()
{
    // Prefix decrement (increment and return this)
    if ( d_pos != 0 )
        d_pos--;
    return *this;
}
MeshElementVectorIterator &MeshElementVectorIterator::operator+=( int n )
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
    return *this;
}
bool MeshElementVectorIterator::operator==( const MeshElementVectorIterator &rhs ) const
{
    return d_data == rhs.d_data && d_pos == rhs.d_pos;
}
bool MeshElementVectorIterator::operator!=( const MeshElementVectorIterator &rhs ) const
{
    return d_data != rhs.d_data || d_pos != rhs.d_pos;
}
MeshElementVectorIterator MeshElementVectorIterator::begin() const
{
    return MeshElementVectorIterator( d_data, 0 );
}
MeshElementVectorIterator MeshElementVectorIterator::end() const
{
    return MeshElementVectorIterator( d_data, d_size );
}
const MeshElement &MeshElementVectorIterator::operator*() const
{
    return d_data->operator[]( d_pos );
}


} // namespace AMP::Mesh
