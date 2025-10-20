#include "AMP/mesh/structured/BoxMesh.h"


namespace AMP::Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
BoxMesh::MeshElementIndexIterator::MeshElementIndexIterator( const MeshElementIndex &first,
                                                             const MeshElementIndex &last,
                                                             const BoxMesh *mesh,
                                                             size_t pos )
    : d_checkBoundary( false ),
      d_isPeriodic(
          { mesh->d_surfaceId[1] == -1, mesh->d_surfaceId[3] == -1, mesh->d_surfaceId[5] == -1 } ),
      d_globalSize( mesh->d_globalSize ),
      d_first( first ),
      d_last( last )
{
    AMP_ASSERT( first.side() == last.side() && first.type() == last.type() );
    d_pos           = pos;
    d_size          = BoxMesh::MeshElementIndex::numElements( d_first, d_last );
    d_checkBoundary = d_first.index( 0 ) < 0 || d_last.index( 0 ) >= d_globalSize[0] ||
                      d_first.index( 1 ) < 0 || d_last.index( 1 ) >= d_globalSize[1] ||
                      d_first.index( 2 ) < 0 || d_last.index( 2 ) >= d_globalSize[2];
}


/********************************************************
 * Return an iterator to the beginning or end            *
 ********************************************************/
BoxMesh::MeshElementIndexIterator BoxMesh::MeshElementIndexIterator::begin() const
{
    auto it2  = *this;
    it2.d_pos = 0;
    return it2;
}
BoxMesh::MeshElementIndexIterator BoxMesh::MeshElementIndexIterator::end() const
{
    auto it2  = *this;
    it2.d_pos = d_size;
    return it2;
}


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
BoxMesh::MeshElementIndexIterator &BoxMesh::MeshElementIndexIterator::operator++()
{
    d_pos++;
    if ( d_pos >= d_size )
        d_pos = d_size;
    return *this;
}
BoxMesh::MeshElementIndexIterator &BoxMesh::MeshElementIndexIterator::operator--()
{
    if ( d_pos != 0 )
        d_pos--;
    return *this;
}


/********************************************************
 * Random access iterators                               *
 ********************************************************/
BoxMesh::MeshElementIndexIterator &BoxMesh::MeshElementIndexIterator::operator+=( int n )
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


/********************************************************
 * Compare two iterators                                 *
 ********************************************************/
bool BoxMesh::MeshElementIndexIterator::operator==(
    const BoxMesh::MeshElementIndexIterator &rhs ) const
{
    return rhs.d_checkBoundary == d_checkBoundary && rhs.d_isPeriodic == d_isPeriodic &&
           rhs.d_globalSize == d_globalSize && rhs.d_first == d_first && rhs.d_last == d_last &&
           rhs.d_pos == d_pos && rhs.d_size == d_size;
}
bool BoxMesh::MeshElementIndexIterator::operator!=(
    const BoxMesh::MeshElementIndexIterator &rhs ) const
{
    return !( ( *this ) == rhs );
}


/********************************************************
 * Dereference the iterator                              *
 ********************************************************/
BoxMesh::MeshElementIndex BoxMesh::MeshElementIndexIterator::operator*() const
{
    if ( d_pos >= d_size )
        return {};
    int i  = d_pos;
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
    return MeshElementIndex( d_first.type(), d_first.side(), i, j, k );
}


} // namespace AMP::Mesh
