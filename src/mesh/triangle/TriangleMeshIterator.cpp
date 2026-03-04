#include "AMP/mesh/triangle/TriangleMeshIterator.h"
#include "AMP/mesh/triangle/TriangleMesh.h"
#include "AMP/mesh/triangle/TriangleMeshElement.h"


namespace AMP::Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
template<uint8_t NG>
TriangleMeshIterator<NG>::TriangleMeshIterator()
{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    d_typeHash = MeshIteratorType;
    d_size     = 0;
    d_pos      = static_cast<size_t>( -1 );
    d_element  = &d_cur_element;
    d_mesh     = nullptr;
}
template<uint8_t NG>
TriangleMeshIterator<NG>::TriangleMeshIterator( const AMP::Mesh::TriangleMesh<NG> *mesh,
                                                std::shared_ptr<const std::vector<ElementID>> list,
                                                size_t pos )
{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    d_typeHash = MeshIteratorType;
    d_size     = 0;
    d_pos      = pos;
    d_element  = &d_cur_element;
    d_mesh     = mesh;
    d_list     = list;
    if ( list ) {
        d_size = list->size();
        d_cur_element =
            TriangleMeshElement<NG>( MeshElementID( mesh->meshID(), ElementID() ), mesh );
        if ( d_pos < d_size )
            d_cur_element.resetElemId( d_list->operator[]( d_pos ) );
    }
}
template<uint8_t NG>
TriangleMeshIterator<NG>::TriangleMeshIterator( const TriangleMeshIterator &rhs )
    : MeshIteratorBase(),
      d_mesh{ rhs.d_mesh },
      d_list{ rhs.d_list },
      d_cur_element{ rhs.d_cur_element }

{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    d_typeHash = rhs.d_typeHash;
    d_size     = rhs.d_size;
    d_pos      = rhs.d_pos;
    d_element  = &d_cur_element;
}
template<uint8_t NG>
TriangleMeshIterator<NG> &TriangleMeshIterator<NG>::operator=( const TriangleMeshIterator &rhs )
{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    if ( this == &rhs )
        return *this;
    d_typeHash    = rhs.d_typeHash;
    d_size        = rhs.d_size;
    d_pos         = rhs.d_pos;
    d_mesh        = rhs.d_mesh;
    d_list        = rhs.d_list;
    d_element     = &d_cur_element;
    d_cur_element = rhs.d_cur_element;
    return *this;
}


/********************************************************
 * Function to clone the iterator                        *
 ********************************************************/
template<uint8_t NG>
std::unique_ptr<MeshIteratorBase> TriangleMeshIterator<NG>::clone() const
{
    return std::make_unique<TriangleMeshIterator>( *this );
}


/********************************************************
 * class name                                            *
 ********************************************************/
template<>
std::string TriangleMeshIterator<1>::className() const
{
    return "TriangleMeshIterator<1>";
}
template<>
std::string TriangleMeshIterator<2>::className() const
{
    return "TriangleMeshIterator<2>";
}
template<>
std::string TriangleMeshIterator<3>::className() const
{
    return "TriangleMeshIterator<3>";
}


/********************************************************
 * Set position                                          *
 ********************************************************/
template<uint8_t NG>
void TriangleMeshIterator<NG>::setPos( size_t pos )
{
    d_pos = pos;
    if ( d_pos < d_size )
        d_cur_element.resetElemId( d_list->operator[]( d_pos ) );
}


/********************************************************
 * Return an iterator to the beginning or end            *
 ********************************************************/
template<uint8_t NG>
MeshIterator TriangleMeshIterator<NG>::begin() const
{
    return MeshIterator::create<TriangleMeshIterator>( d_mesh, d_list, 0 );
}


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
template<uint8_t NG>
MeshIteratorBase &TriangleMeshIterator<NG>::operator++()
{
    // Prefix increment (increment and return this)
    d_pos++;
    if ( d_pos < d_size )
        d_cur_element.resetElemId( d_list->operator[]( d_pos ) );
    return *this;
}
template<uint8_t NG>
MeshIteratorBase &TriangleMeshIterator<NG>::operator--()
{
    // Prefix decrement (increment and return this)
    AMP_INSIST( d_pos > 0, "Decrementing iterator past 0" );
    d_pos--;
    d_cur_element.resetElemId( d_list->operator[]( d_pos ) );
    return *this;
}


/********************************************************
 * Random access iterators                               *
 ********************************************************/
template<uint8_t NG>
MeshIteratorBase &TriangleMeshIterator<NG>::operator+=( int n )
{
    // Check the input
    if ( n >= 0 ) {
        AMP_INSIST( d_pos + n <= d_size, "Iterated past end of iterator" );
    } else { // decrement *this
        AMP_INSIST( -n <= (int64_t) d_pos, "Iterated past beginning of iterator" );
    }
    // Perform the increment and return
    d_pos += n;
    if ( d_pos < d_size )
        d_cur_element.resetElemId( d_list->operator[]( d_pos ) );
    return *this;
}


/********************************************************
 * Compare two iterators                                 *
 ********************************************************/
template<uint8_t NG>
bool TriangleMeshIterator<NG>::operator==( const MeshIteratorBase &rhs ) const
{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    const TriangleMeshIterator *rhs2 = nullptr;
    // Convert rhs to a TriangleMeshIterator* so we can access the base class members
    auto *tmp = reinterpret_cast<const TriangleMeshIterator *>( &rhs );
    if ( tmp->d_typeHash == MeshIteratorType ) {
        rhs2 = tmp; // We can safely cast rhs to a TriangleMeshIterator
    }
    // Perform direct comparisons if we are dealing with two TriangleMeshIterators
    if ( rhs2 != nullptr )
        return *d_list == *rhs2->d_list;
    /* We are comparing a TriangleMeshIterator to an arbitrary iterator
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
template<uint8_t NG>
bool TriangleMeshIterator<NG>::operator!=( const MeshIteratorBase &rhs ) const
{
    return !( *this == rhs );
}


/********************************************************
 *  Write/read restart data (MeshIteratorBase)           *
 ********************************************************/
template<uint8_t NG>
void TriangleMeshIterator<NG>::registerChildObjects( AMP::IO::RestartManager * ) const
{
    AMP_ERROR( "Not finished" );
}
template<uint8_t NG>
void TriangleMeshIterator<NG>::writeRestart( int64_t ) const
{
    AMP_ERROR( "Not finished" );
}
template<uint8_t NG>
TriangleMeshIterator<NG>::TriangleMeshIterator( int64_t fid, AMP::IO::RestartManager *manager )
    : MeshIteratorBase( fid, manager ), d_mesh( nullptr )
{
    AMP_ERROR( "Not finished" );
}


/********************************************************
 *  Explicit instantiations of TriangleMeshIterator      *
 ********************************************************/
template class TriangleMeshIterator<1>;
template class TriangleMeshIterator<2>;
template class TriangleMeshIterator<3>;


} // namespace AMP::Mesh
