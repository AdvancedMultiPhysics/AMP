#include "AMP/mesh/triangle/TriangleMeshIterator.h"
#include "AMP/mesh/triangle/TriangleMesh.h"
#include "AMP/mesh/triangle/TriangleMeshElement.h"


namespace AMP::Mesh {


// unused global variable to prevent compiler warning
static MeshElement nullElement;


/********************************************************
 * Constructors                                          *
 ********************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshIterator<NG, NP, TYPE>::TriangleMeshIterator()
{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_size     = 0;
    d_pos      = -1;
    d_element  = &d_cur_element;
    d_mesh     = nullptr;
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshIterator<NG, NP, TYPE>::TriangleMeshIterator(
    const AMP::Mesh::TriangleMesh<NG, NP> *mesh,
    std::shared_ptr<const std::vector<ElementID>> list,
    size_t pos )
{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    d_typeHash = MeshIteratorType;
    d_iterator = nullptr;
    d_size     = 0;
    d_pos      = pos;
    d_element  = &d_cur_element;
    d_mesh     = mesh;
    d_list     = list;
    if ( list )
        d_size = list->size();
    d_cur_element =
        TriangleMeshElement<NG, NP, TYPE>( MeshElementID( mesh->meshID(), ElementID() ), mesh );
    if ( d_pos < d_size )
        d_cur_element.resetElemId( d_list->operator[]( d_pos ) );
    bool test = true;
    for ( const auto &id : *list )
        test = test && static_cast<uint8_t>( id.type() ) == TYPE;
    AMP_ASSERT( test );
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshIterator<NG, NP, TYPE>::TriangleMeshIterator( const TriangleMeshIterator &rhs )
    : MeshIterator(), d_mesh{ rhs.d_mesh }, d_list{ rhs.d_list }, d_cur_element{ rhs.d_cur_element }

{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    d_typeHash = rhs.d_typeHash;
    d_iterator = nullptr;
    d_size     = rhs.d_size;
    d_pos      = rhs.d_pos;
    d_element  = &d_cur_element;
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
TriangleMeshIterator<NG, NP, TYPE> &
TriangleMeshIterator<NG, NP, TYPE>::operator=( const TriangleMeshIterator &rhs )
{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    if ( this == &rhs )
        return *this;
    d_typeHash    = rhs.d_typeHash;
    d_iterator    = nullptr;
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
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshIterator *TriangleMeshIterator<NG, NP, TYPE>::clone() const
{
    return new TriangleMeshIterator( *this );
}


/********************************************************
 * Return an iterator to the beginning or end            *
 ********************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshIterator TriangleMeshIterator<NG, NP, TYPE>::begin() const
{
    return TriangleMeshIterator( d_mesh, d_list, 0 );
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshIterator TriangleMeshIterator<NG, NP, TYPE>::end() const
{
    return TriangleMeshIterator( d_mesh, d_list, d_size );
}


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshIterator &TriangleMeshIterator<NG, NP, TYPE>::operator++()
{
    // Prefix increment (increment and return this)
    d_pos++;
    if ( d_pos < d_size )
        d_cur_element.resetElemId( d_list->operator[]( d_pos ) );
    return *this;
}
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshIterator &TriangleMeshIterator<NG, NP, TYPE>::operator--()
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
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
MeshIterator &TriangleMeshIterator<NG, NP, TYPE>::operator+=( int n )
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
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
bool TriangleMeshIterator<NG, NP, TYPE>::operator==( const MeshIterator &rhs ) const
{
    static constexpr auto MeshIteratorType = AMP::getTypeID<decltype( *this )>().hash;
    static_assert( MeshIteratorType != 0 );
    const TriangleMeshIterator *rhs2 = nullptr;
    // Convert rhs to a TriangleMeshIterator* so we can access the base class members
    auto *tmp = reinterpret_cast<const TriangleMeshIterator *>( &rhs );
    if ( tmp->d_typeHash == MeshIteratorType ) {
        rhs2 = tmp; // We can safely cast rhs to a TriangleMeshIterator
    } else if ( tmp->d_iterator != nullptr ) {
        tmp = reinterpret_cast<const TriangleMeshIterator *>( tmp->d_iterator );
        if ( tmp->d_typeHash == MeshIteratorType )
            rhs2 = tmp; // We can safely cast rhs.iterator to a TriangleMeshIterator
    }
    // Perform direct comparisions if we are dealing with two TriangleMeshIterators
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
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
bool TriangleMeshIterator<NG, NP, TYPE>::operator!=( const MeshIterator &rhs ) const
{
    return !( ( *this ) == rhs );
}


/********************************************************
 *  Explicit instantiations of TriangleMeshIterator      *
 ********************************************************/
template class TriangleMeshIterator<1, 1, 0>;
template class TriangleMeshIterator<1, 1, 1>;
template class TriangleMeshIterator<1, 1, 2>;
template class TriangleMeshIterator<1, 1, 3>;
template class TriangleMeshIterator<1, 2, 0>;
template class TriangleMeshIterator<1, 2, 1>;
template class TriangleMeshIterator<1, 2, 2>;
template class TriangleMeshIterator<1, 2, 3>;
template class TriangleMeshIterator<1, 3, 0>;
template class TriangleMeshIterator<1, 3, 1>;
template class TriangleMeshIterator<1, 3, 2>;
template class TriangleMeshIterator<1, 3, 3>;
template class TriangleMeshIterator<2, 2, 0>;
template class TriangleMeshIterator<2, 2, 1>;
template class TriangleMeshIterator<2, 2, 2>;
template class TriangleMeshIterator<2, 2, 3>;
template class TriangleMeshIterator<2, 3, 0>;
template class TriangleMeshIterator<2, 3, 1>;
template class TriangleMeshIterator<2, 3, 2>;
template class TriangleMeshIterator<2, 3, 3>;
template class TriangleMeshIterator<3, 3, 0>;
template class TriangleMeshIterator<3, 3, 1>;
template class TriangleMeshIterator<3, 3, 2>;
template class TriangleMeshIterator<3, 3, 3>;


} // namespace AMP::Mesh
