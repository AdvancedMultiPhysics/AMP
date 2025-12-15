#ifndef included_AMP_MeshIterators_inline
#define included_AMP_MeshIterators_inline

#include "AMP/mesh/MeshIterator.h"
#include <iterator>
#include <memory>

namespace AMP::Mesh {


/********************************************************
 *  Comparison operators                                 *
 ********************************************************/
inline bool MeshIterator::operator<( const MeshIterator &rhs ) const
{
    return this->position() < rhs.position();
}
inline bool MeshIterator::operator<=( const MeshIterator &rhs ) const
{
    return this->position() <= rhs.position();
}
inline bool MeshIterator::operator>( const MeshIterator &rhs ) const
{
    return this->position() > rhs.position();
}
inline bool MeshIterator::operator>=( const MeshIterator &rhs ) const
{
    return this->position() >= rhs.position();
}


/********************************************************
 * Function to get the size and position of the iterator *
 ********************************************************/
inline const MeshIterator *MeshIterator::rawIterator() const
{
    return d_iterator == nullptr ? this : d_iterator;
}
inline bool MeshIterator::empty() const
{
    return d_iterator == nullptr ? true : d_iterator->d_size == 0;
}
inline size_t MeshIterator::size() const
{
    return d_iterator == nullptr ? d_size : d_iterator->d_size;
}
inline size_t MeshIterator::position() const
{
    return d_iterator == nullptr ? d_pos : d_iterator->d_pos;
}


/********************************************************
 * Functions for dereferencing the d_iterator            *
 ********************************************************/
inline MeshElement &MeshIterator::operator*()
{
    return d_iterator == nullptr ? *d_element : *d_iterator->d_element;
}
inline MeshElement *MeshIterator::operator->()
{
    return d_iterator == nullptr ? d_element : d_iterator->d_element;
}
inline const MeshElement &MeshIterator::operator*() const
{
    return d_iterator == nullptr ? *d_element : *d_iterator->d_element;
}
inline const MeshElement *MeshIterator::operator->() const
{
    return d_iterator == nullptr ? d_element : d_iterator->d_element;
}
inline MeshElement *MeshIterator::get()
{
    return d_iterator == nullptr ? d_element : d_iterator->d_element;
}
inline const MeshElement *MeshIterator::get() const
{
    return d_iterator == nullptr ? d_element : d_iterator->d_element;
}


} // namespace AMP::Mesh

#endif
