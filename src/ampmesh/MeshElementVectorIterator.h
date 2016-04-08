#ifndef included_AMP_MultiVectorIterator
#define included_AMP_MultiVectorIterator

#include "ampmesh/MeshIterator.h"
#include "utils/shared_ptr.h"
#include <iterator>

namespace AMP {
namespace Mesh {


/**
 * \class MultiVectorIterator
 * \brief A class used to iterate over a set of mesh elements.
 * \details  This class provides routines for iterating over a set
 * of mesh elments that are in a std::vector.
 */
class MultiVectorIterator : public MeshIterator
{
public:
    //! Empty MultiVectorIterator constructor
    MultiVectorIterator();

    //! Default MultiVectorIterator constructor
    MultiVectorIterator( AMP::shared_ptr<std::vector<MeshElement>> elements, size_t pos = 0 );

    /** MultiVectorIterator constructor
     *  Note that this version of the constructor will create a copy of the elements
     */
    MultiVectorIterator( const std::vector<MeshElement> &elements, size_t pos = 0 );

    //! Deconstructor
    virtual ~MultiVectorIterator();

    //! Copy constructor
    MultiVectorIterator( const MultiVectorIterator & );

    //! Assignment operator
    MultiVectorIterator &operator=( const MultiVectorIterator & );

    //! Increment
    MeshIterator &operator++();

    //! Increment
    MeshIterator operator++( int );

    //! Decrement
    MeshIterator &operator--();

    //! Decrement
    MeshIterator operator--( int );

    // Arithmetic operator+
    virtual MeshIterator operator+( int ) const;

    // Arithmetic operator+=
    virtual MeshIterator &operator+=( int N );

    //! Check if two iterators are equal
    bool operator==( const MeshIterator &rhs ) const;

    //! Check if two iterators are not equal
    bool operator!=( const MeshIterator &rhs ) const;

    //! Dereference the iterator
    MeshElement &operator*( void ) override;

    //! Dereference the iterator
    MeshElement *operator->( void ) override;

    //! Dereference the iterator
    const MeshElement &operator*( void ) const override;

    //! Dereference the iterator
    const MeshElement *operator->( void ) const override;

    //! Return an iterator to the begining
    MeshIterator begin() const;

    //! Return an iterator to the begining
    MeshIterator end() const;

    //! Return the number of elements in the iterator
    virtual size_t size() const;

    //! Return the current position (from the beginning) in the iterator
    virtual size_t position() const;

    using MeshIterator::operator+;
    using MeshIterator::operator+=;

protected:
    //! Clone the iterator
    virtual MeshIterator *clone() const;

    // A pointer to a std::vector containing the desired mesh elements
    AMP::shared_ptr<std::vector<MeshElement>> d_elements;
    // An integer containing the current position
    size_t d_pos;
};
}
}

#endif
