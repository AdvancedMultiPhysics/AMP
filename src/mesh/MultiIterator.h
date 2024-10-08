#ifndef included_AMP_MultiIterator
#define included_AMP_MultiIterator

#include "AMP/mesh/MeshIterator.h"

#include <memory>


namespace AMP::Mesh {


/**
 * \class MultiIterator
 * \brief A class used to combine multiple iterators
 * \details  This class provides routines for iterating over a mesh.  More
 *  specifically, this class combines multiple iterators into one.  This
 *  is primarily needed for MultiMesh, but may be used for other applicaitons.
 */
class MultiIterator final : public MeshIterator
{
public:
    //! Empty MultiIterator constructor
    MultiIterator();

    //! Default MultiIterator constructor
    explicit MultiIterator( const std::vector<MeshIterator> &iterators, size_t global_pos = 0 );

    //! Deconstructor
    virtual ~MultiIterator();

    //! Move constructor
    MultiIterator( MultiIterator && ) = default;

    //! Copy constructor
    MultiIterator( const MultiIterator & );

    //! Move operator
    MultiIterator &operator=( MultiIterator && ) = default;

    //! Assignment operator
    MultiIterator &operator=( const MultiIterator & );

    // Increment
    MeshIterator &operator++() override;

    // Decrement
    MeshIterator &operator--() override;

    // Arithmetic operator+=
    MeshIterator &operator+=( int N ) override;

    //! Check if two iterators are equal
    bool operator==( const MeshIterator &rhs ) const override;

    //! Check if two iterators are not equal
    bool operator!=( const MeshIterator &rhs ) const override;

    //! Return an iterator to the begining
    MeshIterator begin() const override;

    //! Return an iterator to the begining
    MeshIterator end() const override;

    using MeshIterator::operator+;
    using MeshIterator::operator+=;

protected:
    //! Clone the iterator
    MeshIterator *clone() const override;

private:
    // Data members
    size_t d_localPos, d_iteratorNum;
    std::vector<size_t> d_iteratorSize;
    std::vector<MeshIterator> d_iterators;
    MeshIterator cur_iterator;
};


} // namespace AMP::Mesh

#endif
