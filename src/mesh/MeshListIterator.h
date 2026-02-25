#ifndef included_AMP_MeshListIterator
#define included_AMP_MeshListIterator

#include "AMP/mesh/MeshIterator.h"
#include "AMP/utils/Utilities.h"

#include <iterator>
#include <memory>


namespace AMP::Mesh {


/**
 * \class MeshListIterator
 * \brief A class used to iterate over a set of mesh elements.
 * \details  This class provides routines for iterating over a set
 * of mesh elments that are in a std::vector.
 */
template<class TYPE = MeshElement>
class MeshListIterator final : public MeshIterator
{
public:
    //! Empty MeshListIterator constructor
    MeshListIterator();

    //! Default MeshListIterator constructor
    explicit MeshListIterator( std::shared_ptr<std::vector<TYPE>> elements, size_t pos = 0 );

    //! Deconstructor
    virtual ~MeshListIterator() = default;

    //! Copy constructor
    MeshListIterator( const MeshListIterator & );

    //! Assignment operator
    MeshListIterator &operator=( const MeshListIterator & );

    //! Increment
    MeshIterator &operator++() override;

    //! Decrement
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

    // A pointer to a std::vector containing the desired mesh elements
    std::shared_ptr<std::vector<TYPE>> d_elements;
};

} // namespace AMP::Mesh

#endif
