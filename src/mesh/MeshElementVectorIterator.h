#ifndef included_AMP_MeshElementVectorIterator
#define included_AMP_MeshElementVectorIterator

#include "AMP/mesh/MeshIterator.h"
#include "AMP/utils/Utilities.h"

#include <iterator>
#include <memory>


namespace AMP::Mesh {


/**
 * \class MeshElementVectorIterator
 * \brief A class used to iterate over a set of mesh elements.
 * \details  This class provides routines for iterating over a set
 * of mesh elments that are in a std::vector.
 */
template<class TYPE = MeshElement>
class MeshElementVectorIterator final : public MeshIterator
{
public:
    //! Empty MeshElementVectorIterator constructor
    MeshElementVectorIterator();

    //! Default MeshElementVectorIterator constructor
    explicit MeshElementVectorIterator( std::shared_ptr<std::vector<TYPE>> elements,
                                        size_t pos = 0 );

    //! Deconstructor
    virtual ~MeshElementVectorIterator() = default;

    //! Copy constructor
    MeshElementVectorIterator( const MeshElementVectorIterator & );

    //! Assignment operator
    MeshElementVectorIterator &operator=( const MeshElementVectorIterator & );

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
