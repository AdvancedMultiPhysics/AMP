#ifndef included_AMP_VectorIterators
#define included_AMP_VectorIterators

#include "AMP/utils/TypeTraits.h"
#include "AMP/utils/UtilityMacros.h"

#include <iterator>


namespace AMP::LinearAlgebra {


// Forward deceleration
class VectorData;


/**
 * \class VectorDataIterator
 * \brief  Iterator for local data in a vector
 *
 * \details Even though a vector may have non-contiguous storage of data, the
 * interface presents a contiguous block of memory to the user:  each element
 * in the vector is given an offset from 0 and the vector is packed.
 * This allows for a random access iterator on the data.
 *
 * Vector::begin() and Vector::end() return this class.  This class
 * uses the DataBlock interface in vectors to access data.  As a result,
 * for some non-AMP managed vectors, this class may not be the most efficient.
 */
template<class TYPE = double>
class VectorDataIterator final
{
public: // iterator_traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = TYPE;
    using difference_type   = ptrdiff_t;
    using pointer           = const TYPE *;
    using reference         = const TYPE &;

private:
    using TYPE2 = typename AMP::remove_cvref_t<TYPE>;
    size_t d_N_blocks, d_CurBlock, d_CurOffset, d_pos, d_size;
    size_t d_hashcode;
    size_t *d_blockSize;
    TYPE **d_data;
    uint64_t *d_typeHash;
    mutable TYPE2 d_dataCast;

    void advance( size_t );
    void recede( size_t );

public:
    //!  Default constructor
    VectorDataIterator();

    //!  Destructor
    ~VectorDataIterator();

    /** \brief Copy constructor
     * \param[in] rhs iterator to copy
     * \details  Copies an iterator
     */
    VectorDataIterator( const VectorDataIterator &rhs );

    /*!
     * Move constructor
     * @param rhs           Iterator to copy
     */
    VectorDataIterator( VectorDataIterator &&rhs );

    /*!
     * Assignment operator
     * @param rhs           Iterator to copy
     */
    VectorDataIterator &operator=( const VectorDataIterator &rhs );

    /*!
     * Move assignment operator
     * @param rhs           Iterator to copy
     */
    VectorDataIterator &operator=( VectorDataIterator &&rhs );


    /** \brief Constructor from a vector
     * \details This will construct an iterator over the local data in the vector.
     *   Vector::begin() and Vector::end() call this function to instantiate a new iterator.
     *   This method should not be invoked outside of this use.
     * \param[in] p  A (non-reference counted) pointer to the vector being iterated over
     * \param[in] position  The local position in the vector.
     */
    explicit VectorDataIterator( VectorData *p, size_t position );


    //! Return a new iterator to the beginning of this iterator
    VectorDataIterator begin() const;


    //! Return a new iterator to the end of this iterator
    VectorDataIterator end() const;


    //! Return the size of the iterator
    inline size_t size() const { return d_size; }


    //! Return the position of the iterator
    inline size_t position() const { return d_pos; }


    /** \brief Dereference the iterator
     * \return Value pointed to by the iterator
     * \warning Setting values in the vector with the iterator requires firing of dataChanged
     * event.
     * \see PetscVector
     * \see DataChangeListener
     * \details  This returns a reference to the data pointed to by the iterator
     */
    inline TYPE &operator*();


    /** \brief Test for equality
     * \return True iff iterators point to the same place on the same vector
     * \details Returns true iff rhs points to the exact same offset of the same vector.
     * It is possible for two iterators to point to the same spot in memory and return false.  This
     * is due to the abstraction of a contiguous block presented by Vector.
     */
    inline bool operator==( const VectorDataIterator &rhs ) const;

    /** \brief Test for inequality
     * \returns True iff !(*this == rhs)
     * \details Returns !(*this == rhs)
     */
    inline bool operator!=( const VectorDataIterator &rhs ) const;

    //! Less than operator
    inline bool operator<( const VectorDataIterator &rhs ) const;

    //! Greater than operator
    inline bool operator>( const VectorDataIterator &rhs ) const;

    //! Less than or equal operator
    inline bool operator<=( const VectorDataIterator &rhs ) const;

    //! Greater than or equal operator
    inline bool operator>=( const VectorDataIterator &rhs ) const;

    /** \brief Increment the iterator
     * \returns a reference to this iterator
     */
    VectorDataIterator &operator++();

    /** \brief Increment the iterator
     * \returns a copy of this iterator
     */
    VectorDataIterator operator++( int );

    /** \brief Decrement the iterator
     * \returns a reference to this iterator
     */
    VectorDataIterator &operator--();

    /** \brief Decrement the iterator
     * \returns a copy of this iterator
     */
    VectorDataIterator operator--( int );

    /** \brief Add a constant to this iterator
      * \param[in] i Offset to move forward
      * \returns A new iterator i steps forward
      * \details  Equivalent to
      * \code
        VectorDataIterator t = *this;
        for ( int k = 0 ; k != i ; k++ )
          ++t;
        \endcode
      * except it is \f$O(1)\f$ in most cases.
      */
    VectorDataIterator operator+( int i );

    /** \brief Add a constant to this iterator
      * \param[in] i Offset to move forward
      * \returns A new iterator i steps forward
      * \details  Equivalent to
      * \code
        for ( int k = 0 ; k != i ; k++ )
          operator ++ ();
        \endcode
      * except it is \f$O(1)\f$ in most cases.
      */
    VectorDataIterator &operator+=( int i );

    /** \brief Subtract a constant to this iterator
      * \param[in] i Offset to move backward
      * \returns A new iterator i steps backward
      * \details  Equivalent to
      * \code
        VectorDataIterator t = *this;
        for ( int k = 0 ; k != i ; k++ )
          --t;
        \endcode
      * except it is \f$O(1)\f$ in most cases.
      */
    VectorDataIterator operator-( int i );

    /** \brief Subtract a constant to this iterator
      * \param [in] i Offset to move backward
      * \returns A new iterator i steps backward
      * \details  Equivalent to
      * \code
        for ( int k = 0 ; k != i ; k++ )
          operator -- ();
        \endcode
      * except it is \f$O(1)\f$ in most cases.
      */
    VectorDataIterator &operator-=( int i );

    /** \brief Compute distance between two iterators
      * \param  rhs The other iterator to measure distance to
      * \returns Number of entities between iterators.  If rhs == this,
      * the answer is 0. Else, if rhs is after this, then the answer is
      * positive, otherwise it is negative.
      * \details If rhs is after this, this is equivalent to
      * \code
        int answer = 0;
        VectorDataIterator  t = *this;
        while ( t != rhs )
        {
          t++;
          answer++;
        }
        return answer;
        \endcode
      * except it is \f$O(1)\f$ in most cases.  If rhs is before this,
      * the equivalent code is the same except t and answer are
      * decremented.
      */
    int operator-( const VectorDataIterator &rhs ) const;
};


} // namespace AMP::LinearAlgebra


#include "AMP/vectors/data/VectorDataIterator.hpp"

#endif
