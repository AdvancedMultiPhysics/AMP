#ifndef included_AMP_MeshIterators
#define included_AMP_MeshIterators

#include "AMP/mesh/MeshElement.h"
#include "AMP/utils/enable_shared_from_this.h"

#include <iterator>
#include <memory>


namespace AMP::IO {
class RestartManager;
}


namespace AMP::Mesh {


/**
 * \class MeshIterator
 * \brief A class used to iterate over elements in a Mesh
 *
 * \details  This class provides routines for iterating over a set of elements.
 *   It is inherited from std::iterator.  The base class provides routines for
 *   the random access iterators, but does so using the increment/decrement routines.
 *   Derived classes may (or may not) override these routines for performance optimizations.
 */
class MeshIterator : public AMP::enable_shared_from_this<AMP::Mesh::MeshIterator>
{
public: // iterator_traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = AMP::Mesh::MeshElement;
    using difference_type   = ptrdiff_t;
    using pointer           = const AMP::Mesh::MeshElement *;
    using reference         = const AMP::Mesh::MeshElement &;

public:
    /**
     *\typedef shared_ptr
     *\brief  Name for the shared pointer.
     *\details  Use this typedef for a reference counted pointer to a mesh manager object.
     */
    typedef std::shared_ptr<MeshIterator> shared_ptr;

    //! Enum for the type of iterator supported
    enum class Type : uint8_t { Forward = 1, Bidirectional = 2, RandomAccess = 3 };


public:
    //! Empty MeshIterator constructor
    MeshIterator();

    //! Move constructor
    MeshIterator( MeshIterator && );

    //! Copy constructor
    MeshIterator( const MeshIterator & );

    //! Move operator
    MeshIterator &operator=( MeshIterator && );

    //! Assignment operator
    MeshIterator &operator=( const MeshIterator & );

    //! Create a mesh element taking ownership
    MeshIterator( MeshIterator * );

    //! Deconstructor
    virtual ~MeshIterator();


public: // Virtual functions
    //! Return the class name
    virtual std::string className() const;

    //! Return an iterator to the begining
    virtual MeshIterator begin() const;

    //! Return an iterator to the begining
    virtual MeshIterator end() const;

    /**
     * \brief Pre-Increment
     * \details  Pre-Increment the mesh iterator and return the reference to the iterator.
     *   This should be the fastest way to increment the iterator.
     */
    virtual MeshIterator &operator++();

    /**
     * \brief Pre-Decrement
     * \details  Pre-Decrement the mesh iterator and return the reference to the iterator.
     *   This should be the fastest way to decrement the iterator.
     *   Note: not all iterators support decrementing the iterator (libmesh).
     */
    virtual MeshIterator &operator--();

    /**
     * \brief Arithmetic operator+=
     * \details  Random access increment to advance the iterator by N.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-increment will be used instead and may reduce performance.
     *   Note: the default behavior of all random access assignement iterators will be to call this
     * function
     *     so derived classes only need to impliment this function for improved performance.
     * \param N  Number to increment by (may be negitive)
     */
    virtual MeshIterator &operator+=( int N );

    /**
     * \brief Arithmetic operator+=
     * \details  Random access increment to advance the iterator by the given iterator.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-increment will be used instead and may reduce performance.
     * \param it  Iterator to add
     */
    virtual MeshIterator &operator+=( const MeshIterator &it );

    //! Dereference iterator with offset
    virtual MeshElement &operator[]( int );

    //! Check if two iterators are equal
    virtual bool operator==( const MeshIterator &rhs ) const;

    //! Check if two iterators are not equal
    virtual bool operator!=( const MeshIterator &rhs ) const;


public: // non-virtual functions
    //! Return the iterator type
    Type type() const;

    //! Return a unique hash id
    uint64_t getID() const;

    //! Return the raw iterator (may be this)
    inline const MeshIterator *rawIterator() const;

    //! Check if the iterator is empty
    inline bool empty() const;

    //! Return the number of elements in the iterator
    inline size_t size() const;

    //! Return the current position (from the beginning) in the iterator
    inline size_t position() const;

    //! Operator <
    inline bool operator<( const MeshIterator & ) const;

    //! Operator <=
    inline bool operator<=( const MeshIterator &it ) const;

    //! Operator >
    inline bool operator>( const MeshIterator & ) const;

    //! Operator >=
    inline bool operator>=( const MeshIterator & ) const;

    //! Dereference the iterator
    inline MeshElement &operator*();

    //! Dereference the iterator
    inline const MeshElement &operator*() const;

    //! Dereference the iterator
    inline MeshElement *operator->();

    //! Dereference the iterator
    inline const MeshElement *operator->() const;

    //! Dereference the iterator
    inline MeshElement *get();

    //! Dereference the iterator
    inline const MeshElement *get() const;

    /**
     * \brief Post-Increment
     * \details  Post-Increment the mesh iterator and return a reference to a temporary iterator.
     *   This should be avoided and pre-increment used whenever possible.
     */
    MeshIterator operator++( int );

    /**
     * \brief Post-Decrement
     * \details  Post-Decrement the mesh iterator and return a reference to a temporary iterator.
     *   This should be avoided and pre-decrement used whenever possible.
     *   Note: not all iterators support decrementing the iterator (libmesh).
     */
    MeshIterator operator--( int );

    /**
     * \brief Arithmetic operator+
     * \details  Random access increment to advance the iterator by N.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-increment will be used instead and may reduce performance.
     *   Note: the default behavior of all random access iterators will be to call this function
     *     so derived classes only need to impliment this function for improved performance.
     * \param N  Number to increment by (may be negitive)
     */
    MeshIterator operator+( int N ) const;

    /**
     * \brief Arithmetic operator+
     * \details  Random access increment to advance the iterator by the given iterator.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-increment will be used instead and may reduce performance.
     * \param it  Iterator to add
     */
    MeshIterator operator+( const MeshIterator &it ) const;

    /**
     * \brief Arithmetic operator-
     * \details  Random access decrement to reverse the iterator by N.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-decrement will be used instead and may reduce performance.
     * \param N  Number to decrement by (may be negitive)
     */
    MeshIterator operator-( int N ) const;

    /**
     * \brief Arithmetic operator-
     * \details  Random access decrement to reverse the iterator by the given iterator.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-decrement will be used instead and may reduce performance.
     * \param it  Iterator to subtract
     */
    MeshIterator operator-( const MeshIterator &it ) const;

    /**
     * \brief Arithmetic operator-=
     * \details  Random access decrement to reverse the iterator by N.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-decrement will be used instead and may reduce performance.
     * \param N  Number to decrement by (may be negitive)
     */
    MeshIterator &operator-=( int N );

    /**
     * \brief Arithmetic operator-=
     * \details  Random access decrement to reverse the iterator by the given iterator.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-decrement will be used instead and may reduce performance.
     * \param it  Iterator to subtract
     */
    MeshIterator &operator-=( const MeshIterator &it );


public: // Write/read restart data
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;
    virtual void writeRestart( int64_t fid ) const;
    MeshIterator( int64_t fid );


protected:
    // Clone the iterator
    virtual MeshIterator *clone() const;

protected:
    // A pointer to the derived class
    MeshIterator *d_iterator;
    // Unique hash for identifying the underlying iterator
    uint32_t d_typeHash;
    // Type of iterator
    Type d_iteratorType;
    // Size of the iterator
    size_t d_size;
    // Position of the iterator
    size_t d_pos;
    // Pointer to the current element
    MeshElement *d_element;
};


} // namespace AMP::Mesh

#include "AMP/mesh/MeshIterator.inline.h"

#endif
