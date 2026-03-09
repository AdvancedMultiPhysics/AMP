#ifndef included_AMP_MeshIterators
#define included_AMP_MeshIterators

#include "AMP/mesh/MeshElement.h"

#include <iterator>
#include <memory>


namespace AMP::IO {
class RestartManager;
}


namespace AMP::Mesh {


class MeshIterator;


/**
 * \class MeshIteratorEnd
 * \brief A base class used to represent the end iterator
 * \details  This is an empty class used to represent the end marker for an iterator.
 */
class MeshIteratorEnd
{
};


/**
 * \class MeshIteratorBase
 * \brief A base class used to iterate over elements in a Mesh
 *
 * \details  This class provides routines for iterating over a set of elements.
 *   It is inherited from std::iterator.  The base class provides routines for
 *   the random access iterators, but does so using the increment/decrement routines.
 *   Derived classes may (or may not) override these routines for performance optimizations.
 */
class MeshIteratorBase
{
public: // iterator_traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = AMP::Mesh::MeshElement;
    using difference_type   = ptrdiff_t;
    using pointer           = const AMP::Mesh::MeshElement *;
    using reference         = const AMP::Mesh::MeshElement &;


public:
    //! Enum for the type of iterator supported
    enum class Type : uint8_t { Forward = 1, Bidirectional = 2, RandomAccess = 3 };


public:
    //! Return the iterator type
    inline Type type() const { return d_iteratorType; }

    //! Return a unique hash id
    inline uint64_t getID() const { return d_typeHash; }

    //! Return the number of elements in the iterator
    inline size_t empty() const { return d_size == 0; }

    //! Return the number of elements in the iterator
    inline size_t size() const { return d_size; }

    //! Return the current position (from the beginning) in the iterator
    inline size_t pos() const { return d_pos; }

    //! Dereference the iterator
    inline const MeshElement *operator->() const { return d_element; }

    //! Dereference the iterator
    inline const MeshElement &operator*() const { return *d_element; }

    //! Return an iterator to the end (use tombstone class)
    inline MeshIteratorEnd end() const { return MeshIteratorEnd(); }

    //! Check if two iterators are equal
    inline bool operator==( MeshIteratorEnd ) const { return d_pos == d_size; }

    //! Check if two iterators are not equal
    inline bool operator!=( MeshIteratorEnd ) const { return d_pos != d_size; }

public:
    //! Virtual destructor
    virtual ~MeshIteratorBase() = default;

    //! Return the class name
    virtual std::string className() const = 0;

    //! Set the position in the iterator
    virtual void setPos( size_t ) = 0;

    // Increment
    virtual MeshIteratorBase &operator++() = 0;

    // Decrement
    virtual MeshIteratorBase &operator--() = 0;

    // Arithmetic operator+=
    virtual MeshIteratorBase &operator+=( int N ) = 0;

    //! Check if two iterators are equal
    virtual bool operator==( const MeshIteratorBase &rhs ) const = 0;

    //! Check if two iterators are not equal
    virtual bool operator!=( const MeshIteratorBase &rhs ) const = 0;

    //! Return an iterator to the beginning
    virtual MeshIterator begin() const = 0;

    //! Clone the iterator
    virtual std::unique_ptr<MeshIteratorBase> clone() const = 0;


public: // Write/read restart data
    virtual void registerChildObjects( AMP::IO::RestartManager * ) const = 0;
    virtual void writeRestart( int64_t ) const                           = 0;
    MeshIteratorBase( int64_t, AMP::IO::RestartManager * );


protected:
    MeshIteratorBase()                           = default;
    MeshIteratorBase( const MeshIteratorBase & ) = delete;
    MeshIteratorBase &operator=( const MeshIteratorBase & ) = delete;
    MeshIteratorBase( MeshIteratorBase && )                 = delete;
    MeshIteratorBase &operator=( MeshIteratorBase && ) = delete;


protected:
    Type d_iteratorType          = Type::RandomAccess; //!< Type of iterator
    uint32_t d_typeHash          = 0;                  //!< Unique hash for the type
    size_t d_size                = 0;                  //!< Size of the iterator
    size_t d_pos                 = 0;                  //!< Position of the iterator
    const MeshElement *d_element = nullptr;            //!< Pointer to the current element
};


/**
 * \class MeshIterator
 * \brief A class used to iterate over elements in a Mesh
 *
 * \details  This class provides routines for iterating over a set of elements.
 *   It is inherited from std::iterator.  This is a wrapper class that contains
 *   a pointer to the underlying iterator.
 */
class MeshIterator final
{
public: // iterator_traits
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = AMP::Mesh::MeshElement;
    using difference_type   = ptrdiff_t;
    using pointer           = const AMP::Mesh::MeshElement *;
    using reference         = const AMP::Mesh::MeshElement &;
    using Type              = MeshIteratorBase::Type;


public:
    //! Create a mesh iterator
    template<class T, typename... Args>
    static MeshIterator create( Args... args )
    {
        return MeshIterator( new T( std::forward<Args>( args )... ) );
    }

    //! Empty MeshIterator constructor
    MeshIterator();

    //! Default constructor
    MeshIterator( std::unique_ptr<MeshIteratorBase> &&p ) : it( p.release() ) {}

    //! Move constructor
    MeshIterator( MeshIterator && );

    //! Copy constructor
    MeshIterator( const MeshIterator & );

    //! Move operator
    MeshIterator &operator=( MeshIterator && );

    //! Assignment operator
    MeshIterator &operator=( const MeshIterator & );

    //! Assignment operator
    MeshIterator &operator=( std::unique_ptr<MeshIteratorBase> && );

    //! Deconstructor
    ~MeshIterator() { delete it; }

    //! Return the class name
    inline std::string className() const { return it->className(); }

    //! Return an iterator to the beginning
    inline MeshIterator begin() const { return it->begin(); }

    //! Return an iterator to the end (use tombstone class)
    inline MeshIteratorEnd end() const { return MeshIteratorEnd(); }

    /**
     * \brief Pre-Increment
     * \details  Pre-Increment the mesh iterator and return the reference to the iterator.
     *   This should be the fastest way to increment the iterator.
     */
    inline MeshIterator &operator++()
    {
        it->operator++();
        return *this;
    }

    /**
     * \brief Pre-Decrement
     * \details  Pre-Decrement the mesh iterator and return the reference to the iterator.
     *   This should be the fastest way to decrement the iterator.
     *   Note: not all iterators support decrementing the iterator (libmesh).
     */
    inline MeshIterator &operator--()
    {
        it->operator--();
        return *this;
    }

    /**
     * \brief Arithmetic operator+=
     * \details  Random access increment to advance the iterator by N.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-increment will be used instead and may reduce performance.
     *   Note: the default behavior of all random access assignment iterators will be to call this
     *       function so derived classes only need to implement this function for improved
     * performance. \param N  Number to increment by (may be negative)
     */
    inline MeshIterator &operator+=( int N )
    {
        it->operator+=( N );
        return *this;
    }

    /**
     * \brief Arithmetic operator+=
     * \details  Random access increment to advance the iterator by the given iterator.
     *   Note: not all iterators support random access (libmesh).
     *   In this case, the pre-increment will be used instead and may reduce performance.
     * \param it  Iterator to add
     */
    MeshIterator &operator+=( const MeshIterator &it );

    //! Check if two iterators are equal
    inline bool operator==( const MeshIterator &rhs ) const { return *it == *rhs.it; }

    //! Check if two iterators are equal
    inline bool operator==( MeshIteratorEnd rhs ) const { return *it == rhs; }

    //! Check if two iterators are equal
    inline bool operator==( const MeshIteratorBase &rhs ) const { return *it == rhs; }

    //! Check if two iterators are not equal
    inline bool operator!=( const MeshIterator &rhs ) const { return *it != *rhs.it; }

    //! Check if two iterators are not equal
    inline bool operator!=( MeshIteratorEnd rhs ) const { return *it != rhs; }

    //! Check if two iterators are not equal
    inline bool operator!=( const MeshIteratorBase &rhs ) const { return *it != rhs; }

    //! Return the iterator type
    inline Type type() const { return it->type(); }

    //! Return a unique hash id
    inline uint64_t getID() const { return it->getID(); }

    //! Return the raw iterator (may be this)
    inline MeshIteratorBase *rawIterator() { return it; }

    //! Return the raw iterator (may be this)
    inline const MeshIteratorBase *rawIterator() const { return it; }

    //! Check if the iterator is empty
    inline bool empty() const { return it->empty(); }

    //! Return the number of elements in the iterator
    inline size_t size() const { return it->size(); }

    //! Return the current position (from the beginning) in the iterator
    inline size_t pos() const { return it->pos(); }

    //! Set the position in the iterator
    inline void setPos( size_t i ) { it->setPos( i ); }

    //! Operator <
    inline bool operator<( const MeshIterator &rhs ) const { return pos() < rhs.pos(); }

    //! Operator <=
    inline bool operator<=( const MeshIterator &rhs ) const { return pos() <= rhs.pos(); }

    //! Operator >
    inline bool operator>( const MeshIterator &rhs ) const { return pos() > rhs.pos(); }

    //! Operator >=
    inline bool operator>=( const MeshIterator &rhs ) const { return pos() >= rhs.pos(); }

    //! Dereference the iterator
    inline const MeshElement &operator*() const { return it->operator*(); }

    //! Dereference the iterator
    inline const MeshElement *operator->() const { return it->operator->(); }

    //! Dereference the iterator
    inline const MeshElement *get() const { return it->operator->(); }

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
     * \param N  Number to increment by (may be negative)
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
     * \param N  Number to decrement by (may be negative)
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
     * \param N  Number to decrement by (may be negative)
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


public: // Advanced functions (use with caution)
    //! Constructor that takes ownership of the raw pointer
    MeshIterator( MeshIteratorBase *p ) : it( p ) {}

    /**
     * \brief Return a pointer to the underlying and release ownership
     * \details  This function will return the raw pointer to the underlying
     *   iterator and release its ownership.  It is up to the user to properly
     *   destroy the returned iterator.
     *   Note: This function invalidates *this and any future calls to this object
     *   may result in a dereferencing a null pointer.
     */
    //! Return a pointer to the underlying and releases the ownership
    MeshIteratorBase *release();


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const;
    void writeRestart( int64_t fid ) const;
    MeshIterator( int64_t fid );


protected:
    MeshIteratorBase *it = nullptr; // A pointer to the derived class
};


} // namespace AMP::Mesh


#endif
