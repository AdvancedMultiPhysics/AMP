#ifndef included_AMP_MeshElementVector
#define included_AMP_MeshElementVector

#include "AMP/utils/UtilityMacros.h"

#include <initializer_list>
#include <memory>
#include <vector>


namespace AMP::Mesh {


class MeshElement;
class MeshElementVectorIterator;


/**
 * \class MeshElementVectorBase
 * \brief A base class for a vector of MeshElements
 */
class MeshElementVectorBase
{
public:
    MeshElementVectorBase( const MeshElementVectorBase & ) = delete;  //!< Copy constructor
    virtual ~MeshElementVectorBase()                       = default; //!< Destructor
    inline bool empty() const { return d_size == 0; }                 //!< Is the vector empty
    inline size_t size() const { return d_size; }              //!< Return the number of elements
    virtual const MeshElement &operator[]( size_t ) const = 0; //!< Access the desired element

protected:
    MeshElementVectorBase() = default;

protected:
    size_t d_size = 0;
};


/**
 * \class MeshElementVector
 * \brief A base class for a vector of MeshElements
 */
template<class TYPE>
class MeshElementVector final : public MeshElementVectorBase
{
public:
    MeshElementVector() = default;
    MeshElementVector( size_t N )
    {
        d_size = N;
        d_data = new TYPE[N];
    }
    MeshElementVector( const TYPE &x ) : MeshElementVector( 1 ) { d_data[0] = x; }
    MeshElementVector( const MeshElementVector & ) = delete;
    virtual ~MeshElementVector() { delete[] d_data; }
    const MeshElement &operator[]( size_t i ) const override
    {
        AMP_DEBUG_ASSERT( i < d_size );
        return d_data[i];
    }
    TYPE &operator[]( size_t i )
    {
        AMP_DEBUG_ASSERT( i < d_size );
        return d_data[i];
    }

protected:
    TYPE *d_data = nullptr;
};


/**
 * \class MeshElementVectorPtr
 * \brief A pointer class to wrap a MeshElementVector
 */
class MeshElementVectorPtr final
{
public:
    template<class T>
    MeshElementVectorPtr( std::unique_ptr<MeshElementVector<T>> p ) : ptr( std::move( p ) )
    {
    }
    MeshElementVectorPtr( std::unique_ptr<MeshElementVectorBase> p ) : ptr( std::move( p ) ) {}
    MeshElementVectorPtr();
    MeshElementVectorPtr( MeshElementVectorPtr && )                 = default;
    MeshElementVectorPtr( const MeshElementVectorPtr & )            = delete;
    MeshElementVectorPtr &operator=( MeshElementVectorPtr && )      = default;
    MeshElementVectorPtr &operator=( const MeshElementVectorPtr & ) = delete;
    inline bool empty() const { return ptr ? ptr->empty() : true; } //!< Is the vector empty
    inline size_t size() const { return ptr->size(); } //!< Return the number of elements
    inline const MeshElement &operator[]( size_t i ) const { return ptr->operator[]( i ); }
    MeshElementVectorIterator begin() const;
    MeshElementVectorIterator end() const;

protected:
    std::unique_ptr<MeshElementVectorBase> ptr;
};


/**
 * \class MeshElementVectorIterator
 * \brief A class used to iterate over a set of mesh elements.
 * \details  This class provides routines for iterating over a set
 * of mesh elments that are in a std::vector.
 */
class MeshElementVectorIterator
{
public: // iterator_traits
    typedef MeshElementVectorIterator Iterator;
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = AMP::Mesh::MeshElement;
    using difference_type   = ptrdiff_t;
    using pointer           = const AMP::Mesh::MeshElement *;
    using reference         = const AMP::Mesh::MeshElement &;

public:
    explicit MeshElementVectorIterator( const MeshElementVectorBase *ptr,
                                        size_t pos = 0 ); //! Default constructor
    virtual ~MeshElementVectorIterator() = default;       //! Destructor
    Iterator &operator++();                               //! Increment
    Iterator &operator--();                               //! Decrement
    Iterator &operator+=( int N );                        // Arithmetic operator+=
    bool operator==( const Iterator &rhs ) const;         //! Check if two iterators are equal
    bool operator!=( const Iterator &rhs ) const;         //! Check if two iterators are not equal
    Iterator begin() const;                               //! Return an iterator to the begining
    Iterator end() const;                                 //! Return an iterator to the begining
    inline size_t size() const { return d_data->size(); } //! Return the size of the iterator
    const MeshElement &operator*() const;                 //! Dereference the iterator

protected:
    size_t d_pos  = 0;
    size_t d_size = 0;
    const MeshElementVectorBase *d_data;
};


} // namespace AMP::Mesh

#endif
