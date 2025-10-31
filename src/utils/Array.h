#ifndef included_AMP_ArrayClass
#define included_AMP_ArrayClass

#include "AMP/utils/ArraySize.h"
#include "AMP/utils/FunctionTable.h"
#include "AMP/utils/Memory.h"

#include <array>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


namespace AMP {


/*!
 * Class Array is a multi-dimensional array class written by Mark Berrill
 */
template<class TYPE, class FUN, class Allocator>
class Array final
{
public: // Typedefs
    typedef TYPE value_type;
    typedef FUN function_table;
    static_assert( !std::is_same_v<TYPE, std::vector<bool>::reference> );
    static_assert( std::is_same_v<typename FUN::value_type, TYPE> );
    static_assert( std::is_same_v<typename Allocator::value_type, void> ||
                   std::is_same_v<typename Allocator::value_type, TYPE> );
    using scalarAllocator_t =
        typename std::allocator_traits<Allocator>::template rebind_alloc<TYPE>;

public: // Constructors / assignment operators
    /*!
     * Create a new empty Array
     */
    Array();

    /*!
     * Create an Array with the given size
     * @param N             Size of the array
     * @param data          Optional raw array to copy the src data
     */
    explicit Array( const ArraySize &N, const TYPE *data = nullptr );

    /*!
     * Create a new 1D Array with the given number of elements
     * @param N             Number of elements in the array
     */
    explicit Array( size_t N );

    /*!
     * Create a new 2D Array with the given number of rows and columns
     * @param N_rows        Number of rows
     * @param N_columns     Number of columns
     */
    explicit Array( size_t N_rows, size_t N_columns );

    /*!
     * Create a new 3D Array with the given number of rows and columns
     * @param N1            Number of rows
     * @param N2            Number of columns
     * @param N3            Number of elements in the third dimension
     */
    explicit Array( size_t N1, size_t N2, size_t N3 );

    /*!
     * Create a new 4D Array with the given number of rows and columns
     * @param N1            Number of elements in the first dimension
     * @param N2            Number of elements in the second dimension
     * @param N3            Number of elements in the third dimension
     * @param N4            Number of elements in the fourth dimension
     */
    explicit Array( size_t N1, size_t N2, size_t N3, size_t N4 );

    /*!
     * Create a new 4D Array with the given number of rows and columns
     * @param N1            Number of elements in the first dimension
     * @param N2            Number of elements in the second dimension
     * @param N3            Number of elements in the third dimension
     * @param N4            Number of elements in the fourth dimension
     * @param N5            Number of elements in the fifth dimension
     */
    explicit Array( size_t N1, size_t N2, size_t N3, size_t N4, size_t N5 );

    /*!
     * Create a 1D Array with the range
     * @param range         Range of the data
     */
    template<typename U = TYPE, typename = typename std::enable_if<std::is_same_v<U, TYPE>>::type>
    explicit Array( const Range<U> &range );

    /*!
     * Create a 1D Array using a string that mimic's MATLAB
     * @param range         Range of the data
     */
    explicit Array( std::string range );

    /*!
     * Create a 1D Array with the given initializer list
     * @param data          Input data
     */
    Array( std::initializer_list<TYPE> data );

    /*!
     * Create a 2D Array with the given initializer lists
     * @param data          Input data
     */
    Array( std::initializer_list<std::initializer_list<TYPE>> data );


    /*!
     * Copy constructor
     * @param rhs           Array to copy
     */
    Array( const Array &rhs );

    /*!
     * Move constructor
     * @param rhs           Array to copy
     */
    Array( Array &&rhs );

    /*!
     * Assignment operator
     * @param rhs           Array to copy
     */
    Array &operator=( const Array &rhs );

    /*!
     * Move assignment operator
     * @param rhs           Array to copy
     */
    Array &operator=( Array &&rhs );

    /*!
     * Assignment operator
     * @param rhs           std::vector to copy
     */
    Array &operator=( const std::vector<TYPE> &rhs );

    //! Is copyable?
    ARRAY_INLINE bool isCopyable() const { return d_isCopyable; }

    //! Set is copyable
    ARRAY_INLINE void setCopyable( bool flag ) { d_isCopyable = flag; }

    //! Is fixed size?
    ARRAY_INLINE bool isFixedSize() const { return d_isFixedSize; }

    //! Set is copyable
    ARRAY_INLINE void setFixedSize( bool flag ) { d_isFixedSize = flag; }

    //! Destructor
    ~Array();


public: // Views/copies/subset
    /*!
     * Create a multi-dimensional Array view to a raw block of data
     * @param N             Number of elements in each dimension
     * @param data          Pointer to the data
     */
    static std::unique_ptr<Array> view( const ArraySize &N, std::shared_ptr<TYPE> data );


    /*!
     * Create a multi-dimensional Array view to a raw block of data
     * @param N             Number of elements in each dimension
     * @param data          Pointer to the data
     */
    static std::unique_ptr<const Array> constView( const ArraySize &N,
                                                   const std::shared_ptr<const TYPE> &data );

    /*!
     * Create an Array view to this array
     */
    Array<TYPE, FUN, Allocator> view();


    /*!
     * Make this object a view of the src
     * @param src           Source vector to take the view of
     */
    void view2( Array &src );

    /*!
     * Make this object a view of the data
     * @param N             Number of elements in each dimension
     * @param data          Pointer to the data
     */
    void view2( const ArraySize &N, std::shared_ptr<TYPE> data );

    /*!
     * Make this object a view of the raw data (expert use only).
     * Use view2( N, shared_ptr(data,[](TYPE*){}) ) instead.
     *   Note: this interface is not recommended as it does not protect from
     *   the src data being deleted while still being used by the Array.
     *   Additionally for maximum performance it does not set the internal shared_ptr
     *   so functions like getPtr and resize will not work correctly.
     * @param ndim          Number of dimensions
     * @param dims          Number of elements in each dimension
     * @param data          Pointer to the data
     * @param isCopyable    Once the view is created, can the array be copied
     * @param isFixedSize   Once the view is created, is the array size fixed
     */
    void viewRaw(
        int ndim, const size_t *dims, TYPE *data, bool isCopyable = true, bool isFixedSize = true )
    {
        viewRaw( ArraySize( ndim, dims ), data, isCopyable, isFixedSize );
    }

    /*!
     * Make this object a view of the raw data (expert use only).
     * Use view2( N, shared_ptr(data,[](TYPE*){}) ) instead.
     *   Note: this interface is not recommended as it does not protect from
     *   the src data being deleted while still being used by the Array.
     *   Additionally for maximum performance it does not set the internal shared_ptr
     *   so functions like getPtr and resize will not work correctly.
     * @param N             Number of elements in each dimension
     * @param data          Pointer to the data
     * @param isCopyable    Once the view is created, can the array be copied
     * @param isFixedSize   Once the view is created, is the array size fixed
     */
    void viewRaw( const ArraySize &N, TYPE *data, bool isCopyable = true, bool isFixedSize = true )
    {
        d_isCopyable  = isCopyable;
        d_isFixedSize = isFixedSize;
        d_ptr.reset();
        d_size = N;
        d_data = data;
    }

    /*!
     * Create an array view of the given data (expert use only).
     * Use view2( N, shared_ptr(data,[](TYPE*){}) ) instead.
     *   Note: this interface is not recommended as it does not protect from
     *   the src data being deleted while still being used by the Array.
     *   Additionally for maximum performance it does not set the internal shared_ptr
     *   so functions like getPtr and resize will not work correctly.
     * @param N             Number of elements in each dimension
     * @param data          Pointer to the data
     */
    static Array staticView( const ArraySize &N, TYPE *data )
    {
        Array x;
        x.viewRaw( N, data, true, true );
        return x;
    }

    /*!
     * Convert an array of one type to another.  This may or may not allocate new memory.
     * @param array         Input array
     */
    template<class TYPE2>
    static inline auto convert( std::shared_ptr<Array<TYPE, FUN, Allocator>> array )
    {
        using FUN2 = FUN::template cloneTo<TYPE2>;
        static_assert( std::is_same_v<typename FUN2::value_type, TYPE2> );
        using ARRAY2 = Array<TYPE2, FUN2, Allocator>;
        auto array2  = std::make_unique<ARRAY2>( array->size() );
        array2.copy( *array );
        return array2;
    }


    /*!
     * Convert an array of one type to another.  This may or may not allocate new memory.
     * @param array         Input array
     */
    template<class TYPE2>
    static inline auto convert( std::shared_ptr<const Array<TYPE, FUN, Allocator>> array )
    {
        using FUN2 = FUN::template cloneTo<TYPE2>;
        static_assert( std::is_same_v<typename FUN2::value_type, TYPE2> );
        using ARRAY2 = Array<TYPE2, FUN2, Allocator>;
        auto array2  = std::make_unique<ARRAY2>( array->size() );
        array2.copy( *array );
        return array2;
    }


    /*!
     * Copy and convert data from another array to this array
     * @param array         Source array
     */
    template<class TYPE2, class FUN2, class Allocator2>
    void inline copy( const Array<TYPE2, FUN2, Allocator2> &array )
    {
        resize( array.size() );
        copy( array.data() );
    }

    /*!
     * Copy and convert data from a raw vector to this array.
     *    Note: The current array must be allocated to the proper size first.
     * @param data          Source data
     */
    template<class TYPE2>
    inline void copy( const TYPE2 *data );

    /*!
     * Copy and convert data from this array to a raw vector.
     * @param data          Source data
     */
    template<class TYPE2>
    inline void copyTo( TYPE2 *data ) const;

    /*!
     * Copy and convert data from this array to a new array
     */
    template<class TYPE2>
    inline auto cloneTo() const
    {
        using FUN2 = FUN::template cloneTo<TYPE2>;
        static_assert( std::is_same_v<typename FUN2::value_type, TYPE2> );
        Array<TYPE2, FUN2, AMP::HostAllocator<void>> dst( this->size() );
        copyTo( dst.data() );
        return dst;
    }


    /*!
     * Reinterpret the array as a new type (use with caution)
     */
    template<class TYPE2>
    inline auto reinterpretArray() const
    {
        static_assert( sizeof( TYPE ) == sizeof( TYPE2 ) );
        auto ptr   = std::reinterpret_pointer_cast<TYPE2>( std::const_pointer_cast<TYPE>( d_ptr ) );
        using FUN2 = FUN::template cloneTo<TYPE2>;
        static_assert( std::is_same_v<typename FUN2::value_type, TYPE2> );
        Array<TYPE2, FUN2, Allocator> y;
        y.view2( d_size, ptr );
        return y;
    }


    /*! swap the raw data pointers for the Arrays after checking for compatibility */
    void swap( Array &other );


    /*!
     * Fill the array with the given value
     * @param y         Value to fill
     */
    ARRAY_INLINE void fill( const TYPE &y )
    {
        for ( auto &x : *this )
            x = y;
    }

    /*!
     * Scale the array by the given value
     * @param y         Value to scale by
     */
    void scale( const TYPE &y );


    /*!
     * Set the values of this array to pow(base, exp)
     * @param base        Base array
     * @param exp         Exponent value
     */
    void pow( const Array &base, const TYPE &exp );


    //! Clear the data in the array
    void clear();


    //! Return the size of the Array
    ARRAY_INLINE int ndim() const { return d_size.ndim(); }


    //! Return the size of the Array
    ARRAY_INLINE const ArraySize &size() const { return d_size; }


    //! Return the size of the Array
    ARRAY_INLINE size_t size( int d ) const { return d_size[d]; }


    //! Return the size of the Array
    ARRAY_INLINE size_t length() const { return d_size.length(); }


    //! Return true if the Array is empty
    ARRAY_INLINE bool empty() const { return d_size.length() == 0; }


    //! Return true if the Array is not empty
    ARRAY_INLINE operator bool() const { return d_size.length() != 0; }


    /*!
     * Resize the Array
     * @param N             Number of elements
     */
    void resize( size_t N );


    /*!
     * Resize the Array
     * @param N_row         Number of rows
     * @param N_col         Number of columns
     */
    void resize( size_t N_row, size_t N_col );

    /*!
     * Resize the Array
     * @param N1            Number of rows
     * @param N2            Number of columns
     * @param N3            Number of elements in the third dimension
     */
    void resize( size_t N1, size_t N2, size_t N3 );

    /*!
     * Resize the Array
     * @param N             Number of elements in each dimension
     */
    void resize( const ArraySize &N );


    /*!
     * Resize the given dimension of the array
     * @param dim           The dimension to resize
     * @param N             Number of elements for the given dimension
     * @param value         Value to initialize new elements
     */
    void resizeDim( int dim, size_t N, const TYPE &value );


    /*!
     * Reshape the Array (total size of array will not change)
     * @param N             Number of elements in each dimension
     */
    void reshape( const ArraySize &N );


    /*!
     * Reshape the Array so that the number of dimensions is the
     *    max of ndim and the largest dim>1.
     * @param ndim          Desired number of dimensions
     */
    ARRAY_INLINE void setNdim( int ndim ) { d_size.setNdim( ndim ); }


    /*!
     * Subset the Array
     * @param index         Index to subset (imin,imax,jmin,jmax,kmin,kmax,...)
     */
    Array subset( const std::vector<size_t> &index ) const;


    /*!
     * Subset the Array
     * @param index         Index to subset (ix:kx:jx,iy:ky:jy,...)
     */
    Array subset( const std::vector<Range<size_t>> &index ) const;


    /*!
     * Copy data from an array into a subset of this array
     * @param index         Index of the subset (imin,imax,jmin,jmax,kmin,kmax,...)
     * @param subset        The subset array to copy from
     */
    void copySubset( const std::vector<size_t> &index, const Array &subset );

    /*!
     * Copy data from an array into a subset of this array
     * @param index         Index of the subset
     * @param subset        The subset array to copy from
     */
    void copySubset( const std::vector<Range<size_t>> &index, const Array &subset );

    /*!
     * Add data from an array into a subset of this array
     * @param index         Index of the subset (imin,imax,jmin,jmax,kmin,kmax,...)
     * @param subset        The subset array to add from
     */
    void addSubset( const std::vector<size_t> &index, const Array &subset );

    /*!
     * Add data from an array into a subset of this array
     * @param index         Index of the subset
     * @param subset        The subset array to add from
     */
    void addSubset( const std::vector<Range<size_t>> &index, const Array &subset );


public: // Accessors
    /*!
     * Access the desired element
     * @param i             The row index
     */
    ARRAY_INLINE TYPE &operator()( size_t i ) { return d_data[d_size.index( i )]; }

    /*!
     * Access the desired element
     * @param i             The row index
     */
    ARRAY_INLINE const TYPE &operator()( size_t i ) const { return d_data[d_size.index( i )]; }

    /*!
     * Access the desired element
     * @param i             The row index
     * @param j             The column index
     */
    ARRAY_INLINE TYPE &operator()( size_t i, size_t j ) { return d_data[d_size.index( i, j )]; }

    /*!
     * Access the desired element
     * @param i             The row index
     * @param j             The column index
     */
    ARRAY_INLINE const TYPE &operator()( size_t i, size_t j ) const
    {
        return d_data[d_size.index( i, j )];
    }

    /*!
     * Access the desired element
     * @param i             The row index
     * @param j             The column index
     * @param k             The third index
     */
    ARRAY_INLINE TYPE &operator()( size_t i, size_t j, size_t k )
    {
        return d_data[d_size.index( i, j, k )];
    }

    /*!
     * Access the desired element
     * @param i             The row index
     * @param j             The column index
     * @param k             The third index
     */
    ARRAY_INLINE const TYPE &operator()( size_t i, size_t j, size_t k ) const
    {
        return d_data[d_size.index( i, j, k )];
    }

    /*!
     * Access the desired element
     * @param i1            The first index
     * @param i2            The second index
     * @param i3            The third index
     * @param i4            The fourth index
     */
    ARRAY_INLINE TYPE &operator()( size_t i1, size_t i2, size_t i3, size_t i4 )
    {
        return d_data[d_size.index( i1, i2, i3, i4 )];
    }

    /*!
     * Access the desired element
     * @param i1            The first index
     * @param i2            The second index
     * @param i3            The third index
     * @param i4            The fourth index
     */
    ARRAY_INLINE const TYPE &operator()( size_t i1, size_t i2, size_t i3, size_t i4 ) const
    {
        return d_data[d_size.index( i1, i2, i3, i4 )];
    }

    /*!
     * Access the desired element
     * @param i1            The first index
     * @param i2            The second index
     * @param i3            The third index
     * @param i4            The fourth index
     * @param i5            The fifth index
     */
    ARRAY_INLINE TYPE &operator()( size_t i1, size_t i2, size_t i3, size_t i4, size_t i5 )
    {
        return d_data[d_size.index( i1, i2, i3, i4, i5 )];
    }

    /*!
     * Access the desired element
     * @param i1            The first index
     * @param i2            The second index
     * @param i3            The third index
     * @param i4            The fourth index
     * @param i5            The fifth index
     */
    ARRAY_INLINE const TYPE &
    operator()( size_t i1, size_t i2, size_t i3, size_t i4, size_t i5 ) const
    {
        return d_data[d_size.index( i1, i2, i3, i4, i5 )];
    }

    /*!
     * Access the desired element
     * @param i             The element index
     */
    ARRAY_INLINE TYPE &operator[]( size_t i ) { return d_data[i]; }

    /*!
     * Access the desired element
     * @param i             The element index
     */
    ARRAY_INLINE const TYPE &operator[]( size_t i ) const { return d_data[i]; }


    /*!
     * Access the desired element as a raw pointer
     * @param i             The global index
     */
    ARRAY_INLINE TYPE *ptr( size_t i ) { return i >= d_size.length() ? nullptr : &d_data[i]; }

    /*!
     * Access the desired element as a raw pointer
     * @param i             The global index
     */
    ARRAY_INLINE const TYPE *ptr( size_t i ) const
    {
        return i >= d_size.length() ? nullptr : &d_data[i];
    }

    //! Get iterator to beginning of data
    ARRAY_INLINE TYPE *begin() { return d_data; }

    //! Get iterator to beginning of data
    ARRAY_INLINE const TYPE *begin() const { return d_data; }

    //! Get iterator to beginning of data
    ARRAY_INLINE TYPE *end() { return d_data + d_size.length(); }

    //! Get iterator to beginning of data
    ARRAY_INLINE const TYPE *end() const { return d_data + d_size.length(); }

    //! Return the pointer to the raw data
    std::shared_ptr<TYPE> getPtr() { return d_ptr; }

    //! Return the pointer to the raw data
    std::shared_ptr<const TYPE> getPtr() const { return d_ptr; }

    //! Return the pointer to the raw data
    ARRAY_INLINE TYPE *data() { return d_data; }

    //! Return the pointer to the raw data
    ARRAY_INLINE const TYPE *data() const { return d_data; }


public: // Operator overloading
    //! Check if two matrices are equal
    // Equality means the dimensions and data have to be identical
    bool operator==( const Array &rhs ) const;

    //! Check if two matrices are not equal
    bool operator!=( const Array &rhs ) const { return !this->operator==( rhs ); }

    //! Add another array
    Array &operator+=( const Array &rhs );

    //! Subtract another array
    Array &operator-=( const Array &rhs );

    //! Add a scalar
    Array &operator+=( const TYPE &rhs );

    //! Subtract a scalar
    Array &operator-=( const TYPE &rhs );


public: // Math operations
    //! Concatenates the arrays along the dimension dim.
    static Array cat( const std::vector<Array> &x, int dim = 0 );

    //! Concatenates the arrays along the dimension dim.
    static Array cat( const std::initializer_list<Array> &x, int dim = 0 );

    //! Concatenates the arrays along the dimension dim.
    static Array cat( size_t N_array, const Array *x, int dim );

    //! Concatenates a given array with the current array
    void cat( const Array &x, int dim = 0 );

    //! Initialize the array with random values (defined from the function table)
    void rand();

    //! Return true if NaNs are present
    bool NaNs() const;

    //! Return the smallest value
    TYPE min() const;

    //! Return the largest value
    TYPE max() const;

    //! Return the sum of all elements
    TYPE sum() const;

    //! Return the mean of all elements
    TYPE mean() const;

    //! Return the min of all elements in a given direction
    Array min( int dir ) const;

    //! Return the max of all elements in a given direction
    Array max( int dir ) const;

    //! Return the sum of all elements in a given direction
    Array sum( int dir ) const;

    //! Return the smallest value
    TYPE min( const std::vector<size_t> &index ) const;

    //! Return the largest value
    TYPE max( const std::vector<size_t> &index ) const;

    //! Return the sum of all elements
    TYPE sum( const std::vector<size_t> &index ) const;

    //! Return the mean of all elements
    TYPE mean( const std::vector<size_t> &index ) const;

    //! Return the smallest value
    TYPE min( const std::vector<Range<size_t>> &index ) const;

    //! Return the largest value
    TYPE max( const std::vector<Range<size_t>> &index ) const;

    //! Return the sum of all elements
    TYPE sum( const std::vector<Range<size_t>> &index ) const;

    //! Return the mean of all elements
    TYPE mean( const std::vector<Range<size_t>> &index ) const;

    //! Find all elements that match the operator
    std::vector<size_t> find( const TYPE &value,
                              std::function<bool( const TYPE &, const TYPE & )> compare ) const;

    //! Find all elements that match the operator (if element is not found return -1)
    int64_t findFirst( const TYPE &value,
                       std::function<bool( const TYPE &, const TYPE & )> compare ) const;

    //! Find all elements that match the operator (if element is not found return -1)
    int64_t findLast( const TYPE &value,
                      std::function<bool( const TYPE &, const TYPE & )> compare ) const;

    //! Print an array
    void
    print( std::ostream &os, const std::string &name = "A", const std::string &prefix = "" ) const;

    //! Transpose an array
    Array reverseDim() const;

    //! Replicate an array a given number of times in each direction
    Array repmat( const ArraySize &N ) const;

    //! Coarsen an array using the given filter
    Array coarsen( const Array &filter ) const;

    //! Coarsen an array using the given filter
    Array coarsen( const ArraySize &ratio, std::function<TYPE( const Array & )> filter ) const;

    /*!
     * Perform a element-wise operation y = f(x)
     * @param[in] fun           The function operation
     * @param[in] x             The input array
     */
    static Array transform( std::function<TYPE( const TYPE & )> fun, const Array &x );

    /*!
     * Perform a element-wise operation z = f(x,y)
     * @param[in] fun           The function operation
     * @param[in] x             The first array
     * @param[in] y             The second array
     */
    static Array transform( std::function<TYPE( const TYPE &, const TYPE & )> fun,
                            const Array &x,
                            const Array &y );

    /*!
     * axpby operation: this = alpha*x + beta*this
     * @param[in] alpha         alpha
     * @param[in] x             x
     * @param[in] beta          beta
     */
    void axpby( const TYPE &alpha, const Array &x, const TYPE &beta );

    /*!
     * Linear interpolation
     * @param[in] x             Position as a decimal index
     */
    inline TYPE interp( const std::vector<double> &x ) const { return interp( x.data() ); }

    /*!
     * Linear interpolation
     * @param[in] x             Position as a decimal index
     */
    TYPE interp( const double *x ) const;

    /**
     * \fn equals (Array & const rhs, TYPE tol )
     * \brief  Determine if two Arrays are equal using an absolute tolerance
     * \param[in] rhs Vector to compare to
     * \param[in] tol Tolerance of comparison
     * \return  True iff \f$||\mathit{rhs} - x||_\infty < \mathit{tol}\f$
     */
    bool equals( const Array &rhs, const TYPE &tol = 0.000001 ) const;

public:
    //! Return the number of bytes required to pack the data
    size_t packSize() const;
    //! Pack the data to a buffer
    size_t pack( std::byte * ) const;
    //! Unpack the data from a buffer
    size_t unpack( const std::byte * );

private:
    scalarAllocator_t d_alloc;
    bool d_isCopyable;           // Can the array be copied
    bool d_isFixedSize;          // Can the array be resized
    ArraySize d_size;            // Size of each dimension
    TYPE *d_data;                // Raw pointer to data in array
    std::shared_ptr<TYPE> d_ptr; // Shared pointer to data in array
    void allocate( const ArraySize &N );

private:
    inline void checkSubsetIndex( const std::vector<Range<size_t>> &range ) const;
    inline std::vector<Range<size_t>> convert( const std::vector<size_t> &index ) const;
    static inline void getSubsetArrays( const std::vector<Range<size_t>> &range,
                                        std::array<size_t, 5> &first,
                                        std::array<size_t, 5> &last,
                                        std::array<size_t, 5> &inc,
                                        std::array<size_t, 5> &N );

private:
    class Deleter
    {
    public:
        Deleter( const scalarAllocator_t &alloc, size_t N ) : d_alloc( alloc ), d_N( N ) {}
        void operator()( TYPE *p )
        {
            if constexpr ( !std::is_trivially_copyable<TYPE>::value ) {
                for ( size_t i = 0; i < d_N; ++i )
                    p[i].~TYPE();
            }
            d_alloc.deallocate( p, d_N );
        }

    private:
        scalarAllocator_t d_alloc;
        size_t d_N;
    };
};


/********************************************************
 *  ostream operator                                     *
 ********************************************************/
inline std::ostream &operator<<( std::ostream &out, const AMP::ArraySize &s )
{
    out << "[" << s[0];
    for ( size_t i = 1; i < s.ndim(); i++ )
        out << "," << s[i];
    out << "]";
    return out;
}
template<class TYPE>
inline std::ostream &operator<<( std::ostream &out, const AMP::Range<TYPE> &r )
{
    out << r.i << ':' << r.k << ':' << r.j;
    return out;
}


/********************************************************
 *  Math operations                                      *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
inline Array<TYPE, FUN, Allocator> operator+( const Array<TYPE, FUN, Allocator> &a,
                                              const Array<TYPE, FUN, Allocator> &b )
{
    auto c = a;
    c += b;
    return c;
}
template<class TYPE, class FUN, class Allocator>
inline Array<TYPE, FUN, Allocator> operator-( const Array<TYPE, FUN, Allocator> &a,
                                              const Array<TYPE, FUN, Allocator> &b )
{
    auto c = a;
    c -= b;
    return c;
}
template<class TYPE, class FUN, class Allocator>
inline Array<TYPE, FUN, Allocator> operator*( const Array<TYPE, FUN, Allocator> &a,
                                              const Array<TYPE, FUN, Allocator> &b )
{
    Array<TYPE, FUN, Allocator> c( FunctionTable<TYPE>::multiplySize( a.size(), b.size() ) );
    FUN::multiply( a.size(), a.data(), b.size(), b.data(), c.size(), c.data() );
    return c;
}
template<class TYPE, class FUN, class Allocator>
inline Array<TYPE, FUN, Allocator> operator*( const Array<TYPE, FUN, Allocator> &a,
                                              const std::vector<TYPE> &b )
{
    Array<TYPE, FUN, Allocator> b2;
    b2.viewRaw( { b.size() }, const_cast<TYPE *>( b.data() ) );
    Array<TYPE, FUN, Allocator> c( FunctionTable<TYPE>::multiplySize( a.size(), b.size() ) );
    FUN::multiply( a.size(), a.data(), b2.size(), b2.data(), c.size(), c.data() );
    return c;
}
template<class TYPE, class FUN, class Allocator>
inline Array<TYPE, FUN, Allocator> operator*( const TYPE &a, const Array<TYPE, FUN, Allocator> &b )
{
    auto c = b;
    c.scale( a );
    return c;
}
template<class TYPE, class FUN, class Allocator>
inline Array<TYPE, FUN, Allocator> operator*( const Array<TYPE, FUN, Allocator> &a, const TYPE &b )
{
    auto c = a;
    c.scale( b );
    return c;
}


/********************************************************
 *  Copy array                                           *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
template<class TYPE2>
inline void AMP::Array<TYPE, FUN, Allocator>::copy( const TYPE2 *data )
{
    if constexpr ( std::is_same_v<TYPE, TYPE2> ) {
        std::copy( data, data + d_size.length(), d_data );
    } else {
        for ( size_t i = 0; i < d_size.length(); i++ )
            d_data[i] = static_cast<TYPE>( data[i] );
    }
}
template<class TYPE, class FUN, class Allocator>
template<class TYPE2>
inline void AMP::Array<TYPE, FUN, Allocator>::copyTo( TYPE2 *data ) const
{
    if constexpr ( std::is_same_v<TYPE, TYPE2> ) {
        std::copy( d_data, d_data + d_size.length(), data );
    } else {
        for ( size_t i = 0; i < d_size.length(); i++ )
            data[i] = static_cast<TYPE2>( d_data[i] );
    }
}


} // namespace AMP


#endif
