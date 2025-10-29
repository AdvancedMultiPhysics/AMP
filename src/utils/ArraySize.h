#ifndef included_AMP_ArraySizeClass
#define included_AMP_ArraySizeClass


#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "AMP/utils/Memory.h"

#if defined( __CUDA_ARCH__ )
    #include <cuda.h>
    #define HOST_DEVICE __host__ __device__
#elif defined( __HIP_DEVICE_COMPILE__ )
    #include "hip/hip_runtime.h"
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif
#if defined( __NVCC__ )
    #define CONSTEXPR HOST_DEVICE constexpr
    #define CONSTEXPR_IF
#elif defined( __HIPCC__ )
    #define CONSTEXPR HOST_DEVICE constexpr
    #define CONSTEXPR_IF constexpr
#else
    #define CONSTEXPR HOST_DEVICE constexpr
    #define CONSTEXPR_IF constexpr
#endif
#if defined( __GNUC__ ) || defined( __clang__ )
    #define ARRAY_INLINE HOST_DEVICE inline __attribute__( ( always_inline ) )
#elif defined( _MSC_VER )
    #define ARRAY_INLINE HOST_DEVICE __forceinline
#else
    #define ARRAY_INLINE HOST_DEVICE inline
#endif
#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG ) && !defined( __NVCC__ ) && \
    !defined( __HIP_DEVICE_COMPILE__ )
    #define CHECK_ARRAY_LENGTH( i, length )                              \
        do {                                                             \
            if ( i >= length )                                           \
                throw std::out_of_range( "Index exceeds array bounds" ); \
        } while ( 0 )
    #define ARRAY_INSIST( test, msg )           \
        do {                                    \
            if ( !( test ) )                    \
                throw std::out_of_range( msg ); \
        } while ( 0 )
#else
    #define CHECK_ARRAY_LENGTH( i, length ) \
        do {                                \
        } while ( 0 )
    #define ARRAY_INSIST( test, msg ) \
        do {                          \
        } while ( 0 )
#endif

namespace AMP {


// Forward declerations
template<class TYPE>
class FunctionTable;
template<class TYPE, class FUN = FunctionTable<TYPE>, class Allocator = AMP::HostAllocator<void>>
class Array;


//! Simple range class
template<class TYPE = size_t>
class Range final
{
public:
    //! Empty constructor
    CONSTEXPR Range() : i( 0 ), j( 0 ), k( 1 ) {}

    /*!
     * Create a range i:k:j (or i:j)
     * @param i_            Starting value
     * @param j_            Ending value
     * @param k_            Increment value
     */
    CONSTEXPR Range( const TYPE &i_, const TYPE &j_, const TYPE &k_ = 1 )
        : i( i_ ), j( j_ ), k( k_ )
    {
    }

    //! Get the number of values in the range
    CONSTEXPR size_t size() const
    {
        if CONSTEXPR_IF ( std::is_integral_v<TYPE> ) {
            int64_t tmp = ( static_cast<int64_t>( j ) - static_cast<int64_t>( i ) ) /
                          static_cast<int64_t>( k );
            return tmp + 1;
        } else if CONSTEXPR_IF ( std::is_floating_point_v<TYPE> ) {
            double tmp = static_cast<double>( ( j - i ) ) / static_cast<double>( k );
            return static_cast<size_t>( floor( tmp + 1e-12 ) + 1 );
        } else if CONSTEXPR_IF ( std::is_same_v<TYPE, std::complex<float>> ||
                                 std::is_same_v<TYPE, std::complex<double>> ) {
            double tmp = std::real( ( j - i ) / ( k ) );
            return static_cast<size_t>( floor( tmp + 1e-12 ) + 1 );
        } else {
            static_assert( !std::is_integral_v<TYPE>, "Unsupported type for range" );
            return 0;
        }
    }

    //! Get the ith values in the range
    CONSTEXPR TYPE get( size_t index ) const
    {
        if CONSTEXPR_IF ( std::is_integral_v<TYPE> ) {
            return static_cast<TYPE>( i + index * k );
        } else if CONSTEXPR_IF ( std::is_floating_point_v<TYPE> ) {
            return static_cast<TYPE>( k * ( i / k + index ) );
        } else if CONSTEXPR_IF ( std::is_same_v<TYPE, std::complex<float>> ||
                                 std::is_same_v<TYPE, std::complex<double>> ) {
            return static_cast<TYPE>( k * ( i / k + static_cast<TYPE>( index ) ) );
        } else {
            static_assert( !std::is_integral_v<TYPE>, "Unsupported type for range" );
            return static_cast<TYPE>( index );
        }
    }

public:
    TYPE i, j, k;
};


//! Simple class to store the array dimensions
class ArraySize final
{
public:
    //! Empty constructor
    CONSTEXPR ArraySize() : d_ndim( 1 ), d_length( 0 ), d_N{ 0, 1, 1, 1, 1 } {}

    /*!
     * Create the vector size
     * @param N1            Number of elements in the first dimension
     */
    CONSTEXPR ArraySize( size_t N1 ) : d_ndim( 1 ), d_length( N1 ), d_N{ N1, 1, 1, 1, 1 } {}

    /*!
     * Create the vector size
     * @param N1            Number of elements in the first dimension
     * @param N2            Number of elements in the second dimension
     */
    CONSTEXPR ArraySize( size_t N1, size_t N2 )
        : d_ndim( 2 ), d_length( N1 * N2 ), d_N{ N1, N2, 1, 1, 1 }
    {
    }

    /*!
     * Create the vector size
     * @param N1            Number of elements in the first dimension
     * @param N2            Number of elements in the second dimension
     * @param N3            Number of elements in the third dimension
     */
    CONSTEXPR ArraySize( size_t N1, size_t N2, size_t N3 )
        : d_ndim( 3 ), d_length( N1 * N2 * N3 ), d_N{ N1, N2, N3, 1, 1 }
    {
    }

    /*!
     * Create the vector size
     * @param N1            Number of elements in the first dimension
     * @param N2            Number of elements in the second dimension
     * @param N3            Number of elements in the third dimension
     * @param N4            Number of elements in the fourth dimension
     */
    CONSTEXPR ArraySize( size_t N1, size_t N2, size_t N3, size_t N4 )
        : d_ndim( 4 ), d_length( N1 * N2 * N3 * N4 ), d_N{ N1, N2, N3, N4, 1 }
    {
    }

    /*!
     * Create the vector size
     * @param N1            Number of elements in the first dimension
     * @param N2            Number of elements in the second dimension
     * @param N3            Number of elements in the third dimension
     * @param N4            Number of elements in the fourth dimension
     * @param N5            Number of elements in the fifth dimension
     */
    CONSTEXPR ArraySize( size_t N1, size_t N2, size_t N3, size_t N4, size_t N5 )
        : d_ndim( 5 ), d_length( N1 * N2 * N3 * N4 * N5 ), d_N{ N1, N2, N3, N4, N5 }
    {
    }

    /*!
     * Create from initializer list
     * @param N             Size of the array
     * @param ndim          Number of dimensions
     */
    template<class TYPE>
    CONSTEXPR ArraySize( std::initializer_list<TYPE> N, int ndim = -1 )
        : d_ndim( 0 ), d_length( 0 ), d_N{ 0, 1, 1, 1, 1 }
    {
        if ( ndim >= 0 ) {
            d_ndim = ndim;
        } else {
            int ndim2 = 0;
            for ( auto it = N.begin(); ndim2 < (int) N.size(); ndim2++, ++it ) {
                if ( *it == (TYPE) -1 )
                    break;
            }
            d_ndim = std::min<int>( ndim2, maxDim() );
        }
        if ( d_ndim == 0 )
            return;
        ARRAY_INSIST( d_ndim <= maxDim(), "Maximum number of dimensions exceeded" );
        auto it  = N.begin();
        d_length = 1;
        for ( size_t i = 0; i < d_ndim; i++, ++it ) {
            d_N[i] = *it;
            d_length *= d_N[i];
        }
    }

    /*!
     * Create from raw pointer
     * @param ndim          Number of dimensions
     * @param dims          Dimensions
     */
    CONSTEXPR ArraySize( size_t ndim, const size_t *dims )
        : d_ndim( ndim ), d_length( 0 ), d_N{ 0, 1, 1, 1, 1 }
    {
        ARRAY_INSIST( d_ndim <= maxDim(), "Maximum number of dimensions exceeded" );
        for ( size_t i = 0; i < ndim; i++ )
            d_N[i] = dims[i];
        d_length = 1;
        for ( unsigned long i : d_N )
            d_length *= i;
        if ( d_ndim == 0 )
            d_length = 0;
    }

    /*!
     * Create from std::array
     * @param N             Size of the array
     */
    template<std::size_t NDIM>
    CONSTEXPR ArraySize( const std::array<size_t, NDIM> &N ) : ArraySize( NDIM, N.data() )
    {
    }

    /*!
     * Create from std::vector
     * @param N             Size of the array
     */
    inline ArraySize( const std::vector<size_t> &N ) : ArraySize( N.size(), N.data() ) {}

    /*!
     * Access the ith dimension
     * @param i             Index to access
     */
    CONSTEXPR size_t operator[]( size_t i ) const { return d_N[i]; }

    //! Return the number of dimensions
    CONSTEXPR uint8_t ndim() const { return d_ndim; }

    //! Return the number of dimensions
    CONSTEXPR size_t size() const { return d_ndim; }

    //! Return the total number of elements in the array
    CONSTEXPR size_t length() const { return d_length; }

    //! Resize the dimension
    CONSTEXPR void resize( uint8_t dim, size_t N )
    {
        ARRAY_INSIST( dim < d_ndim, "Invalid dimension" );
        d_N[dim] = N;
        d_length = 1;
        for ( unsigned long i : d_N )
            d_length *= i;
    }

    /*!
     * Reshape the Array so that the number of dimensions is the
     *    max of ndim and the largest dim>1.
     * @param ndim          Desired number of dimensions
     */
    constexpr void setNdim( uint8_t ndim ) { d_ndim = std::max( ndim, d_ndim ); }

    //! Returns an iterator to the beginning
    CONSTEXPR const size_t *begin() const { return d_N; }

    //! Returns an iterator to the end
    CONSTEXPR const size_t *end() const { return d_N + d_ndim; }

    // Check if two array sizes are equal
    CONSTEXPR bool operator==( const ArraySize &rhs ) const
    {
        return d_ndim == rhs.d_ndim && d_N[0] == rhs.d_N[0] && d_N[1] == rhs.d_N[1] &&
               d_N[2] == rhs.d_N[2] && d_N[3] == rhs.d_N[3] && d_N[4] == rhs.d_N[4];
    }

    // Check if two array sizes are equal (ignoring the dimension)
    CONSTEXPR bool approxEqual( const ArraySize &rhs ) const
    {
        return ( length() == 0 && rhs.length() == 0 ) || memcmp( d_N, rhs.d_N, sizeof( d_N ) ) == 0;
    }

    //! Check if two matrices are not equal
    CONSTEXPR bool operator!=( const ArraySize &rhs ) const { return !operator==( rhs ); }

    //! Maximum supported dimension
    CONSTEXPR static uint8_t maxDim() { return 5u; }

    //! Get the index
    CONSTEXPR size_t index( size_t i ) const
    {
        CHECK_ARRAY_LENGTH( i, d_length );
        return i;
    }

    //! Get the index
    CONSTEXPR size_t index( size_t i1, size_t i2 ) const
    {
        size_t index = i1 + i2 * d_N[0];
        CHECK_ARRAY_LENGTH( index, d_length );
        return index;
    }

    //! Get the index
    CONSTEXPR size_t index( size_t i1, size_t i2, size_t i3 ) const
    {
        size_t index = i1 + d_N[0] * ( i2 + d_N[1] * i3 );
        CHECK_ARRAY_LENGTH( index, d_length );
        return index;
    }

    //! Get the index
    CONSTEXPR size_t index( size_t i1, size_t i2, size_t i3, size_t i4 ) const
    {
        size_t index = i1 + d_N[0] * ( i2 + d_N[1] * ( i3 + d_N[2] * i4 ) );
        CHECK_ARRAY_LENGTH( index, d_length );
        return index;
    }

    //! Get the index
    CONSTEXPR size_t index( size_t i1, size_t i2, size_t i3, size_t i4, size_t i5 ) const
    {
        size_t index = i1 + d_N[0] * ( i2 + d_N[1] * ( i3 + d_N[2] * ( i4 + d_N[3] * i5 ) ) );
        CHECK_ARRAY_LENGTH( index, d_length );
        return index;
    }

    //! Get the index
    constexpr size_t index( const std::array<size_t, 5> &i ) const
    {
        size_t j = 0;
        for ( size_t m = 0, N = 1; m < 5; m++ ) {
            j += i[m] * N;
            N *= d_N[m];
        }
        return j;
    }

    //! Get the index
    constexpr size_t index( std::initializer_list<size_t> i ) const
    {
        size_t N = 1;
        size_t j = 0;
        size_t m = 0;
        for ( size_t k : i ) {
            j += k * N;
            N *= d_N[m++];
        }
        return j;
    }

    //! Convert the index to ijk values
    constexpr std::array<size_t, 5> ijk( size_t index ) const
    {
        CHECK_ARRAY_LENGTH( index, d_length );
        size_t i0 = index % d_N[0];
        index     = index / d_N[0];
        size_t i1 = index % d_N[1];
        index     = index / d_N[1];
        size_t i2 = index % d_N[2];
        index     = index / d_N[2];
        size_t i3 = index % d_N[3];
        index     = index / d_N[3];
        return { i0, i1, i2, i3, index };
    }

    //! Convert the index to ijk values
    CONSTEXPR void ijk( size_t index, size_t *x ) const
    {
        CHECK_ARRAY_LENGTH( index, d_length );
        x[0]  = index % d_N[0];
        index = index / d_N[0];
        x[1]  = index % d_N[1];
        index = index / d_N[1];
        x[2]  = index % d_N[2];
        index = index / d_N[2];
        x[3]  = index % d_N[3];
        index = index / d_N[3];
        x[4]  = index;
    }

private:
    uint8_t d_ndim;
    size_t d_length;
    size_t d_N[5];
};


// Function to concatenate dimensions of two array sizes
CONSTEXPR ArraySize cat( const ArraySize &x, const ArraySize &y )
{
    ARRAY_INSIST( x.ndim() + y.ndim() <= ArraySize::maxDim(),
                  "Maximum number of dimensions exceeded" );
    size_t N[ArraySize::maxDim()] = { 0 };
    for ( int i = 0; i < x.ndim(); i++ )
        N[i] = x[i];
    for ( int i = 0; i < y.ndim(); i++ )
        N[i + x.ndim()] = y[i];
    return ArraySize( x.ndim() + y.ndim(), N );
}


// Function to pop the first dimension and return the resulting array size
CONSTEXPR ArraySize pop( const ArraySize &x )
{
    int ndim      = x.ndim() == 1 ? 1 : x.ndim() - 1;
    size_t dims[] = { x[1], x[2], x[3], x[4] };
    return AMP::ArraySize( ndim, dims );
}


// Remove singleton dimensions
constexpr ArraySize squeeze( const ArraySize &x )
{
    int Nd      = 0;
    size_t N[5] = { 1 };
    for ( size_t i = 0; i < x.maxDim(); i++ ) {
        if ( x[i] != 1 )
            N[Nd++] = x[i];
    }
    return ArraySize( std::max( Nd, 1 ), N );
}


// Operator overloads
CONSTEXPR ArraySize operator*( size_t v, const ArraySize &x )
{
    size_t N[5] = { v * x[0], v * x[1], v * x[2], v * x[3], v * x[4] };
    return ArraySize( x.ndim(), N );
}
CONSTEXPR ArraySize operator*( const ArraySize &x, size_t v )
{
    size_t N[5] = { v * x[0], v * x[1], v * x[2], v * x[3], v * x[4] };
    return ArraySize( x.ndim(), N );
}
CONSTEXPR ArraySize operator*( const ArraySize &x, const ArraySize &y )
{
    int ndim    = std::max( x.ndim(), y.ndim() );
    size_t N[5] = { x[0] * y[0], x[1] * y[1], x[2] * y[2], x[3] * y[3], x[4] * y[4] };
    return ArraySize( ndim, N );
}
CONSTEXPR ArraySize operator-( const ArraySize &x, size_t v )
{
    size_t N[5] = { x[0] - v, x[1] - v, x[2] - v, x[3] - v, x[4] - v };
    return ArraySize( x.ndim(), N );
}
CONSTEXPR ArraySize operator+( const ArraySize &x, size_t v )
{
    size_t N[5] = { x[0] + v, x[1] + v, x[2] + v, x[3] + v, x[4] + v };
    return ArraySize( x.ndim(), N );
}
CONSTEXPR ArraySize operator+( size_t v, const ArraySize &x )
{
    size_t N[5] = { x[0] + v, x[1] + v, x[2] + v, x[3] + v, x[4] + v };
    return ArraySize( x.ndim(), N );
}


} // namespace AMP


#endif
