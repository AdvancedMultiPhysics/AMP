#ifndef included_AMP_ArrayClass_hpp
#define included_AMP_ArrayClass_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMP_MPI_pack.hpp"
#include "AMP/utils/Array.h"
#include "AMP/utils/FunctionTable.h"
#include "AMP/utils/UtilityMacros.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>


/********************************************************
 *  Macros to help instantiate functions                 *
 ********************************************************/
// clang-format off
#ifndef AMP_CXX_STANDARD
    #define AMP_CXX_STANDARD 17
#endif
#if AMP_CXX_STANDARD >= 20
#define instantiateDestructor(TYPE,FUN,A) template AMP::Array<TYPE,FUN,A>::~Array();
#elif defined( __NVCOMPILER )
#define instantiateDestructor(TYPE,FUN,A) template AMP::Array<TYPE,FUN,A>::~Array();
#else
#define instantiateDestructor(TYPE,FUN,A) template AMP::Array<TYPE,FUN,A>::~Array<TYPE,FUN,A>();
#endif
#define instantiateArrayConstructors2( TYPE, FUN, A )                                    \
    instantiateDestructor(TYPE,FUN,A)                                                    \
    template AMP::Array<TYPE,FUN,A>::Array();                                            \
    template AMP::Array<TYPE,FUN,A>::Array( const AMP::ArraySize&, TYPE const* );        \
    template AMP::Array<TYPE,FUN,A>::Array( size_t );                                    \
    template AMP::Array<TYPE,FUN,A>::Array( size_t, size_t );                            \
    template AMP::Array<TYPE,FUN,A>::Array( size_t, size_t, size_t );                    \
    template AMP::Array<TYPE,FUN,A>::Array( size_t, size_t, size_t, size_t );            \
    template AMP::Array<TYPE,FUN,A>::Array( size_t, size_t, size_t, size_t, size_t );    \
    template AMP::Array<TYPE,FUN,A>::Array( std::initializer_list<TYPE> );               \
    template AMP::Array<TYPE,FUN,A>::Array( std::initializer_list<std::initializer_list<TYPE>> ); \
    template AMP::Array<TYPE,FUN,A>::Array( const AMP::Array<TYPE,FUN,A>& );             \
    template AMP::Array<TYPE,FUN,A>::Array( AMP::Array<TYPE,FUN,A>&& );                  \
    template AMP::Array<TYPE,FUN,A> AMP::Array<TYPE,FUN,A>::view();                      \
    template void AMP::Array<TYPE,FUN,A>::allocate( const AMP::ArraySize& );             \
    template void AMP::Array<TYPE,FUN,A>::reshape( const AMP::ArraySize& );              \
    template std::unique_ptr<const AMP::Array<TYPE,FUN,A>>                               \
        AMP::Array<TYPE,FUN,A>::constView( const AMP::ArraySize&, const std::shared_ptr<TYPE const>& ); \
    template void AMP::Array<TYPE,FUN,A>::viewRaw( const AMP::ArraySize&, TYPE*, bool, bool ); \
    template void AMP::Array<TYPE,FUN,A>::view2( const AMP::ArraySize&, std::shared_ptr<TYPE> ); \
    template AMP::Array<TYPE,FUN,A>& AMP::Array<TYPE,FUN,A>::operator=( const AMP::Array<TYPE,FUN,A>& ); \
    template AMP::Array<TYPE,FUN,A>& AMP::Array<TYPE,FUN,A>::operator=( AMP::Array<TYPE,FUN,A>&& ); \
    template TYPE* AMP::Array<TYPE,FUN,A>::data();                                       \
    template TYPE const* AMP::Array<TYPE,FUN,A>::data() const;                           \
    template void AMP::Array<TYPE,FUN,A>::resize( size_t );                              \
    template void AMP::Array<TYPE,FUN,A>::resize( size_t, size_t );                      \
    template void AMP::Array<TYPE,FUN,A>::resize( size_t, size_t, size_t );              \
    template void AMP::Array<TYPE,FUN,A>::resize( AMP::ArraySize const& );               \
    template void AMP::Array<TYPE,FUN,A>::clear();                                       \
    template int AMP::Array<TYPE,FUN,A>::ndim() const;                                   \
    template const AMP::ArraySize& AMP::Array<TYPE,FUN,A>::size() const;                 \
    template size_t AMP::Array<TYPE,FUN,A>::size( int ) const;                           \
    template size_t AMP::Array<TYPE,FUN,A>::length() const;                              \
    template bool AMP::Array<TYPE,FUN,A>::empty() const;                                 \
    template AMP::Array<TYPE,FUN,A>& AMP::Array<TYPE,FUN,A>::operator=( const std::vector<TYPE>& ); \
    template bool AMP::Array<TYPE,FUN,A>::operator==( const AMP::Array<TYPE,FUN,A>& ) const
#define instantiateArrayConstructors( TYPE )                              \
    instantiateArrayConstructors2( TYPE, AMP::FunctionTable<TYPE>, std::allocator<void> )
#define PACK_UNPACK_ARRAY( TYPE )                                         \
    template size_t AMP::packSize( const AMP::Array<TYPE> & );            \
    template size_t AMP::pack( const AMP::Array<TYPE> &, std::byte * );   \
    template size_t AMP::unpack( AMP::Array<TYPE> &, const std::byte * )
#define PACK_UNPACK_ARRAY2( TYPE )                                        \
    template size_t AMP::Array<TYPE>::packSize() const;                   \
    template size_t AMP::Array<TYPE>::pack( std::byte * ) const;          \
    template size_t AMP::Array<TYPE>::unpack( const std::byte * )
// clang-format on


namespace AMP {


/********************************************************
 *  External instantiations                              *
 ********************************************************/
extern template class Array<bool>;
extern template class Array<char>;
extern template class Array<uint8_t>;
extern template class Array<uint16_t>;
extern template class Array<uint32_t>;
extern template class Array<uint64_t>;
extern template class Array<int8_t>;
extern template class Array<int16_t>;
extern template class Array<int32_t>;
extern template class Array<int64_t>;
extern template class Array<double>;
extern template class Array<float>;


/********************************************************
 *  Constructors                                         *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array()
    : d_isCopyable( true ), d_isFixedSize( false ), d_data( nullptr )
{
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( const ArraySize &N, const TYPE *data )
    : d_isCopyable( true ), d_isFixedSize( false )
{
    allocate( N );
    if ( data )
        copy( data );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( size_t N ) : d_isCopyable( true ), d_isFixedSize( false )
{
    allocate( ArraySize( N ) );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( size_t N_rows, size_t N_cols )
    : d_isCopyable( true ), d_isFixedSize( false )
{
    allocate( ArraySize( N_rows, N_cols ) );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( size_t N1, size_t N2, size_t N3 )
    : d_isCopyable( true ), d_isFixedSize( false )
{
    allocate( ArraySize( N1, N2, N3 ) );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( size_t N1, size_t N2, size_t N3, size_t N4 )
    : d_isCopyable( true ), d_isFixedSize( false )
{
    allocate( ArraySize( N1, N2, N3, N4 ) );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( size_t N1, size_t N2, size_t N3, size_t N4, size_t N5 )
    : d_isCopyable( true ), d_isFixedSize( false )
{
    allocate( ArraySize( N1, N2, N3, N4, N5 ) );
}
template<class TYPE, class FUN, class Allocator>
template<typename U, typename>
Array<TYPE, FUN, Allocator>::Array( const Range<U> &range )
    : d_isCopyable( true ), d_isFixedSize( false )
{
    size_t N = range.size();
    allocate( { N } );
    for ( size_t i = 0; i < N; i++ )
        d_data[i] = range.get( i );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( std::string str ) : d_isCopyable( true ), d_isFixedSize( false )
{
    allocate( 0 );
    if ( (int) std::count( str.begin(), str.end(), ' ' ) == (int) str.length() ) {
        // Empty string
        return;
    }
    // Remove unnecessary whitespace
    while ( str.front() == ' ' )
        str.erase( 0, 1 );
    while ( str.back() == ' ' )
        str.resize( str.length() - 1 );
    while ( str.find( ',' ) != std::string::npos )
        str[str.find( ',' )] = ' ';
    while ( str.find( "  " ) != std::string::npos )
        str.replace( str.find( "  " ), 2, " " );
    // Check if the string is of the format [...]
    if ( str.front() == '[' && str.back() == ']' ) {
        str.erase( 0, 1 );
        str.resize( str.length() - 1 );
        *this = Array( str );
        return;
    }
    // Check if we are dealing with a 2D array
    if ( str.find( ';' ) != std::string::npos ) {
        size_t i1 = 0;
        size_t i2 = str.find( ';' );
        std::vector<Array> x;
        while ( i2 > i1 ) {
            auto tmp = str.substr( i1, i2 - i1 );
            x.emplace_back( Array( tmp ) );
            i1 = i2 + 1;
            i2 = str.find( ';', i1 + 1 );
            if ( i2 == std::string::npos )
                i2 = str.length();
        }
        for ( auto &y : x )
            y.reshape( { 1, y.length() } );
        *this = cat( x, 0 );
        return;
    }
    // Begin parsing the array constructor
    size_t i1 = 0;
    size_t i2 = str.find( ' ' );
    std::vector<TYPE> data;
    while ( i2 > i1 ) {
        auto tmp = str.substr( i1, i2 - i1 );
        int type = std::count( tmp.begin(), tmp.end(), ':' );
        if ( type == 0 ) {
            data.push_back( std::stod( tmp ) );
        } else if ( type == 1 ) {
            size_t k = tmp.find( ':' );
            TYPE x1  = std::stod( tmp.substr( 0, k ) );
            TYPE x2  = std::stod( tmp.substr( k + 1 ) );
            Range<TYPE> f( x1, x2 );
            for ( size_t i = 0; i < f.size(); i++ )
                data.push_back( f.get( i ) );
        } else if ( type == 2 ) {
            size_t k1 = tmp.find( ':' );
            size_t k2 = tmp.find( ':', k1 + 1 );
            TYPE x1   = std::stod( tmp.substr( 0, k1 ) );
            TYPE dx   = std::stod( tmp.substr( k1 + 1, k2 - k1 - 1 ) );
            TYPE x2   = std::stod( tmp.substr( k2 + 1 ) );
            Range<TYPE> f( x1, x2, dx );
            for ( size_t i = 0; i < f.size(); i++ )
                data.push_back( f.get( i ) );
        } else {
            throw std::logic_error( "Failed to parse string constructor: " + str );
        }
        i1 = i2;
        i2 = str.find( ' ', i1 + 1 );
        if ( i2 == std::string::npos )
            i2 = str.length();
    }
    allocate( data.size() );
    if constexpr ( std::is_same_v<TYPE, bool> ) {
        for ( size_t i = 0; i < data.size(); i++ )
            d_data[i] = data[i];
    } else {
        copy( data.data() );
    }
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( std::initializer_list<TYPE> x )
    : d_isCopyable( true ), d_isFixedSize( false )
{
    allocate( { x.size() } );
    auto it = x.begin();
    for ( size_t i = 0; i < x.size(); ++i, ++it )
        d_data[i] = *it;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( std::initializer_list<std::initializer_list<TYPE>> x )
    : d_isCopyable( true ), d_isFixedSize( false )
{
    size_t Nx = x.size();
    size_t Ny = 0;
    for ( const auto y : x )
        Ny = std::max<size_t>( Ny, y.size() );
    allocate( { Nx, Ny } );
    auto itx = x.begin();
    for ( size_t i = 0; i < x.size(); ++i, ++itx ) {
        auto ity = itx->begin();
        for ( size_t j = 0; j < itx->size(); ++j, ++ity ) {
            d_data[i + j * Nx] = *ity;
        }
    }
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::allocate( const ArraySize &N )
{
    if ( d_isFixedSize )
        throw std::logic_error( "Array cannot be resized" );
    d_size = N;
    d_data = nullptr;
    d_ptr.reset();
    size_t length = d_size.length();
    if ( length > 0 ) {
        try {
            d_data = d_alloc.allocate( length );
            if constexpr ( !std::is_trivially_copyable<TYPE>::value ) {
                for ( size_t i = 0; i < length; ++i )
                    new ( d_data + i ) TYPE();
            }
            d_ptr.reset( d_data, Deleter( d_alloc, length ) );
        } catch ( ... ) {
            throw std::logic_error( "Failed to allocate array" );
        }
    }
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( const Array &rhs )
    : d_alloc( rhs.d_alloc ), d_isCopyable( true ), d_isFixedSize( false )
{
    if ( !rhs.d_isCopyable )
        throw std::logic_error( "Array cannot be copied" );
    allocate( rhs.size() );
    copy( rhs.d_data );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::Array( Array &&rhs )
    : d_alloc( rhs.d_alloc ),
      d_isCopyable( rhs.d_isCopyable ),
      d_isFixedSize( rhs.d_isFixedSize ),
      d_size( rhs.d_size ),
      d_data( rhs.d_data ),
      d_ptr( std::move( rhs.d_ptr ) )
{
    rhs.d_size = ArraySize();
    rhs.d_data = nullptr;
    rhs.d_ptr  = nullptr;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> &Array<TYPE, FUN, Allocator>::operator=( const Array &rhs )
{
    if ( this == &rhs )
        return *this;
    if ( !rhs.d_isCopyable )
        throw std::logic_error( "Array cannot be copied" );
    d_alloc = rhs.d_alloc;
    allocate( rhs.size() );
    copy( rhs.d_data );
    return *this;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> &Array<TYPE, FUN, Allocator>::operator=( Array &&rhs )
{
    if ( this == &rhs )
        return *this;
    d_alloc       = rhs.d_alloc;
    d_isCopyable  = rhs.d_isCopyable;
    d_isFixedSize = rhs.d_isFixedSize;
    d_size        = rhs.d_size;
    d_data        = rhs.d_data;
    d_ptr         = std::move( rhs.d_ptr );
    rhs.d_size    = ArraySize();
    rhs.d_data    = nullptr;
    rhs.d_ptr     = nullptr;
    return *this;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> &Array<TYPE, FUN, Allocator>::operator=( const std::vector<TYPE> &rhs )
{
    allocate( ArraySize( rhs.size() ) );
    if constexpr ( std::is_same_v<TYPE, bool> ) {
        for ( size_t i = 0; i < rhs.size(); i++ )
            d_data[i] = rhs[i];
    } else {
        copy( rhs.data() );
    }
    return *this;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>::~Array()
{
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::clear()
{
    d_isCopyable  = true;
    d_isFixedSize = false;
    allocate( {} );
}


/********************************************************
 *  Copy/move values from one array to another (resize)  *
 ********************************************************/
template<class TYPE>
static inline void moveValues( const ArraySize &N1, const ArraySize &N2, TYPE *data1, TYPE *data2 )
{
    for ( size_t i5 = 0; i5 < std::min( N1[4], N2[4] ); i5++ ) {
        for ( size_t i4 = 0; i4 < std::min( N1[3], N2[3] ); i4++ ) {
            for ( size_t i3 = 0; i3 < std::min( N1[2], N2[2] ); i3++ ) {
                for ( size_t i2 = 0; i2 < std::min( N1[1], N2[1] ); i2++ ) {
                    for ( size_t i1 = 0; i1 < std::min( N1[0], N2[0] ); i1++ ) {
                        size_t index1 = N1.index( i1, i2, i3, i4, i5 );
                        size_t index2 = N2.index( i1, i2, i3, i4, i5 );
                        data2[index2] = std::move( data1[index1] );
                    }
                }
            }
        }
    }
}
template<class TYPE>
static inline void
copyValues( const ArraySize &N1, const ArraySize &N2, const TYPE *data1, TYPE *data2 )
{
    for ( size_t i5 = 0; i5 < std::min( N1[4], N2[4] ); i5++ ) {
        for ( size_t i4 = 0; i4 < std::min( N1[3], N2[3] ); i4++ ) {
            for ( size_t i3 = 0; i3 < std::min( N1[2], N2[2] ); i3++ ) {
                for ( size_t i2 = 0; i2 < std::min( N1[1], N2[1] ); i2++ ) {
                    for ( size_t i1 = 0; i1 < std::min( N1[0], N2[0] ); i1++ ) {
                        size_t index1 = N1.index( i1, i2, i3, i4, i5 );
                        size_t index2 = N2.index( i1, i2, i3, i4, i5 );
                        data2[index2] = data1[index1];
                    }
                }
            }
        }
    }
}


/********************************************************
 *  Resize the array                                     *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::resize( size_t N )
{
    resize( ArraySize( N ) );
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::resize( size_t N_row, size_t N_col )
{
    resize( ArraySize( N_row, N_col ) );
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::resize( size_t N1, size_t N2, size_t N3 )
{
    resize( ArraySize( N1, N2, N3 ) );
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::resize( const ArraySize &N )
{
    // Check if the array actually changed size
    bool equal = true;
    for ( size_t i = 0; i < ArraySize::maxDim(); i++ )
        equal = equal && N[i] == d_size[i];
    if ( equal ) {
        d_size = N;
        return;
    }
    // Store the old data
    auto N0    = d_size;
    auto data0 = d_ptr;
    // Allocate new data
    allocate( N );
    // Copy the old values
    if ( N.length() > 0 && d_size.length() > 0 ) {
        if ( data0.use_count() <= 1 ) {
            // We own the data, use std:move
            moveValues( N0, N, data0.get(), d_data );
        } else if constexpr ( std::is_copy_constructible_v<TYPE> ) {
            // We do not own the data, copy
            copyValues( N0, N, data0.get(), d_data );
        } else {
            throw std::logic_error( "No copy constructor" );
        }
    }
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::resizeDim( int dim, size_t N, const TYPE &value )
{
    if ( dim < 0 || dim > d_size.ndim() )
        throw std::out_of_range( "Invalid dimension" );
    size_t N0 = d_size[dim];
    auto size = d_size;
    size.resize( dim, N );
    resize( size );
    size_t n1 = 1, n2 = 1;
    for ( int d = 0; d < dim; d++ )
        n1 *= size[d];
    for ( size_t d = dim + 1; d < size.ndim(); d++ )
        n2 *= size[d];
    for ( size_t k = 0; k < n2; k++ ) {
        for ( size_t j = N0; j < N; j++ ) {
            for ( size_t i = 0; i < n1; i++ ) {
                d_data[i + j * n1 + k * n1 * N] = value;
            }
        }
    }
}


/********************************************************
 *  Reshape the array                                     *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::reshape( const ArraySize &N )
{
    if ( N.length() != d_size.length() )
        throw std::logic_error( "reshape is not allowed to change the array size" );
    d_size = N;
}


/********************************************************
 *  Subset the array                                     *
 ********************************************************/
// Helper function to check subset indices
template<class TYPE, class FUN, class Allocator>
inline void
Array<TYPE, FUN, Allocator>::checkSubsetIndex( const std::vector<Range<size_t>> &range ) const
{
    bool test = (int) range.size() == d_size.ndim();
    for ( size_t d = 0; d < range.size(); d++ )
        test = test && range[d].j < d_size[d];
    if ( !test )
        throw std::logic_error( "indices for subset are invalid" );
}
template<class TYPE, class FUN, class Allocator>
std::vector<Range<size_t>>
Array<TYPE, FUN, Allocator>::convert( const std::vector<size_t> &index ) const
{
    std::vector<Range<size_t>> range( d_size.ndim() );
    if ( index.size() % 2 != 0 || static_cast<int>( index.size() / 2 ) < d_size.ndim() )
        throw std::logic_error( "indices for subset are invalid" );
    for ( int d = 0; d < d_size.ndim(); d++ )
        range[d] = Range<size_t>( index[2 * d + 0], index[2 * d + 1] );
    return range;
}
// Helper function to return dimensions for the subset array
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::getSubsetArrays( const std::vector<Range<size_t>> &index,
                                                   std::array<size_t, 5> &first,
                                                   std::array<size_t, 5> &last,
                                                   std::array<size_t, 5> &inc,
                                                   std::array<size_t, 5> &N )
{
    first.fill( 0 );
    last.fill( 0 );
    inc.fill( 1 );
    N.fill( 1 );
    size_t ndim = index.size();
    for ( size_t d = 0; d < ndim; d++ ) {
        first[d] = index[d].i;
        last[d]  = index[d].j;
        inc[d]   = index[d].k;
        N[d]     = ( last[d] - first[d] + inc[d] ) / inc[d];
    }
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>
Array<TYPE, FUN, Allocator>::subset( const std::vector<Range<size_t>> &index ) const
{
    // Get the subset indicies
    checkSubsetIndex( index );
    std::array<size_t, 5> first, last, inc, N1;
    getSubsetArrays( index, first, last, inc, N1 );
    ArraySize S1( d_size.ndim(), N1.data() );
    // Create the new array
    Array<TYPE, FUN, Allocator> subset_array( S1 );
    // Fill the new array
    static_assert( ArraySize::maxDim() == 5, "Not programmed for more than 5 dimensions" );
    TYPE *subset_data = subset_array.data();
    for ( size_t i4 = first[4], k1 = 0; i4 <= last[4]; i4 += inc[4] ) {
        for ( size_t i3 = first[3]; i3 <= last[3]; i3 += inc[3] ) {
            for ( size_t i2 = first[2]; i2 <= last[2]; i2 += inc[2] ) {
                for ( size_t i1 = first[1]; i1 <= last[1]; i1 += inc[1] ) {
                    for ( size_t i0 = first[0]; i0 <= last[0]; i0 += inc[0], k1++ ) {
                        size_t k2       = d_size.index( i0, i1, i2, i3, i4 );
                        subset_data[k1] = d_data[k2];
                    }
                }
            }
        }
    }
    return subset_array;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>
Array<TYPE, FUN, Allocator>::subset( const std::vector<size_t> &index ) const
{
    auto range = convert( index );
    return subset( range );
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::copySubset( const std::vector<Range<size_t>> &index,
                                              const Array<TYPE, FUN, Allocator> &subset )
{
    // Get the subset indices
    checkSubsetIndex( index );
    std::array<size_t, 5> first, last, inc, N1;
    getSubsetArrays( index, first, last, inc, N1 );
    // Copy the sub-array
    static_assert( ArraySize::maxDim() == 5, "Not programmed for more than 5 dimensions" );
    const TYPE *src_data = subset.data();
    for ( size_t i4 = first[4], k1 = 0; i4 <= last[4]; i4 += inc[4] ) {
        for ( size_t i3 = first[3]; i3 <= last[3]; i3 += inc[3] ) {
            for ( size_t i2 = first[2]; i2 <= last[2]; i2 += inc[2] ) {
                for ( size_t i1 = first[1]; i1 <= last[1]; i1 += inc[1] ) {
                    for ( size_t i0 = first[0]; i0 <= last[0]; i0 += inc[0], k1++ ) {
                        size_t k2  = d_size.index( i0, i1, i2, i3, i4 );
                        d_data[k2] = src_data[k1];
                    }
                }
            }
        }
    }
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::addSubset( const std::vector<Range<size_t>> &index,
                                             const Array<TYPE, FUN, Allocator> &subset )
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        // Get the subset indices
        checkSubsetIndex( index );
        std::array<size_t, 5> first, last, inc, N1;
        getSubsetArrays( index, first, last, inc, N1 );
        // add the sub-array
        static_assert( ArraySize::maxDim() == 5, "Not programmed for more than 5 dimensions" );
        for ( size_t i4 = first[4], k1 = 0; i4 <= last[4]; i4 += inc[4] ) {
            for ( size_t i3 = first[3]; i3 <= last[3]; i3 += inc[3] ) {
                for ( size_t i2 = first[2]; i2 <= last[2]; i2 += inc[2] ) {
                    for ( size_t i1 = first[1]; i1 <= last[1]; i1 += inc[1] ) {
                        for ( size_t i0 = first[0]; i0 <= last[0]; i0 += inc[0], k1++ ) {
                            size_t k2 = d_size.index( i0, i1, i2, i3, i4 );
                            d_data[k2] += subset.d_data[k1];
                        }
                    }
                }
            }
        }
    } else {
        AMP_ERROR( "addSubset not supported for non-arithmetic types" );
    }
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::copySubset( const std::vector<size_t> &index,
                                              const Array<TYPE, FUN, Allocator> &subset )
{
    auto range = convert( index );
    copySubset( range, subset );
}

template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::addSubset( const std::vector<size_t> &index,
                                             const Array<TYPE, FUN, Allocator> &subset )
{
    auto range = convert( index );
    addSubset( range, subset );
}


/********************************************************
 *  Operator overloading                                 *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
bool Array<TYPE, FUN, Allocator>::operator==( const Array &rhs ) const
{
    if ( this == &rhs )
        return true;
    if ( d_size != rhs.d_size )
        return false;
    bool match = true;
    for ( size_t i = 0; i < d_size.length(); i++ )
        match = match && d_data[i] == rhs.d_data[i];
    return match;
}


/********************************************************
 *  Get a view of an C array                             *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
std::unique_ptr<Array<TYPE, FUN, Allocator>>
Array<TYPE, FUN, Allocator>::view( const ArraySize &N, std::shared_ptr<TYPE> data )
{
    auto array    = std::make_unique<Array<TYPE, FUN, Allocator>>();
    array->d_size = N;
    array->d_ptr  = data;
    array->d_data = array->d_ptr.get();
    return array;
}
template<class TYPE, class FUN, class Allocator>
std::unique_ptr<const Array<TYPE, FUN, Allocator>>
Array<TYPE, FUN, Allocator>::constView( const ArraySize &N,
                                        std::shared_ptr<const TYPE> const &data )
{
    auto array    = std::make_unique<Array<TYPE, FUN, Allocator>>();
    array->d_size = N;
    array->d_ptr  = std::const_pointer_cast<TYPE>( data );
    array->d_data = array->d_ptr.get();
    return array;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::view()
{
    Array<TYPE, FUN, Allocator> x;
    x.d_alloc       = d_alloc;
    x.d_isCopyable  = d_isCopyable;
    x.d_isFixedSize = d_isFixedSize;
    x.d_size        = d_size;
    x.d_data        = d_data;
    x.d_ptr         = d_ptr;
    return x;
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::view2( Array<TYPE, FUN, Allocator> &src )
{
    view2( src.size(), src.getPtr() );
    d_data = src.d_data;
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::view2( const ArraySize &N, std::shared_ptr<TYPE> data )
{
    d_size = N;
    d_ptr  = data;
    d_data = d_ptr.get();
}


/********************************************************
 *  Basic functions                                      *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::swap( Array &other )
{
    // check that dimensions match
    if ( d_size != other.d_size )
        throw std::logic_error( "length of arrays do not match" );
    // swap the data
    std::swap( d_data, other.d_data );
    std::swap( d_ptr, other.d_ptr );
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::pow( const Array<TYPE, FUN, Allocator> &baseArray,
                                       const TYPE &exp )
{
    // not insisting on the shapes being the same
    // but insisting on the total size being the same
    if ( d_size.length() != baseArray.length() )
        throw std::logic_error( "length of arrays do not match" );

    const auto base_data = baseArray.data();
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        for ( size_t i = 0; i < d_size.length(); i++ )
            d_data[i] = std::pow( base_data[i], exp );
    } else {
        throw std::logic_error( "pow not supported for non-arithmetic types" );
    }
}


/********************************************************
 *  Replicate the array                                  *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::repmat( const ArraySize &N_rep ) const
{
    auto N2 = N_rep * d_size;
    std::array<size_t, 5> N1, Nr;
    N1.fill( 1 );
    Nr.fill( 1 );
    for ( size_t d = 0; d < N_rep.ndim(); d++ ) {
        N1[d] = d_size[d];
        Nr[d] = N_rep[d];
    }
    Array<TYPE, FUN, Allocator> y( N2 );
    static_assert( ArraySize::maxDim() <= 5, "Not programmed for dimensions > 5" );
    TYPE *y2 = y.data();
    for ( size_t i4 = 0; i4 < N1[4]; i4++ ) {
        for ( size_t i3 = 0; i3 < N1[3]; i3++ ) {
            for ( size_t i2 = 0; i2 < N1[2]; i2++ ) {
                for ( size_t i1 = 0; i1 < N1[1]; i1++ ) {
                    for ( size_t i0 = 0; i0 < N1[0]; i0++ ) {
                        TYPE x = operator()( i0, i1, i2, i3, i4 );
                        for ( size_t j4 = 0; j4 < Nr[4]; j4++ ) {
                            for ( size_t j3 = 0; j3 < Nr[3]; j3++ ) {
                                for ( size_t j2 = 0; j2 < Nr[2]; j2++ ) {
                                    for ( size_t j1 = 0; j1 < Nr[1]; j1++ ) {
                                        for ( size_t j0 = 0; j0 < Nr[0]; j0++ ) {
                                            size_t index = N2.index( i0 + j0 * N1[0],
                                                                     i1 + j1 * N1[1],
                                                                     i2 + j2 * N1[2],
                                                                     i3 + j3 * N1[3],
                                                                     i4 + j4 * N1[4] );
                                            y2[index]    = x;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return y;
}


/********************************************************
 *  Simple math operations                               *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
bool Array<TYPE, FUN, Allocator>::NaNs() const
{
    if constexpr ( std::is_floating_point_v<TYPE> ) {
        bool test = false;
        for ( size_t i = 0; i < d_size.length(); i++ )
            test = test || d_data[i] != d_data[i];
        return test;
    } else {
        return false;
    }
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::mean( void ) const
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        TYPE x = sum() / d_size.length();
        return x;
    } else {
        AMP_ERROR( "mean is not supported for non-arithmetic types" );
    }
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::min( int dir ) const
{
    auto size_ans = d_size;
    size_ans.resize( dir, 1 );
    Array<TYPE, FUN, Allocator> ans( size_ans );
    size_t N1 = 1, N2 = 1, N3 = 1;
    for ( int d = 0; d < std::min<int>( dir, d_size.ndim() ); d++ )
        N1 *= d_size[d];
    N2 = d_size[dir];
    for ( size_t d = dir + 1; d < d_size.ndim(); d++ )
        N3 *= d_size[d];
    TYPE *data2 = ans.d_data;
    for ( size_t i3 = 0; i3 < N3; i3++ ) {
        for ( size_t i1 = 0; i1 < N1; i1++ ) {
            TYPE x = d_data[i1 + i3 * N1 * N2];
            for ( size_t i2 = 0; i2 < N2; i2++ )
                x = std::min( x, d_data[i1 + i2 * N1 + i3 * N1 * N2] );
            data2[i1 + i3 * N1] = x;
        }
    }
    return ans;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::max( int dir ) const
{
    auto size_ans = d_size;
    size_ans.resize( dir, 1 );
    Array<TYPE, FUN, Allocator> ans( size_ans );
    size_t N1 = 1, N2 = 1, N3 = 1;
    for ( int d = 0; d < std::min<int>( dir, d_size.ndim() ); d++ )
        N1 *= d_size[d];
    N2 = d_size[dir];
    DISABLE_WARNINGS // Suppress false array subscript is above array bounds
        for ( size_t d = dir + 1; d < d_size.ndim(); d++ ) N3 *= d_size[d];
    ENABLE_WARNINGS // Enable warnings
        TYPE *data2 = ans.d_data;
    for ( size_t i3 = 0; i3 < N3; i3++ ) {
        for ( size_t i1 = 0; i1 < N1; i1++ ) {
            TYPE x = d_data[i1 + i3 * N1 * N2];
            for ( size_t i2 = 0; i2 < N2; i2++ )
                x = std::max( x, d_data[i1 + i2 * N1 + i3 * N1 * N2] );
            data2[i1 + i3 * N1] = x;
        }
    }
    return ans;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::sum( int dir ) const
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        auto size_ans = d_size;
        size_ans.resize( dir, 1 );
        Array<TYPE, FUN, Allocator> ans( size_ans );
        size_t N1 = 1, N2 = 1, N3 = 1;
        for ( int d = 0; d < std::min<int>( dir, d_size.ndim() ); d++ )
            N1 *= d_size[d];
        N2 = d_size[dir];
        DISABLE_WARNINGS
        for ( size_t d = dir + 1; d < d_size.ndim(); d++ )
            N3 *= d_size[d];
        ENABLE_WARNINGS
        TYPE *data2 = ans.d_data;
        for ( size_t i3 = 0; i3 < N3; i3++ ) {
            for ( size_t i1 = 0; i1 < N1; i1++ ) {
                TYPE x = 0;
                for ( size_t i2 = 0; i2 < N2; i2++ )
                    x += d_data[i1 + i2 * N1 + i3 * N1 * N2];
                data2[i1 + i3 * N1] = x;
            }
        }
        return ans;
    } else {
        AMP_ERROR( "sum is not supported for non-arithmetic types" );
    }
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::min( const std::vector<Range<size_t>> &range ) const
{
    // Get the subset indicies
    checkSubsetIndex( range );
    std::array<size_t, 5> first, last, inc, N1;
    getSubsetArrays( range, first, last, inc, N1 );
    static_assert( ArraySize::maxDim() <= 5, "Function programmed for more than 5 dimensions" );
    TYPE x = std::numeric_limits<TYPE>::max();
    for ( size_t i4 = first[4]; i4 <= last[4]; i4 += inc[4] ) {
        for ( size_t i3 = first[3]; i3 <= last[3]; i3 += inc[3] ) {
            for ( size_t i2 = first[2]; i2 <= last[2]; i2 += inc[2] ) {
                for ( size_t i1 = first[1]; i1 <= last[1]; i1 += inc[1] ) {
                    for ( size_t i0 = first[0]; i0 <= last[0]; i0 += inc[0] ) {
                        size_t k1 = d_size.index( i0, i1, i2, i3, i4 );
                        x         = std::min( x, d_data[k1] );
                    }
                }
            }
        }
    }
    return x;
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::max( const std::vector<Range<size_t>> &range ) const
{
    // Get the subset indicies
    checkSubsetIndex( range );
    std::array<size_t, 5> first, last, inc, N1;
    getSubsetArrays( range, first, last, inc, N1 );
    static_assert( ArraySize::maxDim() <= 5, "Function programmed for more than 5 dimensions" );
    TYPE x = std::numeric_limits<TYPE>::min();
    for ( size_t i4 = first[4]; i4 <= last[4]; i4 += inc[4] ) {
        for ( size_t i3 = first[3]; i3 <= last[3]; i3 += inc[3] ) {
            for ( size_t i2 = first[2]; i2 <= last[2]; i2 += inc[2] ) {
                for ( size_t i1 = first[1]; i1 <= last[1]; i1 += inc[1] ) {
                    for ( size_t i0 = first[0]; i0 <= last[0]; i0 += inc[0] ) {
                        size_t k1 = d_size.index( i0, i1, i2, i3, i4 );
                        x         = std::max( x, d_data[k1] );
                    }
                }
            }
        }
    }
    return x;
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::sum( const std::vector<Range<size_t>> &range ) const
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        // Get the subset indicies
        checkSubsetIndex( range );
        std::array<size_t, 5> first, last, inc, N1;
        getSubsetArrays( range, first, last, inc, N1 );
        static_assert( ArraySize::maxDim() <= 5, "Function programmed for more than 5 dimensions" );
        TYPE x = 0;
        for ( size_t i4 = first[4]; i4 <= last[4]; i4 += inc[4] ) {
            for ( size_t i3 = first[3]; i3 <= last[3]; i3 += inc[3] ) {
                for ( size_t i2 = first[2]; i2 <= last[2]; i2 += inc[2] ) {
                    for ( size_t i1 = first[1]; i1 <= last[1]; i1 += inc[1] ) {
                        for ( size_t i0 = first[0]; i0 <= last[0]; i0 += inc[0] ) {
                            size_t k1 = d_size.index( i0, i1, i2, i3, i4 );
                            x += d_data[k1];
                        }
                    }
                }
            }
        }
        return x;
    } else {
        AMP_ERROR( "sum not supported for non-arithmetic types" );
    }
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::mean( const std::vector<Range<size_t>> &range ) const
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        // Get the subset indicies
        checkSubsetIndex( range );
        std::array<size_t, 5> first, last, inc, N1;
        getSubsetArrays( range, first, last, inc, N1 );
        static_assert( ArraySize::maxDim() <= 5, "Function programmed for more than 5 dimensions" );
        size_t n = 1;
        for ( auto &d : N1 )
            n *= d;
        TYPE x = sum( range ) / n;
        return x;
    } else {
        AMP_ERROR( "mean not supported for non-arithmetic types" );
    }
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::min( const std::vector<size_t> &index ) const
{
    auto range = convert( index );
    return min( range );
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::max( const std::vector<size_t> &index ) const
{
    auto range = convert( index );
    return max( range );
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::sum( const std::vector<size_t> &index ) const
{
    auto range = convert( index );
    return sum( range );
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::mean( const std::vector<size_t> &index ) const
{
    auto range = convert( index );
    return mean( range );
}


/********************************************************
 *  Find all elements that match the given operation     *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
std::vector<size_t>
Array<TYPE, FUN, Allocator>::find( const TYPE &value,
                                   std::function<bool( const TYPE &, const TYPE & )> compare ) const
{
    std::vector<size_t> result;
    result.reserve( d_size.length() );
    for ( size_t i = 0; i < d_size.length(); i++ ) {
        if ( compare( d_data[i], value ) )
            result.push_back( i );
    }
    return result;
}
template<class TYPE, class FUN, class Allocator>
int64_t Array<TYPE, FUN, Allocator>::findFirst(
    const TYPE &value, std::function<bool( const TYPE &, const TYPE & )> compare ) const
{
    for ( size_t i = 0; i < d_size.length(); i++ ) {
        if ( compare( d_data[i], value ) )
            return i;
    }
    return -1;
}
template<class TYPE, class FUN, class Allocator>
int64_t Array<TYPE, FUN, Allocator>::findLast(
    const TYPE &value, std::function<bool( const TYPE &, const TYPE & )> compare ) const
{
    for ( size_t i = d_size.length(); i > 0; i-- ) {
        if ( compare( d_data[i - 1], value ) )
            return i - 1;
    }
    return -1;
}


/********************************************************
 *  Print an array to an output stream                   *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::print( std::ostream &os,
                                         const std::string &name,
                                         const std::string &prefix ) const
{
    if ( d_size.ndim() == 1 ) {
        for ( size_t i = 0; i < d_size[0]; i++ )
            os << prefix << name << "[" << i << "] = " << d_data[i] << std::endl;
    } else if ( d_size.ndim() == 2 ) {
        if ( !name.empty() )
            os << prefix << name << ":" << std::endl;
        for ( size_t i = 0; i < d_size[0]; i++ ) {
            for ( size_t j = 0; j < d_size[1]; j++ )
                os << prefix << "  " << operator()( i, j );
            os << std::endl;
        }
    } else {
        throw std::logic_error( "Not programmed for this dimension" );
    }
}


/********************************************************
 *  Reverse dimensions (transpose)                       *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::reverseDim() const
{
    if ( ndim() == 1 )
        return *this;
    auto &N = d_size;
    Array<TYPE, FUN, Allocator> y;
    if ( ndim() == 2 ) {
        ArraySize N2( N[1], N[0] );
        y.allocate( N2 );
        for ( size_t i0 = 0; i0 < d_size[0]; i0++ ) {
            for ( size_t i1 = 0; i1 < d_size[1]; i1++ ) {
                y( i1, i0 ) = d_data[d_size.index( i0, i1 )];
            }
        }
    } else if ( ndim() == 3 ) {
        ArraySize N2( N[2], N[1], N[0] );
        y.allocate( N2 );
        for ( size_t i0 = 0; i0 < d_size[0]; i0++ ) {
            for ( size_t i1 = 0; i1 < d_size[1]; i1++ ) {
                for ( size_t i2 = 0; i2 < d_size[2]; i2++ ) {
                    y( i2, i1, i0 ) = d_data[d_size.index( i0, i1, i2 )];
                }
            }
        }
    } else if ( ndim() == 4 ) {
        ArraySize N2( N[3], N[2], N[1], N[0] );
        y.allocate( N2 );
        for ( size_t i0 = 0; i0 < d_size[0]; i0++ ) {
            for ( size_t i1 = 0; i1 < d_size[1]; i1++ ) {
                for ( size_t i2 = 0; i2 < d_size[2]; i2++ ) {
                    for ( size_t i3 = 0; i3 < d_size[3]; i3++ ) {
                        y( i3, i2, i1, i0 ) = d_data[d_size.index( i0, i1, i2, i3 )];
                    }
                }
            }
        }
    } else if ( ndim() == 5 ) {
        ArraySize N2( N[4], N[3], N[2], N[1], N[0] );
        y.allocate( N2 );
        for ( size_t i0 = 0; i0 < d_size[0]; i0++ ) {
            for ( size_t i1 = 0; i1 < d_size[1]; i1++ ) {
                for ( size_t i2 = 0; i2 < d_size[2]; i2++ ) {
                    for ( size_t i3 = 0; i3 < d_size[3]; i3++ ) {
                        for ( size_t i4 = 0; i4 < d_size[4]; i4++ ) {
                            y( i4, i3, i2, i1, i0 ) = d_data[d_size.index( i0, i1, i2, i3, i4 )];
                        }
                    }
                }
            }
        }
    }
    return y;
}


/********************************************************
 *  Coarsen the array                                    *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>
Array<TYPE, FUN, Allocator>::coarsen( const Array<TYPE, FUN, Allocator> &filter ) const
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        auto S2 = size();
        for ( size_t i = 0; i < S2.size(); i++ ) {
            size_t s = S2[i] / filter.size( i );
            S2.resize( i, s );
            if ( S2[i] * filter.size( i ) != size( i ) )
                throw std::invalid_argument( "Array must be multiple of filter size" );
        }
        Array<TYPE, FUN, Allocator> y( S2 );
        if ( d_size.ndim() > 3 )
            throw std::logic_error( "Function not programmed for more than 3 dimensions" );
        const auto &Nh = filter.d_size;
        for ( size_t k1 = 0; k1 < y.d_size[2]; k1++ ) {
            for ( size_t j1 = 0; j1 < y.d_size[1]; j1++ ) {
                for ( size_t i1 = 0; i1 < y.d_size[0]; i1++ ) {
                    TYPE tmp = 0;
                    for ( size_t k2 = 0; k2 < Nh[2]; k2++ ) {
                        for ( size_t j2 = 0; j2 < Nh[1]; j2++ ) {
                            for ( size_t i2 = 0; i2 < Nh[0]; i2++ ) {
                                tmp += filter( i2, j2, k2 ) * operator()( i1 *Nh[0] + i2,
                                                                          j1 * Nh[1] + j2,
                                                                          k1 * Nh[2] + k2 );
                            }
                        }
                    }
                    y( i1, j1, k1 ) = tmp;
                }
            }
        }
        return y;
    } else {
        AMP_ERROR( "coarsen not supported for non-arithmetic types" );
    }
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::coarsen(
    const ArraySize &ratio,
    std::function<TYPE( const Array<TYPE, FUN, Allocator> & )> filter ) const
{
    if ( ratio.size() != d_size.ndim() )
        throw std::logic_error( "ratio size does not match ndim" );
    auto S2 = size();
    for ( size_t i = 0; i < S2.size(); i++ ) {
        S2.resize( i, S2[i] / ratio[i] );
        if ( S2[i] * ratio[i] != size( i ) )
            throw std::invalid_argument( "Array must be multiple of filter size" );
    }
    Array<TYPE, FUN, Allocator> tmp( ratio );
    Array<TYPE, FUN, Allocator> y( S2 );
    if ( d_size.ndim() > 3 )
        throw std::logic_error( "Function not programmed for more than 3 dimensions" );
    for ( size_t k1 = 0; k1 < y.d_size[2]; k1++ ) {
        for ( size_t j1 = 0; j1 < y.d_size[1]; j1++ ) {
            for ( size_t i1 = 0; i1 < y.d_size[0]; i1++ ) {
                for ( size_t k2 = 0; k2 < ratio[2]; k2++ ) {
                    for ( size_t j2 = 0; j2 < ratio[1]; j2++ ) {
                        for ( size_t i2 = 0; i2 < ratio[0]; i2++ ) {
                            tmp( i2, j2, k2 ) = operator()(
                                i1 *ratio[0] + i2, j1 * ratio[1] + j2, k1 * ratio[2] + k2 );
                        }
                    }
                }
                y( i1, j1, k1 ) = filter( tmp );
            }
        }
    }
    return y;
}


/********************************************************
 *  Concatenates the arrays                              *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::cat( const Array<TYPE, FUN, Allocator> &x, int dim )
{
    std::vector<Array<TYPE, FUN, Allocator>> tmp( 2 );
    tmp[0].view2( *this );
    tmp[1].view2( const_cast<Array<TYPE, FUN, Allocator> &>( x ) );
    *this = cat( tmp, dim );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::cat( const std::initializer_list<Array> &x,
                                                              int dim )
{
    return cat( x.size(), x.begin(), dim );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> Array<TYPE, FUN, Allocator>::cat( const std::vector<Array> &x, int dim )
{
    return cat( x.size(), x.data(), dim );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>
Array<TYPE, FUN, Allocator>::cat( size_t N_array, const Array *x, int dim )
{
    if ( N_array == 0 )
        return Array<TYPE, FUN, Allocator>();
    // Check that the dimensions match
    bool check = true;
    for ( size_t i = 1; i < N_array; i++ ) {
        check = check && x[i].ndim() == x[0].ndim();
        for ( int d = 0; d < x[0].ndim(); d++ )
            if ( d != dim )
                check = check && x[i].size( d ) == x[0].size( d );
    }
    if ( !check )
        throw std::logic_error( "Array dimensions do not match for concatenation" );
    // Create the output array
    auto size = x[0].d_size;
    for ( size_t i = 1; i < N_array; i++ )
        size.resize( dim, size[dim] + x[i].size( dim ) );
    Array<TYPE, FUN, Allocator> out( size );
    size_t N1 = 1;
    size_t N2 = size[dim];
    size_t N3 = 1;
    for ( int d = 0; d < dim; d++ )
        N1 *= size[d];
    for ( size_t d = dim + 1; d < size.ndim(); d++ )
        N3 *= size[d];
    TYPE *data = out.data();
    for ( size_t i = 0, i0 = 0; i < N_array; i++ ) {
        const TYPE *src = x[i].data();
        size_t N22      = x[i].size( dim );
        for ( size_t j2 = 0; j2 < N3; j2++ ) {
            for ( size_t i1 = 0; i1 < N22; i1++ ) {
                for ( size_t j1 = 0; j1 < N1; j1++ ) {
                    data[j1 + ( i1 + i0 ) * N1 + j2 * N1 * N2] = src[j1 + i1 * N1 + j2 * N1 * N22];
                }
            }
        }
        i0 += N22;
    }
    return out;
}


/********************************************************
 *  Interpolate                                          *
 ********************************************************/
template<class T>
inline constexpr bool is_compatible_double = std::is_floating_point_v<T> || std::is_integral_v<T>;
template<class TYPE>
inline TYPE Array_interp_1D( double x, int N, const TYPE *data )
{
    if constexpr ( is_compatible_double<TYPE> ) {
        int i = floor( x );
        i     = std::max( i, 0 );
        i     = std::min( i, N - 2 );
        return ( i + 1 - x ) * data[i] + ( x - i ) * data[i + 1];
    } else {
        throw std::logic_error( "Invalid conversion" );
    }
}
template<class TYPE>
inline TYPE Array_interp_2D( double x, double y, int Nx, int Ny, const TYPE *data )
{
    if constexpr ( is_compatible_double<TYPE> ) {
        int i             = floor( x );
        i                 = std::max( i, 0 );
        i                 = std::min( i, Nx - 2 );
        double dx         = x - i;
        double dx2        = 1.0 - dx;
        int j             = floor( y );
        j                 = std::max( j, 0 );
        j                 = std::min( j, Ny - 2 );
        double dy         = y - j;
        double dy2        = 1.0 - dy;
        const double f[4] = { (double) data[i + j * Nx],
                              (double) data[i + 1 + j * Nx],
                              (double) data[i + ( j + 1 ) * Nx],
                              (double) data[i + 1 + ( j + 1 ) * Nx] };
        return ( dx * f[1] + dx2 * f[0] ) * dy2 + ( dx * f[3] + dx2 * f[2] ) * dy;
    } else {
        throw std::logic_error( "Invalid conversion" );
    }
}
template<class TYPE>
inline TYPE
Array_interp_3D( double x, double y, double z, int Nx, int Ny, int Nz, const TYPE *data )
{
    if constexpr ( is_compatible_double<TYPE> ) {
        int i             = floor( x );
        i                 = std::max( i, 0 );
        i                 = std::min( i, Nx - 2 );
        double dx         = x - i;
        double dx2        = 1.0 - dx;
        int j             = floor( y );
        j                 = std::max( j, 0 );
        j                 = std::min( j, Ny - 2 );
        double dy         = y - j;
        double dy2        = 1.0 - dy;
        int k             = floor( z );
        k                 = std::max( k, 0 );
        k                 = std::min( k, Nz - 2 );
        double dz         = z - k;
        double dz2        = 1.0 - dz;
        const double f[8] = { (double) data[i + j * Nx + k * Nx * Ny],
                              (double) data[i + 1 + j * Nx + k * Nx * Ny],
                              (double) data[i + ( j + 1 ) * Nx + k * Nx * Ny],
                              (double) data[i + 1 + ( j + 1 ) * Nx + k * Nx * Ny],
                              (double) data[i + j * Nx + ( k + 1 ) * Nx * Ny],
                              (double) data[i + 1 + j * Nx + ( k + 1 ) * Nx * Ny],
                              (double) data[i + ( j + 1 ) * Nx + ( k + 1 ) * Nx * Ny],
                              (double) data[i + 1 + ( j + 1 ) * Nx + ( k + 1 ) * Nx * Ny] };
        double h0         = ( dx * f[1] + dx2 * f[0] ) * dy2 + ( dx * f[3] + dx2 * f[2] ) * dy;
        double h1         = ( dx * f[5] + dx2 * f[4] ) * dy2 + ( dx * f[7] + dx2 * f[6] ) * dy;
        return h0 * dz2 + h1 * dz;
    } else {
        throw std::logic_error( "Invalid conversion" );
    }
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::interp( const double *x ) const
{
    int ndim = 0, dim[5];
    double x2[5];
    for ( int d = 0; d < d_size.ndim(); d++ ) {
        if ( d_size[d] > 1 ) {
            x2[ndim]  = x[d];
            dim[ndim] = d_size[d];
            ndim++;
        }
    }
    TYPE f = 0;
    if ( ndim == 0 ) {
        // No data, do nothing
    } else if ( ndim == 1 ) {
        f = Array_interp_1D( x2[0], dim[0], d_data );
    } else if ( ndim == 2 ) {
        f = Array_interp_2D( x2[0], x2[1], dim[0], dim[1], d_data );
    } else if ( ndim == 3 ) {
        f = Array_interp_3D( x2[0], x2[1], x2[2], dim[0], dim[1], dim[2], d_data );
    } else {
        throw std::logic_error( "Not finished" );
    }
    return f;
}


/********************************************************
 *  Math operations (should call the Math class)         *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::rand()
{
    FUN::rand( d_size.length(), d_data );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> &
Array<TYPE, FUN, Allocator>::operator+=( const Array<TYPE, FUN, Allocator> &rhs )
{
    AMP_ASSERT( d_size == rhs.d_size );
    FUN::px( d_size.length(), rhs.d_data, d_data );
    return *this;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> &
Array<TYPE, FUN, Allocator>::operator-=( const Array<TYPE, FUN, Allocator> &rhs )
{
    AMP_ASSERT( d_size == rhs.d_size );
    FUN::mx( d_size.length(), rhs.d_data, d_data );
    return *this;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> &Array<TYPE, FUN, Allocator>::operator+=( const TYPE &rhs )
{
    FUN::px( d_size.length(), rhs, d_data );
    return *this;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator> &Array<TYPE, FUN, Allocator>::operator-=( const TYPE &rhs )
{
    FUN::mx( d_size.length(), rhs, d_data );
    return *this;
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::min() const
{
    return FUN::min( d_size.length(), d_data );
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::max() const
{
    return FUN::max( d_size.length(), d_data );
}
template<class TYPE, class FUN, class Allocator>
TYPE Array<TYPE, FUN, Allocator>::sum() const
{
    return FUN::sum( d_size.length(), d_data );
}
template<class TYPE, class FUN, class Allocator>
void Array<TYPE, FUN, Allocator>::axpby( const TYPE &alpha,
                                         const Array<TYPE, FUN, Allocator> &x,
                                         const TYPE &beta )
{
    AMP_ASSERT( x.d_size == d_size );
    FUN::axpby( alpha, d_size.length(), x.d_data, beta, d_data );
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>
Array<TYPE, FUN, Allocator>::transform( std::function<TYPE( const TYPE & )> fun,
                                        const Array<TYPE, FUN, Allocator> &x )
{
    Array<TYPE, FUN, Allocator> y( x.size() );
    FUN::transform( fun, x.length(), x.data(), y.data() );
    return y;
}
template<class TYPE, class FUN, class Allocator>
Array<TYPE, FUN, Allocator>
Array<TYPE, FUN, Allocator>::transform( std::function<TYPE( const TYPE &, const TYPE & )> fun,
                                        const Array<TYPE, FUN, Allocator> &x,
                                        const Array<TYPE, FUN, Allocator> &y )
{
    Array<TYPE, FUN, Allocator> z( x.size() );
    FUN::transform( fun, x.length(), x.data(), y.data(), z.data() );
    return z;
}
template<class TYPE, class FUN, class Allocator>
bool Array<TYPE, FUN, Allocator>::equals( const Array &rhs, const TYPE &tol ) const
{
    return FUN::equals( d_size.length(), d_data, rhs.d_data, tol );
}


} // namespace AMP


/********************************************************
 *  Pack/Unpack                                          *
 ********************************************************/
template<class TYPE, class FUN, class Allocator>
size_t AMP::Array<TYPE, FUN, Allocator>::packSize() const
{
    size_t N = 0;
    N += AMP::packSize( d_isCopyable );
    N += AMP::packSize( d_isFixedSize );
    N += AMP::packSize( d_size );
    if constexpr ( std::is_trivially_copyable_v<TYPE> ) {
        N += length() * sizeof( TYPE );
    } else {
        for ( size_t i = 0; i < length(); i++ )
            N += AMP::packSize( d_data[i] );
    }
    return N;
}
template<class TYPE, class FUN, class Allocator>
size_t AMP::Array<TYPE, FUN, Allocator>::pack( std::byte *buf ) const
{
    size_t N = 0;
    N += AMP::pack( d_isCopyable, &buf[N] );
    N += AMP::pack( d_isFixedSize, &buf[N] );
    N += AMP::pack( d_size, &buf[N] );
    if constexpr ( std::is_trivially_copyable_v<TYPE> ) {
        memcpy( &buf[N], d_data, length() * sizeof( TYPE ) );
        N += length() * sizeof( TYPE );
    } else {
        for ( size_t i = 0; i < length(); i++ )
            N += AMP::pack( d_data[i], &buf[N] );
    }
    return N;
}
template<class TYPE, class FUN, class Allocator>
size_t AMP::Array<TYPE, FUN, Allocator>::unpack( const std::byte *buf )
{
    size_t N = 0;
    bool copy, fixed;
    ArraySize size;
    N += AMP::unpack( copy, &buf[N] );
    N += AMP::unpack( fixed, &buf[N] );
    N += AMP::unpack( size, &buf[N] );
    resize( size );
    d_isCopyable  = copy;
    d_isFixedSize = fixed;
    if constexpr ( std::is_trivially_copyable_v<TYPE> ) {
        // clang-format off
        DISABLE_WARNINGS
        memcpy( d_data, &buf[N], length() * sizeof( TYPE ) );
        N += length() * sizeof( TYPE );
        ENABLE_WARNINGS
        // clang-format on
    } else {
        for ( size_t i = 0; i < length(); i++ )
            N += AMP::unpack( d_data[i], &buf[N] );
    }
    return N;
}


#endif
