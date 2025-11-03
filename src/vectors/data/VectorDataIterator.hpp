#ifndef included_AMP_VectorIterators_tmpl
#define included_AMP_VectorIterators_tmpl

#include "AMP/vectors/data/VectorData.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
template<typename TYPE>
VectorDataIterator<TYPE>::VectorDataIterator()
    : d_N_blocks( 0 ),
      d_CurBlock( 0 ),
      d_CurOffset( 0 ),
      d_pos( 0 ),
      d_size( 0 ),
      d_hashcode( 0 ),
      d_blockSize( nullptr ),
      d_data( nullptr ),
      d_typeHash( nullptr ),
      d_dataCast( 0 )
{
}
template<typename TYPE>
VectorDataIterator<TYPE>::VectorDataIterator( const VectorDataIterator &rhs )
    : d_N_blocks( rhs.d_N_blocks ),
      d_CurBlock( rhs.d_CurBlock ),
      d_CurOffset( rhs.d_CurOffset ),
      d_pos( rhs.d_pos ),
      d_size( rhs.d_size ),
      d_hashcode( rhs.d_hashcode ),
      d_blockSize( nullptr ),
      d_data( nullptr ),
      d_typeHash( nullptr ),
      d_dataCast( rhs.d_dataCast )
{
    d_data      = new TYPE *[d_N_blocks];
    d_blockSize = new size_t[d_N_blocks];
    for ( size_t i = 0; i < d_N_blocks; i++ ) {
        d_data[i]      = rhs.d_data[i];
        d_blockSize[i] = rhs.d_blockSize[i];
    }
    if ( rhs.d_typeHash ) {
        d_typeHash = new uint64_t[d_N_blocks];
        for ( size_t i = 0; i < d_N_blocks; i++ )
            d_typeHash[i] = rhs.d_typeHash[i];
    }
}
template<typename TYPE>
VectorDataIterator<TYPE>::VectorDataIterator( VectorDataIterator &&rhs )
    : d_N_blocks( rhs.d_N_blocks ),
      d_CurBlock( rhs.d_CurBlock ),
      d_CurOffset( rhs.d_CurOffset ),
      d_pos( rhs.d_pos ),
      d_size( rhs.d_size ),
      d_hashcode( rhs.d_hashcode ),
      d_blockSize( nullptr ),
      d_data( nullptr ),
      d_typeHash( nullptr ),
      d_dataCast( rhs.d_dataCast )
{
    std::swap( d_blockSize, rhs.d_blockSize );
    std::swap( d_data, rhs.d_data );
    std::swap( d_typeHash, rhs.d_typeHash );
}
template<typename TYPE>
VectorDataIterator<TYPE> &VectorDataIterator<TYPE>::operator=( const VectorDataIterator &rhs )
{
    if ( this == &rhs )
        return *this;
    delete[] d_data;
    delete[] d_blockSize;
    d_N_blocks  = rhs.d_N_blocks;
    d_CurBlock  = rhs.d_CurBlock;
    d_CurOffset = rhs.d_CurOffset;
    d_pos       = rhs.d_pos;
    d_size      = rhs.d_size;
    d_hashcode  = rhs.d_hashcode;
    d_data      = new TYPE *[d_N_blocks];
    d_blockSize = new size_t[d_N_blocks];
    for ( size_t i = 0; i < d_N_blocks; i++ ) {
        d_data[i]      = rhs.d_data[i];
        d_blockSize[i] = rhs.d_blockSize[i];
    }
    d_typeHash = nullptr;
    if ( rhs.d_typeHash ) {
        d_typeHash = new uint64_t[d_N_blocks];
        for ( size_t i = 0; i < d_N_blocks; i++ )
            d_typeHash[i] = rhs.d_typeHash[i];
    }
    return *this;
}
template<typename TYPE>
VectorDataIterator<TYPE> &VectorDataIterator<TYPE>::operator=( VectorDataIterator &&rhs )
{
    if ( this == &rhs )
        return *this;
    d_N_blocks  = rhs.d_N_blocks;
    d_CurBlock  = rhs.d_CurBlock;
    d_CurOffset = rhs.d_CurOffset;
    d_pos       = rhs.d_pos;
    d_size      = rhs.d_size;
    d_hashcode  = rhs.d_hashcode;
    d_typeHash  = nullptr;
    std::swap( d_blockSize, rhs.d_blockSize );
    std::swap( d_data, rhs.d_data );
    std::swap( d_typeHash, rhs.d_typeHash );
    d_dataCast = rhs.d_dataCast;
    return *this;
}
template<typename TYPE>
VectorDataIterator<TYPE>::VectorDataIterator( VectorData *vec, size_t position )
    : d_N_blocks( vec->numberOfDataBlocks() ),
      d_CurBlock( 0 ),
      d_CurOffset( 0 ),
      d_pos( 0 ),
      d_size( 0 ),
      d_hashcode( 0 ),
      d_blockSize( nullptr ),
      d_data( nullptr ),
      d_typeHash( nullptr ),
      d_dataCast( 0 )
{
    d_data       = new TYPE *[d_N_blocks];
    d_blockSize  = new size_t[d_N_blocks];
    bool allType = true;
    for ( size_t i = 0; i < d_N_blocks; i++ ) {
        d_data[i]      = vec->getRawDataBlock<TYPE>( i );
        d_blockSize[i] = vec->sizeOfDataBlock( i );
        d_size += d_blockSize[i];
        d_hashcode ^= std::hash<TYPE *>()( d_data[i] ) + 0x9e3779b9 + ( d_hashcode << 6 ) +
                      ( d_hashcode >> 2 );
        allType = allType && vec->isBlockType<TYPE>( i );
    }
    if ( !allType ) {
        if constexpr ( std::is_const_v<TYPE> ) {
            d_typeHash = new uint64_t[d_N_blocks];
            for ( size_t i = 0; i < d_N_blocks; i++ ) {
                d_typeHash[i] = vec->getType( i ).hash;
            }
        } else {
            AMP_ERROR( "Implicit conversion of non-const data is not allowed" );
        }
    }
    advance( position );
}
template<typename TYPE>
VectorDataIterator<TYPE>::~VectorDataIterator()
{
    delete[] d_blockSize;
    delete[] d_data;
    delete[] d_typeHash;
}


/****************************************************************
 * Function to return an iterator to the begining/end            *
 ****************************************************************/
template<typename TYPE>
inline VectorDataIterator<TYPE> VectorDataIterator<TYPE>::begin() const
{
    auto it        = VectorDataIterator( *this );
    it.d_CurBlock  = 0;
    it.d_CurOffset = 0;
    it.d_pos       = 0;
    return it;
}
template<typename TYPE>
inline VectorDataIterator<TYPE> VectorDataIterator<TYPE>::end() const
{
    auto it = VectorDataIterator( *this );
    if ( d_N_blocks > 0 ) {
        it.d_CurBlock  = d_N_blocks - 1;
        it.d_CurOffset = d_blockSize[d_N_blocks - 1];
        it.d_pos       = d_size;
    }
    return it;
}


/****************************************************************
 * Function to advance/receed the iterator                       *
 ****************************************************************/
template<typename TYPE>
inline void VectorDataIterator<TYPE>::advance( size_t i )
{
    AMP_INSIST( d_pos + i <= d_size, "Attempted to iterate past the end of the iterator" );
    size_t togo = i;
    while ( togo > 0 ) {
        if ( togo >= d_blockSize[d_CurBlock] - d_CurOffset ) {
            // We need to advance to the next data block
            d_pos += d_blockSize[d_CurBlock] - d_CurOffset;
            togo -= d_blockSize[d_CurBlock] - d_CurOffset;
            d_CurOffset = 0;
            d_CurBlock++;
        } else {
            // We need to advance to the correct position within the current data block
            d_pos += togo;
            d_CurOffset += togo;
            togo = 0;
        }
    }
}
template<typename TYPE>
inline void VectorDataIterator<TYPE>::recede( size_t i )
{
    AMP_INSIST( d_pos >= i, "Attempted to iterate past the beginning of the iterator" );
    size_t togo = i;
    while ( togo > 0 ) {
        if ( togo > d_CurOffset ) {
            // We need to advance to the next data block
            d_pos -= d_CurOffset;
            togo -= d_CurOffset;
            d_CurBlock--;
            d_CurOffset = d_blockSize[d_CurBlock];
        } else {
            // We need to advance to the correct position within the current data block
            d_pos -= togo;
            d_CurOffset -= togo;
            togo = 0;
        }
    }
}


/****************************************************************
 * Increment/Decrement Operators                                 *
 ****************************************************************/
template<typename TYPE>
VectorDataIterator<TYPE> &VectorDataIterator<TYPE>::operator++()
{
    // Prefix increment (increment and return this)
    advance( 1 );
    return *this;
}
template<typename TYPE>
VectorDataIterator<TYPE> VectorDataIterator<TYPE>::operator++( int )
{
    // Postfix increment (increment and return temporary object)
    VectorDataIterator<TYPE> tmp( *this ); // Create a temporary variable
    advance( 1 );                          // apply operator
    return tmp;                            // return temporary result
}
template<typename TYPE>
VectorDataIterator<TYPE> &VectorDataIterator<TYPE>::operator--()
{
    // Prefix decrement (decrement and return this)
    recede( 1 );
    return *this;
}
template<typename TYPE>
VectorDataIterator<TYPE> VectorDataIterator<TYPE>::operator--( int )
{
    // Postfix decrement (decrement and return temporary object)
    VectorDataIterator<TYPE> tmp( *this ); // Create a temporary variable
    recede( 1 );                           // apply operator
    return tmp;                            // return temporary result
}
template<typename TYPE>
VectorDataIterator<TYPE> &VectorDataIterator<TYPE>::operator+=( int offset )
{
    if ( offset > 0 )
        advance( offset );
    if ( offset < 0 )
        recede( -offset );
    return *this;
}
template<typename TYPE>
VectorDataIterator<TYPE> &VectorDataIterator<TYPE>::operator-=( int offset )
{
    if ( offset > 0 )
        recede( offset );
    if ( offset < 0 )
        advance( -offset );
    return *this;
}
template<typename TYPE>
VectorDataIterator<TYPE> VectorDataIterator<TYPE>::operator+( int offset )
{
    VectorDataIterator ans( *this );
    if ( offset > 0 )
        ans.advance( offset );
    if ( offset < 0 )
        ans.recede( -offset );
    return ans;
}
template<typename TYPE>
VectorDataIterator<TYPE> VectorDataIterator<TYPE>::operator-( int offset )
{
    VectorDataIterator ans( *this );
    if ( offset > 0 )
        ans.recede( offset );
    if ( offset < 0 )
        ans.advance( -offset );
    return ans;
}


/****************************************************************
 * Difference Operators                                          *
 ****************************************************************/
template<typename TYPE>
int VectorDataIterator<TYPE>::operator-( const VectorDataIterator<TYPE> &rhs ) const
{
    return d_pos - rhs.d_pos;
}


/****************************************************************
 * Assigment Operators                                           *
 ****************************************************************/
template<typename TYPE>
inline TYPE &VectorDataIterator<TYPE>::operator*()
{
    constexpr auto hash = getTypeID<TYPE>().hash;
    if ( !d_typeHash ) {
        return d_data[d_CurBlock][d_CurOffset];
    } else if ( d_typeHash[d_CurBlock] == hash ) {
        return d_data[d_CurBlock][d_CurOffset];
    } else {
        constexpr auto dblHash = getTypeID<double>().hash;
        constexpr auto fltHash = getTypeID<float>().hash;
        if ( d_typeHash[d_CurBlock] == dblHash ) {
            auto ptr   = reinterpret_cast<const double *>( d_data[d_CurBlock] );
            d_dataCast = ptr[d_CurOffset];
        } else if ( d_typeHash[d_CurBlock] == fltHash ) {
            auto ptr   = reinterpret_cast<const float *>( d_data[d_CurBlock] );
            d_dataCast = ptr[d_CurOffset];
        } else {
            AMP_ERROR( "Unknown data type" );
        }
        return d_dataCast;
    }
}


/****************************************************************
 * Assignment Operators                                          *
 ****************************************************************/
template<typename TYPE>
TYPE &VectorDataIterator<TYPE>::operator[]( int i )
{
    if ( d_typeHash )
        AMP_ERROR( "random-access iterators do not support implicit conversion" );
    VectorDataIterator<TYPE> tmp( *this ); // Create a temporary variable
    if ( i > 0 )
        tmp.advance( i );
    if ( i < 0 )
        tmp.recede( -i );
    return tmp.d_data[tmp.d_CurBlock][tmp.d_CurOffset];
}


/****************************************************************
 * Boolean Operators                                             *
 ****************************************************************/
template<typename TYPE>
inline bool VectorDataIterator<TYPE>::operator==( const VectorDataIterator<TYPE> &rhs ) const
{
    return d_hashcode == rhs.d_hashcode && d_pos == rhs.d_pos;
}
template<typename TYPE>
inline bool VectorDataIterator<TYPE>::operator!=( const VectorDataIterator<TYPE> &rhs ) const
{
    return d_hashcode != rhs.d_hashcode || d_pos != rhs.d_pos;
}
template<typename TYPE>
inline bool VectorDataIterator<TYPE>::operator<( const VectorDataIterator<TYPE> &rhs ) const
{
    return d_hashcode == rhs.d_hashcode ? ( d_pos < rhs.d_pos ) : ( d_hashcode < rhs.d_hashcode );
}
template<typename TYPE>
inline bool VectorDataIterator<TYPE>::operator>( const VectorDataIterator<TYPE> &rhs ) const
{
    return d_hashcode == rhs.d_hashcode ? ( d_pos > rhs.d_pos ) : ( d_hashcode > rhs.d_hashcode );
}
template<typename TYPE>
inline bool VectorDataIterator<TYPE>::operator<=( const VectorDataIterator<TYPE> &rhs ) const
{
    return d_hashcode == rhs.d_hashcode ? ( d_pos <= rhs.d_pos ) : ( d_hashcode <= rhs.d_hashcode );
}
template<typename TYPE>
inline bool VectorDataIterator<TYPE>::operator>=( const VectorDataIterator<TYPE> &rhs ) const
{
    return d_hashcode == rhs.d_hashcode ? ( d_pos >= rhs.d_pos ) : ( d_hashcode >= rhs.d_hashcode );
}


} // namespace AMP::LinearAlgebra


#endif
