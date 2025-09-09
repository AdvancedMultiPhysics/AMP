#include "AMP/utils/Array.h"
#include "AMP/utils/Array.hpp"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/typeid.h"

#include <bitset>
#include <complex>
#include <cstddef>
#include <string>
#include <string_view>


#define instantiateFull( T )      \
    template class AMP::Array<T>; \
    PACK_UNPACK_ARRAY( T );       \
    template AMP::Array<T>::Array( AMP::Range<T> const & )

#define instantiateData( T )           \
    instantiateArrayConstructors( T ); \
    PACK_UNPACK_ARRAY( T );            \
    PACK_UNPACK_ARRAY2( T )


/********************************************************
 *  Basic numeric Ts                                  *
 ********************************************************/
instantiateFull( char );
instantiateFull( uint8_t );
instantiateFull( uint16_t );
instantiateFull( uint32_t );
instantiateFull( uint64_t );
instantiateFull( int8_t );
instantiateFull( int16_t );
instantiateFull( int32_t );
instantiateFull( int64_t );
instantiateFull( long long );
instantiateFull( float );
instantiateFull( double );
instantiateFull( long double );


/********************************************************
 *  std::complex                                         *
 ********************************************************/
#define instantiateComplex( T )                                                                    \
    instantiateArrayConstructors( T );                                                             \
    template AMP::Array<T>::Array( const Range<T> &range );                                        \
    template AMP::Array<T> AMP::Array<T>::repmat( AMP::ArraySize const & ) const;                  \
    template void AMP::Array<T>::copySubset( std::vector<size_t> const &, AMP::Array<T> const & ); \
    template AMP::Array<T> AMP::Array<T>::subset( std::vector<size_t> const & ) const;             \
    template bool AMP::Array<T>::NaNs() const;                                                     \
    template void AMP::Array<T, AMP::FunctionTable>::rand();                                       \
    PACK_UNPACK_ARRAY( T );                                                                        \
    PACK_UNPACK_ARRAY2( T )
instantiateComplex( std::complex<float> );
instantiateComplex( std::complex<double> );


/********************************************************
 *  bool                                                 *
 ********************************************************/
template class AMP::Array<bool>;
PACK_UNPACK_ARRAY( bool );


/********************************************************
 *  byte, typeid, string                                 *
 ********************************************************/
instantiateData( std::byte );
instantiateData( AMP::typeID );
instantiateData( std::string );
instantiateData( std::string_view );
PACK_UNPACK_ARRAY( std::byte * );


/********************************************************
 *  MPI routines                                         *
 ********************************************************/
#include "AMP/utils/AMP_MPI.I"
template AMP::Array<int> AMP::AMP_MPI::bcast<AMP::Array<int>>( AMP::Array<int> const &, int ) const;
