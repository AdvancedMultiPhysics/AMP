#include "AMP/utils/Array.h"
#include "AMP/utils/Array.hpp"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/typeid.h"

#include <complex>
#include <cstddef>


namespace AMP {


/********************************************************
 *  Explicit instantiations of Array                     *
 ********************************************************/
#if defined( __INTEL_COMPILER )
DISABLE_WARNINGS
#endif
template class Array<bool>;
template class Array<char>;
template class Array<uint8_t>;
template class Array<uint16_t>;
template class Array<uint32_t>;
template class Array<uint64_t>;
template class Array<int8_t>;
template class Array<int16_t>;
template class Array<int32_t>;
template class Array<int64_t>;
template class Array<long long>;
template class Array<float>;
template class Array<double>;
template class Array<long double>;
instantiateArrayConstructors( std::byte );
instantiateArrayConstructors( std::string );
instantiateArrayConstructors( std::string_view );
instantiateArrayConstructors( typeID );


/********************************************************
 *  Explicit instantiations of Array<std::complex>       *
 ********************************************************/
instantiateArrayConstructors( std::complex<float> );
instantiateArrayConstructors( std::complex<double> );
template Array<std::complex<float>>::Array( const Range<std::complex<float>> &range );
template Array<std::complex<double>>::Array( const Range<std::complex<double>> &range );
template Array<std::complex<float>>
Array<std::complex<float>>::repmat( std::vector<unsigned long> const & ) const;
template Array<std::complex<double>>
Array<std::complex<double>>::repmat( std::vector<unsigned long> const & ) const;
template void Array<std::complex<float>>::copySubset( std::vector<unsigned long> const &,
                                                      Array<std::complex<float>> const & );
template void Array<std::complex<double>>::copySubset( std::vector<unsigned long> const &,
                                                       Array<std::complex<double>> const & );
template Array<std::complex<float>>
Array<std::complex<float>>::subset( std::vector<unsigned long> const & ) const;
template Array<std::complex<double>>
Array<std::complex<double>>::subset( std::vector<unsigned long> const & ) const;
template bool Array<std::complex<float>>::NaNs() const;
template bool Array<std::complex<double>>::NaNs() const;
template void AMP::Array<std::complex<double>, AMP::FunctionTable>::rand();

#if defined( __INTEL_COMPILER )
ENABLE_WARNINGS
#endif


/********************************************************
 *  Pack/Unpack                                          *
 ********************************************************/
PACK_UNPACK_ARRAY( std::byte * );
PACK_UNPACK_ARRAY( bool );
PACK_UNPACK_ARRAY( char );
PACK_UNPACK_ARRAY( uint8_t );
PACK_UNPACK_ARRAY( uint16_t );
PACK_UNPACK_ARRAY( uint32_t );
PACK_UNPACK_ARRAY( uint64_t );
PACK_UNPACK_ARRAY( int8_t );
PACK_UNPACK_ARRAY( int16_t );
PACK_UNPACK_ARRAY( int32_t );
PACK_UNPACK_ARRAY( int64_t );
PACK_UNPACK_ARRAY( long long );
PACK_UNPACK_ARRAY( float );
PACK_UNPACK_ARRAY( double );
PACK_UNPACK_ARRAY( long double );
PACK_UNPACK_ARRAY( std::complex<float> );
PACK_UNPACK_ARRAY( std::complex<double> );
PACK_UNPACK_ARRAY( std::byte );
PACK_UNPACK_ARRAY( std::string );
PACK_UNPACK_ARRAY( std::string_view );
PACK_UNPACK_ARRAY2( std::byte * );
PACK_UNPACK_ARRAY2( std::string );
PACK_UNPACK_ARRAY2( std::complex<float> );
PACK_UNPACK_ARRAY2( std::complex<double> );


} // namespace AMP
