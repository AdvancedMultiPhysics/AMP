#include "AMP/utils/typeid.h"

#include <memory>


namespace AMP {


/********************************************************************
 * Run some compile-time tests                                       *
 ********************************************************************/
template<class T>
static constexpr bool check( std::string_view name )
{
    auto type = getTypeID<T>();
    return std::string_view( type.name ) == name && type.bytes == sizeof( T ) && type.hash != 0;
}
static_assert( sizeof( uint8_t ) == sizeof( unsigned char ) );
static_assert( check<int8_t>( "int8_t" ) );
static_assert( check<int16_t>( "int16_t" ) );
static_assert( check<int32_t>( "int32_t" ) );
static_assert( check<int64_t>( "int64_t" ) );
static_assert( check<uint8_t>( "uint8_t" ) );
static_assert( check<uint16_t>( "uint16_t" ) );
static_assert( check<uint32_t>( "uint32_t" ) );
static_assert( check<uint64_t>( "uint64_t" ) );
static_assert( check<bool>( "bool" ) );
static_assert( check<char>( "char" ) );
static_assert( check<int>( "int32_t" ) );
static_assert( check<unsigned char>( "uint8_t" ) );
static_assert( check<float>( "float" ) );
static_assert( check<double>( "double" ) );
static_assert( check<std::string>( "std::string" ) );
static_assert( check<std::string_view>( "std::string_view" ) );
static_assert( check<std::complex<float>>( "std::complex<float>" ) );
static_assert( check<std::complex<double>>( "std::complex<double>" ) );
static_assert( check<const double>( "double" ) );
static_assert( check<const double &>( "double" ) );
// static_assert( check<std::shared_ptr<double>>( "std::shared_ptr<double>" ) ); // Fails windows
// static_assert( check<double *>( "double*" ) );  // Fails clang-16
// static_assert( check<const double *>( "const double*" ) ); // Fails clang-16
// static_assert( check<double const *>( "const double*" ) ); // Fails clang-16


} // namespace AMP
