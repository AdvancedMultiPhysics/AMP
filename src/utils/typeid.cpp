#include "AMP/utils/typeid.h"

#include <memory>
#include <vector>


namespace AMP {


/********************************************************************
 * Run some compile-time tests                                       *
 ********************************************************************/
template<class T>
static constexpr bool checkTraits()
{
    constexpr auto type = getTypeID<T>();
    static_assert( type.is_void() == std::is_void_v<T> );
    static_assert( type.is_null_pointer() == std::is_null_pointer_v<T> );
    static_assert( type.is_integral() == std::is_integral_v<T> );
    static_assert( type.is_floating_point() == std::is_floating_point_v<T> );
    static_assert( type.is_array() == std::is_array_v<T> );
    static_assert( type.is_enum() == std::is_enum_v<T> );
    static_assert( type.is_union() == std::is_union_v<T> );
    static_assert( type.is_class() == std::is_class_v<T> );
    static_assert( type.is_function() == std::is_function_v<T> );
    static_assert( type.is_pointer() == std::is_pointer_v<T> );
    static_assert( type.is_member_pointer() == std::is_member_pointer_v<T> );
    static_assert( type.is_member_object_pointer() == std::is_member_object_pointer_v<T> );
    static_assert( type.is_member_function_pointer() == std::is_member_function_pointer_v<T> );
    static_assert( type.is_fundamental() == std::is_fundamental_v<T> );
    static_assert( type.is_scalar() == std::is_scalar_v<T> );
    static_assert( type.is_object() == std::is_object_v<T> );
    static_assert( type.is_compound() == std::is_compound_v<T> );
    static_assert( type.is_trivially_copyable() == std::is_trivially_copyable_v<T> );
    static_assert( type.has_unique_object_representations() ==
                   std::has_unique_object_representations_v<T> );
    static_assert( type.is_empty() == std::is_empty_v<T> );
    static_assert( type.is_polymorphic() == std::is_polymorphic_v<T> );
    static_assert( type.is_abstract() == std::is_abstract_v<T> );
    static_assert( type.is_final() == std::is_final_v<T> );
    static_assert( type.is_aggregate() == std::is_aggregate_v<T> );
    static_assert( type.is_signed() == std::is_signed_v<T> );
    static_assert( type.is_arithmetic() == std::is_arithmetic_v<T> );
    static_assert( type.is_unsigned() == std::is_unsigned_v<T> );
    static_assert( type.is_member_pointer() == std::is_member_pointer_v<T> );
    return true;
}
template<class T>
static constexpr bool check( std::string_view name )
{
    checkTraits<T>();
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
static_assert( checkTraits<std::vector<double>>() );


} // namespace AMP
