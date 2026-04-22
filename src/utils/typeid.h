#ifndef included_AMP_typeid
#define included_AMP_typeid

#include <complex>
#include <cstdint>
#include <ostream>
#include <string_view>
#include <type_traits>

#include "AMP/AMP_TPLs.h"

#ifndef AMP_CXX_STANDARD
    #define AMP_CXX_STANDARD 17
#endif
#if AMP_CXX_STANDARD >= 20
    #include <source_location>
#endif


namespace TypeID_Helpers {

// Helper function to copy a string
constexpr void copy( char *dst, const char *src, size_t N )
{
    for ( size_t i = 0; i < N; i++ )
        dst[i] = 0;
    for ( size_t i = 0; i < ( N - 1 ) && src[i] != 0; i++ )
        dst[i] = src[i];
}

// Helper function to replace substrings
constexpr void replace( char *str, size_t N, std::string_view match, std::string_view replace )
{
    std::string_view str2( str, N );
    str2   = str2.substr( 0, str2.find( (char) 0 ) );
    auto i = str2.find( match );
    while ( i != std::string::npos ) {
        if ( match.size() == replace.size() ) {
            for ( size_t j = 0; j < match.size(); j++ )
                str[i + j] = replace[j];
        } else if ( match.size() > replace.size() ) {
            for ( size_t j = 0; j < replace.size(); j++ )
                str[i + j] = replace[j];
            size_t D = match.size() - replace.size();
            for ( size_t j = i + replace.size(); j < N - D; j++ )
                str[j] = str[j + D];
            for ( size_t j = N - D; j < N; j++ )
                str[j] = 0;
        } else {
            throw std::logic_error( "Not finished" );
        }
        i = str2.find( match );
    }
}
constexpr void deblank( char *str, size_t N )
{
    const char *whitespaces = " \t\f\v\n\r\0";
    std::string_view str2( str, N );
    str2   = str2.substr( 0, str2.find( (char) 0 ) );
    auto i = str2.find_first_not_of( whitespaces );
    auto j = str2.find_last_not_of( whitespaces );
    if ( i == std::string::npos )
        return;
    str2 = str2.substr( i, j - i + 1 );
    for ( size_t k = 0; k < str2.size(); k++ )
        str[k] = str2[k];
    for ( size_t k = str2.size(); k < N; k++ )
        str[k] = 0;
}

// Perform murmur hash (constexpr version that assumes key.size() is a multiple of 8)
template<std::size_t N>
constexpr uint64_t MurmurHash64A( const char *key )
{
    static_assert( N % 8 == 0 );
    const uint64_t seed = 0x65ce2a5d390efa53LLU;
    const uint64_t m    = 0xc6a4a7935bd1e995LLU;
    const int r         = 47;
    uint64_t h          = seed ^ ( N * m );
    for ( size_t i = 0; i < N; i += 8 ) {
        uint64_t k = ( uint64_t( key[i] ) << 56 ) ^ ( uint64_t( key[i + 1] ) << 48 ) ^
                     ( uint64_t( key[i + 2] ) << 40 ) ^ ( uint64_t( key[i + 3] ) << 32 ) ^
                     ( uint64_t( key[i + 4] ) << 24 ) ^ ( uint64_t( key[i + 5] ) << 16 ) ^
                     ( uint64_t( key[i + 6] ) << 8 ) ^ ( uint64_t( key[i + 7] ) );
        k *= m;
        k ^= k >> r;
        k *= m;
        h ^= k;
        h *= m;
    }
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

} // namespace TypeID_Helpers


/*!
 * @brief  Get the type name
 * @details  This will return a string for the type name.
 *    Note that the string may be different on different platforms or
 *    different compilers.  This function cannot be in a namespace to
 *    ensure the proper namespace is used for the type.
 *    If the name does not fit within the buffer it may be truncated.
 *    The truncation may occur before cleaning up the name which may
 *    result in a shorter name than expected.
 * @param[in] N             The size of the name buffer
 * @param[out] name         The name of the type (null terminated, may be truncated)
 */
template<typename T>
constexpr void getTypeName( uint64_t N, char *name )
{
    using namespace TypeID_Helpers;
    if constexpr ( std::is_same_v<T, bool> ) {
        copy( name, "bool", N );
    } else if constexpr ( std::is_same_v<T, char> ) {
        copy( name, "char", N );
    } else if constexpr ( std::is_same_v<T, int8_t> ) {
        copy( name, "int8_t", N );
    } else if constexpr ( std::is_same_v<T, uint8_t> || std::is_same_v<T, unsigned char> ) {
        copy( name, "uint8_t", N );
    } else if constexpr ( std::is_same_v<T, int16_t> ) {
        copy( name, "int16_t", N );
    } else if constexpr ( std::is_same_v<T, uint16_t> ) {
        copy( name, "uint16_t", N );
    } else if constexpr ( std::is_same_v<T, int> || std::is_same_v<T, int32_t> ) {
        copy( name, "int32_t", N );
    } else if constexpr ( std::is_same_v<T, unsigned> || std::is_same_v<T, uint32_t> ) {
        copy( name, "uint32_t", N );
    } else if constexpr ( std::is_same_v<T, int64_t> ) {
        copy( name, "int64_t", N );
    } else if constexpr ( std::is_same_v<T, uint64_t> ) {
        copy( name, "uint64_t", N );
    } else if constexpr ( std::is_same_v<T, float> ) {
        copy( name, "float", N );
    } else if constexpr ( std::is_same_v<T, double> ) {
        copy( name, "double", N );
    } else if constexpr ( std::is_same_v<T, std::complex<float>> ) {
        copy( name, "std::complex<float>", N );
    } else if constexpr ( std::is_same_v<T, std::complex<double>> ) {
        copy( name, "std::complex<double>", N );
    } else if constexpr ( std::is_same_v<T, std::string> ) {
        copy( name, "std::string", N );
    } else if constexpr ( std::is_same_v<T, std::string_view> ) {
        copy( name, "std::string_view", N );
    } else {
        // Get the type name from the function
#if defined( __clang__ ) || defined( __GNUC__ )
        constexpr std::string_view name0 = __PRETTY_FUNCTION__;
        std::string_view name2           = name0;
#elif defined( _MSC_VER )
        constexpr std::string_view name0 = __FUNCSIG__;
        std::string_view name2           = name0;
#elif AMP_CXX_STANDARD >= 20
        // Note this fails for nvhpc (does not contain template type)
        auto source = std::source_location::current();
        std::string_view name2( source.function_name() );
#else
    #error "Not finished";
#endif
        // Try to get just the type of interest
        if ( name2.find( "T = " ) != std::string::npos ) {
            name2 = name2.substr( name2.find( "T = " ) + 4 );
            if ( name2.find( ';' ) != std::string::npos )
                name2 = name2.substr( 0, name2.find( ';' ) );
            else
                name2 = name2.substr( 0, name2.rfind( ']' ) );
        }
        if ( name2.find( "getTypeName<" ) != std::string::npos ) {
            auto i1 = name2.find( "getTypeName<" );
            auto i2 = name2.rfind( ">" );
            name2   = name2.substr( i1 + 12, i2 - i1 - 12 );
        }
        if ( name2[0] == ' ' )
            name2.remove_prefix( 1 );
        // Copy the function name
        name2 = name2.substr( 0, N - 1 );
        for ( size_t i = 0; i < N; i++ )
            name[i] = 0;
        for ( size_t i = 0; i < std::min( name2.size(), N - 1 ); i++ )
            name[i] = name2[i];
        // Cleanup some common format issues to make the typeid more consistent
        // clang-format off
        name[N - 1] = 0;
        replace( name, N, "class ", "" );
        replace( name, N, "struct ", "" );
        replace( name, N, "std::__cxx11::basic_string<char>", "std::string" );
        replace( name, N, "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>", "std::string" );
        replace( name, N, "std::basic_string<char,struct std::char_traits<char>,class std::allocator<char>>", "std::string" );
        replace( name, N, "std::__cxx11::basic_string_view<char>", "std::string_view" );
        replace( name, N, "std::__cxx11::basic_string_view<char, std::char_traits<char>>", "std::string_view" );
        replace( name, N, "std::basic_string_view<char,struct std::char_traits<char>>", "std::string_view" );
        replace( name, N, "std::basic_string_view<char>", "std::string_view" );
        replace( name, N, "std::basic_string_view<char, std::char_traits<char>>", "std::string_view" );
        replace( name, N, "std::__debug::", "std::" );
        replace( name, N, " *", "* " );
        replace( name, N, " >", ">" );
        replace( name, N, "* [", "*[" );
        deblank( name, N );
        // clang-format on
    }
}


namespace AMP {


//! Class to store type info
struct alignas( 8 ) typeID {
    uint32_t bytes  = 0;     //!< Size of object (bytes)
    uint32_t hash   = 0;     //!< Hash of function
    uint32_t traits = 0;     //! Store some basic properties
    char name[116]  = { 0 }; //!< Name of function (may be truncated, null-terminated)
    constexpr bool operator==( uint32_t rhs ) const { return hash == rhs; }
    constexpr bool operator!=( uint32_t rhs ) const { return hash != rhs; }
    constexpr bool operator==( const typeID &rhs ) const { return hash == rhs.hash; }
    constexpr bool operator!=( const typeID &rhs ) const { return hash != rhs.hash; }
    constexpr bool is_void() const { return traits & 0x1; }
    constexpr bool is_null_pointer() const { return traits & 0x2; }
    constexpr bool is_integral() const { return traits & 0x4; }
    constexpr bool is_floating_point() const { return traits & 0x8; }
    constexpr bool is_array() const { return traits & 0x10; }
    constexpr bool is_enum() const { return traits & 0x20; }
    constexpr bool is_union() const { return traits & 0x40; }
    constexpr bool is_class() const { return traits & 0x80; }
    constexpr bool is_function() const { return traits & 0x100; }
    constexpr bool is_pointer() const { return traits & 0x200; }
    constexpr bool is_member_object_pointer() const { return traits & 0x400; }
    constexpr bool is_member_function_pointer() const { return traits & 0x800; }
    constexpr bool is_fundamental() const { return traits & 0x1000; }
    constexpr bool is_scalar() const { return traits & 0x2000; }
    constexpr bool is_object() const { return traits & 0x4000; }
    constexpr bool is_compound() const { return traits & 0x8000; }
    constexpr bool is_trivially_copyable() const { return traits & 0x10000; }
    constexpr bool has_unique_object_representations() const { return traits & 0x20000; }
    constexpr bool is_empty() const { return traits & 0x40000; }
    constexpr bool is_polymorphic() const { return traits & 0x80000; }
    constexpr bool is_abstract() const { return traits & 0x100000; }
    constexpr bool is_final() const { return traits & 0x200000; }
    constexpr bool is_aggregate() const { return traits & 0x400000; }
    constexpr bool is_signed() const { return traits & 0x800000; }
    constexpr bool is_arithmetic() const { return is_integral() || is_floating_point(); }
    constexpr bool is_unsigned() const { return is_arithmetic() && !is_signed(); }
    constexpr bool is_member_pointer() const
    {
        return is_member_object_pointer() || is_member_function_pointer();
    }
};
static_assert( sizeof( typeID ) == 128 );


//! Get the type info (does not resolve dynamic types)
template<typename T0>
constexpr typeID getTypeIDEval()
{
    using namespace TypeID_Helpers;
    typeID id = {};
    // Remove const/references
    using T1 = typename std::remove_reference_t<T0>;
    using T2 = typename std::remove_cv_t<T1>;
    using T  = typename std::remove_cv_t<T2>;
    // Get the name of the class
    char name[4096] = { 0 };
    getTypeName<T>( sizeof( name ), name );
    copy( id.name, name, sizeof( id.name ) );
    // Create the hash
    if ( name[0] != 0 )
        id.hash = MurmurHash64A<sizeof( name )>( name );
    // Set the size
    id.bytes = sizeof( T );
    // Set the properties
    auto set = [&id]( uint8_t i, bool val ) {
        if ( val )
            id.traits |= ( (uint32_t) 0x1 ) << i;
    };
    set( 0, std::is_void_v<T0> );
    set( 1, std::is_null_pointer_v<T0> );
    set( 2, std::is_integral_v<T0> );
    set( 3, std::is_floating_point_v<T0> );
    set( 4, std::is_array_v<T0> );
    set( 5, std::is_enum_v<T0> );
    set( 6, std::is_union_v<T0> );
    set( 7, std::is_class_v<T0> );
    set( 8, std::is_function_v<T0> );
    set( 9, std::is_pointer_v<T0> );
    set( 10, std::is_member_object_pointer_v<T0> );
    set( 11, std::is_member_function_pointer_v<T0> );
    set( 12, std::is_fundamental_v<T0> );
    set( 13, std::is_scalar_v<T0> );
    set( 14, std::is_object_v<T0> );
    set( 15, std::is_compound_v<T0> );
    set( 16, std::is_trivially_copyable_v<T0> );
    set( 17, std::has_unique_object_representations_v<T0> );
    set( 18, std::is_empty_v<T0> );
    set( 19, std::is_polymorphic_v<T0> );
    set( 20, std::is_abstract_v<T0> );
    set( 21, std::is_final_v<T0> );
    set( 22, std::is_aggregate_v<T0> );
    set( 23, std::is_signed_v<T0> );
    return id;
}
//! Get the type info (does not resolve dynamic types)
template<typename TYPE>
constexpr typeID getTypeID()
{
    constexpr auto id = getTypeIDEval<TYPE>();
    static_assert( id != 0 );
    return id;
}


// Print the typeid name
inline std::ostream &operator<<( std::ostream &out, const typeID &id )
{
    out << id.name;
    return out;
}


} // namespace AMP

#endif
