#ifndef included_AMP_typeid
#define included_AMP_typeid

#include <complex>
#include <cstdint>
#include <string_view>
#include <type_traits>


namespace AMP {


//! Class to store type info
struct alignas( 8 ) typeID {
    uint32_t bytes = 0;     // Size of object (bytes)
    uint32_t hash  = 0;     // Hash of function
    char name[120] = { 0 }; // Name of function (may be truncated, null-terminated)
    constexpr bool operator==( uint32_t rhs ) const { return hash == rhs; }
    constexpr bool operator!=( uint32_t rhs ) const { return hash != rhs; }
    constexpr bool operator==( const typeID &rhs ) const { return hash == rhs.hash; }
    constexpr bool operator!=( const typeID &rhs ) const { return hash != rhs.hash; }
};
static_assert( sizeof( typeID ) == 128 );


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


//! Get the type name
template<typename T>
constexpr void getTypeName( uint64_t N, char *name )
{
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
        if ( name2.find( "T = " ) != std::string::npos ) {
            name2 = name2.substr( name2.find( "T = " ) + 4 );
            if ( name2.find( ';' ) != std::string::npos )
                name2 = name2.substr( 0, name2.find( ';' ) );
            else
                name2 = name2.substr( 0, name2.rfind( ']' ) );
        }
#elif defined( _MSC_VER )
        constexpr std::string_view name0 = __FUNCSIG__;
        std::string_view name2           = name0;
        if ( name2.find( "getTypeName<" ) != std::string::npos ) {
            auto i1 = name2.find( "getTypeName<" );
            auto i2 = name2.rfind( ">" );
            name2   = name2.substr( i1 + 12, i2 - i1 - 12 );
        }
        if ( name2[0] == ' ' )
            name2.remove_prefix( 1 );
#else
    // Not finished, one possible workaround, pass default class name as string_view
    #error "Not finished";
#endif
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


//! Perform murmur hash (constexpr version that assumes key.size() is a multiple of 8)
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


//! Get the type info (does not resolve dynamic types)
template<typename T0>
constexpr typeID getTypeIDEval()
{
    typeID id = {};
    // Remove const/references
    using T1 = typename std::remove_reference_t<T0>;
    using T2 = typename std::remove_cv_t<T1>;
    using T  = typename std::remove_cv_t<T2>;
    // Get the name of the class
    char name[1024] = { 0 };
    getTypeName<T>( sizeof( name ), name );
    copy( id.name, name, sizeof( id.name ) );
    // Create the hash
    if ( name[0] != 0 )
        id.hash = MurmurHash64A<sizeof( name )>( name );
    // Set the size
    id.bytes = sizeof( T );
    return id;
}
template<typename TYPE>
constexpr typeID getTypeID()
{
    constexpr auto id = getTypeIDEval<TYPE>();
    static_assert( id != 0 );
    return id;
}


} // namespace AMP

#endif
