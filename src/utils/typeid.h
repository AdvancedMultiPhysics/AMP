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
        // Get the name of the function to create the type name
        char name0[1024] = { 0 };
#if defined( __clang__ )
        copy( name0, __PRETTY_FUNCTION__, sizeof( name0 ) );
#elif defined( __GNUC__ )
        copy( name0, __PRETTY_FUNCTION__, sizeof( name0 ) );
#elif defined( _MSC_VER )
        copy( name0, __FUNCSIG__, sizeof( name0 ) );
#else
    // Not finished, one possible workaround, pass default class name as string_view
    #error "Not finished";
#endif
        // Get the type name from the function
        std::string_view name2( name0 );
        if ( name2.find( "]" ) != std::string::npos )
            name2 = name2.substr( 0, name2.find( "]" ) );
        if ( name2.find( "T = " ) != std::string::npos ) {
            name2 = name2.substr( name2.find( "T = " ) + 4 );
            name2 = name2.substr( 0, name2.find( ';' ) );
        }
        if ( name2.rfind( " = " ) != std::string::npos ) {
            name2 = name2.substr( name2.rfind( " = " ) + 3 );
        }
        name2 = name2.substr( 0, N - 1 );
        for ( size_t i = 0; i < N; i++ )
            name[i] = 0;
        for ( size_t i = 0; i < std::min( name2.size(), N - 1 ); i++ )
            name[i] = name2[i];
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
    char name[128] = { 0 };
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
