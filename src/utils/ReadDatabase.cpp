#include "AMP/utils/Array.hpp"
#include "AMP/utils/Database.h"
#include "AMP/utils/Database.hpp"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>


namespace AMP {


/********************************************************************
 * Helper functions                                                  *
 ********************************************************************/
static inline bool strcmpi( std::string_view s1, std::string_view s2 )
{
    if ( s1.size() != s2.size() )
        return false;
    bool equal = true;
    for ( size_t i = 0; i < s1.size(); i++ ) {
        char a    = s1[i];
        char b    = s2[i];
        bool test = ( a & 0x1F ) == ( b & 0x1F ) && a > 64 && b > 64;
        equal     = equal && test;
    }
    return equal;
}
template<class TYPE>
static TYPE readValue( std::string_view str );
template<>
double readValue<double>( std::string_view str )
{
    double data = 0;
    if ( strcmpi( str, "inf" ) || strcmpi( str, "infinity" ) ) {
        data = std::numeric_limits<double>::infinity();
    } else if ( strcmpi( str, "-inf" ) || strcmpi( str, "-infinity" ) ) {
        data = -std::numeric_limits<double>::infinity();
    } else if ( strcmpi( str, "nan" ) || strcmpi( str, "-nan" ) ) {
        data = std::numeric_limits<double>::quiet_NaN();
    } else if ( str.find( '/' ) != std::string::npos ) {
        AMP_ERROR( "Error reading value (double): " + std::string( str ) );
    } else {
        char *pos = nullptr;
        data      = std::strtod( str.data(), &pos );
        if ( static_cast<size_t>( pos - str.data() ) == str.size() + 1 )
            AMP_ERROR( "Error reading value (double): " + std::string( str ) );
    }
    return data;
}
template<>
int readValue<int>( std::string_view str )
{
    char *pos = nullptr;
    int data  = strtol( str.data(), &pos, 10 );
    if ( static_cast<size_t>( pos - str.data() ) == str.size() + 1 )
        AMP_ERROR( "Error reading value (int): " + std::string( str ) );
    return data;
}
template<class TYPE>
static AMP::Range<TYPE> readRange( std::string_view str )
{
    AMP::Range<TYPE> z;
    auto i1 = str.find_first_of( ':' );
    auto i2 = str.find_first_of( ':', i1 + 1 );
    auto v1 = readValue<TYPE>( str.substr( 0, i1 ) );
    if ( i2 == std::string::npos ) {
        auto v2 = readValue<TYPE>( str.substr( i1 + 1 ) );
        z       = AMP::Range<TYPE>( v1, v2 );
    } else {
        auto v2 = readValue<TYPE>( str.substr( i2 + 1 ) );
        auto dx = readValue<TYPE>( str.substr( i1 + 1, i2 - i1 - 1 ) );
        z       = AMP::Range<TYPE>( v1, v2, dx );
    }
    return z;
}
template<>
AMP::Range<int> readValue<AMP::Range<int>>( std::string_view str )
{
    return readRange<int>( str );
}
template<>
AMP::Range<double> readValue<AMP::Range<double>>( std::string_view str )
{
    return readRange<double>( str );
}
template<class TYPE>
static std::tuple<TYPE, Units> readPair( std::string_view str )
{
    auto str0 = str;
    auto tmp  = deblank( std::move( str0 ) );
    if constexpr ( std::is_same_v<TYPE, std::complex<double>> ) {
        // We are trying to read a complex number
        if ( str[0] != '(' ) {
            // Read a double and convert to complex
            auto [value, unit] = readPair<double>( str );
            return std::make_tuple( std::complex<double>( value ), unit );
        }
        size_t p1 = str.find( ',' );
        size_t p2 = str.find( ')' );
        std::string_view s1( &str[1], p1 - 1 );
        std::string_view s2( &str[p1 + 1], p2 - p1 - 1 );
        auto value = std::complex<double>( readValue<double>( s1 ), readValue<double>( s2 ) );
        return std::make_tuple( value, Units( tmp.substr( p2 + 1 ) ) );
    } else {
        // Read a scalar type
        size_t index = tmp.find( ' ' );
        if ( index != std::string::npos ) {
            return std::make_tuple( readValue<TYPE>( tmp.substr( 0, index ) ),
                                    Units( tmp.substr( index + 1 ) ) );
        } else {
            return std::make_tuple( readValue<TYPE>( tmp ), Units() );
        }
    }
}


/********************************************************************
 * Read the input file into memory                                   *
 ********************************************************************/
static std::vector<char> readFile( const std::string &filename, Database::source_location src )
{
    // Read the input file into memory
    FILE *fid = fopen( filename.data(), "rb" );
    DATABASE_INSIST( fid, src, "Error opening file %s", filename.data() );
    fseek( fid, 0, SEEK_END );
    size_t bytes = ftell( fid );
    rewind( fid );
    std::vector<char> data( bytes + 1, 0 );
    [[maybe_unused]] size_t result = fread( data.data(), 1, bytes, fid );
    fclose( fid );
    return data;
}


/********************************************************************
 * Read input database file                                          *
 ********************************************************************/
enum class token_type {
    newline,
    line_comment,
    block_start,
    block_stop,
    quote,
    comma,
    equal,
    define,
    bracket,
    end_bracket,
    end,
    import_data
};
static constexpr size_t length( token_type type )
{
    size_t len = 0;
    if ( type == token_type::newline || type == token_type::quote || type == token_type::equal ||
         type == token_type::bracket || type == token_type::end_bracket ||
         type == token_type::end ) {
        len = 1;
    } else if ( type == token_type::line_comment || type == token_type::block_start ||
                type == token_type::block_stop || type == token_type::define ) {
        len = 2;
    } else if ( type == token_type::import_data ) {
        len = 3;
    }
    return len;
}
static inline std::tuple<size_t, token_type> find_next_token( const char *buffer )
{
    size_t i = 0;
    while ( true ) {
        if ( buffer[i] == '\n' || buffer[i] == '\r' ) {
            return std::make_tuple( i + 1, token_type::newline );
        } else if ( buffer[i] == 0 ) {
            return std::make_tuple( i + 1, token_type::end );
        } else if ( buffer[i] == '"' ) {
            return std::make_tuple( i + 1, token_type::quote );
        } else if ( buffer[i] == ',' ) {
            return std::make_tuple( i + 1, token_type::comma );
        } else if ( buffer[i] == '=' ) {
            return std::make_tuple( i + 1, token_type::equal );
        } else if ( buffer[i] == ':' ) {
            if ( buffer[i + 1] == '=' )
                return std::make_tuple( i + 2, token_type::define );
        } else if ( buffer[i] == '{' ) {
            return std::make_tuple( i + 1, token_type::bracket );
        } else if ( buffer[i] == '}' ) {
            return std::make_tuple( i + 1, token_type::end_bracket );
        } else if ( buffer[i] == '/' ) {
            if ( buffer[i + 1] == '/' ) {
                return std::make_tuple( i + 2, token_type::line_comment );
            } else if ( buffer[i + 1] == '*' ) {
                return std::make_tuple( i + 2, token_type::block_start );
            }
        } else if ( buffer[i] == '#' ) {
            return std::make_tuple( i + 1, token_type::line_comment );
        } else if ( buffer[i] == '*' ) {
            if ( buffer[i + 1] == '/' )
                return std::make_tuple( i + 2, token_type::block_stop );
        } else if ( std::string_view( &buffer[i], 3 ) == "<<<" ) {
            return std::make_tuple( i + 3, token_type::import_data );
        }
        i++;
        if ( i >= 10000000 ) {
            AMP_WARNING( "Token not found" );
            break;
        }
    }
    return std::make_tuple<size_t, token_type>( 0, token_type::end );
}
static size_t skip_comment( const char *buffer )
{
    auto tmp          = find_next_token( buffer );
    auto comment_type = std::get<1>( tmp );
    size_t pos        = 0;
    if ( comment_type == token_type::line_comment ) {
        // Line comment
        while ( std::get<1>( tmp ) != token_type::newline &&
                std::get<1>( tmp ) != token_type::end ) {
            pos += std::get<0>( tmp );
            tmp = find_next_token( &buffer[pos] );
        }
        pos += std::get<0>( tmp );
    } else {
        /* Block comment */
        while ( std::get<1>( tmp ) != token_type::block_stop ) {
            if ( comment_type == token_type::block_start && std::get<1>( tmp ) == token_type::end )
                throw std::logic_error( "Encountered end of file before block comment end" );
            pos += std::get<0>( tmp );
            tmp = find_next_token( &buffer[pos] );
        }
        pos += std::get<0>( tmp );
    }
    return pos;
}
enum class class_type {
    STRING,
    BOOL,
    INT,
    FLOAT,
    COMPLEX,
    BOX,
    ARRAY,
    EQUATION,
    DATABASE_ENTRY,
    RANGE,
    UNKNOWN
};
class_type operator+( class_type x, class_type y )
{
    if ( x == y )
        return x;
    auto INT   = class_type::INT;
    auto FLOAT = class_type::FLOAT;
    if ( ( x == INT || x == FLOAT ) && ( y == INT || y == FLOAT ) )
        return FLOAT;
    return class_type::UNKNOWN;
}
static class_type getRangeType( std::string_view value );
static class_type getType( std::string_view value0,
                           const std::map<std::string, const KeyData *> &databaseKeys = {} )
{
    // Check for empty string
    if ( value0.empty() )
        return class_type::INT;
    // Check if we have units
    auto value = value0;
    if ( value.find( ' ' ) != std::string::npos ) {
        value = deblank( value.substr( 0, value.find( ' ' ) ) );
    }
    // Check if we are dealing with a simple bool
    if ( strcmpi( value, "true" ) || strcmpi( value, "false" ) )
        return class_type::BOOL;
    // Check if we could be an int or float
    bool is_int   = true;
    bool is_float = true;
    for ( char c : value ) {
        if ( c < 42 || c == 46 || c >= 58 )
            is_int = false;
        if ( ( c < 42 || c >= 58 ) && ( c != 69 && c != 101 ) )
            is_float = false;
    }
    if ( is_int )
        return class_type::INT;
    if ( is_float )
        return class_type::FLOAT;
    // Check for special floating point types
    if ( strcmpi( value, "inf" ) || strcmpi( value, "nan" ) )
        return class_type::FLOAT;
    // Check if we are a database entry
    if ( databaseKeys.find( std::string( value0 ) ) != databaseKeys.end() )
        return class_type::DATABASE_ENTRY;
    // Check if we are dealing with a range
    size_t N = std::count( value.begin(), value.end(), ':' );
    if ( N == 1 || N == 2 ) {
        auto type = getRangeType( value );
        if ( type == class_type::INT || type == class_type::FLOAT )
            return class_type::RANGE;
    }
    // Return unknown
    return class_type::UNKNOWN;
}
static class_type getRangeType( std::string_view value )
{
    auto i1   = value.find_first_of( ':' );
    auto i2   = std::min<size_t>( value.find_first_of( ':', i1 + 1 ), value.size() );
    auto type = getType( value.substr( 0, i1 ) ) + getType( value.substr( i1 + 1, i2 - i1 - 1 ) );
    if ( i2 < value.size() )
        type = type + getType( value.substr( i2 + 1 ) );
    return type;
}
template<class T>
static bool isType( const std::vector<const KeyData *> &data )
{
    bool test = true;
    for ( auto tmp : data )
        test = test && tmp->isType<T>();
    return test;
}
template<class T>
static bool isType( const std::vector<std::unique_ptr<AMP::KeyData>> &data )
{
    bool test = true;
    for ( auto &tmp : data )
        test = test && tmp->isType<T>();
    return test;
}
// Convert the string value to the database value
static std::tuple<std::unique_ptr<KeyData>, std::set<std::string>>
createKeyData( std::string_view key,
               class_type data_type,
               std::vector<std::string_view> &values,
               const std::map<std::string, const KeyData *> &databaseKeys )
{
    std::unique_ptr<KeyData> data;
    std::set<std::string> usedKeys;
    if ( values.empty() ) {
        data = std::make_unique<EmptyKeyData>();
    } else if ( values.size() == 1 && values[0].empty() ) {
        data = std::make_unique<EmptyKeyData>();
    } else if ( data_type == class_type::STRING ) {
        // We are dealing with strings
        for ( auto &value : values ) {
            if ( value.empty() )
                continue;
            if ( value[0] != '"' || value.back() != '"' )
                throw std::logic_error( "Error parsing string for key: " + std::string( key ) );
            value = value.substr( 1, value.size() - 2 );
        }
        if ( values.size() == 1 ) {
            std::string str( values[0] );
            data = std::make_unique<KeyDataScalar<std::string>>( std::move( str ) );
        } else {
            Array<std::string> data2( values.size() );
            for ( size_t i = 0; i < values.size(); i++ )
                data2( i ) = std::string( values[i].data(), values[i].size() );
            data = std::make_unique<KeyDataArray<std::string>>( std::move( data2 ) );
        }
    } else if ( data_type == class_type::EQUATION ) {
        // We are dealing with equations
        Array<std::string_view> data2( values.size() );
        Units unit;
        for ( size_t i = 0; i < values.size(); i++ ) {
            size_t j   = values[i].find( ';' );
            data2( i ) = deblank( values[i].substr( 0, j + 1 ) );
            auto str   = deblank( values[i].substr( j + 1 ) );
            if ( unit.isNull() && !str.empty() )
                unit = Units( str );
        }
        if ( data2.length() > 1 )
            AMP_ERROR( "Arrays of equations are not currently supported" );
        data = std::make_unique<EquationKeyData>( std::string( data2( 0 ) ), unit );
    } else if ( data_type == class_type::BOOL ) {
        // We are dealing with logical values
        Array<bool> data2( values.size() );
        for ( size_t i = 0; i < values.size(); i++ ) {
            if ( !strcmpi( values[i], "true" ) && !strcmpi( values[i], "false" ) )
                throw std::logic_error( "Error converting " + std::string( key ) +
                                        " to logical array" );
            data2( i ) = strcmpi( values[i], "true" );
        }
        if ( values.size() == 1 ) {
            data = std::make_unique<KeyDataScalar<bool>>( data2( 0 ) );
        } else {
            data = std::make_unique<KeyDataArray<bool>>( std::move( data2 ) );
        }
    } else if ( data_type == class_type::INT ) {
        // We are dealing with integer values
        Array<int> data2( values.size() );
        Units unit;
        for ( size_t i = 0; i < values.size(); i++ ) {
            Units unit2;
            std::tie( data2( i ), unit2 ) = readPair<int>( values[i] );
            if ( !unit2.isNull() )
                unit = unit2;
        }
        if ( values.size() == 1 ) {
            data = std::make_unique<KeyDataScalar<int>>( data2( 0 ), unit );
        } else {
            data = std::make_unique<KeyDataArray<int>>( std::move( data2 ), unit );
        }
    } else if ( data_type == class_type::FLOAT ) {
        // We are dealing with floating point values
        Array<double> data2( values.size() );
        Units unit;
        for ( size_t i = 0; i < values.size(); i++ ) {
            Units unit2;
            std::tie( data2( i ), unit2 ) = readPair<double>( values[i] );
            if ( !unit2.isNull() )
                unit = unit2;
        }
        if ( values.size() == 1 ) {
            data = std::make_unique<KeyDataScalar<double>>( data2( 0 ), unit );
        } else {
            data = std::make_unique<KeyDataArray<double>>( std::move( data2 ), unit );
        }
    } else if ( data_type == class_type::COMPLEX ) {
        // We are dealing with complex values
        Array<std::complex<double>> data2( values.size() );
        Units unit;
        for ( size_t i = 0; i < values.size(); i++ ) {
            Units unit2;
            std::tie( data2( i ), unit2 ) = readPair<std::complex<double>>( values[i] );
            if ( !unit2.isNull() )
                unit = unit2;
        }
        if ( values.size() == 1 ) {
            data = std::make_unique<KeyDataScalar<std::complex<double>>>( data2( 0 ), unit );
        } else {
            data = std::make_unique<KeyDataArray<std::complex<double>>>( std::move( data2 ), unit );
        }
    } else if ( data_type == class_type::BOX ) {
        Array<DatabaseBox> data2( values.size() );
        for ( size_t i = 0; i < values.size(); i++ )
            data2( i ) = DatabaseBox( values[i] );
        if ( values.size() == 1 ) {
            data = std::make_unique<KeyDataScalar<DatabaseBox>>( data2( 0 ) );
        } else {
            data = std::make_unique<KeyDataArray<DatabaseBox>>( std::move( data2 ) );
        }
    } else if ( data_type == class_type::ARRAY ) {
        // We are dealing with an Array
        size_t k = 0;
        for ( size_t i = 0; i < values[0].size(); i++ ) {
            if ( values[0][i] == ']' )
                k = i;
        }
        auto array_str = values[0].substr( 0, k + 1 );
        auto unit_str  = values[0].substr( k + 1 );
        Units unit( unit_str );
        if ( deblank( array_str.substr( 1, array_str.size() - 2 ) ).empty() ) {
            // We are dealing with an empty array
            data = std::make_unique<KeyDataArray<double>>( Array<double>( 0 ), unit );
        } else {
            // Get the array size
            size_t ndim = 1, dims[10] = { 1 };
            for ( size_t i = 1, d = 0; i < array_str.size() - 1; i++ ) {
                if ( array_str[i] == '[' ) {
                    d++;
                    ndim    = std::max( ndim, d + 1 );
                    dims[d] = 1;
                } else if ( array_str[i] == ']' ) {
                    d--;
                } else if ( array_str[i] == ',' ) {
                    dims[d]++;
                }
            }
            size_t dims2[10] = { 0 };
            for ( size_t d = 0; d < ndim; d++ )
                dims2[d] = dims[ndim - d - 1];
            ArraySize size( ndim, dims2 );
            // Get the Array values
            values.clear();
            values.resize( size.length() );
            for ( size_t i1 = 0, i2 = 0, j = 0; j < values.size(); j++ ) {
                while ( array_str[i1] == '[' || array_str[i1] == ']' || array_str[i1] == ',' ||
                        array_str[i1] == ' ' || array_str[i1] == '\n' )
                    i1++;
                i2 = i1 + 1;
                while ( array_str[i2] != '[' && array_str[i2] != ']' && array_str[i2] != ',' &&
                        array_str[i2] != ' ' )
                    i2++;
                values[j] = deblank( array_str.substr( i1, i2 - i1 ) );
                i1        = i2;
            }
            // Create the array
            if ( array_str.find( '"' ) != std::string::npos ) {
                // String array
                Array<std::string> A( size );
                for ( size_t i = 0; i < size.length(); i++ )
                    A( i ) = std::string( values[i].substr( 1, values[i].size() - 2 ) );
                data = std::make_unique<KeyDataArray<std::string>>( std::move( A ), unit );
            } else if ( array_str.find( '(' ) != std::string::npos ) {
                // Complex array
                Array<std::complex<double>> A( size );
                A.fill( 0.0 );
                for ( size_t i = 0; i < size.length(); i++ )
                    std::tie( A( i ), std::ignore ) = readPair<std::complex<double>>( values[i] );
                data = std::make_unique<KeyDataArray<std::complex<double>>>( std::move( A ), unit );
            } else if ( array_str.find( "true" ) != std::string::npos ||
                        array_str.find( "false" ) != std::string::npos ) {
                // Bool array
                Array<bool> A( size );
                A.fill( false );
                for ( size_t i = 0; i < values.size(); i++ ) {
                    if ( !strcmpi( values[i], "true" ) && !strcmpi( values[i], "false" ) )
                        throw std::logic_error( "Error converting " + std::string( key ) +
                                                " to logical array" );
                    A( i ) = strcmpi( values[i], "true" );
                }
                data = std::make_unique<KeyDataArray<bool>>( std::move( A ), unit );
            } else if ( array_str.find( '.' ) != std::string::npos ||
                        array_str.find( 'e' ) != std::string::npos ) {
                // Floating point array
                Array<double> A( size );
                A.fill( 0.0 );
                for ( size_t i = 0; i < size.length(); i++ )
                    std::tie( A( i ), std::ignore ) = readPair<double>( values[i] );
                data = std::make_unique<KeyDataArray<double>>( std::move( A ), unit );
            } else {
                // Integer point array
                Array<int> A( size );
                A.fill( 0 );
                for ( size_t i = 0; i < size.length(); i++ )
                    std::tie( A( i ), std::ignore ) = readPair<int>( values[i] );
                data = std::make_unique<KeyDataArray<int>>( std::move( A ), unit );
            }
        }
    } else if ( data_type == class_type::DATABASE_ENTRY ) {
        std::vector<const KeyData *> data2( values.size(), nullptr );
        for ( size_t i = 0; i < values.size(); i++ ) {
            data2[i] = databaseKeys.find( std::string( values[i] ) )->second;
            usedKeys.insert( std::string( values[i] ) );
        }
        if ( values.size() == 1 ) {
            data = data2[0]->clone();
        } else {
            auto unit = data2[0]->unit();
            for ( size_t i = 0; i < values.size(); i++ ) {
                if ( data2[i]->arraySize().length() != 1 )
                    throw std::logic_error(
                        "Using multiple values from database only works for scalars: " +
                        std::string( key ) );
                AMP_INSIST( unit == data2[i]->unit(),
                            "Copying array of values requires all values to share the same units" );
            }
            if ( isType<bool>( data2 ) ) {
                AMP::Array<bool> x( values.size() );
                for ( size_t i = 0; i < values.size(); i++ )
                    x( i ) = dynamic_cast<const KeyDataScalar<bool> *>( data2[i] )->get();
                data = std::make_unique<KeyDataArray<bool>>( std::move( x ), unit );
            } else if ( isType<int>( data2 ) ) {
                AMP::Array<int> x( values.size() );
                for ( size_t i = 0; i < values.size(); i++ )
                    x( i ) = dynamic_cast<const KeyDataScalar<int> *>( data2[i] )->get();
                data = std::make_unique<KeyDataArray<int>>( std::move( x ), unit );
            } else if ( isType<double>( data2 ) ) {
                AMP::Array<double> x( values.size() );
                for ( size_t i = 0; i < values.size(); i++ )
                    x( i ) = dynamic_cast<const KeyDataScalar<double> *>( data2[i] )->get();
                data = std::make_unique<KeyDataArray<double>>( std::move( x ), unit );
            } else if ( isType<std::string>( data2 ) ) {
                AMP::Array<std::string> x( values.size() );
                for ( size_t i = 0; i < values.size(); i++ )
                    x( i ) = dynamic_cast<const KeyDataScalar<std::string> *>( data2[i] )->get();
                data = std::make_unique<KeyDataArray<std::string>>( std::move( x ), unit );
            } else {
                throw std::logic_error(
                    "Using multiple values from database - unable to convert data: " +
                    std::string( key ) );
            }
        }
    } else if ( data_type == class_type::RANGE ) {
        // We are dealing with a range
        AMP_INSIST( values.size() == 1, "Arrays of Ranges are not supported" );
        auto type = getRangeType( values[0] );
        if ( type == class_type::INT ) {
            auto [range, unit] = readPair<AMP::Range<int>>( values[0] );
            data = std::make_unique<KeyDataArray<int>>( AMP::Array<int>( range ), unit );
        } else if ( type == class_type::FLOAT ) {
            auto [range, unit] = readPair<AMP::Range<double>>( values[0] );
            data = std::make_unique<KeyDataArray<double>>( AMP::Array<double>( range ), unit );
        } else {
            throw std::logic_error( "Unknown type for range" );
        }
    } else if ( data_type == class_type::UNKNOWN ) {
        // Treat unknown data as a string
        if ( values.size() == 1 ) {
            std::string str( values[0] );
            data = std::make_unique<KeyDataScalar<std::string>>( std::move( str ) );
        } else {
            Array<std::string> data2( values.size() );
            for ( size_t i = 0; i < values.size(); i++ )
                data2( i ) = std::string( values[i].data(), values[i].size() );
            data = std::make_unique<KeyDataArray<std::string>>( std::move( data2 ) );
        }
    } else {
        throw std::logic_error( "Internal error" );
    }
    return std::make_tuple( std::move( data ), std::move( usedKeys ) );
}
static std::tuple<size_t, std::unique_ptr<KeyData>, std::set<std::string>>
read_value( std::string_view buffer,
            std::string_view key,
            const std::map<std::string, const KeyData *> &databaseKeys =
                std::map<std::string, const KeyData *>() )
{
    AMP_ASSERT( !buffer.empty() );
    // Split the value to an array of values
    size_t pos      = 0;
    token_type type = token_type::end;
    std::vector<std::string_view> values;
    class_type data_type = class_type::UNKNOWN;
    auto nextChar        = []( const char *c ) {
        while ( *c == ' ' )
            c++;
        return *c;
    };
    std::set<std::string> usedKeys;
    while ( type != token_type::newline ) {
        AMP_ASSERT( pos < buffer.size() );
        while ( buffer[pos] == ' ' || buffer[pos] == '\t' )
            pos++;
        size_t pos0 = pos;
        if ( buffer[pos0] == '@' && buffer[pos0 + 1] == '(' ) {
            // We are dealing with an equation
            data_type = class_type::EQUATION;
            while ( buffer[pos] != ';' && buffer[pos] != 0 && buffer[pos] != '\n' )
                pos++;
            AMP_INSIST( buffer[pos] == ';', "Equations must terminate with a ';'" );
            size_t i;
            std::tie( i, type ) = find_next_token( &buffer[pos] );
            pos += i;
        } else if ( buffer[pos0] == '(' ) {
            // We are dealing with a complex number
            data_type = class_type::COMPLEX;
            while ( buffer[pos] != ')' )
                pos++;
            size_t i;
            std::tie( i, type ) = find_next_token( &buffer[pos] );
            pos += i;
        } else if ( buffer[pos0] == '"' ) {
            // We are in a string
            data_type = class_type::STRING;
            pos++;
            while ( buffer[pos] != '"' )
                pos++;
            pos++;
            size_t i;
            std::tie( i, type ) = find_next_token( &buffer[pos] );
            pos += i;
        } else if ( buffer[pos0] == '[' && nextChar( &buffer[pos0 + 1] ) == '(' ) {
            // We are (probably) reading a SAMRAI box
            data_type = class_type::BOX;
            while ( buffer[pos] != ']' )
                pos++;
            size_t i;
            std::tie( i, type ) = find_next_token( &buffer[pos] );
            pos += i;
            // Check that it is a box and not an array of complex numbers
        } else if ( buffer[pos0] == '[' ) {
            // We are reading a multi-dimensional array
            data_type = class_type::ARRAY;
            int count = 1;
            pos       = pos0 + 1;
            while ( count != 0 && pos < buffer.size() ) {
                if ( buffer[pos] == '[' )
                    count++;
                if ( buffer[pos] == ']' )
                    count--;
                pos++;
            }
            size_t i;
            std::tie( i, type ) = find_next_token( &buffer[pos] );
            pos += i;
        } else {
            std::tie( pos, type ) = find_next_token( &buffer[pos0] );
            pos += pos0;
            if ( buffer[pos - 1] == '"' ) {
                while ( buffer[pos] != '"' )
                    pos++;
                size_t pos2           = pos + 1;
                std::tie( pos, type ) = find_next_token( &buffer[pos2] );
                pos += pos2;
            }
        }
        std::string_view tmp( &buffer[pos0], pos - pos0 - length( type ) );
        if ( !tmp.empty() ) {
            if ( tmp.back() == ',' )
                tmp = std::string_view( tmp.data(), tmp.size() - 1 );
        }
        tmp = deblank( tmp );
        values.push_back( deblank( tmp ) );
        if ( type == token_type::comma ) {
            // We have multiple values
            continue;
        }
        if ( type == token_type::line_comment || type == token_type::block_start ) {
            // We encountered a comment
            pos += skip_comment( &buffer[pos - length( type )] ) - length( type );
            break;
        }
    }
    // Check the type of each entry
    if ( data_type == class_type::UNKNOWN ) {
        data_type = getType( values[0], databaseKeys );
        for ( size_t i = 1; i < values.size(); i++ ) {
            auto type2 = getType( values[i], databaseKeys );
            if ( type2 == data_type ) {
                continue;
            } else if ( type2 == class_type::UNKNOWN ) {
                data_type = class_type::UNKNOWN;
                break;
            } else if ( ( type2 == class_type::INT || type2 == class_type::FLOAT ) &&
                        ( data_type == class_type::INT || data_type == class_type::FLOAT ) ) {
                data_type = class_type::FLOAT;
            } else if ( type2 != data_type ) {
                throw std::logic_error( "Mismatched types in '" + std::string( key ) + "'" );
            }
        }
    }
    if ( data_type == class_type::UNKNOWN ) {
        throw std::logic_error( "Unknown type in '" + std::string( key ) + "':\n   " +
                                std::string( buffer.data(), pos ) );
    }
    // Convert the string value to the database value
    auto data = createKeyData( key, data_type, values, databaseKeys );
    AMP_ASSERT( std::get<0>( data ) );
    return std::make_tuple(
        pos, std::move( std::get<0>( data ) ), std::move( std::get<1>( data ) ) );
}
template<class TYPE>
static std::unique_ptr<KeyData> applyOperator( const std::vector<std::unique_ptr<KeyData>> &data,
                                               const std::vector<char> &op )
{
    auto y = data[0]->getArray<TYPE>();
    auto u = data[0]->unit();
    for ( size_t i = 1; i < data.size(); i++ ) {
        auto x = data[i]->getArray<TYPE>();
        AMP_ASSERT( x.size() == y.size() );
        for ( size_t j = 0; j < x.length(); j++ ) {
            if ( op[i] == '+' ) {
                y( j ) += x( j );
                AMP_ASSERT( u == data[i]->unit() );
            } else if ( op[i] == '-' ) {
                y( j ) -= x( j );
                AMP_ASSERT( u == data[i]->unit() );
            } else if ( op[i] == '*' ) {
                y( j ) *= x( j );
                u *= data[i]->unit();
            } else if ( op[i] == '/' ) {
                y( j ) /= x( j );
                u /= data[i]->unit();
            } else {
                AMP_ERROR( "Unknown operator" );
            }
        }
    }
    return std::make_unique<KeyDataArray<TYPE>>( std::move( y ), u );
}
static std::tuple<size_t, std::unique_ptr<KeyData>, std::set<std::string>>
read_operator_value( std::string_view buffer,
                     std::string_view key,
                     const std::map<std::string, const KeyData *> &databaseKeys )
{
    size_t i        = 0;
    size_t pos      = 0;
    token_type type = token_type::equal;
    while ( type != token_type::line_comment && type != token_type::block_start &&
            type != token_type::newline && type != token_type::end ) {
        std::tie( i, type ) = find_next_token( &buffer[pos] );
        pos += i;
    }
    if ( type == token_type::line_comment && type != token_type::block_start )
        pos -= 2;
    auto buffer2           = buffer.substr( 0, pos );
    std::vector<int> index = { -1 };
    std::vector<char> ops  = { '\0' };
    for ( int i = 0, s = 0; i < (int) buffer2.size(); i++ ) {
        if ( buffer2[i] == '"' )
            s++;
        bool isOp =
            buffer2[i] == '+' || buffer2[i] == '-' || buffer2[i] == '*' || buffer2[i] == '/';
        if ( isOp && s % 2 == 0 ) {
            index.push_back( i );
            ops.push_back( buffer2[i] );
        }
    }
    index.push_back( buffer2.size() );
    ops.push_back( '\0' );
    AMP_ASSERT( ops.size() >= 3 );
    std::vector<std::unique_ptr<KeyData>> objs;
    std::set<std::string> usedKeys;
    for ( size_t i = 0; i < index.size() - 1; i++ ) {
        auto valBuf =
            std::string( buffer2.substr( index[i] + 1, index[i + 1] - index[i] - 1 ) ) + '\n';
        auto tmp = read_value( valBuf.data(), key, databaseKeys );
        objs.push_back( std::move( std::get<1>( tmp ) ) );
        for ( auto &key : std::get<2>( tmp ) )
            usedKeys.insert( key );
    }
    std::unique_ptr<KeyData> data;
    if ( isType<std::string>( objs ) ) {
        auto y = objs[0]->getArray<std::string>();
        for ( size_t i = 1; i < objs.size(); i++ ) {
            AMP_ASSERT( ops[i] == '+' );
            y += objs[i]->getArray<std::string>();
        }
        data = std::make_unique<KeyDataArray<std::string>>( std::move( y ) );
    } else if ( isType<bool>( objs ) ) {
        AMP_ERROR( "Not finished" );
    } else if ( isType<int64_t>( objs ) || isType<double>( objs ) ) {
        data = applyOperator<double>( objs, ops );
    } else if ( isType<std::complex<double>>( objs ) ) {
        data = applyOperator<std::complex<double>>( objs, ops );
    } else {
        AMP_ERROR( "Not finished" );
    }
    return std::make_tuple( pos, std::move( data ), std::move( usedKeys ) );
}
static std::string generateMsg( const std::string &errMsgPrefix,
                                const char *msg,
                                int line,
                                const std::string_view &key = "" )
{
    auto out = errMsgPrefix + msg;
    if ( !key.empty() )
        out += ": '" + std::string( key ) + "'";
    return out + " in input at line " + std::to_string( line );
}
static std::tuple<size_t, std::set<std::string>>
loadDatabase( const std::string &errMsgPrefix,
              std::string_view buffer,
              Database &db,
              std::map<std::string, const KeyData *> databaseKeys = {},
              int line0                                           = 0 )
{
    size_t pos = 0;
    std::set<std::string> usedKeys;
    auto updateUsedKeys = [&db, &usedKeys]( const std::set<std::string> &keys ) {
        for ( auto &key : keys ) {
            if ( !db.getData( key ) )
                usedKeys.insert( key );
        }
    };
    while ( pos < buffer.size() ) {
        size_t i;
        token_type type;
        std::tie( i, type ) = find_next_token( &buffer[pos] );
        std::string_view tmp( &buffer[pos], i - length( type ) );
        const auto key = deblank( tmp );
        int line =
            line0 + std::count_if( &buffer[0], &buffer[pos], []( char c ) { return c == '\n'; } );
        if ( type == token_type::line_comment || type == token_type::block_start ) {
            // Comment
            size_t k = skip_comment( &buffer[pos] );
            pos += k;
        } else if ( type == token_type::newline ) {
            if ( !key.empty() ) {
                auto msg = errMsgPrefix + "Key is not assigned a value: " + std::string( key );
                AMP_ERROR( generateMsg( errMsgPrefix, "Key is not assigned a value", line, key ) );
            }
            pos += i;
        } else if ( type == token_type::equal ) {
            // Reading key/value pair
            AMP_INSIST( !key.empty(), generateMsg( errMsgPrefix, "Empty key", line ) );
            pos += i;
            std::unique_ptr<KeyData> data;
            try {
                auto tmp = read_value( &buffer[pos], key, databaseKeys );
                i        = std::get<0>( tmp );
                data     = std::move( std::get<1>( tmp ) );
                updateUsedKeys( std::get<2>( tmp ) );
            } catch ( StackTrace::abort_error &err ) {
                auto msg = generateMsg( errMsgPrefix, "Error loading key", line, key );
                msg += "\nCaught error in file " + std::string( err.source.file_name() ) +
                       " at line " + std::to_string( err.source.line() ) + "\n" + "   " +
                       err.message;
                AMP_ERROR( msg );
            } catch ( std::exception &err ) {
                auto msg = generateMsg( errMsgPrefix, "Error loading key", line, key );
                msg += "\nUnhandled exception:\n   " + std::string( err.what() );
                AMP_ERROR( msg );
            }
            if ( !data )
                AMP_ERROR(
                    generateMsg( errMsgPrefix, "Error loading key (empty data)", line, key ) );
            databaseKeys[std::string( key )] = data.get();
            db.putData( key, std::move( data ) );
            pos += i;
        } else if ( type == token_type::define ) {
            // We are defining a token in terms of others
            pos += i + 1;
            auto [j, data, keys] = read_operator_value( &buffer[pos], key, databaseKeys );
            updateUsedKeys( keys );
            databaseKeys[std::string( key )] = data.get();
            db.putData( key, std::move( data ) );
            pos += j;
        } else if ( type == token_type::bracket ) {
            // Read database
            AMP_INSIST( !key.empty(), generateMsg( errMsgPrefix, "Empty key", line ) );
            pos += i;
            auto database = std::make_unique<Database>();
            auto tmp =
                loadDatabase( errMsgPrefix, buffer.substr( pos ), *database, databaseKeys, line );
            pos += std::get<0>( tmp );
            updateUsedKeys( std::get<1>( tmp ) );
            database->setName( std::string( key ) );
            databaseKeys[std::string( key )] = database.get();
            db.putData( key, std::move( database ) );
        } else if ( type == token_type::end_bracket || type == token_type::end ) {
            // Finished with the database
            pos += i;
            break;
        } else if ( type == token_type::import_data ) {
            // Import another file into the current database
            auto pos2 = buffer.substr( pos ).find( ">>>" );
            AMP_INSIST( pos2 != std::string::npos,
                        generateMsg( errMsgPrefix, "Error reading line:", line ) );
            auto filename = deblank( buffer.substr( pos + 3, pos2 - 3 ) );
            pos += pos2 + 3;
            if ( ( filename[0] == '"' && filename.back() == '"' ) ||
                 ( filename[0] == '\'' && filename.back() == '\'' ) )
                filename = filename.substr( 1, filename.size() - 2 );
            auto db2 = Database::parseInputFile( std::string( filename ) );
            for ( auto &key : db2->getAllKeys() )
                db.putData( key, db2->getData( key )->clone() );
        } else {
            throw std::logic_error( "Error loading data" );
        }
    }
    return std::tie( pos, usedKeys );
}


/********************************************************************
 * Read YAML file                                                    *
 ********************************************************************/
static inline std::tuple<std::string_view, std::string_view> splitYAML( std::string_view line )
{
    size_t pos = line.find_first_not_of( ' ' );
    if ( line[pos] == '-' )
        pos++;
    size_t pos2 = line.find( ':' );
    auto key    = deblank( line.substr( pos, pos2 - pos ) );
    auto value  = deblank( line.substr( pos2 + 1 ) );
    return std::tie( key, value );
}
static inline std::unique_ptr<KeyData> makeKeyData( std::vector<Database> &&data )
{
    if ( data.size() == 1 )
        return std::make_unique<Database>( std::move( data[0] ) );
    return std::make_unique<DatabaseVector>( std::move( data ) );
}
static inline size_t getLine( const char *buffer, size_t pos )
{
    size_t i = pos;
    while ( buffer[i] != 0 && buffer[i] != '\n' ) {
        i++;
    }
    return i;
}
size_t loadYAMLDatabase( const char *buffer, Database &db, size_t pos = 0, size_t indent = 0 )
{
    std::string lastKey;
    while ( buffer[pos] != 0 ) {
        // Get the next line
        auto pos2 = getLine( buffer, pos );
        std::string_view line( &buffer[pos], pos2 - pos );
        // Remove the comments
        line = line.substr( 0, line.find( '#' ) );
        // Find the first non-whitespace character
        size_t p = line.find_first_not_of( ' ' );
        if ( p < indent )
            return pos; // End of list
        // Remove empty space
        line = deblank( line );
        if ( line.empty() ) {
            pos = pos2 + 1;
            continue;
        }
        if ( line[0] == '-' ) {
            // We are dealing with a new item (database)
            auto p2 = line.find( ':' );
            std::string name;
            if ( p2 != std::string::npos )
                name = deblank( line.substr( p2 + 1 ) );
            else
                name = deblank( line.substr( 1 ) );
            auto db2 = std::make_unique<Database>( name );
            pos      = loadYAMLDatabase( buffer, *db2, pos2 + 1, p + 1 );
            db.putDatabase( name, std::move( db2 ) );
            continue;
        }
        auto [key, value] = splitYAML( line );
        AMP_ASSERT( !key.empty() );
        if ( value.empty() ) {
            // Treat the key as a new database to load
            Database tmp;
            pos = loadYAMLDatabase( buffer, tmp, pos2 + 1, p + 1 );
            for ( auto key2 : tmp.getAllKeys() )
                db.putData( key2, tmp.getData( key2 )->clone() );
            continue;
        } else if ( value == "|" ) {
            // Special case with block scalars
            pos2++;
            size_t pos3 = getLine( buffer, pos2 );
            std::string_view line2( &buffer[pos2 + 1], pos3 - pos2 );
            size_t p0 = line2.find_first_not_of( ' ' );
            Array<double> x;
            while ( true ) {
                pos3      = getLine( buffer, pos2 );
                line2     = std::string_view( &buffer[pos2], pos3 - pos2 );
                size_t p2 = line2.find_first_not_of( ' ' );
                if ( p2 < p0 )
                    break;
                pos2  = pos3 + 1;
                line2 = deblank( line2 );
                std::string line3( line2 );
                line3 = AMP::Utilities::strrep( line3, "  ", " " );
                line3 = AMP::Utilities::strrep( line3, " ", "," );
                line3 += '\n';
                auto tmp = read_value( line3, key );
                auto y   = std::get<1>( tmp )->convertToDouble();
                size_t i = x.size( 0 );
                x.resize( i + 1, y.length() );
                for ( size_t j = 0; j < y.length(); j++ )
                    x( i, j ) = y( j );
            }
            auto data = std::make_unique<KeyDataArray<double>>( std::move( x ) );
            db.putData( key, std::move( data ) );
        } else if ( !value.empty() ) {
            std::unique_ptr<KeyData> entry;
            try {
                auto tmp = read_value( value.data(), key );
                entry    = std::move( std::get<1>( tmp ) );
            } catch ( ... ) {
                entry = std::make_unique<KeyDataScalar<std::string>>( std::string( value ) );
            }
            try {
                if ( entry->convertToDouble() == Array<double>( 1, 0 ) )
                    entry = std::make_unique<KeyDataScalar<std::string>>( std::string( value ) );
            } catch ( ... ) {
            }
            if ( !entry )
                AMP_ERROR( "Unable to parse value: " + std::string( value ) );
            db.putData( key, std::move( entry ) );
        } else {
            AMP_ERROR( "Not finished" );
        }
        pos = pos2 + 1;
    }
    return pos;
}
std::unique_ptr<KeyData> Database::readYAML( std::string_view filename, source_location src )
{
    // Read the file into memory
    auto buffer = readFile( std::string( filename ), src );
    // Read the file
    std::vector<Database> data;
    for ( size_t i = 0; i < buffer.size(); ) {
        data.resize( data.size() + 1 );
        i = loadYAMLDatabase( &buffer[i], data.back() ) + 1;
    }
    // Return the result
    return makeKeyData( std::move( data ) );
}


/********************************************************************
 * Read input database file                                          *
 ********************************************************************/
void Database::readDatabase( const std::string &filename, source_location src )
{
    // Read the input file into memory
    auto buffer = readFile( filename, src );
    // Create the database entries
    loadDatabase( "Error loading database from file \"" + filename + "\"\n", buffer.data(), *this );
}
std::unique_ptr<Database> Database::createFromString( std::string_view data )
{
    auto db = std::make_unique<Database>();
    loadDatabase( "Error creating database from file\n", data, *db );
    return db;
}


} // namespace AMP
