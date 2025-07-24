#ifndef included_AMP_Database_hpp
#define included_AMP_Database_hpp

#include "AMP/IO/HDF.h"
#include "AMP/utils/AMP_MPI_pack.hpp"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/UtilityMacros.h"

#include <cstddef>
#include <iomanip>
#include <limits>
#include <tuple>
#include <type_traits>


#define DATABASE_ERROR( SRC, ... )                                          \
    do {                                                                    \
        char msg[1000];                                                     \
        snprintf( msg, sizeof( msg ), __VA_ARGS__ );                        \
        if ( SRC.empty() )                                                  \
            StackTrace::Utilities::abort( msg, SOURCE_LOCATION_CURRENT() ); \
        else                                                                \
            StackTrace::Utilities::abort( msg, SRC );                       \
    } while ( 0 )
#define DATABASE_WARNING( SRC, ... )                                                          \
    do {                                                                                      \
        char msg[4096] = "WARNING: ";                                                         \
        char *ptr      = &msg[9];                                                             \
        ptr += snprintf( ptr, 4000, __VA_ARGS__ );                                            \
        if ( SRC.empty() ) {                                                                  \
            sprintf( ptr, "\nWarning called in %s on line %i", __FILE__, __LINE__ );          \
        } else {                                                                              \
            sprintf( ptr, "\nWarning called in %s on line %i", SRC.file_name(), SRC.line() ); \
        }                                                                                     \
        AMP::pout << msg << std::endl;                                                        \
        AMP::plog << msg << std::endl << std::flush;                                          \
    } while ( 0 )
#define DATABASE_INSIST( TEST, SRC, ... )                                       \
    do {                                                                        \
        if ( !( TEST ) ) {                                                      \
            char msg[1000];                                                     \
            snprintf( msg, sizeof( msg ), __VA_ARGS__ );                        \
            if ( SRC.empty() )                                                  \
                StackTrace::Utilities::abort( msg, SOURCE_LOCATION_CURRENT() ); \
            else                                                                \
                StackTrace::Utilities::abort( msg, SRC );                       \
        }                                                                       \
    } while ( 0 )


namespace AMP {


/********************************************************************
 * Helper function to perform data conversion                        *
 ********************************************************************/
template<class TYPE>
static inline bool compare( const TYPE &x, const TYPE &y )
{
    bool test = x == y;
    if constexpr ( std::is_same_v<TYPE, double> ) {
        test = test || fabs( x - y ) <= 1e-12 * fabs( x + y );
        test = test || ( ( x != x ) && ( y != y ) );
    } else if constexpr ( std::is_same_v<TYPE, float> ) {
        test = test || fabs( x - y ) <= 1e-7 * fabs( x + y );
        test = test || ( ( x != x ) && ( y != y ) );
    } else if constexpr ( std::is_same_v<TYPE, std::complex<double>> ) {
        test = test || std::abs( x - y ) <= 1e-12 * std::abs( x + y );
        test = test || ( ( x != x ) && ( y != y ) );
    } else if constexpr ( std::is_same_v<TYPE, std::complex<float>> ) {
        test = test || std::abs( x - y ) <= 1e-7 * std::abs( x + y );
        test = test || ( ( x != x ) && ( y != y ) );
    }
    return test;
}
template<class TYPE>
static constexpr TYPE getTol()
{
    if constexpr ( std::is_same_v<TYPE, float> ) {
        return 1e-6;
    } else if constexpr ( std::is_same_v<TYPE, double> ) {
        return 1e-12;
    } else {
        return 0;
    }
}
template<class TYPE>
static constexpr TYPE abs( const TYPE &x )
{
    if constexpr ( std::is_signed_v<TYPE> ) {
        return x < 0 ? -x : x;
    } else {
        return x;
    }
}
template<class TYPE1, class TYPE2>
Array<TYPE2> convert( const Array<TYPE1> &x )
{
    if constexpr ( std::is_same_v<TYPE1, TYPE2> ) {
        return x;
    } else if constexpr ( std::is_same_v<TYPE1, bool> && std::is_arithmetic_v<TYPE2> ) {
        Array<TYPE2> y( x.size() );
        for ( size_t i = 0; i < x.length(); i++ )
            y( i ) = static_cast<TYPE2>( x( i ) ? 1 : 0 );
        return y;
    } else if constexpr ( std::is_arithmetic_v<TYPE1> && std::is_arithmetic_v<TYPE2> ) {
        Array<TYPE2> y( x.size() );
        y.fill( 0 );
        bool pass = true;
        auto tol  = getTol<TYPE1>();
        for ( size_t i = 0; i < x.length(); i++ ) {
            y( i ) = static_cast<TYPE2>( x( i ) );
            pass   = pass && abs( static_cast<TYPE1>( y( i ) - x( i ) ) ) <= tol * x( i );
        }
        if ( !pass ) {
            std::string type1 = typeid( TYPE1 ).name();
            std::string type2 = typeid( TYPE2 ).name();
            std::string msg = "Converting " + type1 + "-" + type2 + " results in loss of precision";
            AMP_WARNING( msg );
        }
        return y;
    } else {
        std::string type1 = typeid( TYPE1 ).name();
        std::string type2 = typeid( TYPE2 ).name();
        std::string msg =
            "Invalid conversion: " + type1 + "-" + type2 + " results in loss of precision";
        throw std::logic_error( msg );
    }
}


/********************************************************************
 * Basic classes for primitive data types                            *
 ********************************************************************/
template<class TYPE>
inline void printValue( std::ostream &os, const TYPE &value )
{
    if constexpr ( std::is_floating_point_v<TYPE> ) {
        if ( AMP::Utilities::isNaN( value ) ) {
            os << "nan";
        } else if ( AMP::Utilities::isInf( value ) ) {
            os << "inf";
        } else if ( AMP::Utilities::isInf( -value ) ) {
            os << "-inf";
        } else {
            os << std::setprecision( 14 ) << value;
        }
    } else if constexpr ( std::is_same_v<TYPE, bool> ) {
        if ( value )
            os << "true";
        else
            os << "false";
    } else if constexpr ( std::is_same_v<TYPE, char> || std::is_same_v<TYPE, std::string> ) {
        os << '"' << value << '"';
    } else if constexpr ( std::is_integral_v<TYPE> ) {
        os << static_cast<int64_t>( value );
    } else {
        os << value;
    }
}
class EmptyKeyData final : public KeyData
{
public:
    EmptyKeyData() {}
    virtual ~EmptyKeyData() {}
    typeID getClassType() const override { return getTypeID<EmptyKeyData>(); }
    std::unique_ptr<KeyData> clone() const override { return std::make_unique<EmptyKeyData>(); }
    void print( std::ostream &os, std::string_view = "", bool = true, bool = false ) const override
    {
        os << std::endl;
    }
    typeID getDataType() const override { return AMP::typeID(); }
    bool is_floating_point() const override { return true; }
    bool is_integral() const override { return true; }
    ArraySize arraySize() const override { return ArraySize(); }
    Array<double> convertToDouble() const override { return Array<double>(); }
    Array<int64_t> convertToInt64() const override { return Array<int64_t>(); }
    bool operator==( const KeyData &rhs ) const override { return rhs.convertToDouble().empty(); }
    size_t packSize() const override { return 0; }
    size_t pack( std::byte * ) const override { return 0; }
    size_t unpack( const std::byte * ) override { return 0; }
    void writeHDF5( int64_t, const std::string & ) const override {}
    void readHDF5( int64_t, const std::string & ) override {}
};
class EquationKeyData final : public KeyData
{
public:
    EquationKeyData() = default;
    EquationKeyData( std::string_view eq, const Units &unit = Units() );
    EquationKeyData( MathExpr eq, const Units &unit = Units() );
    virtual ~EquationKeyData();
    typeID getClassType() const override { return getTypeID<EquationKeyData>(); }
    std::unique_ptr<KeyData> clone() const override;
    void print( std::ostream &, std::string_view = "", bool = true, bool = false ) const override;
    typeID getDataType() const override;
    bool is_floating_point() const override;
    bool is_integral() const override;
    ArraySize arraySize() const override;
    Array<double> convertToDouble() const override;
    Array<int64_t> convertToInt64() const override;
    bool operator==( const KeyData &rhs ) const override;
    const auto &getEq() const { return *d_eq; }
    size_t packSize() const override;
    size_t pack( std::byte *buf ) const override;
    size_t unpack( const std::byte * ) override;
    void writeHDF5( int64_t, const std::string & ) const override;
    void readHDF5( int64_t, const std::string & ) override;

private:
    MathExpr *d_eq = nullptr;
};
template<class TYPE>
class KeyDataScalar final : public KeyData
{
public:
    KeyDataScalar() = default;
    explicit KeyDataScalar( TYPE data, const Units &unit = Units() )
        : KeyData( unit ), d_data( std::move( data ) )
    {
        static_assert( !std::is_same_v<TYPE, std::vector<bool>::reference> );
    }
    virtual ~KeyDataScalar() {}
    typeID getClassType() const override { return getTypeID<KeyDataScalar>(); }
    std::unique_ptr<KeyData> clone() const override
    {
        return std::make_unique<KeyDataScalar>( d_data, d_unit );
    }
    void print( std::ostream &os,
                std::string_view indent = "",
                bool                    = true,
                bool printType          = false ) const override
    {
        os << indent;
        printValue( os, d_data );
        if ( !d_unit.isNull() )
            os << " " << d_unit.str();
        if ( printType )
            os << "  // " << getDataType().name;
        os << std::endl;
    }
    typeID getDataType() const override { return getTypeID<TYPE>(); }
    bool is_floating_point() const override { return std::is_floating_point<TYPE>(); }
    bool is_integral() const override { return std::is_integral<TYPE>(); }
    ArraySize arraySize() const override { return ArraySize( 1 ); }
    Array<double> convertToDouble() const override
    {
        if constexpr ( std::is_arithmetic_v<TYPE> ) {
            Array<TYPE> x( 1 );
            x( 0 ) = d_data;
            return convert<TYPE, double>( x );
        } else {
            DATABASE_ERROR( SOURCE_LOCATION_CURRENT(), "Unable to convert type" );
            return {};
        }
    }
    Array<int64_t> convertToInt64() const override
    {
        if constexpr ( std::is_arithmetic_v<TYPE> ) {
            Array<TYPE> x( 1 );
            x( 0 ) = d_data;
            return convert<TYPE, int64_t>( x );
        } else {
            DATABASE_ERROR( SOURCE_LOCATION_CURRENT(), "Unable to convert type" );
            return {};
        }
    }
    bool operator==( const KeyData &rhs ) const override;
    const TYPE &get() const { return d_data; }
    size_t packSize() const override { return AMP::packSize( d_unit ) + AMP::packSize( d_data ); }
    size_t pack( std::byte *buf ) const override
    {
        size_t N = 0;
        N += AMP::pack( d_unit, &buf[N] );
        N += AMP::pack( d_data, &buf[N] );
        return N;
    }
    size_t unpack( const std::byte *buf ) override
    {
        size_t N = 0;
        N += AMP::unpack( d_unit, &buf[N] );
        N += AMP::unpack( d_data, &buf[N] );
        return N;
    }
    void writeHDF5( int64_t fid, const std::string &name ) const override
    {
        if constexpr ( AMP::is_shared_ptr_v<TYPE> ) {
            typedef typename TYPE::element_type TYPE1;
            typedef typename std::remove_cv_t<TYPE1> TYPE2;
            AMP::IO::writeHDF5<TYPE2>( fid, name, *d_data );
        } else {
            typedef typename std::remove_reference_t<TYPE> TYPE1;
            typedef typename std::remove_cv_t<TYPE1> TYPE2;
            AMP::IO::writeHDF5<TYPE2>( fid, name, d_data );
        }
    }
    void readHDF5( [[maybe_unused]] int64_t fid, [[maybe_unused]] const std::string &name ) override
    {
        if constexpr ( AMP::is_shared_ptr_v<TYPE> ) {
            typedef typename TYPE::element_type TYPE2;
            if constexpr ( std::is_const_v<TYPE2> )
                AMP_ERROR( "Unable to read into const object" );
            else
                AMP::IO::readHDF5( fid, name, *d_data );
        } else if constexpr ( std::is_const_v<TYPE> ) {
            AMP_ERROR( "Unable to read into const object" );
        } else {
            typedef typename std::remove_reference_t<TYPE> TYPE2;
            AMP::IO::readHDF5<TYPE2>( fid, name, d_data );
        }
    }

private:
    TYPE d_data = TYPE();
};
template<class TYPE>
class KeyDataArray final : public KeyData
{
public:
    KeyDataArray() = default;
    explicit KeyDataArray( Array<TYPE> data, const Units &unit = Units() )
        : KeyData( unit ), d_data( std::move( data ) )
    {
        static_assert( !std::is_same_v<TYPE, std::vector<bool>::reference> );
        data.clear(); // Suppress cppclean warning
    }
    virtual ~KeyDataArray() {}
    typeID getClassType() const override { return getTypeID<KeyDataArray>(); }
    std::unique_ptr<KeyData> clone() const override
    {
        return std::make_unique<KeyDataArray>( d_data, d_unit );
    }
    void print( std::ostream &os,
                std::string_view indent = "",
                bool                    = true,
                bool printType          = false ) const override
    {
        os << indent;
        if ( d_data.ndim() == 1 ) {
            printResursive( os, d_data );
        } else {
            os << "[";
            printResursive( os, d_data );
            os << "]";
        }
        if ( !d_unit.isNull() )
            os << " " << d_unit.str();
        if ( printType )
            os << "  // " << getDataType().name;
        os << std::endl;
    }
    static void printResursive( std::ostream &os, const Array<TYPE> &x )
    {
        int ndim = x.ndim();
        if ( ndim > 1 ) {
            for ( size_t i = 0; i < x.size( ndim - 1 ); i++ ) {
                if ( i > 0 )
                    os << ",";
                os << "[";
                size_t dims[5] = { 1 };
                for ( int d = 0; d < ndim - 1; d++ )
                    dims[d] = x.size( d );
                ArraySize s( ndim - 1, dims );
                Array<TYPE> y;
                y.viewRaw( s, const_cast<TYPE *>( &x( i * s.length() ) ) );
                printResursive( os, y );
                os << ']';
            }
        } else {
            for ( size_t i = 0; i < x.length(); i++ ) {
                if ( i > 0 )
                    os << ", ";
                printValue( os, x( i ) );
            }
        }
    }
    typeID getDataType() const override { return getTypeID<TYPE>(); }
    bool is_floating_point() const override { return std::is_floating_point<TYPE>(); }
    bool is_integral() const override { return std::is_integral<TYPE>(); }
    ArraySize arraySize() const override { return d_data.size(); }
    Array<double> convertToDouble() const override { return convert<TYPE, double>( d_data ); }
    Array<int64_t> convertToInt64() const override { return convert<TYPE, int64_t>( d_data ); }
    bool operator==( const KeyData &rhs ) const override;
    const Array<TYPE> &get() const { return d_data; }
    size_t packSize() const override { return AMP::packSize( d_unit ) + AMP::packSize( d_data ); }
    size_t pack( std::byte *buf ) const override
    {
        size_t N = 0;
        N += AMP::pack( d_unit, &buf[N] );
        N += AMP::pack( d_data, &buf[N] );
        return N;
    }
    size_t unpack( const std::byte *buf ) override
    {
        size_t N = 0;
        N += AMP::unpack( d_unit, &buf[N] );
        N += AMP::unpack( d_data, &buf[N] );
        return N;
    }
    void writeHDF5( int64_t fid, const std::string &name ) const override
    {
        if constexpr ( AMP::is_shared_ptr_v<TYPE> ) {
            typedef typename TYPE::element_type TYPE1;
            typedef typename AMP::remove_cvref_t<TYPE1> TYPE2;
            if constexpr ( std::is_copy_constructible_v<TYPE2> ) {
                AMP::Array<TYPE2> y( d_data.size() );
                for ( size_t i = 0; i < d_data.length(); i++ )
                    y( i ) = *d_data( i );
                AMP::IO::writeHDF5( fid, name, y );
            } else {
                throw std::logic_error( name + " is not copy constructible" );
            }
        } else {
            AMP::IO::writeHDF5( fid, name, d_data );
        }
    }
    void readHDF5( int64_t fid, const std::string &name ) override
    {
        if constexpr ( AMP::is_shared_ptr_v<TYPE> ) {
            typedef typename TYPE::element_type TYPE1;
            typedef typename AMP::remove_cvref_t<TYPE1> TYPE2;
            if constexpr ( std::is_copy_constructible_v<TYPE2> ) {
                AMP::Array<TYPE2> y;
                AMP::IO::readHDF5( fid, name, y );
                d_data.resize( y.size() );
                for ( size_t i = 0; i < d_data.length(); i++ )
                    d_data( i ) = std::make_shared<TYPE2>( y( i ) );
            } else {
                throw std::logic_error( name + " is not copy constructible" );
            }
        } else {
            AMP::IO::readHDF5( fid, name, d_data );
        }
    }

private:
    Array<TYPE> d_data;
};
class DatabaseVector final : public KeyData
{
public:
    DatabaseVector() = default;
    explicit DatabaseVector( std::vector<Database> &&data ) : d_data( std::move( data ) ) {}
    explicit DatabaseVector( const std::vector<Database> &data )
    {
        d_data.resize( data.size() );
        for ( size_t i = 0; i < data.size(); i++ )
            d_data[i] = std::move( *data[i].cloneDatabase() );
    }
    DatabaseVector( DatabaseVector &&rhs ) : d_data( std::move( rhs.d_data ) ) {}
    DatabaseVector &operator=( DatabaseVector &&rhs )
    {
        d_data = std::move( rhs.d_data );
        return *this;
    }
    virtual ~DatabaseVector() {}
    typeID getClassType() const override { return getTypeID<DatabaseVector>(); }
    std::unique_ptr<KeyData> clone() const override
    {
        return std::make_unique<DatabaseVector>( d_data );
    }
    void print( std::ostream &os,
                std::string_view indent = "",
                bool sort               = true,
                bool printType          = false ) const override
    {
        std::string indent2 = std::string( indent ) + "   ";
        for ( const auto &data : d_data ) {
            data.print( os, indent2, sort, printType );
        }
    }
    typeID getDataType() const override { return getTypeID<Database>(); }
    bool is_floating_point() const override { return false; }
    bool is_integral() const override { return false; }
    ArraySize arraySize() const override { return ArraySize( d_data.size() ); }
    Array<double> convertToDouble() const override
    {
        throw std::logic_error( "Cannot convert DatabaseVector to double" );
    }
    Array<int64_t> convertToInt64() const override
    {
        throw std::logic_error( "Cannot convert DatabaseVector to int64" );
    }
    bool operator==( const KeyData &rhs ) const override
    {
        auto rhs2 = dynamic_cast<const DatabaseVector *>( &rhs );
        if ( rhs2 == nullptr )
            throw std::logic_error( "rhs is not a DatabaseVector" );
        return d_data == rhs2->d_data;
    }
    const std::vector<Database> &get() const { return d_data; }
    size_t packSize() const override
    {
        size_t bytes = sizeof( size_t );
        for ( const auto &db : d_data )
            bytes += db.packSize();
        return bytes;
    }
    size_t pack( std::byte *buf ) const override
    {
        size_t N = AMP::pack( d_data.size(), buf );
        for ( const auto &db : d_data )
            N += db.pack( &buf[N] );
        return N;
    }
    size_t unpack( const std::byte *buf ) override
    {
        size_t N_db;
        size_t N = AMP::unpack( N_db, buf );
        d_data.clear();
        d_data.resize( N_db );
        for ( auto &db : d_data )
            N += db.unpack( &buf[N] );
        return N;
    }
    void writeHDF5( int64_t fid, const std::string &name ) const override
    {
        AMP::IO::writeHDF5( fid, name, d_data );
    }
    void readHDF5( int64_t fid, const std::string &name ) override
    {
        AMP::IO::readHDF5( fid, name, d_data );
    }

private:
    std::vector<Database> d_data;
};


/********************************************************************
 * KeyDataScalar::operator==                                         *
 ********************************************************************/
template<class TYPE>
bool KeyDataScalar<TYPE>::operator==( const KeyData &rhs ) const
{
    auto tmp1 = dynamic_cast<const KeyDataScalar<TYPE> *>( &rhs );
    auto tmp2 = dynamic_cast<const KeyDataArray<TYPE> *>( &rhs );
    if ( tmp1 ) {
        return compare( d_data, tmp1->d_data );
    } else if ( tmp2 ) {
        if ( tmp2->get().size() != 1 )
            return false;
        return compare( d_data, tmp2->get()( 0 ) );
    } else if ( ( is_floating_point() || is_integral() ) &&
                ( rhs.is_floating_point() || rhs.is_integral() ) ) {
        auto data1 = convertToDouble();
        auto data2 = rhs.convertToDouble();
        if ( data1.size() != data2.size() )
            return false;
        bool test = true;
        for ( size_t i = 0; i < data1.length(); i++ )
            test = test && compare( data1( i ), data2( i ) );
        return test;
    }
    return false;
}
template<class TYPE>
bool KeyDataArray<TYPE>::operator==( const KeyData &rhs ) const
{
    auto tmp1 = dynamic_cast<const KeyDataScalar<TYPE> *>( &rhs );
    auto tmp2 = dynamic_cast<const KeyDataArray<TYPE> *>( &rhs );
    if ( tmp1 ) {
        if ( d_data.size() != 1 )
            return false;
        return compare( d_data( 0 ), tmp1->get() );
    } else if ( tmp2 ) {
        return compare( d_data, tmp2->get() );
    } else if ( ( is_floating_point() || is_integral() ) &&
                ( rhs.is_floating_point() || rhs.is_integral() ) ) {
        auto data1 = convertToDouble();
        auto data2 = rhs.convertToDouble();
        if ( data1.size() != data2.size() )
            return false;
        bool test = true;
        for ( size_t i = 0; i < data1.length(); i++ )
            test = test && compare( data1( i ), data2( i ) );
        return test;
    }
    return false;
}


/********************************************************************
 * Get data                                                          *
 ********************************************************************/
template<class TYPE>
void scaleData( Array<TYPE> &data, double factor );
template<class TYPE>
void scaleData( TYPE &data, double factor );
template<class TYPE>
TYPE KeyData::getScalar( const Units &unit, std::string_view name, source_location src ) const
{
    size_t length = arraySize().length();
    DATABASE_INSIST( length == 1, src, "Variable %s is not a scalar", name.data() );
    TYPE data       = TYPE();
    double factor   = convertUnits( unit, name );
    auto scalarData = dynamic_cast<const KeyDataScalar<TYPE> *>( this );
    auto arrayData  = dynamic_cast<const KeyDataArray<TYPE> *>( this );
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        if ( scalarData ) {
            data = scalarData->get();
        } else if ( arrayData ) {
            const auto &data2 = arrayData->get();
            data              = data2( 0 );
        } else if ( is_integral() ) {
            auto data2 = convert<int64_t, TYPE>( convertToInt64() );
            data       = data2( 0 );
        } else if ( is_floating_point() ) {
            auto data2 = convert<double, TYPE>( convertToDouble() );
            data       = data2( 0 );
        } else {
            DATABASE_ERROR( src, "Unable to convert data for key %s", name.data() );
        }
        if ( factor != 1.0 )
            scaleData( data, factor );
    } else {
        DATABASE_INSIST( factor == 1.0, src, "Only arithmetic types can convert units" );
        if ( scalarData ) {
            data = scalarData->get();
        } else if ( arrayData ) {
            const auto &data2 = arrayData->get();
            data              = data2( 0 );
        } else {
            DATABASE_ERROR( src, "Unable to convert data for key %s", name.data() );
        }
    }
    return data;
}
template<class TYPE>
Array<TYPE> KeyData::getArray( const Units &unit, std::string_view name, source_location src ) const
{
    Array<TYPE> data;
    if ( arraySize().length() == 0 )
        return data;
    double factor   = convertUnits( unit, name );
    auto scalarData = dynamic_cast<const KeyDataScalar<TYPE> *>( this );
    auto arrayData  = dynamic_cast<const KeyDataArray<TYPE> *>( this );
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        if ( scalarData ) {
            const auto &data2 = scalarData->get();
            data.resize( 1 );
            data( 0 ) = data2;
        } else if ( arrayData ) {
            data = arrayData->get();
        } else if ( is_integral() ) {
            data = std::move( convert<int64_t, TYPE>( convertToInt64() ) );
        } else if ( is_floating_point() ) {
            data = std::move( convert<double, TYPE>( convertToDouble() ) );
        } else {
            DATABASE_ERROR( src, "Unable to convert data for key %s", name.data() );
        }
        if ( factor != 1.0 )
            scaleData( data, factor );
    } else {
        DATABASE_INSIST( factor == 1.0, src, "Only arithmetic types can convert units" );
        if ( scalarData ) {
            const auto &data2 = scalarData->get();
            data.resize( 1 );
            data( 0 ) = data2;
        } else if ( arrayData ) {
            data = arrayData->get();
        } else {
            DATABASE_ERROR( src, "Unable to convert data for key %s", name.data() );
        }
    }
    return data;
}
template<class TYPE>
TYPE Database::getScalar( std::string_view key, const Units &unit, source_location src ) const
{
    auto keyData = getData( key );
    DATABASE_INSIST( keyData, src, "Variable %s was not found in database", key.data() );
    return keyData->getScalar<TYPE>( unit, key, src );
}
template<class TYPE>
Array<TYPE> Database::getArray( std::string_view key, const Units &unit, source_location src ) const
{
    auto keyData = getData( key );
    DATABASE_INSIST( keyData, src, "Variable %s was not found in database", key.data() );
    return keyData->getArray<TYPE>( unit, key, src );
}
template<class TYPE>
std::vector<TYPE>
Database::getVector( std::string_view key, const Units &unit, source_location src ) const
{
    auto data = getArray<TYPE>( key, unit, src );
    DATABASE_INSIST(
        data.ndim() <= 1, src, "Variable %s cannot be converted to a vector", key.data() );
    return std::vector<TYPE>( data.begin(), data.end() );
}


/********************************************************************
 * Get data with default                                             *
 ********************************************************************/
template<class TYPE>
TYPE Database::getWithDefault( std::string_view key,
                               const IdentityType<const TYPE &> value,
                               const Units &unit,
                               source_location src ) const
{
    // Check if the key exists and return if it does not
    auto keyData = getData( key );
    if ( !keyData )
        return value;
    // Call the appropriate getScalar/getArray/getVector function
    if constexpr ( is_vector_v<TYPE> ) {
        return getVector<typename TYPE::value_type>( key, unit, src );
    } else if constexpr ( is_Array_v<TYPE> ) {
        return getArray<typename TYPE::value_type>( key, unit, src );
    } else {
        return getScalar<TYPE>( key, unit, src );
    }
}


/********************************************************************
 * Put data                                                          *
 ********************************************************************/
template<class TYPE>
void Database::putScalar(
    std::string_view key, TYPE value, Units unit, Check check, source_location src )
{
    if constexpr ( std::is_same_v<TYPE, std::vector<bool>::reference> ) {
        // Guard against storing a bit reference (store a bool instead)
        putScalar<bool>( key, value, unit, check, src );
    } else if constexpr ( std::is_same_v<TYPE, char *> || std::is_same_v<TYPE, const char *> ||
                          std::is_same_v<TYPE, std::string_view> ) {
        // Guard against storing a char* or string_view (store a std::string instead)
        putScalar<std::string>( key, std::string( value ), unit, check, src );
    } else {
        // Store the scalar value
        auto keyData = std::make_unique<KeyDataScalar<TYPE>>( std::move( value ), unit );
        putData( key, std::move( keyData ), check, src );
    }
}
template<class TYPE>
void Database::putVector( std::string_view key,
                          const std::vector<TYPE> &data,
                          Units unit,
                          Check check,
                          source_location src )
{
    Array<TYPE> x;
    x            = data;
    auto keyData = std::make_unique<KeyDataArray<TYPE>>( std::move( x ), unit );
    putData( key, std::move( keyData ), check, src );
}
template<class TYPE>
void Database::putArray(
    std::string_view key, Array<TYPE> data, Units unit, Check check, source_location src )
{
    auto keyData = std::make_unique<KeyDataArray<TYPE>>( std::move( data ), unit );
    putData( key, std::move( keyData ), check, src );
}


/********************************************************************
 * isType                                                            *
 ********************************************************************/
template<class TYPE>
bool Database::isType( std::string_view key, source_location src ) const
{
    auto data = getData( key );
    DATABASE_INSIST( data, src, "Variable %s was not found in database", key.data() );
    if constexpr ( std::is_same_v<TYPE, std::vector<bool>::reference> ) {
        // Guard against checking a bit reference (use a bool instead)
        return data->isType<bool>();
    } else {
        return data->isType<TYPE>();
    }
}


} // namespace AMP

#endif
