#include "AMP/utils/AMP_MPI_pack.hpp"
#include "AMP/utils/Array.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Database.hpp"
#include "AMP/utils/FactoryStrategy.hpp"
#include "AMP/utils/MathExpr.h"
#include "AMP/utils/Utilities.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <typeinfo>


namespace AMP {


/********************************************************************
 * Scale data                                                        *
 ********************************************************************/
template<class TYPE>
void scaleData( TYPE &data, [[maybe_unused]] double factor )
{
    if constexpr ( std::is_same_v<TYPE, bool> ) {
        throw std::logic_error( "Unable to scale bool" );
    } else if constexpr ( std::is_same_v<TYPE, std::complex<float>> ) {
        data = static_cast<float>( factor ) * data;
    } else if constexpr ( std::is_same_v<TYPE, std::complex<double>> ) {
        data = factor * data;
    } else if constexpr ( std::is_arithmetic_v<TYPE> ) {
        data = static_cast<TYPE>( factor ) * data;
    } else {
        std::string type = typeid( TYPE ).name();
        throw std::logic_error( "Unable to scale " + type );
    }
}
template<class TYPE>
void scaleData( Array<TYPE> &data, [[maybe_unused]] double factor )
{
    if constexpr ( std::is_same_v<TYPE, bool> ) {
        throw std::logic_error( "Unable to scale bool" );
    } else if constexpr ( std::is_arithmetic_v<TYPE> ) {
        data.scale( factor );
    } else {
        std::string type = typeid( TYPE ).name();
        throw std::logic_error( "Unable to scale " + type );
    }
}


/********************************************************************
 * EquationKeyData                                                   *
 ********************************************************************/
EquationKeyData::EquationKeyData( std::string_view eq, const Units &unit ) : KeyData( unit )
{
    eq = deblank( eq );
    AMP_ASSERT( eq.size() > 4 );
    size_t i = eq.find( ')' );
    if ( eq[0] != '@' || eq[1] != '(' || eq.back() != ';' || i == std::string::npos )
        AMP_ERROR( "Equation appears invalid: " + std::string( eq ) );
    auto eq_var   = deblank( eq.substr( 2, i - 2 ) );
    auto equation = std::string( deblank( eq.substr( i + 1, eq.size() - i - 2 ) ) );
    std::vector<std::string> vars;
    if ( !eq_var.empty() ) {
        for ( size_t i = 0; i < eq_var.size(); ) {
            size_t j = std::min( eq_var.find( ',', i ), eq_var.size() );
            vars.emplace_back( deblank( eq_var.substr( i, j - i ) ) );
            i = j + 1;
        }
    }
    d_eq = new MathExpr( equation, vars );
}
EquationKeyData::EquationKeyData( MathExpr eq, const Units &unit )
    : KeyData( unit ), d_eq( new MathExpr( std::move( eq ) ) )
{
}
EquationKeyData::~EquationKeyData() { delete d_eq; }
std::unique_ptr<KeyData> EquationKeyData::clone() const
{
    return std::make_unique<EquationKeyData>( d_eq->clone(), d_unit );
}
void EquationKeyData::print( std::ostream &os, std::string_view indent, bool, bool printType ) const
{
    os << indent << "@(";
    if ( d_eq ) {
        auto vars = d_eq->getVars();
        auto expr = d_eq->getExpr();
        if ( !vars.empty() ) {
            os << vars[0];
            for ( size_t i = 1; i < vars.size(); i++ )
                os << "," << vars[i];
        }
        os << ") " << expr << ";";
    }
    if ( printType )
        os << "  // " << getDataType().name;
    os << std::endl;
}
typeID EquationKeyData::getDataType() const { return AMP::getTypeID<MathExpr>(); }
bool EquationKeyData::is_floating_point() const { return d_eq->getVars().empty(); }
bool EquationKeyData::is_integral() const
{
    if ( d_eq->getVars().empty() ) {
        double v = ( *d_eq )();
        if ( static_cast<double>( static_cast<int64_t>( v ) ) == v )
            return true;
    }
    return false;
}
ArraySize EquationKeyData::arraySize() const
{
    return d_eq->getVars().empty() ? ArraySize( 1 ) : ArraySize();
}
Array<double> EquationKeyData::convertToDouble() const
{
    if ( d_eq->getVars().empty() ) {
        double v = ( *d_eq )();
        return Array<double>( { 1 }, &v );
    }
    return Array<double>();
}
Array<int64_t> EquationKeyData::convertToInt64() const
{
    if ( d_eq->getVars().empty() ) {
        double v   = ( *d_eq )();
        int64_t v2 = v;
        if ( v2 == v )
            return Array<int64_t>( { 1 }, &v2 );
    }
    return Array<int64_t>();
}
bool EquationKeyData::operator==( const KeyData &rhs ) const
{
    auto rhs2 = dynamic_cast<const EquationKeyData *>( &rhs );
    if ( rhs2 == nullptr )
        return false;
    if ( d_eq == rhs2->d_eq )
        return true;
    if ( !d_eq || !rhs2->d_eq )
        return false;
    return *d_eq == *rhs2->d_eq;
}
size_t EquationKeyData::packSize() const
{
    std::string expr;
    std::vector<std::string> vars;
    if ( d_eq ) {
        expr = d_eq->getExpr();
        vars = d_eq->getVars();
    }
    return AMP::packSize( expr ) + AMP::packSize( vars );
}
size_t EquationKeyData::pack( std::byte *buf ) const
{
    std::string expr;
    std::vector<std::string> vars;
    if ( d_eq ) {
        expr = d_eq->getExpr();
        vars = d_eq->getVars();
    }
    size_t N = 0;
    N += AMP::pack( expr, &buf[N] );
    N += AMP::pack( vars, &buf[N] );
    return N;
}
size_t EquationKeyData::unpack( const std::byte *buf )
{
    std::string expr;
    std::vector<std::string> vars;
    size_t N = 0;
    N += AMP::unpack( expr, &buf[N] );
    N += AMP::unpack( vars, &buf[N] );
    delete d_eq;
    d_eq = nullptr;
    if ( !expr.empty() )
        d_eq = new MathExpr( expr, vars );
    return N;
}
void EquationKeyData::writeHDF5( int64_t fid, const std::string &name ) const
{
    hid_t gid = AMP::IO::createGroup( fid, name );
    AMP::IO::writeHDF5( gid, "expr", d_eq->getExpr() );
    AMP::IO::writeHDF5( gid, "vars", d_eq->getVars() );
    AMP::IO::closeGroup( gid );
}
void EquationKeyData::readHDF5( int64_t fid, const std::string &name )
{
    std::string expr;
    std::vector<std::string> vars;
    hid_t gid = AMP::IO::openGroup( fid, name );
    AMP::IO::readHDF5( gid, "expr", expr );
    AMP::IO::readHDF5( gid, "vars", vars );
    AMP::IO::closeGroup( gid );
    delete d_eq;
    d_eq = nullptr;
    if ( !expr.empty() )
        d_eq = new MathExpr( expr, vars );
}


/********************************************************************
 * DatabaseBox                                                       *
 ********************************************************************/
DatabaseBox::DatabaseBox() : d_dim( 0 )
{
    d_lower.fill( 0 );
    d_upper.fill( 0 );
}
DatabaseBox::DatabaseBox( int dim, const int *lower, const int *upper ) : d_dim( dim )
{
    AMP_ASSERT( dim <= 5 );
    d_lower.fill( 0 );
    d_upper.fill( 0 );
    for ( int d = 0; d < dim; d++ ) {
        d_lower[d] = lower[d];
        d_upper[d] = upper[d];
    }
}
DatabaseBox::DatabaseBox( std::string_view str ) : d_dim( 0 )
{
    d_lower.fill( 0 );
    d_upper.fill( 0 );
    size_t j     = 0;
    size_t k     = 0;
    char tmp[10] = { 0 };
    for ( size_t i = 0; i < str.length(); i++ ) {
        if ( str[i] == ' ' || str[i] == '(' || str[i] == '[' || str[i] == ']' ) {
            continue;
        } else if ( str[i] == ',' || str[i] == ')' ) {
            if ( str[i] == ')' && d_dim == 0 ) {
                d_dim = k + 1;
                while ( str[i] != ',' && i + 1 < str.length() ) {
                    ++i;
                }
            }
            tmp[j] = 0;
            j      = 0;
            int v  = atoi( tmp );
            if ( k < d_dim || d_dim == 0 )
                d_lower[k++] = v;
            else
                d_upper[( k++ ) - d_dim] = v;
        } else {
            tmp[j++] = str[i];
        }
    }
}
uint8_t &DatabaseBox::dim() { return d_dim; }
uint8_t DatabaseBox::dim() const { return d_dim; }
bool DatabaseBox::empty() const
{
    if ( d_dim == 0 )
        return true;
    for ( int d = 0; d < d_dim; d++ ) {
        if ( d_upper[d] < d_lower[d] )
            return true;
    }
    return false;
}
int &DatabaseBox::lower( uint8_t d )
{
    AMP_ASSERT( d_dim <= 5 && d < d_dim );
    return d_lower[d];
}
int DatabaseBox::lower( uint8_t d ) const
{
    AMP_ASSERT( d_dim <= 5 && d < d_dim );
    return d_lower[d];
}
int &DatabaseBox::upper( uint8_t d )
{
    AMP_ASSERT( d_dim <= 5 && d < d_dim );
    return d_upper[d];
}
int DatabaseBox::upper( uint8_t d ) const
{
    AMP_ASSERT( d_dim <= 5 && d < d_dim );
    return d_upper[d];
}
bool DatabaseBox::operator==( const DatabaseBox &box ) const
{
    bool equal = d_dim == box.d_dim;
    for ( int d = 0; d < d_dim; d++ ) {
        equal = equal && d_lower[d] == box.d_lower[d];
        equal = equal && d_upper[d] == box.d_upper[d];
    }
    return equal;
}
std::ostream &operator<<( std::ostream &out, const DatabaseBox &box )
{
    if ( box.empty() )
        out << "[(),()]";
    out << "[(" << box.lower( 0 );
    for ( int d = 1; d < box.dim(); d++ )
        out << "," << box.lower( d );
    out << "),(" << box.upper( 0 );
    for ( int d = 1; d < box.dim(); d++ )
        out << "," << box.upper( d );
    out << ")]";
    return out;
}


/********************************************************************
 * Explicit instantiations                                           *
 ********************************************************************/
#define instantiate( FUN )       \
    FUN( bool );                 \
    FUN( char );                 \
    FUN( int8_t );               \
    FUN( int16_t );              \
    FUN( int32_t );              \
    FUN( int64_t );              \
    FUN( uint8_t );              \
    FUN( uint16_t );             \
    FUN( uint32_t );             \
    FUN( uint64_t );             \
    FUN( float );                \
    FUN( double );               \
    FUN( long double );          \
    FUN( std::complex<float> );  \
    FUN( std::complex<double> ); \
    FUN( std::string );          \
    FUN( DatabaseBox )
#define instantiateConvert( TYPE )                                       \
    template Array<TYPE> convert( const Array<bool> & );                 \
    template Array<TYPE> convert( const Array<char> & );                 \
    template Array<TYPE> convert( const Array<int8_t> & );               \
    template Array<TYPE> convert( const Array<int16_t> & );              \
    template Array<TYPE> convert( const Array<int32_t> & );              \
    template Array<TYPE> convert( const Array<int64_t> & );              \
    template Array<TYPE> convert( const Array<uint8_t> & );              \
    template Array<TYPE> convert( const Array<uint16_t> & );             \
    template Array<TYPE> convert( const Array<uint32_t> & );             \
    template Array<TYPE> convert( const Array<uint64_t> & );             \
    template Array<TYPE> convert( const Array<float> & );                \
    template Array<TYPE> convert( const Array<double> & );               \
    template Array<TYPE> convert( const Array<long double> & );          \
    template Array<TYPE> convert( const Array<std::complex<float>> & );  \
    template Array<TYPE> convert( const Array<std::complex<double>> & ); \
    template Array<TYPE> convert( const Array<std::string> & );          \
    template Array<TYPE> convert( const Array<DatabaseBox> & )
#define instantiateScaleData( TYPE )                             \
    template void scaleData<TYPE>( TYPE & data, double factor ); \
    template void scaleData<TYPE>( Array<TYPE> & data, double factor )
#define instantiateKeyDataScalar( TYPE ) template class KeyDataScalar<TYPE>
#define instantiateKeyDataArray( TYPE ) template class KeyDataArray<TYPE>
#define instantiateIsType( TYPE ) \
    template bool Database::isType<TYPE>( std::string_view, source_location ) const
#define instantiatePutScalar( TYPE ) \
    template void Database::putScalar<TYPE>( std::string_view, TYPE, Units, Check, source_location )
#define instantiateGetScalar( TYPE )                                                            \
    template TYPE Database::getScalar<TYPE>( std::string_view, const Units &, source_location ) \
        const
#define instantiateGetArray( TYPE )                                                             \
    template Array<TYPE> Database::getArray( std::string_view, const Units &, source_location ) \
        const
#define instantiateGetVector( TYPE )                \
    template std::vector<TYPE> Database::getVector( \
        std::string_view, const Units &, source_location ) const
#define instantiatePutVector( TYPE )         \
    template void Database::putVector<TYPE>( \
        std::string_view, const std::vector<TYPE> &, Units, Check, source_location )
#define instantiatePutArray( TYPE )         \
    template void Database::putArray<TYPE>( \
        std::string_view, Array<TYPE>, Units, Check, source_location )
#define instantiateGetWithDefault( TYPE )                                                     \
    template TYPE Database::getWithDefault<TYPE>(                                             \
        std::string_view, IdentityType<TYPE const &>, const Units &, source_location ) const; \
    template std::vector<TYPE> Database::getWithDefault<std::vector<TYPE>>(                   \
        std::string_view,                                                                     \
        IdentityType<std::vector<TYPE> const &>,                                              \
        const Units &,                                                                        \
        source_location ) const;                                                              \
    template Array<TYPE> Database::getWithDefault<Array<TYPE>>(                               \
        std::string_view, IdentityType<Array<TYPE> const &>, const Units &, source_location ) \
        const

instantiate( instantiateConvert );        // convert
instantiate( instantiateScaleData );      // scaleData
instantiate( instantiateKeyDataScalar );  // KeyDataScalar
instantiate( instantiateKeyDataArray );   // KeyDataArray
instantiate( instantiateIsType );         // Database::isType
instantiate( instantiateGetScalar );      // Database::getScalar
instantiate( instantiateGetVector );      // Database::getVector
instantiate( instantiateGetArray );       // Database::getArray
instantiate( instantiatePutScalar );      // Database::putScalar
instantiate( instantiatePutVector );      // Database::putVector
instantiate( instantiatePutArray );       // Database::putArray
instantiate( instantiateGetWithDefault ); // Database::getWithDefault
template void
Database::putScalar<const char *>( std::string_view, const char *, Units, Check, source_location );
template void Database::putScalar<std::vector<bool>::reference>(
    std::string_view key, std::vector<bool>::reference, Units, Check, source_location );


} // namespace AMP


/********************************************************
 *  Register KeyData with factory                        *
 ********************************************************/
#define REGISTER_KEYDATA( TYPE ) \
    d_factories[AMP::getTypeID<TYPE>().name] = []() { return std::make_unique<TYPE>(); }
#define REGISTER_KEYDATA2( TYPE )                 \
    REGISTER_KEYDATA( AMP::KeyDataScalar<TYPE> ); \
    REGISTER_KEYDATA( AMP::KeyDataArray<TYPE> )
template<>
void AMP::FactoryStrategy<AMP::KeyData>::registerDefault()
{
    REGISTER_KEYDATA2( bool );
    REGISTER_KEYDATA2( char );
    REGISTER_KEYDATA2( int8_t );
    REGISTER_KEYDATA2( int16_t );
    REGISTER_KEYDATA2( int32_t );
    REGISTER_KEYDATA2( int64_t );
    REGISTER_KEYDATA2( uint8_t );
    REGISTER_KEYDATA2( uint16_t );
    REGISTER_KEYDATA2( uint32_t );
    REGISTER_KEYDATA2( uint64_t );
    REGISTER_KEYDATA2( float );
    REGISTER_KEYDATA2( double );
    REGISTER_KEYDATA2( long double );
    REGISTER_KEYDATA2( std::complex<float> );
    REGISTER_KEYDATA2( std::complex<double> );
    REGISTER_KEYDATA2( std::string );
    REGISTER_KEYDATA2( DatabaseBox );
    REGISTER_KEYDATA( AMP::Database );
    REGISTER_KEYDATA( AMP::EmptyKeyData );
    REGISTER_KEYDATA( AMP::DatabaseVector );
    REGISTER_KEYDATA( AMP::EquationKeyData );
    d_factories["AMP::KeyDataScalar<std::basic_string<char>>"] = []() {
        return std::make_unique<AMP::KeyDataScalar<std::string>>();
    };
    d_factories["AMP::KeyDataArray<std::basic_string<char>>"] = []() {
        return std::make_unique<AMP::KeyDataArray<std::string>>();
    };
}


/********************************************************
 *  Explicit instantiations of Array<DatabaseBox>        *
 ********************************************************/
#include "AMP/utils/Array.hpp"
instantiateArrayConstructors( AMP::DatabaseBox );
PACK_UNPACK_ARRAY( AMP::DatabaseBox );
PACK_UNPACK_ARRAY2( AMP::DatabaseBox );
