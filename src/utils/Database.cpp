#include "AMP/utils/Database.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Database.hpp"
#include "AMP/utils/MathExpr.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <complex>
#include <cstring>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>


namespace AMP {


/********************************************************************
 * Constructors/destructor                                           *
 ********************************************************************/
Database::Database() : KeyData(), d_check( Check::WarnOverwrite ) {}
Database::Database( std::string name )
    : KeyData(), d_check( Check::WarnOverwrite ), d_name( std::move( name ) )
{
}
Database::Database( Database &&rhs )
{
    std::swap( d_check, rhs.d_check );
    std::swap( d_name, rhs.d_name );
    std::swap( d_used, rhs.d_used );
    std::swap( d_hash, rhs.d_hash );
    std::swap( d_keys, rhs.d_keys );
    std::swap( d_data, rhs.d_data );
}
Database::Database( const Database &rhs ) : KeyData() { copy( rhs ); }
Database &Database::operator=( const Database &rhs )
{
    if ( &rhs != this )
        copy( rhs );
    return *this;
}
Database &Database::operator=( Database &&rhs )
{
    if ( this != &rhs ) {
        std::swap( d_check, rhs.d_check );
        std::swap( d_name, rhs.d_name );
        std::swap( d_used, rhs.d_used );
        std::swap( d_hash, rhs.d_hash );
        std::swap( d_keys, rhs.d_keys );
        std::swap( d_data, rhs.d_data );
    }
    return *this;
}


/********************************************************************
 * Set default behavior for database                                 *
 ********************************************************************/
void Database::setDefaultAddKeyBehavior( Check check, bool setChildren )
{
    d_check = check;
    if ( setChildren ) {
        for ( auto tmp : d_data ) {
            auto db = std::dynamic_pointer_cast<Database>( tmp );
            if ( db )
                db->setDefaultAddKeyBehavior( check, true );
        }
    }
}


/********************************************************************
 * Clone the database                                                *
 ********************************************************************/
std::unique_ptr<KeyData> Database::clone() const { return std::make_unique<Database>( *this ); }
std::unique_ptr<Database> Database::cloneDatabase() const
{
    return std::make_unique<Database>( *this );
}
void Database::copy( const Database &rhs )
{
    d_check = rhs.d_check;
    d_name  = rhs.d_name;
    d_hash  = rhs.d_hash;
    d_keys  = rhs.d_keys;
    d_used  = std::vector<bool>( rhs.d_used.size(), false );
    d_data.resize( rhs.d_data.size() );
    for ( size_t i = 0; i < d_data.size(); i++ )
        d_data[i] = rhs.d_data[i]->clone();
    rhs.d_used = std::vector<bool>( rhs.d_used.size(), true );
}


/********************************************************************
 * Check if the databases are equivalent                             *
 ********************************************************************/
bool Database::operator==( const Database &rhs ) const
{
    auto keys1 = getAllKeys( true );
    auto keys2 = rhs.getAllKeys( true );
    if ( keys1 != keys2 )
        return false;
    for ( const auto &key : keys1 ) {
        auto d1 = getData( key );
        auto d2 = rhs.getData( key );
        if ( *d1 != *d2 )
            return false;
    }
    return true;
}
bool Database::operator==( const KeyData &rhs ) const
{
    auto db = dynamic_cast<const Database *>( &rhs );
    if ( !db )
        return false;
    return operator==( *db );
}


/********************************************************************
 * Get the data object                                               *
 ********************************************************************/
bool Database::keyExists( std::string_view key ) const
{
    auto hash = hashString( key );
    int index = find( hash, false );
    return index != -1;
}
void Database::deleteData( std::string_view key )
{
    auto hash = hashString( key );
    int index = find( hash, false );
    std::swap( d_hash[index], d_hash.back() );
    std::swap( d_keys[index], d_keys.back() );
    std::swap( d_data[index], d_data.back() );
    d_hash.pop_back();
    d_keys.pop_back();
    d_data.pop_back();
}
KeyData *Database::getData( std::string_view key )
{
    auto hash = hashString( key );
    int index = find( hash, true );
    return index == -1 ? nullptr : d_data[index].get();
}
const KeyData *Database::getData( std::string_view key ) const
{
    auto hash = hashString( key );
    int index = find( hash, true );
    return index == -1 ? nullptr : d_data[index].get();
}
typeID Database::getDataType( std::string_view key ) const
{
    auto hash = hashString( key );
    int index = find( hash, true );
    return index == -1 ? typeID() : d_data[index]->getDataType();
}
bool Database::isDatabase( std::string_view key, source_location src ) const
{
    auto hash = hashString( key );
    int index = find( hash, false );
    DATABASE_INSIST( index != -1, src, "Variable %s is not in database", key.data() );
    auto ptr2 = dynamic_cast<const Database *>( d_data[index].get() );
    return ptr2 != nullptr;
}
std::shared_ptr<Database> Database::getDatabase( std::string_view key, source_location src )
{
    auto hash = hashString( key );
    int index = find( hash, true );
    if ( index == -1 )
        return nullptr;
    auto ptr2 = std::dynamic_pointer_cast<Database>( d_data[index] );
    DATABASE_INSIST( ptr2, src, "Variable %s is not a database", key.data() );
    return ptr2;
}
std::shared_ptr<const Database> Database::getDatabase( std::string_view key,
                                                       source_location src ) const
{
    auto hash = hashString( key );
    int index = find( hash, true );
    if ( index == -1 )
        return nullptr;
    auto ptr2 = std::dynamic_pointer_cast<const Database>( d_data[index] );
    DATABASE_INSIST( ptr2, src, "Variable %s is not a database", key.data() );
    return ptr2;
}
const Database &Database::operator()( std::string_view key, source_location src ) const
{
    auto hash = hashString( key );
    int index = find( hash, true );
    DATABASE_INSIST( index != -1, src, "Variable %s is not in database", key.data() );
    auto ptr2 = std::dynamic_pointer_cast<const Database>( d_data[index] );
    DATABASE_INSIST( ptr2, src, "Variable %s is not a database", key.data() );
    return *ptr2;
}
std::vector<std::string> Database::getAllKeys( bool sort ) const
{
    auto keys = d_keys;
    if ( sort )
        std::sort( keys.begin(), keys.end() );
    return keys;
}
void Database::putData( std::string_view key,
                        std::unique_ptr<KeyData> data,
                        Check check,
                        source_location src )
{
    if ( check == Check::GetDatabaseDefault )
        check = d_check;
    auto hash = hashString( key );
    int index = find( hash, false );
    if ( index != -1 ) {
        if ( check == Check::Error )
            DATABASE_ERROR(
                src, "Error: Variable '%s' already exists in database", std::string( key ).data() );
        if ( check == Check::WarnOverwrite || check == Check::WarnKeep )
            DATABASE_WARNING( src,
                              "Warning: variable '%s' already exists in database",
                              std::string( key ).data() );
        if ( check == Check::Overwrite || check == Check::WarnOverwrite )
            d_data[index] = std::move( data );
    } else {
        d_used.emplace_back( false );
        d_hash.emplace_back( hash );
        d_keys.emplace_back( key );
        d_data.emplace_back( std::move( data ) );
    }
}
void Database::erase( std::string_view key, bool check )
{
    auto hash = hashString( key );
    int index = find( hash, false );
    if ( index == -1 ) {
        if ( check )
            AMP_ERROR( std::string( key ) + " does not exist in database" );
        return;
    }
    d_used[index] = d_used.back();
    d_hash[index] = d_hash.back();
    std::swap( d_keys[index], d_keys.back() );
    std::swap( d_data[index], d_data.back() );
    d_used.pop_back();
    d_hash.pop_back();
    d_keys.pop_back();
    d_data.pop_back();
}
void Database::clear()
{
    d_hash.clear();
    d_keys.clear();
    d_data.clear();
    d_used.clear();
}


/********************************************************************
 * Convert units                                                     *
 ********************************************************************/
double KeyData::convertUnits( const Units &unit, std::string_view key ) const
{
    if ( unit.isNull() )
        return 1.0;
    DATABASE_INSIST(
        !d_unit.isNull(), SOURCE_LOCATION_CURRENT(), "Field %s must have units", key.data() );
    return d_unit.convert( unit );
}


/********************************************************************
 * Is the data of the given type                                     *
 ********************************************************************/
template<>
bool KeyData::isType<std::string>() const
{
    return getDataType() == getTypeID<std::string>();
}
template<>
bool KeyData::isType<bool>() const
{
    return getDataType() == getTypeID<bool>();
}
template<>
bool KeyData::isType<std::complex<float>>() const
{
    return getDataType() == getTypeID<std::complex<float>>();
}
template<>
bool KeyData::isType<std::complex<double>>() const
{
    return getDataType() == getTypeID<std::complex<double>>();
}
template<>
bool KeyData::isType<double>() const
{
    if ( getDataType() == getTypeID<double>() )
        return true;
    return is_floating_point() || is_integral();
}
template<>
bool KeyData::isType<DatabaseBox>() const
{
    return getDataType() == getTypeID<DatabaseBox>();
}
template<>
bool KeyData::isType<MathExpr>() const
{
    if ( dynamic_cast<const EquationKeyData *>( this ) )
        return true;
    if ( isType<double>() && arraySize().length() == 1 )
        return true;
    return false;
}
template<>
bool KeyData::isType<Database>() const
{
    return dynamic_cast<const Database *>( this );
}
template<class TYPE>
bool KeyData::isType() const
{
    constexpr auto id0 = getTypeID<TYPE>();
    if ( getDataType() == id0 )
        return true;
    if ( is_integral() ) {
        auto data2 = convertToInt64();
        bool pass  = true;
        for ( auto tmp : data2 )
            pass = pass && static_cast<int64_t>( static_cast<TYPE>( tmp ) ) == tmp;
        return pass;
    }
    if ( is_floating_point() ) {
        auto data2 = convertToDouble();
        bool pass  = true;
        for ( auto tmp : data2 )
            pass = pass && static_cast<double>( static_cast<TYPE>( tmp ) ) == tmp;
        return pass;
    }
    return false;
}
template bool KeyData::isType<char>() const;
template bool KeyData::isType<uint8_t>() const;
template bool KeyData::isType<uint16_t>() const;
template bool KeyData::isType<uint32_t>() const;
template bool KeyData::isType<uint64_t>() const;
template bool KeyData::isType<int8_t>() const;
template bool KeyData::isType<int16_t>() const;
template bool KeyData::isType<int32_t>() const;
template bool KeyData::isType<int64_t>() const;
template bool KeyData::isType<float>() const;
template bool KeyData::isType<long double>() const;
bool Database::isString( std::string_view key, source_location src ) const
{
    auto data = getData( key );
    DATABASE_INSIST( data, src, "Variable %s was not found in database", key.data() );
    return data->isType<std::string>();
}

bool Database::isEquation( std::string_view key, source_location src ) const
{
    auto data = getData( key );
    DATABASE_INSIST( data, src, "Variable %s was not found in database", key.data() );
    return data->isType<MathExpr>();
}
std::unique_ptr<MathExpr>
Database::getEquation( std::string_view key, const Units &unit, source_location src ) const
{
    auto data = getData( key );
    DATABASE_INSIST( data, src, "Variable %s was not found in database", key.data() );
    double factor = data->convertUnits( unit, key );
    auto eq_data  = dynamic_cast<const EquationKeyData *>( data );
    if ( eq_data ) {
        auto expr = eq_data->getEq().getExpr();
        auto vars = eq_data->getEq().getVars();
        if ( factor != 1.0 )
            expr = std::to_string( factor ) + "*( " + expr + ")";
        return std::make_unique<MathExpr>( expr, vars );
    }
    if ( data->isType<double>() && data->arraySize().length() == 1 ) {
        double v = factor * data->convertToDouble()( 0 );
        return std::make_unique<MathExpr>( std::to_string( v ) );
    }
    DATABASE_ERROR( src, "Variable %s was not an equation", key.data() );
    return nullptr;
}


/********************************************************************
 * Print the database                                                *
 ********************************************************************/
void Database::print( std::ostream &os, std::string_view indent, bool sort, bool printType ) const
{
    auto keys = getAllKeys( sort ); //  We want the keys in sorted order
    for ( const auto &key : keys ) {
        os << indent << key;
        auto hash  = hashString( key );
        auto index = find( hash, false );
        auto data  = d_data[index].get();
        auto db    = dynamic_cast<const Database *>( data );
        auto dbVec = dynamic_cast<const DatabaseVector *>( data );
        if ( db ) {
            os << " {\n";
            db->print( os, std::string( indent ) + "   ", sort, printType );
            os << indent << "}\n";
        } else if ( dbVec ) {
            os << ":\n";
            dbVec->print( os, indent, sort, printType );
        } else {
            os << " = ";
            data->print( os, "", sort, printType );
        }
    }
}
std::string Database::print( std::string_view indent, bool sort, bool printType ) const
{
    std::stringstream ss;
    print( ss, indent, sort, printType );
    return ss.str();
}
std::vector<std::string> Database::getUnused( bool recursive ) const
{
    AMP_ASSERT( d_keys.size() == d_used.size() );
    std::vector<std::string> unused;
    for ( size_t i = 0; i < d_used.size(); i++ ) {
        if ( !d_used[i] ) {
            unused.push_back( d_keys[i] );
        } else if ( recursive ) {
            auto db = std::dynamic_pointer_cast<const Database>( d_data[i] );
            if ( db ) {
                for ( auto tmp : db->getUnused( true ) )
                    unused.push_back( db->d_name + "::" + tmp );
            }
        }
    }
    return unused;
}


/********************************************************************
 * Read input database file                                          *
 ********************************************************************/
std::shared_ptr<Database> Database::parseInputFile( const std::string &filename )
{
    std::shared_ptr<Database> db;
    auto extension = filename.substr( filename.rfind( '.' ) + 1 );
    if ( extension == "yml" ) {
        std::shared_ptr<KeyData> data = readYAML( filename );
        auto db2                      = std::dynamic_pointer_cast<Database>( data );
        auto dbVec                    = std::dynamic_pointer_cast<DatabaseVector>( data );
        if ( db2 ) {
            db = db2;
        } else if ( dbVec ) {
            db = std::make_shared<Database>( filename );
            for ( const auto &db3 : dbVec->get() ) {
                auto key = db3.getName();
                AMP_ASSERT( !key.empty() );
                AMP_ASSERT( !db->keyExists( key ) );
                db->putDatabase( key, db3.cloneDatabase() );
            }
        } else {
            AMP_ERROR( "Unknown keyData" );
        }
        db->setName( filename );
    } else {
        db = std::make_shared<Database>( filename );
        db->readDatabase( filename );
    }
    return db;
}
Array<double> Database::convertToDouble() const
{
    throw std::logic_error( "convertData on a database is not valid" );
}
Array<int64_t> Database::convertToInt64() const
{
    throw std::logic_error( "convertData on a database is not valid" );
}
bool Database::is_floating_point() const
{
    throw std::logic_error( "convertData on a database is not valid" );
}
bool Database::is_integral() const
{
    throw std::logic_error( "convertData on a database is not valid" );
}


// Print a database to an output stream
template<class TYPE>
static void printVar( const std::string &name,
                      const std::vector<TYPE> &data,
                      std::ostream &os,
                      const std::string &indent )
{
    os << indent << name << " = ";
    if ( !data.empty() ) {
        os << data[0];
        for ( size_t i = 1; i < data.size(); i++ )
            os << ", " << data[i];
    }
    os << std::endl;
}
void Utilities::printDatabase( const Database &db, std::ostream &os, const std::string &indent )
{
    db.print( os, indent );
}


} // namespace AMP

/********************************************************
 *  Explicit instantiations of Array<Database>           *
 ********************************************************/
#include "AMP/utils/Array.hpp"
instantiateArrayConstructors( AMP::Database );
instantiateArrayConstructors( AMP::Database::Check );
