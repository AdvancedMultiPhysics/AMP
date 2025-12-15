#ifndef included_AMP_Database
#define included_AMP_Database

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/TypeTraits.h"
#include "AMP/utils/Units.h"
#include "AMP/utils/typeid.h"

#include "StackTrace/source_location.h"

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>


// Forward declare SAMRAI's database
namespace SAMRAI::tbox {
class Database;
class DatabaseBox;
} // namespace SAMRAI::tbox


// AMP namespace
namespace AMP {


// Forward declares
class MathExpr;


//! Base class to hold data of a given type
class KeyData
{
public:
    using source_location = StackTrace::source_location;
    //! Destructor
    virtual ~KeyData() {}
    //! Return class type
    virtual typeID getClassType() const = 0;
    //! Copy the data
    virtual std::unique_ptr<KeyData> clone() const = 0;
    //! Print the data to a stream
    virtual void print( std::ostream &os,
                        std::string_view indent = "",
                        bool sort               = true,
                        bool printType          = false ) const = 0;
    //! Return true if the type is a floating point type
    virtual bool is_floating_point() const = 0;
    //! Return true if the type is a integer point type
    virtual bool is_integral() const = 0;
    // Check if the entry can be stored as the given type
    template<class TYPE>
    bool isType() const;
    // Get the fundamental type (e.g. double, int, float, ...)
    virtual typeID getDataType() const = 0;
    //! Return the array size
    virtual ArraySize arraySize() const = 0;
    //! Return the data as a Array<double> (throw error if this is not valid)
    virtual Array<double> convertToDouble() const = 0;
    //! Return the data as a Array<int64_t> (throw error if this is not valid)
    virtual Array<int64_t> convertToInt64() const = 0;
    //! Check if two sets of data are equal
    virtual bool operator==( const KeyData &rhs ) const = 0;
    //! Check if two sets of data are not equal
    inline bool operator!=( const KeyData &rhs ) const { return !operator==( rhs ); }
    //! Return the units
    const Units &unit() const { return d_unit; }
    //! Return the conversion factor (if used)
    double convertUnits( const Units &, std::string_view = "" ) const;
    //! Return the number of bytes required to pack the data
    virtual size_t packSize() const = 0;
    //! Pack the data to a buffer
    virtual size_t pack( std::byte * ) const = 0;
    //! Unpack the data from a buffer
    virtual size_t unpack( const std::byte * ) = 0;
    //! Write the data to HDF5
    virtual void writeHDF5( int64_t fid, const std::string &name ) const = 0;
    //! Read the data from HDF5
    virtual void readHDF5( int64_t fid, const std::string &name ) = 0;
    //! Convert the data to a scalar of the given type
    template<class TYPE>
    TYPE getScalar( const Units &unit     = {},
                    std::string_view name = "",
                    source_location src   = source_location::current() ) const;
    //! Convert the data to a scalar of the given type
    template<class TYPE>
    Array<TYPE> getArray( const Units &unit     = {},
                          std::string_view name = "",
                          source_location src   = source_location::current() ) const;

protected:
    KeyData() {}
    KeyData( const Units &unit ) : d_unit( unit ) {}
    KeyData( KeyData && )      = delete;
    KeyData( const KeyData & ) = delete;

    KeyData &operator=( KeyData && ) = delete;
    KeyData &operator=( const KeyData & ) = delete;

protected:
    Units d_unit;
};


//! Class to a database
class Database final : public KeyData
{
public:
    //! enum to control behavior when trying to add existing keys
    enum class Check : uint8_t {
        Overwrite,     ///< Overwrite the data
        Keep,          ///< Keep the existing data
        WarnOverwrite, ///< Overwrite the data but print a warning (default)
        WarnKeep,      ///< Keep the existing data but print a warning
        Error,         ///< Throw an error
        GetDatabaseDefault
    };

    template<typename T>
    struct IdentityTypeStruct {
        typedef T type;
    };
    template<typename T>
    using IdentityType = typename IdentityTypeStruct<const T &>::type;

public:
    //! Empty constructor
    Database();

    //! Basic constructor
    explicit Database( std::string name );

    /**
     * Open an database file.
     * @param filename       Name of input file to open
     */
    static std::shared_ptr<Database> parseInputFile( const std::string &filename );

    // Read a YAML database
    static std::unique_ptr<KeyData> readYAML( std::string_view filename,
                                              source_location src = source_location::current() );

    /** \brief Create a database from key/value pairs
     * \details  This function will create a database from a set of key/value pairs
     *    of the form: create( "key1", value1, "key2", value2, ... ).
     *    Note that for simplicity each value must either be a scalar value
     *       (int, double, string, etc) or a std::vector/Array of scalar values
     * \param[in]  args         The input arguments
     */
    template<class... Args>
    static std::unique_ptr<Database> create( Args... args );

    /** \brief Create a database from key/value/unit triplets
     * \details  This function will create a database from a set of key/value/unit triplets
     *    of the form: create( "key1", value1, unit1, "key2", value2, unit2, ... ).
     *    Note that for simplicity each value must either be a scalar value
     *       (int, double, string, etc) or a std::vector/Array of scalar values
     * \param[in]  args         The input arguments
     */
    template<class... Args>
    static std::unique_ptr<Database> createWithUnits( Args... args );

    /**
     * Create database from string
     * @param data       String containing the database data
     */
    static std::unique_ptr<Database> createFromString( std::string_view data );

    //! Copy constructor
    Database( const Database & );

    //! Assignment operator
    Database &operator=( const Database & );

    //! Move constructor
    Database( Database &&rhs );

    //! Move assignment operator
    Database &operator=( Database &&rhs );

    //! Destructor
    virtual ~Database() = default;

    /**
     * Open an database file
     * @param filename       Name of input file to open
     * @param[in] src        Source location in case of error
     */
    void readDatabase( const std::string &filename,
                       source_location src = source_location::current() );

    //! Return class type
    typeID getClassType() const override { return getTypeID<Database>(); }

    //! Get the default behavior when adding keys
    inline Check getDefaultAddKeyBehavior() const { return d_check; }

    //! Set the default behavior when adding keys
    /** \brief  Set the default behavior when adding keys
     * \details  This function will specify the default behavior when a user adds
     *     a key that is already in the database.
     * \param[in] check         The default behavior to set
     * \param[in] setChildren   Change the behavior of any children
     */
    void setDefaultAddKeyBehavior( Check check, bool setChildren );

    //! Copy the data
    std::unique_ptr<KeyData> clone() const override;

    //! Copy the data
    void copy( const Database &rhs );

    //! Copy the data
    std::unique_ptr<Database> cloneDatabase() const;

    //! Get the name of the database
    inline const std::string &getName() const { return d_name; }

    //! Get the name of the database
    inline void setName( std::string name ) { d_name = std::move( name ); }

    /**
     * Return true if the specified key exists in the database and false
     *     otherwise.
     * @param[in] key           Key name to lookup.
     */
    bool keyExists( std::string_view key ) const;


    /**
     * Delete the key if it exists
     * @param[in] key           Key name to delete.
     */
    void deleteData( std::string_view key );


    /** \brief Return all keys in the database.
     * \details  This function will return the list of the keys available.
     *    The user may specify if they want the keys to be returned in
     *    sorted order.
     * \param[in] sort         Sort the keys (default is true)
     */
    std::vector<std::string> getAllKeys( bool sort = true ) const;


    //! Return true if the database is empty
    inline size_t empty() const { return d_data.size() == 0; }


    //! Return the number of entries in the database
    inline size_t size() const { return d_data.size(); }


    //! Return true if the databases are equivalent
    bool operator==( const Database &rhs ) const;


    //! Return true if the databases are equivalent
    bool operator==( const KeyData &rhs ) const override;


    //! Return the number of entries in the database
    inline bool operator!=( const Database &rhs ) const { return !operator==( rhs ); }


    /**
     * Get the key as a string
     * @param[in] key       Key name in database.
     * @param[in] src       Source location in case of error
     */
    inline std::string getString( std::string_view key,
                                  source_location src = source_location::current() ) const
    {
        return getScalar<std::string>( key, {}, src );
    }


    /**
     * Get the scalar entry from the database with the specified key
     * name.  If the specified key does not exist in the database or
     * is not a scalar of the given type, then an error message is printed and
     * the program exits.
     *
     * @param[in] key       Key name in database.
     * @param[in] unit      Desired units
     * @param[in] src       Source location in case of error
     */
    template<class TYPE>
    TYPE getScalar( std::string_view key,
                    const Units &unit   = Units(),
                    source_location src = source_location::current() ) const;


    /**
     * Get the scalar entry from the database with the specified key
     * name.  If the specified key does not exist in the database the
     * the default value will be printed
     *
     * @param[in] key       Key name in database
     * @param[in] value     Default value
     * @param[in] unit      Desired units
     * @param[in] src       Source location in case of error
     */
    template<class TYPE>
    TYPE getWithDefault( std::string_view key,
                         IdentityType<const TYPE &> value,
                         const Units &unit   = Units(),
                         source_location src = source_location::current() ) const;


    /**
     * Get the vector entries from the database with the specified key
     * name.  If the specified key does not exist in the database or
     * is not of the given type, then an error message is printed and
     * the program exits.
     *
     * @param[in] key       Key name in database.
     * @param[in] unit      Desired units
     * @param[in] src       Source location in case of error
     */
    template<class TYPE>
    Array<TYPE> getArray( std::string_view key,
                          const Units &unit   = Units(),
                          source_location src = source_location::current() ) const;


    /**
     * Get the vector entries from the database with the specified key
     * name.  If the specified key does not exist in the database or
     * is not of the given type, then an error message is printed and
     * the program exits.
     *
     * @param[in] key       Key name in database.
     * @param[in] unit      Desired units
     * @param[in] src       Source location in case of error
     */
    template<class TYPE>
    std::vector<TYPE> getVector( std::string_view key,
                                 const Units &unit   = Units(),
                                 source_location src = source_location::current() ) const;


    /**
     * Put the scalar entry into the database with the specified key name.
     * @param[in] key       Key name in database.
     * @param[in] value     Value to store
     * @param[in] unit      Desired units
     * @param[in] check     Optional value to indicate the behavior if the key exists
     * @param[in] src       Source location in case of error
     */
    template<class TYPE>
    void putScalar( std::string_view key,
                    TYPE value,
                    Units unit          = Units(),
                    Check check         = Check::GetDatabaseDefault,
                    source_location src = source_location::current() );


    /**
     * Put the vector entries into the database with the specified key
     * name.  If the specified key does not exist in the database or
     * is not of the given type, then an error message is printed and
     * the program exits.
     *
     * @param[in] key       Key name in database.
     * @param[in] data      Data to store
     * @param[in] unit      Desired units
     * @param[in] check     Optional value to indicate the behavior if the key exists
     * @param[in] src       Source location in case of error
     */
    template<class TYPE>
    void putArray( std::string_view key,
                   Array<TYPE> data,
                   Units unit          = Units(),
                   Check check         = Check::GetDatabaseDefault,
                   source_location src = source_location::current() );


    /**
     * Put the vector entries into the database with the specified key
     * name.  If the specified key does not exist in the database or
     * is not of the given type, then an error message is printed and
     * the program exits.
     *
     * @param[in] key       Key name in database.
     * @param[in] data      Data to store
     * @param[in] unit      Desired units
     * @param[in] check     Optional value to indicate the behavior if the key exists
     * @param[in] src       Source location in case of error
     */
    template<class TYPE>
    void putVector( std::string_view key,
                    const std::vector<TYPE> &data,
                    Units unit          = Units(),
                    Check check         = Check::GetDatabaseDefault,
                    source_location src = source_location::current() );


    /**
     * Get a raw pointer to the data for a key in the database.
     * If the specified key does not exist, a null pointer is returned.
     *
     * @param key Key name in database.
     */
    KeyData *getData( std::string_view key );

    /**
     * Get a raw pointer to the data for a key in the database.
     * If the specified key does not exist, a null pointer is returned.
     *
     * @param key Key name in database.
     */
    const KeyData *getData( std::string_view key ) const;


    /**
     * Put the data for a key in the database.
     *
     * @param[in] key       Key name in database.
     * @param[in] data      Data to store
     * @param[in] check     Optional value to indicate the behavior if the key exists
     * @param[in] src       Source location in case of error
     */
    void putData( std::string_view key,
                  std::unique_ptr<KeyData> data,
                  Check check         = Check::GetDatabaseDefault,
                  source_location src = source_location::current() );


    //! Check if the key is a database object
    bool isDatabase( std::string_view key, source_location src = source_location::current() ) const;


    //! Check if the named entry is a string
    bool isString( std::string_view key, source_location src = source_location::current() ) const;

    /**
     * Check if the named entry is an equation
     * Note: scalar values can be represented as an equation and will return true
     *
     * @param[in] key       Key name in database
     * @param[in] src       Source location in case of error
     */
    bool isEquation( std::string_view key, source_location src = source_location::current() ) const;


    /**
     * Return an equation for the key
     * Note: scalar values can be represented as an equation and will return a new scalar equation
     *
     * @param[in] key       Key name in database
     * @param[in] unit      Desired units
     * @param[in] src       Source location in case of error
     */
    std::unique_ptr<MathExpr> getEquation( std::string_view key,
                                           const Units &unit   = Units(),
                                           source_location src = source_location::current() ) const;


    //! Check if the entry can be stored as the given type
    template<class TYPE>
    bool isType( std::string_view key, source_location src = source_location::current() ) const;

    //! Get the fundamental type (e.g. double, int, float, ...)
    typeID getDataType( std::string_view key ) const;

    /**
     * Get a raw pointer to the database for a key in the database.
     * If the specified key does not exist, a null pointer is returned.
     *
     * @param[in] key       Key name in database
     * @param[in] src       Source location in case of error
     */
    std::shared_ptr<Database> getDatabase( std::string_view key,
                                           source_location src = source_location::current() );

    /**
     * Get a raw pointer to the database for a key in the database.
     * If the specified key does not exist, a null pointer is returned.
     *
     * @param[in] key       Key name in database
     * @param[in] src       Source location in case of error
     */
    std::shared_ptr<const Database>
    getDatabase( std::string_view key, source_location src = source_location::current() ) const;


    /**
     * Access the database
     *
     * @param[in] key       Key name in database
     * @param[in] src       Source location in case of error
     */
    const Database &operator()( std::string_view key,
                                source_location src = source_location::current() ) const;


    /**
     * Put the database for a key in the database.
     * If the specified key already exists in the database an error is thrown.
     *
     * @param key       Key name in database.
     * @param db        Database to store
     */
    inline void putDatabase( std::string_view key, std::unique_ptr<Database> db )
    {
        putData( key, std::move( db ) );
    }


    /**
     * Create a new empty database in the current database and return a pointer.
     * If the specified key already exists in the database an error is thrown.
     * This is equivalent to:
     * \code
         auto tmp = std::make_unique<Database>( key );
         this->putDatabase( key, std::move( tmp ) );
         return this->getDatabase( key );
     * \endcode
     *
     * @param key       Key name in database.
     */
    inline std::shared_ptr<Database> createAddDatabase( std::string_view key )
    {
        putDatabase( key, std::make_unique<Database>( std::string( key ) ) );
        return getDatabase( key );
    }


    /**
     * Create a database and return it
     * If the specified key already exists in the database an error is thrown.
     *
     * @param key       Key name in database.
     */
    inline std::shared_ptr<Database> putDatabase( std::string_view key )
    {
        putData( key, std::make_unique<Database>( std::string( key.data(), key.size() ) ) );
        return getDatabase( key );
    }


    /**
     * Erase the given key
     * If the specified key does not exists and check is true, an error is thrown.
     *
     * @param key       Key name in database
     * @param check     Check if the key exists
     */
    void erase( std::string_view key, bool check = true );


    /**
     * Clear all keys from the database
     */
    void clear();


    /**
     * Print the data to a stream
     * @param os        Output stream
     * @param indent    Indenting to use before each line
     * @param sort      Sort the entries before printing
     * @param printType Print a comment with the stored datatype
     */
    void print( std::ostream &os,
                std::string_view indent = "",
                bool sort               = true,
                bool printType          = false ) const override;


    //! Print the type
    typeID getDataType() const override { return AMP::getTypeID<Database>(); }


    /**
     * Print the data to a string
     * @param indent    Indenting to use before each line
     * @param sort      Sort the entries before printing
     * @param printType Print a comment with the stored datatype
     * @return          Output string
     */
    std::string
    print( std::string_view indent = "", bool sort = true, bool printType = false ) const;


    /**
     * Get unused entries
     * @param recursive Check sub databases (pre-pending by DatabaseName::)
     * @return          Output string
     */
    std::vector<std::string> getUnused( bool recursive = true ) const;


public: // Pack/unpack data
    size_t packSize() const override;
    size_t pack( std::byte * ) const override;
    size_t unpack( const std::byte * ) override;
    void writeHDF5( int64_t fid, const std::string &name ) const override;
    void readHDF5( int64_t fid, const std::string &name ) override;

#ifdef AMP_USE_SAMRAI
public: // SAMRAI interfaces
    //! Construct a database from a SAMRAI database
    Database( SAMRAI::tbox::Database & );

    //! Create a SAMRAI database from this
    std::shared_ptr<SAMRAI::tbox::Database> cloneToSAMRAI() const;

    //! Create a database using SAMRAI and then convert to an AMP database
    static std::shared_ptr<Database> readThroughSAMRAI( const std::string &filename );

#endif


protected: // Internal data and functions
    Check d_check;
    std::string d_name;
    std::vector<uint32_t> d_hash;
    std::vector<std::string> d_keys;
    std::vector<std::shared_ptr<KeyData>> d_data;
    mutable std::vector<bool> d_used;

    // Function to add arguments to the database
    template<class TYPE, class... Args>
    void addArgs( std::string_view key, TYPE value, Args... args );

    // Function to add arguments to the database
    template<class TYPE, class... Args>
    void addArgsWithUnits( std::string_view key, TYPE value, const Units &unit, Args... args );

    // Hash a string
    static inline uint32_t hashString( std::string_view s )
    {
        return std::hash<std::string_view>{}( s );
    }

    // Find an entry
    inline int find( uint32_t hash, bool use ) const
    {
        for ( size_t i = 0; i < d_hash.size(); i++ ) {
            if ( hash == d_hash[i] ) {
                if ( use && !d_used[i] )
                    d_used[i] = true;
                return i;
            }
        }
        return -1;
    }

    // Functions inherited from KeyData that really aren't valid
    Array<double> convertToDouble() const override;
    Array<int64_t> convertToInt64() const override;
    ArraySize arraySize() const override { return ArraySize( 1 ); }
    bool is_floating_point() const override;
    bool is_integral() const override;
};


//! Class to store a box
class DatabaseBox final
{
public:
    //! Empty constructor
    DatabaseBox();

    //! Default constructor
    DatabaseBox( int dim, const int *lower, const int *upper );

    //! Construct from a string of the format "[(0,0,0), (7,7,7)]"
    explicit DatabaseBox( std::string_view str );

#ifdef AMP_USE_SAMRAI
    //! Construct a DatabaseBox from a SAMRAI DatabaseBox
    DatabaseBox( const SAMRAI::tbox::DatabaseBox & );

    //! Create a SAMRAI DatabaseBox from this
    SAMRAI::tbox::DatabaseBox cloneToSAMRAI() const;
#endif

    //! Return a non-const reference to the number of dimensions
    uint8_t &dim();

    //! Return the number of dimensions
    uint8_t dim() const;

    //! Return whether the box is empty.
    //! A box is empty if it has dimension zero or if any part of the
    //! upper index is less than its corresponding part of the lower index.
    bool empty() const;

    //! Return non-const reference to the specified component of the lower index
    int &lower( uint8_t i );

    //! Return the specified component of the lower index
    int lower( uint8_t i ) const;

    //! Return non-const reference to the specified component of the upper index
    int &upper( uint8_t i );

    //! Return the specified component of the upper index
    int upper( uint8_t i ) const;

    bool operator==( const DatabaseBox &box ) const;

private:
    uint8_t d_dim;
    std::array<int, 5> d_lower, d_upper;
};


/********************************************************************
 * Register KeyData with the factory                                 *
 ********************************************************************/
void registerKeyData( const std::string &name, std::function<std::unique_ptr<KeyData>()> fun );


/********************************************************************
 * Overload ostream operator                                         *
 ********************************************************************/
std::ostream &operator<<( std::ostream &out, const DatabaseBox & );


/********************************************************************
 * Create database from arguments                                    *
 ********************************************************************/
template<class TYPE, class... Args>
inline void Database::addArgs( std::string_view key, TYPE value, Args... args )
{
    if constexpr ( is_vector_v<TYPE> ) {
        putVector( key, value );
    } else if constexpr ( is_Array_v<TYPE> ) {
        putArray( key, value );
    } else if constexpr ( std::is_same_v<TYPE, std::string> ||
                          std::is_same_v<TYPE, std::string_view> ) {
        putScalar( key, value );
    } else if constexpr ( has_size_v<TYPE> || is_initializer_list_v<TYPE> ) {
        typedef decltype( *value.begin() ) TYPE2;
        typedef typename AMP::remove_cvref_t<TYPE2> TYPE3;
        std::vector<TYPE3> data( value.begin(), value.end() );
        putVector( key, std::move( data ) );
    } else {
        putScalar( key, value );
    }
    if constexpr ( sizeof...( args ) > 0 )
        addArgs( args... );
}
template<class TYPE, class... Args>
inline void
Database::addArgsWithUnits( std::string_view key, TYPE value, const Units &unit, Args... args )
{
    if constexpr ( is_vector_v<TYPE> ) {
        putVector( key, value, unit );
    } else if constexpr ( is_Array_v<TYPE> ) {
        putArray( key, value );
    } else if constexpr ( std::is_same_v<TYPE, std::string> ||
                          std::is_same_v<TYPE, std::string_view> ) {
        putScalar( key, value, unit );
    } else if constexpr ( has_size_v<TYPE> || is_initializer_list_v<TYPE> ) {
        typedef decltype( *value.begin() ) TYPE2;
        typedef typename AMP::remove_cvref_t<TYPE2> TYPE3;
        std::vector<TYPE3> data( value.begin(), value.end() );
        putVector( key, std::move( data ), unit );
    } else {
        putScalar( key, value, unit );
    }
    if constexpr ( sizeof...( args ) > 0 )
        addArgsWithUnits( args... );
}
template<class... Args>
inline std::unique_ptr<Database> Database::create( Args... args )
{
    constexpr size_t n = sizeof...( args );
    static_assert( n % 2 == 0 );
    auto db = std::make_unique<Database>();
    if ( n == 0 )
        return db;
    db->addArgs( args... );
    return db;
}
template<class... Args>
inline std::unique_ptr<Database> Database::createWithUnits( Args... args )
{
    constexpr size_t n = sizeof...( args );
    static_assert( n % 3 == 0 );
    auto db = std::make_unique<Database>();
    if ( n == 0 )
        return db;
    db->addArgsWithUnits( args... );
    return db;
}


} // namespace AMP


#endif
