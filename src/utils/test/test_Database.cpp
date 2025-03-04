#include "AMP/AMP_TPLs.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Array.hpp"
#include "AMP/utils/Database.h"
#include "AMP/utils/Database.hpp"
#include "AMP/utils/MathExpr.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/to_tuple.h"

#include <chrono>
#include <complex>
#include <fstream>
#include <random>
#include <sstream>


using namespace AMP;


std::default_random_engine generator( 123 );


bool equal( double a, double b ) { return fabs( a - b ) <= 1e-12 * fabs( a + b ); }


// Add the result to the tests
void checkResult( AMP::UnitTest &ut, bool pass, const std::string &msg )
{
    if ( pass )
        ut.passes( msg );
    else
        ut.failure( msg );
}


// Generate random number
template<class TYPE>
static TYPE random()
{
    if constexpr ( std::is_floating_point_v<TYPE> ) {
        std::uniform_real_distribution<double> dist( 0, 1 );
        return static_cast<TYPE>( dist( generator ) );
    } else if constexpr ( std::is_integral_v<TYPE> ) {
        std::uniform_int_distribution<int> dist( 0, 1000000000 );
        return static_cast<TYPE>( dist( generator ) );
    } else if constexpr ( std::is_same_v<TYPE, std::complex<double>> ) {
        std::uniform_real_distribution<double> dist( 0, 1 );
        return std::complex<double>( random<double>(), random<double>() );
    } else if constexpr ( std::is_same_v<TYPE, std::complex<float>> ) {
        std::uniform_real_distribution<float> dist( 0, 1 );
        return std::complex<float>( random<float>(), random<float>() );
    } else {
        AMP_ERROR( "Invalid TYPE: " + std::string( AMP::getTypeID<TYPE>().name ) );
    }
    return 0;
}


// Test a type in the database
template<class TYPE>
static void addType( Database &db, UnitTest &ut )
{
    bool pass = true;
    std::string typeName( AMP::getTypeID<TYPE>().name );
    TYPE rand = random<TYPE>();
    db.putScalar<TYPE>( "scalar-" + typeName, rand );
    db.putVector<TYPE>( "vector-" + typeName, { rand } );
    auto v1 = db.getScalar<TYPE>( "scalar-" + typeName );
    auto v2 = db.getVector<TYPE>( "vector-" + typeName );
    pass    = pass && v1 == rand && v2.size() == 1 && v2[0] == rand;
    pass    = pass && db.isType<TYPE>( "scalar-" + typeName );
    pass    = pass && db.isType<TYPE>( "vector-" + typeName );
    if ( std::is_floating_point_v<TYPE> ) {
        auto v3 = db.getScalar<double>( "scalar-" + typeName );
        auto v4 = db.getVector<double>( "vector-" + typeName );
        pass    = pass && static_cast<TYPE>( v3 ) == rand;
        pass    = pass && v4.size() == 1 && static_cast<TYPE>( v4[0] ) == rand;
        pass    = pass && db.isType<double>( "scalar-" + typeName );
        pass    = pass && !db.isType<int>( "scalar-" + typeName );
    } else if ( std::is_integral_v<TYPE> ) {
        auto v3 = db.getScalar<int>( "scalar-" + typeName );
        auto v4 = db.getVector<int>( "vector-" + typeName );
        pass    = pass && static_cast<TYPE>( v3 ) == rand;
        pass    = pass && v4.size() == 1 && static_cast<TYPE>( v4[0] ) == rand;
        pass    = pass && db.isType<double>( "scalar-" + typeName );
        pass    = pass && db.isType<int>( "scalar-" + typeName );
    }
    if ( !pass )
        ut.failure( typeName );
}


// Run some basic tests
template<class TYPE>
static bool isType( std::shared_ptr<const AMP::Database> db, const std::string_view &key )
{
    return db->getDataType( key ) == AMP::getTypeID<TYPE>() && db->isType<TYPE>( key );
}
void runBasicTests( UnitTest &ut )
{
    // Create a database with some different types of data
    Database db;
    db.putScalar<std::string>( "string", "test" );
    db.putScalar<bool>( "true", true );
    db.putScalar<bool>( "false", false );
    db.putScalar<double>( "double", 3.14 );
    db.putScalar<int>( "int", -2 );
    db.putScalar<double>( "inf", std::numeric_limits<double>::infinity() );
    db.putScalar<double>( "nan", std::numeric_limits<double>::quiet_NaN() );
    db.putVector<int>( "i3", { 1, 1, 0 } );
    db.putVector<double>( "x", { 1.1, 2.2, 3.3 } );
    db.putScalar<double>( "x1", 1.5, "cm" );
    db.putVector<double>( "x2", { 2.5 }, "mm" );
    db.putVector<std::shared_ptr<Database>>( "dummy", {} );
    db.deleteData( "dummy" );

    // Test adding some different types
    addType<uint8_t>( db, ut );
    addType<uint16_t>( db, ut );
    addType<uint32_t>( db, ut );
    addType<uint64_t>( db, ut );
    addType<int8_t>( db, ut );
    addType<int16_t>( db, ut );
    addType<int32_t>( db, ut );
    addType<int64_t>( db, ut );
    addType<float>( db, ut );
    addType<double>( db, ut );
    addType<long double>( db, ut );
    addType<std::complex<double>>( db, ut );

    // Try to read/add a database that ends in a comment and has units
    const char beamDatabaseText[] =
        "prof = \"gaussian\"     // Temporal profile\n"
        "z_shape = \"hard_hat\"  // Shape along the line\n"
        "y_shape = \"gaussian\"  // Shape perpendicular to the line\n"
        "geometry = 1            // Beam geometry\n"
        "x      = 0 cm           // Position\n"
        "FWHM   = 120 ps         // Full Width Half Max\n"
        "E      = 0.4 J          // Energy in the beam\n"
        "delay  = 0 ps           // Delay the beam respect to zero\n"
        "diam   = 30 um          // Beam diameter\n"
        "length = 0.4 cm         // Beam length\n "
        "lambda = 0.8 um         // Wavelength of laser\n"
        "angle  = 0 degrees      // Angle of beam with repect to normal\n"
        "array  = [ [ [ 1, 2, 3, 4 ], [5,6,7,8],[9,10,11,12]],\n"
        "         [[0,1,2,3],[4,5,6,7],[8,9,10,11]]] m   // Multi-dimentional array\n"
        "empty  = []             // Empty array\n"
        "end  = \"end\"          // String\n";
    auto beam = Database::createFromString( beamDatabaseText );
    db.putDatabase( "beam", std::move( beam ) );

    // Check the arrays
    auto empty = db.getDatabase( "beam" )->getData( "empty" );
    if ( !empty )
        ut.failure( "empty array (1)" );
    else if ( empty->arraySize().length() != 0 )
        ut.failure( "empty array (2)" );
    auto array = db.getDatabase( "beam" )->getArray<int>( "array", "m" );
    if ( array.size() != ArraySize( 4, 3, 2 ) )
        ut.failure( "array size" );

    // Write the database to a file
    std::ofstream inputfile;
    inputfile.open( "test_Database.out" );
    db.print( inputfile, "", false, true );
    inputfile.close();

    // Read the database and check that everything matches
    auto db2 = Database::parseInputFile( "test_Database.out" );
    if ( !isType<std::string>( db2, "string" ) )
        ut.failure( "string is a database?" );
    if ( !isType<std::string>( db2, "string" ) ||
         db2->getScalar<std::string>( "string" ) != "test" )
        ut.failure( "string" );
    if ( !isType<bool>( db2, "true" ) || !db2->getScalar<bool>( "true" ) )
        ut.failure( "true" );
    if ( !isType<bool>( db2, "false" ) || db2->getScalar<bool>( "false" ) )
        ut.failure( "false" );
    if ( !isType<double>( db2, "double" ) || db2->getScalar<double>( "double" ) != 3.14 )
        ut.failure( "double" );
    if ( !isType<int>( db2, "int" ) || db2->getScalar<int>( "int" ) != -2 )
        ut.failure( "int" );
    if ( !isType<int>( db2, "i3" ) )
        ut.failure( "i3" );
    if ( !isType<double>( db2, "x" ) )
        ut.failure( "x" );
    if ( db2->getScalar<double>( "inf" ) != std::numeric_limits<double>::infinity() )
        ut.failure( "inf" );
    if ( db2->getScalar<double>( "nan" ) == db2->getScalar<double>( "nan" ) )
        ut.failure( "nan" );
    if ( !equal( db2->getScalar<double>( "x1", "m" ), 0.015 ) ||
         !equal( db2->getScalar<double>( "x2", "m" ), 0.0025 ) ||
         !equal( db2->getVector<double>( "x1", "um" )[0], 15000 ) ||
         !equal( db2->getVector<double>( "x2", "um" )[0], 2500 ) )
        ut.failure( "units" );
    checkResult( ut, db == *db2, "print" );

    // Add more types that are not compatible with print
    addType<std::complex<float>>( db, ut );

    // Try to clone the database
    auto db3 = db.cloneDatabase();
    checkResult( ut, db == *db3, "clone" );

    // Test copy/move operators
    bool pass = true;
    Database db4( std::move( *db3 ) );
    pass = pass && db == db4;
    Database db5;
    db5  = std::move( db4 );
    pass = pass && db == db5;
    Database db6;
    db6.copy( db5 );
    pass = pass && db == db5;
    checkResult( ut, pass, "copy/move" );

    // Test creating a database from key/value pairs
    int v1                   = 4;
    double v2                = 2.0;
    std::string v3           = "string";
    std::vector<double> v4   = { 1.5, 0.2 };
    std::array<double, 3> v5 = { 2.4, 3.7, 5.6 };
    std::set<double> v6      = { 1.2, 1.3, 1.4 };
    auto db7                 = Database::create(
        "v1", v1, "v2", v2, "v3", v3, "v4", v4, "v5", v5, "v6", v6, "v7", "test" );
    pass = db7->getScalar<int>( "v1" ) == v1 && db7->getScalar<double>( "v2" ) == v2 &&
           db7->getString( "v3" ) == v3 && db7->getVector<double>( "v4" ) == v4 &&
           db7->getVector<double>( "v5" ) == std::vector<double>( v5.begin(), v5.end() ) &&
           db7->getVector<double>( "v6" ) == std::vector<double>( v6.begin(), v6.end() ) &&
           db7->getString( "v7" ) == "test";
    checkResult( ut, pass, "Database::create" );

    // Test creating a database from key/value/unit triplets
    // clang-format off
    db7 = Database::createWithUnits( "v1", v1,   "", "v2", v2, "cm", "v3", v3, "", "v4", v4, "m",
                                     "v5", v5, "kg", "v6", v6,  "J", "v7", "test", "" );
    // clang-format on
    pass = db7->getScalar<int>( "v1" ) == v1 && db7->getScalar<double>( "v2", "m" ) == 1e-2 * v2 &&
           db7->getString( "v3" ) == v3 && db7->getVector<double>( "v4", "m" ) == v4 &&
           db7->getVector<double>( "v5", "kg" ) == std::vector<double>( v5.begin(), v5.end() ) &&
           db7->getVector<double>( "v6", "J" ) == std::vector<double>( v6.begin(), v6.end() ) &&
           db7->getString( "v7" ) == "test";
    checkResult( ut, pass, "Database::createUnits" );

    // Test getting some values in different units
    Database db8;
    db8.putScalar<double>( "I1", 2.3, "W/cm^2" );
    db8.putScalar<double>( "I2", 2.3e4, "J/(s*m^2)" );
    db8.putScalar<double>( "I3", 2.3e7, "ergs/(s*cm^2)" );
    auto I1 = db8.getScalar<double>( "I1", "W/cm^2" );
    auto I2 = db8.getScalar<double>( "I2", "W/cm^2" );
    auto I3 = db8.getScalar<double>( "I3", "W/cm^2" );
    pass    = equal( I1, 2.3 ) && equal( I2, 2.3 ) && equal( I3, 2.3 );
    checkResult( ut, pass, "Database::units" );

    // Run some extra Units tests
    pass = Units( Units( "W/m^2" ).str() ) == Units( "uW/mm^2" );
    pass = pass && Units( ( Units( "V" ) * Units( "A" ) ).str() ) == Units( "W" );
    checkResult( ut, pass, "Units" );

    // Test getting a key that doesn't exists
    try {
        db.getScalar<double>( "garbage" );
        ut.failure( "Unknown key" );
    } catch ( StackTrace::abort_error &err ) {
        std::string file = std::string( err.source.file_name() );
        if ( file.find( "test_Database.cpp" ) != std::string::npos )
            ut.passes( "Unknown key (source file/line detected)" );
        else
            ut.passes( "Unknown key (default error)" );
    } catch ( ... ) {
        ut.failure( "Unknown key (unknown exception)" );
    }
}


// Run tests using an input file
void runFileTests( UnitTest &ut, const std::string &filename )
{
    auto prefix = filename + " - ";
    // Read the file
    auto t1 = std::chrono::system_clock::now();
    auto db = Database::parseInputFile( filename );
    auto t2 = std::chrono::system_clock::now();
    int us  = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    printf( "Time to read %s: %i us\n", filename.c_str(), us );
    checkResult( ut, !db->empty(), prefix + "!empty()" );
    // Create database from string
    if ( AMP::IO::getSuffix( filename ) != "yml" ) {
        std::ifstream ifstream( filename );
        std::stringstream buffer;
        buffer << ifstream.rdbuf();
        t1       = std::chrono::system_clock::now();
        auto db2 = Database::createFromString( buffer.str() );
        t2       = std::chrono::system_clock::now();
        us       = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
        printf( "Time to create from string: %i us\n", us );
        checkResult( ut, *db2 == *db, prefix + "createFromString" );
    }
    // Clone the database
    t1       = std::chrono::system_clock::now();
    auto db3 = db->cloneDatabase();
    t2       = std::chrono::system_clock::now();
    us       = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    printf( "Time to clone database: %i us\n", us );
    // Check the database
    t1 = std::chrono::system_clock::now();
    checkResult( ut, *db3 == *db, prefix + "clone" );
    t2 = std::chrono::system_clock::now();
    us = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    printf( "Time to compare database: %i us\n", us / 2 );
    // Write the database to a file
    std::ofstream inputfile;
    inputfile.open( "test_Database.2.out" );
    db->print( inputfile, "", false, true );
    inputfile.close();
    // Check that data loaded correctly
    if ( filename == "laser_plasma_input.txt" ) {
        bool pass = !db->getVector<std::string>( "material" ).empty();
        checkResult( ut, pass, "Found material" );
    }
    if ( filename == "library.yml" ) {
        auto name = ( *db )( "3d" )( "plastics" )( "pmma" ).getString( "name" );
        bool pass = name == "PMMA - Poly(methyl methacrylate)";
        checkResult( ut, pass, "Found material" );
    }
    if ( filename == "Johnson.yml" ) {
        auto data = ( *db )( "tabulated nk" ).getArray<double>( "data" );
        bool pass = data.size() == AMP::ArraySize( 49, 3 );
        checkResult( ut, pass, "Found material" );
    }
    if ( filename == "input_Database" ) {
        auto db2 = db->getDatabase( "db1" );
        checkResult( ut, db2->getScalar<int>( "eq1" ) == 9, "eq1 evaluates to a scalar int" );
        auto eq1 = db2->getEquation( "eq1" );
        auto eq2 = db2->getEquation( "eq2" );
        checkResult( ut, ( *eq1 )() == 9, "eq1" );
        checkResult( ut, ( *eq2 )( { 3.0 } ) == 12, "eq2" );
        auto db3 = db->getDatabase( "db2" )->getDatabase( "db" );
        AMP_ASSERT( db3 );
        checkResult( ut, *db2 == *db3, "db2->db == db3" );
    }
    printf( "\n" );
    // Try sending/receiving database
    auto db2 = AMP::AMP_MPI( AMP_COMM_WORLD ).bcast( db, 0 );
    checkResult( ut, *db == *db2, filename + " - send/recv" );
}


// Test converting a struct to a database
struct myClass {
    int a;
    double b;
    float c;
    std::complex<double> d;
};
void testStructToDatabase( AMP::UnitTest &ut )
{
#ifndef DISABLE_TO_TUPLE
    // Create a simple struct
    myClass x;
    x.a = 5;
    x.b = 3.14;
    x.c = 2.1;
    x.d = std::complex<double>( 1.4, 0.9 );

    // Check the conversion to a tuple
    auto t = AMP::to_tuple( x );
    static_assert(
        std::is_same_v<std::tuple<int, double, float, std::complex<double>>, decltype( t )> );
    bool pass = std::get<0>( t ) == x.a && std::get<1>( t ) == x.b && std::get<2>( t ) == x.c &&
                std::get<3>( t ) == x.d;
    checkResult( ut, pass, "Convert struct to tuple" );
#else
    ut.expected_failure( "to_tuple is not supported by compiler" );
#endif
}


// Main
int main( int argc, char *argv[] )
{
    AMPManager::startup( argc, argv );
    UnitTest ut;

    // Run the tests
    runBasicTests( ut );
    testStructToDatabase( ut );
    for ( int i = 1; i < argc; i++ )
        runFileTests( ut, argv[i] );

    // Return
    int N_errors = ut.NumFailGlobal();
    ut.report();
    ut.reset();
    AMPManager::shutdown();
    return N_errors;
}
