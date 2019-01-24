#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/InputDatabase.h"
#include "AMP/utils/InputManager.h"
#include "AMP/utils/MemoryDatabase.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/shared_ptr.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>


/************************************************************************
 *                                                                       *
 * This tests whether we can create and use an InputManager object       *
 *                                                                       *
 ************************************************************************/
void readInputDatabase( AMP::UnitTest *ut )
{
    std::string input_file = "input_Database";
    std::string log_file   = "output_Database";

    // Create input database and parse all data in input file.
    auto input_db = AMP::make_shared<AMP::InputDatabase>( "input_db" );
    AMP::InputManager::getManager()->parseInputFile( input_file, input_db );

    auto tmp_db = input_db->getDatabase( "Try" );
    int number  = tmp_db->getInteger( "number" );

    if ( number > 0 ) {
        auto intArray = tmp_db->getIntegerArray( "intArray" );
        if ( (int) intArray.size() != number )
            ut->failure( "intArray was the wrong size" );
        auto doubleArray = tmp_db->getDoubleArray( "doubleArray" );
        if ( (int) doubleArray.size() != number )
            ut->failure( "doubleArray was the wrong size" );
    }

    std::cout << "tmp_db created and destroyed successfully." << std::endl;

    input_db.reset();

    ut->passes( "Get Database Works and the Destructor of AMP::Database works." );
}


/************************************************************************
 *                                                                       *
 * This tests whether we can put/get keys with a database                *
 *                                                                       *
 ************************************************************************/
template<class DATABASE>
void testCreateDatabase( AMP::UnitTest *ut )
{
    AMP::shared_ptr<DATABASE> db( new DATABASE( "database" ) );
    db->create( "database" );

    int lower[3] = { 0, 0, 0 }, upper[3] = { 10, 10, 10 };
    AMP::DatabaseBox box( 3, lower, upper );
    AMP_ASSERT( box == box );
    AMP_ASSERT( box.getDimension() == 3 );
    for ( int i = 0; i < 3; i++ ) {
        AMP_ASSERT( box.lower( i ) == lower[i] );
        AMP_ASSERT( box.upper( i ) == upper[i] );
    }

    db->putScalar( "scalar_int", (int) 1 );
    db->putScalar( "scalar_float", (float) 1 );
    db->putScalar( "scalar_double", (double) 1 );
    db->putScalar( "scalar_complex", std::complex<double>( 1, 0 ) );
    db->putScalar( "scalar_char", (char) 1 );
    db->putScalar( "scalar_bool", true );
    db->putDatabaseBox( "box", box );
    db->putDatabaseBoxArray( "box_array", std::vector<AMP::DatabaseBox>( 1, box ) );

    AMP_ASSERT( db->keyExists( "scalar_int" ) );

    AMP_ASSERT( db->isInteger( "scalar_int" ) );
    AMP_ASSERT( db->isFloat( "scalar_float" ) );
    AMP_ASSERT( db->isDouble( "scalar_double" ) );
    AMP_ASSERT( db->isComplex( "scalar_complex" ) );
    AMP_ASSERT( db->isChar( "scalar_char" ) );
    AMP_ASSERT( db->isBool( "scalar_bool" ) );
    AMP_ASSERT( db->isDatabaseBox( "box" ) );

    AMP_ASSERT( db->getInteger( "scalar_int" ) == 1 );
    AMP_ASSERT( db->getFloat( "scalar_float" ) == 1.0 );
    AMP_ASSERT( db->getDouble( "scalar_double" ) == 1.0 );
    AMP_ASSERT( db->getComplex( "scalar_complex" ) == std::complex<double>( 1, 0 ) );
    AMP_ASSERT( db->getChar( "scalar_char" ) == 1 );
    AMP_ASSERT( db->getBool( "scalar_bool" ) == true );
    AMP_ASSERT( db->getDatabaseBox( "box" ) == box );
    AMP_ASSERT( db->getDatabaseBoxArray( "box_array" ).operator[]( 0 ) == box );

    AMP_ASSERT( db->getIntegerWithDefault( "scalar_int", 0 ) == 1 );
    AMP_ASSERT( db->getFloatWithDefault( "scalar_float", 0 ) == 1.0 );
    AMP_ASSERT( db->getDoubleWithDefault( "scalar_double", 0 ) == 1.0 );
    AMP_ASSERT( db->getComplexWithDefault( "scalar_complex", std::complex<double>( 0, 0 ) ) ==
                std::complex<double>( 1, 0 ) );
    AMP_ASSERT( db->getCharWithDefault( "scalar_char", 0 ) == 1 );
    AMP_ASSERT( db->getBoolWithDefault( "scalar_bool", false ) == true );

    AMP_ASSERT( db->getIntegerArray( "scalar_int" ).size() == 1 );
    AMP_ASSERT( db->getFloatArray( "scalar_float" ).size() == 1 );
    AMP_ASSERT( db->getDoubleArray( "scalar_double" ).size() == 1 );
    AMP_ASSERT( db->getComplexArray( "scalar_complex" ).size() == 1 );
    AMP_ASSERT( db->getCharArray( "scalar_char" ).size() == 1 );
    AMP_ASSERT( db->getBoolArray( "scalar_bool" ).size() == 1 );

    ut->passes( "Create database passes" );
}


//---------------------------------------------------------------------------//

int main( int argc, char *argv[] )
{
    AMP::AMP_MPI::start_MPI( argc, argv );
    AMP::UnitTest ut;

    readInputDatabase( &ut );
    testCreateDatabase<AMP::MemoryDatabase>( &ut );
    testCreateDatabase<AMP::InputDatabase>( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMP_MPI::stop_MPI();
    return num_failed;
}