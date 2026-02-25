#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Units.h"
#include "AMP/utils/Utilities.h"


using AMP::Units;
using AMP::UnitType;


inline bool approx_equal( double a, double b )
{
    double e = a >= b ? a - b : b - a;
    return e < 1e-8 * b;
}


// Check unit prefixes
void checkPrefix( AMP::UnitTest &ut )
{
    bool pass = true;
    pass      = pass && Units::convert( Units::getUnitPrefix( "y" ) ) == 1e-24;
    pass      = pass && Units::convert( Units::getUnitPrefix( "z" ) ) == 1e-21;
    pass      = pass && Units::convert( Units::getUnitPrefix( "a" ) ) == 1e-18;
    pass      = pass && Units::convert( Units::getUnitPrefix( "f" ) ) == 1e-15;
    pass      = pass && Units::convert( Units::getUnitPrefix( "p" ) ) == 1e-12;
    pass      = pass && Units::convert( Units::getUnitPrefix( "n" ) ) == 1e-9;
    pass      = pass && Units::convert( Units::getUnitPrefix( "u" ) ) == 1e-6;
    pass      = pass && Units::convert( Units::getUnitPrefix( "m" ) ) == 1e-3;
    pass      = pass && Units::convert( Units::getUnitPrefix( "c" ) ) == 1e-2;
    pass      = pass && Units::convert( Units::getUnitPrefix( "d" ) ) == 0.1;
    pass      = pass && Units::convert( Units::getUnitPrefix( "" ) ) == 1;
    pass      = pass && Units::convert( Units::getUnitPrefix( "da" ) ) == 10;
    pass      = pass && Units::convert( Units::getUnitPrefix( "h" ) ) == 100;
    pass      = pass && Units::convert( Units::getUnitPrefix( "k" ) ) == 1e3;
    pass      = pass && Units::convert( Units::getUnitPrefix( "M" ) ) == 1e6;
    pass      = pass && Units::convert( Units::getUnitPrefix( "G" ) ) == 1e9;
    pass      = pass && Units::convert( Units::getUnitPrefix( "T" ) ) == 1e12;
    pass      = pass && Units::convert( Units::getUnitPrefix( "P" ) ) == 1e15;
    pass      = pass && Units::convert( Units::getUnitPrefix( "E" ) ) == 1e18;
    pass      = pass && Units::convert( Units::getUnitPrefix( "Z" ) ) == 1e21;
    pass      = pass && Units::convert( Units::getUnitPrefix( "Y" ) ) == 1e24;
    auto list = Units::getAllPrefixes();
    for ( size_t i = 0; i < list.size(); i++ ) {}
    if ( pass )
        ut.passes( "getUnitPrefix" );
    else
        ut.failure( "getUnitPrefix" );
}


// Check basic operators
void checkOperators( AMP::UnitTest &ut )
{
    bool pass = true;
    Units x( "V" );
    Units y( "A" );
    pass = pass && ( x == x ) && ( y == y );
    pass = pass && x != y;
    pass = pass && Units( "568.261251 m" ) != Units( "568.26125 m" );
    pass = pass && Units( "568.26125001 m" ) == Units( "568.26125 m" );
    pass = pass && x * y == Units( "W" );
    pass = pass && x.pow( 3 ) == Units( "V^3" );
    pass = pass && x / y == Units( "V/A" );
    // Try catching an error
    try {
        Units z( "m/s garbage" );
        pass = false;
    } catch ( const std::logic_error &err ) {
        if ( err.what() != std::string_view( "Unknown unit: garbage" ) ) {
            std::cout << "Caught unexpected message: " << err.what() << std::endl;
            pass = false;
        }
    } catch ( ... ) {
        pass = false;
        std::cout << "Caught unknown exception\n";
    }
    if ( pass )
        ut.passes( "checkOperators" );
    else
        ut.failure( "checkOperators" );
}


// Check unit type
void checkType( AMP::UnitTest &ut )
{
    bool pass = true;
    pass      = pass && Units( "meter" ).getType() == UnitType::length;
    pass      = pass && Units( "gram" ).getType() == UnitType::mass;
    pass      = pass && Units( "second" ).getType() == UnitType::time;
    pass      = pass && Units( "ampere" ).getType() == UnitType::current;
    pass      = pass && Units( "kelvin" ).getType() == UnitType::temperature;
    pass      = pass && Units( "joule" ).getType() == UnitType::energy;
    pass      = pass && Units( "erg" ).getType() == UnitType::energy;
    pass      = pass && Units( "watt" ).getType() == UnitType::power;
    pass      = pass && Units( "mole" ).getType() == UnitType::mole;
    pass      = pass && Units( "candela" ).getType() == UnitType::intensity;
    pass      = pass && Units( "degree" ).getType() == UnitType::angle;
    pass      = pass && Units( "radian" ).getType() == UnitType::angle;
    if ( pass )
        ut.passes( "Units::getType" );
    else
        ut.failure( "Units::getType" );
}


// Check some basic conversions
void check( AMP::UnitTest &ut,
            const std::string &u1,
            const std::string &u2 = "",
            double scale          = 1.0 )
{
    auto testname = u1;
    Units x( u1 );
    Units x2( x.str() );
    Units x3( x.printSI() );
    bool pass = x == x2 && x == x3;
    if ( u2 != "" ) {
        testname += " -> " + u2;
        pass = pass && approx_equal( x.convert( Units( u2 ) ), scale );
    } else {
        std::cout << u1 << " --> " << x.printFull() << std::endl;
    }
    if ( pass ) {
        ut.passes( testname );
    } else {
        std::cout << u1 << " --> " << x.printFull() << std::endl;
        ut.failure( testname );
    }
}
void check( AMP::UnitTest &ut, const std::string &u1, const std::string &u2, const std::string &u3 )
{
    auto testname = u1 + " - " + u2 + " - " + u3;
    Units x1( u1 );
    Units x2( u2 );
    Units x3( u3 );
    if ( x1 == x2 && x1 == x3 )
        ut.passes( testname );
    else
        ut.failure( testname );
}


void testPrefixUnits( AMP::UnitTest &ut )
{
    // Test combinations of units and prefixes to make sure there are not read errors
    auto units    = Units::getAllUnits();
    auto prefixes = Units::getAllPrefixes();
    bool pass     = true;
    for ( const auto &u : units ) {
        Units u1( u );
        for ( const auto &p : prefixes ) {
            double s = Units::convert( Units::getUnitPrefix( p ) );
            Units u2( p + u );
            double s2 = 0.0;
            try {
                s2 = u2.convert( u1 );
            } catch ( ... ) {
            }
            if ( !approx_equal( s2, s ) ) {
                pass = false;
                std::cout << u1.str() << "   " << u2.str() << "   " << s << std::endl;
            }
        }
    }
    if ( pass ) {
        ut.passes( "unit-prefix combinations" );
    } else {
        ut.failure( "unit-prefix combinations" );
    }
}


// Main
int main( int argc, char *argv[] )
{
    constexpr char ohm[]   = { (char) 206, (char) 169, (char) 0 }; // Ohm symbol in UTF-16
    constexpr char micro[] = { (char) 206, (char) 188, (char) 0 }; // micro symbol in UTF-16

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Run some basic checks
    AMP_ASSERT( Units().isNull() );
    AMP_ASSERT( Units( "" ).isNull() );
    checkPrefix( ut );
    checkType( ut );
    checkOperators( ut );

    // Test some simple types and conversions
    check( ut, "meter" );
    check( ut, "um", std::string( micro ) + "m" );
    check( ut, "uW/mm^2" );
    check( ut, "kg^-1*m^-2*s^3*A^2" );
    check( ut, "1.2e-6", " ", 1.2e-6 );
    check( ut, "3.2 x 10^4 uL", "L", 0.032 );
    check( ut, "W/m^2", "uW/mm^2" );
    check( ut, "radian", "rad" );
    check( ut, "steradian", "sr" );

    // Test derived SI units
    check( ut, "hertz", "Hz", "s^-1" );
    check( ut, "newton", "N", "kg*m*s^-2" );
    check( ut, "pascal", "Pa", "N/m^2" );
    check( ut, "joule", "J", "Pa*m^3" );
    check( ut, "watt", "W", "J/s" );
    check( ut, "coulomb", "C", "s*A" );
    check( ut, "volt", "V", "J/C" );
    check( ut, "farad", "F", "C/V" );
    check( ut, "ohm", ohm, "V/A" );
    check( ut, "siemens", "S", "ohm^-1" );
    check( ut, "weber", "Wb", "V*s" );
    check( ut, "tesla", "T", "Wb/m^2" );
    check( ut, "henry", "H", "Wb/A" );
    check( ut, "lumen", "lm", "cd*sr" );
    check( ut, "lux", "lx", "lm/m^2" );
    check( ut, "becquerel", "Bq", "s^-1" );
    check( ut, "gray", "Gy", "J/kg" );
    check( ut, "sievert", "Sv", "J/kg" );
    check( ut, "katal", "kat", "mol/s" );

    // Test English units
    check( ut, "inch", "in", "25.4 mm" );
    check( ut, "foot", "ft", "12 in" );
    check( ut, "yard", "yd", "3 ft" );
    check( ut, "furlong", "fur", "220 yd" );
    check( ut, "mile", "mi", "5280 ft" );
    check( ut, "acre", "4840 yd^2" );
    check( ut, "qt", "pt", 2 );
    check( ut, "gal", "pt", 8 );
    check( ut, "lb", "oz", 16 );
    check( ut, "teaspoon", "milliliters", 4.92892 );
    check( ut, "tablespoon", "milliliters", 14.7868 );
    check( ut, "cup", "milliliters", 236.588 );
    check( ut, "pint", "milliliters", 473.1764727459 );
    check( ut, "quart", "milliliters", 946.3529454918 );
    check( ut, "gallon", "milliliters", 3785.411781967 );
    check( ut, "qt", "pt", 2 );
    check( ut, "gal", "pt", 8 );
    check( ut, "lb", "oz", 16 );
    check( ut, "pt", "litre", 0.4731764727459 );
    check( ut, "pt", "0.4731764727459 litre" );
    check( ut, "oz", "g", 28.349523125 );
    check( ut, "ton", "lb", 2240 );

    // Test atomic units
    check( ut, "hartree", "J", 4.359744722207185e-18 );
    check( ut, "bohr", "m", 5.2917721090380e-11 );

    // Test other conversions
    check( ut, "%", " ", 0.01 );
    check( ut, "%", "percent" );
    check( ut, "m^0", " " );
    check( ut, "barye", "Ba", "0.1 Pa" );
    check( ut, "eV", "K", 11604.51996505152 );
    check( ut, "K", "eV", 8.617331893190124e-5 );
    check( ut, "minute", "s", 60 );
    check( ut, "hour", "hr", "60 minutes" );
    check( ut, "day", "hour", 24 );
    check( ut, "week", "day", 7 );
    check( ut, "mmHg", "Pa", 133.322387415 );
    check( ut, "ampere*second", "coulomb" );
    check( ut, "ergs/(s*cm^2)", "W/(m^2)", 1e-3 );
    check( ut, "	dyn	", "N", 1e-5 );
    check( ut, "(N/m^3)m(m^3/s)^2(1/m^5)", "kg/s^4" );
    check( ut, "(lbf/ft^3)ft(ft^3/s)^2(1/in^5)", "kg/s^4", 3.631430055671e6 );

    // Test reading with scalar
    std::string str = "3.14159";
    auto t1         = AMP::Units( str );
    auto t2         = AMP::Units( AMP::UnitType::unitless, 3.14159 );
    if ( t1 == t2 )
        ut.passes( "Read scalar" );
    else
        ut.failure( "Read scalar" );

    // Test combinations of units and prefixes
    testPrefixUnits( ut );

    // Print all supported units
    auto list = Units::getAllUnits();
    printf( "\nSupported Units:\n" );
    for ( size_t i = 0; i < list.size(); i++ ) {
        printf( " %11s", list[i].data() );
        if ( i % 10 == 9 )
            printf( "\n" );
    }
    printf( "\n" );

    // Return
    int N_errors = ut.NumFailGlobal();
    ut.report();
    ut.reset();
    AMP::AMPManager::shutdown();
    return N_errors;
}
