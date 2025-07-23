#include "AMP/utils/Units.h"
#include "AMP/utils/Constants.h"
#include "AMP/utils/Utilities.h"


namespace AMP {


/********************************************************************
 * Write a string for the units                                      *
 ********************************************************************/
std::string Units::str() const
{
    if ( isNull() ) {
        return {};
    } else if ( d_unit[0] != 0 ) {
        return printUnit();
    } else {
        return printSI();
    }
}
std::string Units::printUnit() const { return std::string( d_unit.data(), 0, d_unit.size() ); }
std::string Units::printSIBase() const
{
    constexpr const char *d_SI_units[] = { "s", "m", "kg", "A", "K", "mol", "cd", "rad", "sr" };
    std::string s;
    for ( size_t i = 0; i < d_SI.size(); i++ ) {
        if ( d_SI[i] != 0 ) {
            s += d_SI_units[i];
            if ( d_SI[i] != 1 )
                s += "^" + std::to_string( d_SI[i] );
            s += " ";
        }
    }
    if ( !s.empty() )
        s.resize( s.size() - 1 );
    return s;
}
std::string Units::printSI() const
{
    auto str = printSIBase();
    if ( d_scale != 1.0 )
        str = Utilities::stringf( "%0.12e ", d_scale ) + str;
    if ( str.empty() )
        str = " ";
    return str;
}
std::string Units::printFull() const
{
    return std::string( d_unit.data(), 0, d_unit.size() ) + " --- " + printSI();
}


/********************************************************************
 * Run some compile-time tests                                       *
 ********************************************************************/
static constexpr bool approx_equal( double a, double b )
{
    double e = a >= b ? a - b : b - a;
    return e < 1e-8 * b;
}
static_assert( sizeof( Units ) == 48 );
static_assert( std::is_final_v<Units> );
static_assert( std::is_trivially_copyable_v<Units> );
static_assert( !std::is_arithmetic_v<Units> );
static_assert( Units().isNull() );
static_assert( Units( "" ).isNull() );
static_assert( atoi( " 2303785 " ) == 2303785 );
static_assert( atoi( " +2303785 " ) == 2303785 );
static_assert( atoi( " -2303785 " ) == -2303785 );
static_assert( strtod( " 2303785 " ) == 2303785 );
static_assert( strtod( "2303785.42" ) - 2303785.42 < 1e-12 );
static_assert( strtod( " -2303785.42E-4 " ) + 230.378542 < 1e-12 );
static_assert( Units::convert( Units::getUnitPrefix( "q" ) ) == 1e-30 );
static_assert( Units::convert( Units::getUnitPrefix( "r" ) ) == 1e-27 );
static_assert( Units::convert( Units::getUnitPrefix( "y" ) ) == 1e-24 );
static_assert( Units::convert( Units::getUnitPrefix( "z" ) ) == 1e-21 );
static_assert( Units::convert( Units::getUnitPrefix( "a" ) ) == 1e-18 );
static_assert( Units::convert( Units::getUnitPrefix( "f" ) ) == 1e-15 );
static_assert( Units::convert( Units::getUnitPrefix( "p" ) ) == 1e-12 );
static_assert( Units::convert( Units::getUnitPrefix( "n" ) ) == 1e-9 );
static_assert( Units::convert( Units::getUnitPrefix( "u" ) ) == 1e-6 );
static_assert( Units::convert( Units::getUnitPrefix( "m" ) ) == 1e-3 );
static_assert( Units::convert( Units::getUnitPrefix( "c" ) ) == 1e-2 );
static_assert( Units::convert( Units::getUnitPrefix( "d" ) ) == 0.1 );
static_assert( Units::convert( Units::getUnitPrefix( "" ) ) == 1 );
static_assert( Units::convert( Units::getUnitPrefix( "da" ) ) == 10 );
static_assert( Units::convert( Units::getUnitPrefix( "h" ) ) == 100 );
static_assert( Units::convert( Units::getUnitPrefix( "k" ) ) == 1e3 );
static_assert( Units::convert( Units::getUnitPrefix( "M" ) ) == 1e6 );
static_assert( Units::convert( Units::getUnitPrefix( "G" ) ) == 1e9 );
static_assert( Units::convert( Units::getUnitPrefix( "T" ) ) == 1e12 );
static_assert( Units::convert( Units::getUnitPrefix( "P" ) ) == 1e15 );
static_assert( Units::convert( Units::getUnitPrefix( "E" ) ) == 1e18 );
static_assert( Units::convert( Units::getUnitPrefix( "Z" ) ) == 1e21 );
static_assert( Units::convert( Units::getUnitPrefix( "Y" ) ) == 1e24 );
static_assert( Units::convert( Units::getUnitPrefix( "R" ) ) == 1e27 );
static_assert( Units::convert( Units::getUnitPrefix( "Q" ) ) == 1e30 );
static_assert( Units( "meter" ).getType() == UnitType::length );
static_assert( Units( "gram" ).getType() == UnitType::mass );
static_assert( Units( "second" ).getType() == UnitType::time );
static_assert( Units( "ampere" ).getType() == UnitType::current );
static_assert( Units( "kelvin" ).getType() == UnitType::temperature );
static_assert( Units( "joule" ).getType() == UnitType::energy );
static_assert( Units( "erg" ).getType() == UnitType::energy );
static_assert( Units( "watt" ).getType() == UnitType::power );
static_assert( Units( "mole" ).getType() == UnitType::mole );
static_assert( Units( "candela" ).getType() == UnitType::intensity );
static_assert( Units( "degree" ).getType() == UnitType::angle );
static_assert( Units( "radian" ).getType() == UnitType::angle );
static_assert( Units( "V" ) * Units( "A" ) == Units( "W" ) );
static_assert( Units( "W/m^2" ) == Units( "uW/mm^2" ) );
static_assert( approx_equal( Units( "J" ).convert( Units( "ergs" ) ), 1e7 ) );
static_assert( approx_equal( Units( "eV" ).convert( Units( "K" ) ), 11604.51996505152 ) );
static_assert( approx_equal( Units( "qt" ).convert( Units( "pt" ) ), 2 ) );
static_assert( approx_equal( Units( "gal" ).convert( Units( "pt" ) ), 8 ) );
static_assert( approx_equal( Units( "lb" ).convert( Units( "oz" ) ), 16 ) );
static_assert( approx_equal( Units( "ergs/(s*cm^2)" ).convert( Units( "W/(m^2)" ) ), 1e-3 ) );
static_assert( approx_equal( Units( "pt" ).convert( Units( "litre" ) ), 0.4731764727459 ) );
static_assert( approx_equal( Units( "oz" ).convert( Units( "g" ) ), 28.349523125 ) );
static_assert( approx_equal( Units( "ton" ).convert( Units( "lb" ) ), 2240 ) );
constexpr char microOhm[] = { (char) 206, (char) 188, (char) 206, (char) 169, (char) 0 }; // UTF-16
static_assert( approx_equal( Units( "ohm" ).convert( Units( microOhm ) ), 1e6 ) );
} // namespace AMP


/************************************************************************
 * read/write HDF5                                                      *
 ***********************************************************************/
#include "AMP/IO/HDF.hpp"
#include "AMP/utils/Array.hpp"
#ifdef AMP_USE_HDF5
template<>
hid_t AMP::IO::getHDF5datatype<AMP::Units>()
{
    AMP_ERROR( "Not finished" );
    return 0;
}
template<>
void AMP::IO::writeHDF5Array<AMP::Units>( hid_t,
                                          const std::string &,
                                          const AMP::Array<AMP::Units> & )
{
    AMP_ERROR( "Not finished" );
}
template<>
void AMP::IO::readHDF5Array<AMP::Units>( hid_t, const std::string &, AMP::Array<AMP::Units> & )
{
    AMP_ERROR( "Not finished" );
}
template<>
void AMP::IO::writeHDF5Scalar<AMP::Units>( hid_t fid,
                                           const std::string &name,
                                           const AMP::Units &data )
{
    AMP::IO::writeHDF5( fid, name, sizeof( data ), &data );
}
template<>
void AMP::IO::readHDF5Scalar<AMP::Units>( hid_t fid, const std::string &name, AMP::Units &data )
{
    AMP::IO::readHDF5( fid, name, sizeof( data ), &data );
}
#endif
INSTANTIATE_HDF5( AMP::Units );
instantiateArrayConstructors( AMP::Units );
