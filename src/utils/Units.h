#ifndef included_AMP_Units
#define included_AMP_Units

#include <array>
#include <math.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>


namespace AMP {

//! Enum to hold prefix
enum class UnitPrefix : int8_t {
    quecto  = 0,
    ronto   = 1,
    yocto   = 2,
    zepto   = 3,
    atto    = 4,
    femto   = 5,
    pico    = 6,
    nano    = 7,
    micro   = 8,
    milli   = 9,
    centi   = 10,
    deci    = 11,
    none    = 12,
    deca    = 13,
    hecto   = 14,
    kilo    = 15,
    mega    = 16,
    giga    = 17,
    tera    = 18,
    peta    = 19,
    exa     = 20,
    zetta   = 21,
    yotta   = 22,
    ronna   = 23,
    quetta  = 24,
    unknown = -1
};


//! Enum to hold type
enum class UnitType : int8_t {
    unitless              = 0,
    time                  = 1,
    length                = 2,
    mass                  = 3,
    current               = 4,
    temperature           = 5,
    mole                  = 6,
    intensity             = 7,
    angle                 = 8,
    solidAngle            = 9,
    energy                = 10,
    power                 = 11,
    frequency             = 12,
    force                 = 13,
    pressure              = 14,
    electricCharge        = 15,
    electricalPotential   = 16,
    capacitance           = 17,
    resistance            = 18,
    electricalConductance = 19,
    magneticFlux          = 20,
    magneticFluxDensity   = 21,
    inductance            = 22,
    luminousFlux          = 23,
    illuminance           = 24,
    unknown               = -1
};


/**
 * \class Units
 * \brief  Provides a class for storing units
 * \details
 * \verbatim
 This class provides routines for creating and storing units.
 A user can specify a unit and get the scaling factor to a compatible unit.
 All user-provided units are converted to SI base units with a scaling factor.
 Currently we support all major SI units and some additional units with prefixes.
 Note: currently the majority of the routines are constexpr so that the creation
    of a unit can be done at compile-time.
 Note: To prevent ambiguity some unit abbreviations are not supported:
    min (minute) - ambiguous with min (milli-inch)
 Note: To prevent ambiguity some units are not supported:
    year: could be 365 days, 365.25 days, 365.2425 days
    horsepower:
        Mechanical horsepower - 745.6998715822702 W
        Metric horsepower - 735.49875 W
        Electric horsepower - 746 W
        Boiler horsepower - 9,812.5 W
        Hydraulic horsepower - 745.69987 W
        Air horsepower - 745.69987 W
 \endverbatim
 */
class alignas( 8 ) Units final
{
public:
    using unit_type = std::array<char, 31>;
    using SI_type   = std::array<int8_t, 9>;

public:
    //! Empty constructor
    constexpr Units();

    /**
     * \brief  Construct the units from a const char array
     * \details  This is the default constructor for a string view.
     *    It can create a unit from a string of the format "W/(m^2)"
     * \param unit      Input string
     */
    constexpr Units( const char *unit );

    /**
     * \brief  Construct the units from a const char array
     * \details  This is the default constructor for a string view.
     *    It can create a unit from a string of the format "W/(m^2)"
     * \param unit      Input string
     */
    constexpr Units( const std::string_view &unit );

    /**
     * \brief  Construct the units from a const char array
     * \details  This is the default constructor for a string view.
     *    It can create a unit from a string of the format "W/(m^2)"
     * \param unit      Input string
     * \param value     Value of constant scalar
     */
    constexpr Units( const std::string_view &unit, double value );

    /**
     * \brief  Construct the units
     * \details  Construct units directly by specifying the SI units and scale
     * \param unit      Input string
     * \param scale     Value of constant scalar
     */
    explicit constexpr Units( const SI_type &unit, double scale = 1.0 );

    /**
     * \brief  Construct the units
     * \details  Construct units directly by specifying the SI units and scale
     * \param unit      Input string
     * \param scale     Value of constant scalar
     */
    explicit constexpr Units( const UnitType &unit, double scale = 1.0 );

    /**
     * \brief  Get the unit type
     * \details  This returns the type of unit (if it is reducible).
     *   e.g. time, length, energy, power, etc.
     */
    constexpr UnitType getType() const noexcept;

    /**
     * \brief  Get the prefix from a string
     * \details  This returns an enum representing the given input string
     * \param str       Input string
     */
    constexpr static UnitPrefix getUnitPrefix( const std::string_view &str ) noexcept;

    /**
     * \brief  Check if two units are compatible
     * \details  Check if two units are compatible with each other.
     *     I.e. if convert will succeed.
     * \param unit      Unit to compare
     */
    constexpr bool compatible( const Units &unit ) noexcept;

    /**
     * \brief  Convert the unit to a new type
     * \details  This function returns the scaling factor to convert between
     *    two different units.  For example, converting "J" to "ergs" returns 1e7.
     *    If the two units are not compatible an exception will be thrown.
     * \param unit      Desired unit
     */
    constexpr double convert( const Units &unit ) const;

    /**
     * \brief  Convert the prefix to a double
     * \details  This returns the value of the given prefix
     * \param x         Enum for the prefix
     */
    constexpr static double convert( UnitPrefix x ) noexcept;

    //! Operator ==
    constexpr bool operator==( const Units &rhs ) const noexcept;

    //! Operator !=
    constexpr bool operator!=( const Units &rhs ) const noexcept { return !operator==( rhs ); }

    //! Operator *=
    constexpr void operator*=( const Units &rhs ) noexcept;

    //! Operator *=
    constexpr void operator*=( double x ) noexcept { d_scale *= x; }

    //! Operator /=
    constexpr void operator/=( const Units &rhs ) noexcept;

    //! Raise the unit to the given power
    constexpr Units pow( int p ) const noexcept;

    //! Check if unit is null
    constexpr bool isNull() const { return d_scale == 0; }


public:
    //! Get a string representation of the units
    std::string str() const;

    //! Get a string representation for the prefix
    static constexpr std::string_view getPrefixStr( UnitPrefix ) noexcept;

    //! Get a string representation of the units in SI units (with scaling factor)
    std::string printSI() const;

    //! Get a string representation of the units in SI units (with scaling factor)
    std::string printUnit() const;

    //! Get the full unit and conversion string
    std::string printFull() const;


public:
    //! Get all supported unit prefixes
    static inline std::vector<std::string> getAllPrefixes();

    //! Get all supported units
    static inline std::vector<std::string> getAllUnits();

    // Convert string to int
    static constexpr int atoi( std::string_view, bool = true );

    // Convert string to double
    static constexpr double strtod( std::string_view );


protected:
    unit_type d_unit;
    SI_type d_SI;
    double d_scale;

protected:
    std::string printSIBase() const;
    static constexpr Units read( std::string_view str );
    static constexpr Units read2( std::string_view str );
    static constexpr Units readUnit( const std::string_view &str, bool throwErr = true );
    static constexpr SI_type combine( const SI_type &a, const SI_type &b );
    static constexpr SI_type getSI( UnitType );
    static constexpr std::pair<size_t, char> findToken( const std::string_view &, size_t );
    static constexpr size_t findPar( const std::string_view &, size_t );

protected:
    friend constexpr Units pow( Units base, int exponent );
};


inline std::ostream &operator<<( std::ostream &os, const Units &unit )
{
    os << unit.str();
    return os;
}


} // namespace AMP


#include "AMP/utils/Units.hpp"


#endif
