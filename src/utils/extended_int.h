#ifndef included_extended_int
#define included_extended_int

#include <array>
#include <limits>
#include <stdint.h>
#include <stdlib.h>
#include <string>


namespace AMP::extended {


/** \class int64N
 *
 * This class provides an arbitrary precision integer
 */
template<uint8_t N>
class alignas( 8 ) int64N
{
public:
    //! Empty constructor
    constexpr int64N();

    //! Copy constructor
    constexpr int64N( const int64N & ) = default;

    //! Assignment operator
    constexpr int64N<N> &operator=( const int64N<N> & ) = default;

    //! Move constructor
    constexpr int64N( int64N && ) = default;

    //! Move operator
    constexpr int64N<N> &operator=( int64N<N> && ) = default;

    //! Create from int
    explicit constexpr int64N( int rhs );

    //! Create from int64
    explicit constexpr int64N( int64_t );

    //! Create from int64N<N>
    template<uint8_t N2>
    explicit constexpr int64N( const int64N<N2> & );

    //! Create from string
    explicit constexpr int64N( const char * );

    //! Conversion to int
    constexpr operator int() const;

    //! Conversion to int64
    constexpr operator int64_t() const;

    //! Conversion to double
    constexpr operator double() const;

    //! Get the hexadecimal number as a char array
    constexpr std::array<char, 16 * N + 3> hex( bool fixedWidth = true ) const;

    //! Get the string value as a hexadecimal number
    inline operator std::string() const { return hex().data(); }

    //! Overload arimetic operators
    constexpr int64N operator!() const;
    constexpr int64N &operator+=( const int64N & );
    constexpr int64N &operator+=( const int64_t );
    constexpr int64N &operator+=( const uint64_t );
    constexpr int64N &operator-=( const int64N & );
    constexpr int64N &operator*=( const int64N & );
    constexpr int64N &operator/=( const int64N & );

    //! Overload comparison operators
    constexpr bool operator==( const int64N & ) const;
    constexpr bool operator!=( const int64N & ) const;
    constexpr bool operator>( const int64N & ) const;
    constexpr bool operator<( const int64N & ) const;
    constexpr bool operator>=( const int64N & ) const;
    constexpr bool operator<=( const int64N & ) const;
    constexpr bool operator==( int ) const;
    constexpr bool operator!=( int ) const;
    constexpr bool operator>( int ) const;
    constexpr bool operator<( int ) const;
    constexpr bool operator>=( int ) const;
    constexpr bool operator<=( int ) const;

    //! Get the sign of the number
    constexpr int sign() const;

    //! Convert to the 2's compliment of the number (equivalent to multiplying by -1)
    constexpr void compliment();

    //! Bitshift operators
    constexpr int64N &operator<<=( unsigned );
    constexpr int64N &operator>>=( unsigned );

protected:
    uint64_t data[N];

    template<uint8_t N2>
    friend class int64N;
};


// Define some types
typedef int64N<2> int128_t;
typedef int64N<4> int256_t;
typedef int64N<8> int512_t;
typedef int64N<16> int1024_t;
typedef int64N<32> int2048_t;


// Arithmetic operator overloading
template<uint8_t N>
constexpr int64N<N> operator+( const int64N<N> &x, const int64N<N> &y );
template<uint8_t N>
constexpr int64N<N> operator-( const int64N<N> &x, const int64N<N> &y );
template<uint8_t N>
constexpr int64N<N> operator*( const int64N<N> &x, const int64N<N> &y );
template<uint8_t N>
constexpr int64N<N> operator/( const int64N<N> &x, const int64N<N> &y );
template<uint8_t N>
constexpr int64N<N> operator-( const int64N<N> &x );
template<uint8_t N>
constexpr int64N<N> operator-( const int64_t &x, const int64N<N> &y );


// ostream
template<uint8_t N>
std::ostream &operator<<( std::ostream &out, const int64N<N> &x );


// bitshift operators
template<uint8_t N>
constexpr int64N<N> operator<<( const int64N<N> &, const int );
template<uint8_t N>
constexpr int64N<N> operator>>( const int64N<N> &, const int );
} // namespace AMP::extended


// numeric_limits
namespace std {
template<uint8_t N>
class numeric_limits<AMP::extended::int64N<N>>
{
public: // Member constants
    static constexpr bool is_specialized           = true;
    static constexpr bool is_signed                = true;
    static constexpr bool is_integer               = true;
    static constexpr bool is_exact                 = true;
    static constexpr bool has_infinity             = false;
    static constexpr bool has_quiet_NaN            = false;
    static constexpr bool has_signaling_NaN        = false;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss          = false;
    static constexpr float_round_style round_style = round_toward_zero;
    static constexpr bool is_iec559                = false;
    static constexpr bool is_bounded               = true;
    static constexpr bool is_modulo                = false;
    static constexpr int digits                    = 64 * N - 1;
    static constexpr int digits10                  = ( 64 * N - 1 ) * 0.301029995663981;
    static constexpr int max_digits10              = 0;
    static constexpr int radix                     = 2;
    static constexpr int min_exponent              = 0;
    static constexpr int min_exponent10            = 0;
    static constexpr int max_exponent              = 0;
    static constexpr int max_exponent10            = 0;
    static constexpr bool traps                    = true;
    static constexpr bool tinyness_before          = false;

public: // Member functions
    static constexpr AMP::extended::int64N<N> min()
    {
        return AMP::extended::int64N<N>( 1 ) << 64 * N - 1;
    }
    static constexpr AMP::extended::int64N<N> lowest() { return min(); }
    static constexpr AMP::extended::int64N<N> max() { return !min(); }
    static constexpr AMP::extended::int64N<N> epsilon() throw();
    static constexpr AMP::extended::int64N<N> round_error() throw();
    static constexpr AMP::extended::int64N<N> infinity() throw();
    static constexpr AMP::extended::int64N<N> quiet_NaN() throw();
    static constexpr AMP::extended::int64N<N> signaling_NaN() throw();
    static constexpr AMP::extended::int64N<N> denorm_min() throw();
};


} // namespace std


#include "AMP/utils/extended_int.hpp"

#endif
