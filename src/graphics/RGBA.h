#ifndef included_AMP_RGBA
#define included_AMP_RGBA

#include <cstdint>


namespace AMP::Graphics {


class ARGB32;


//! Helper function to convert 4 uint8_t to uint32_t
constexpr uint32_t store( uint8_t a, uint8_t b, uint8_t c, uint8_t d )
{
    return ( (uint32_t) a ) << 24 | ( (uint32_t) b ) << 16 | ( (uint32_t) c ) << 8 | d;
}


//! Structure to store RGBA data
class RGBA32 final
{
public:
    constexpr RGBA32() : data( 0xFF ){};
    constexpr explicit RGBA32( uint8_t x ) : data( store( x, x, x, 255 ) ){};
    constexpr explicit RGBA32( uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255 )
        : data( store( r, g, b, a ) )
    {
    }
    constexpr explicit RGBA32( uint32_t x ) : data( x ) {}
    constexpr explicit RGBA32( ARGB32 x );
    constexpr uint8_t red() const { return ( data & 0xFF000000 ) >> 24; }
    constexpr uint8_t green() const { return ( data & 0xFF0000 ) >> 16; }
    constexpr uint8_t blue() const { return ( data & 0xFF00 ) >> 8; }
    constexpr uint8_t alpha() const { return data & 0xFF; }
    constexpr operator uint32_t() { return data; }
    constexpr operator uint8_t() { return red() * 0.299 + green() * 0.587 + blue() * 0.114; }
    constexpr operator uint16_t() { return red() * 76.843 + green() * 150.859 + blue() * 29.298; }
    constexpr operator float() { return operator double(); }
    constexpr operator double()
    {
        return red() * 0.001172549019608 + green() * 0.002301960784314 + blue() * 0.000447058823529;
    }
    constexpr bool operator==( const RGBA32 &rhs ) const { return data == rhs.data; }

private:
    uint32_t data; // Store data as 32-bit unsigned integer
};


//! Structure to store RGBA data
class ARGB32 final
{
public:
    constexpr ARGB32() : data( 0xFF000000 ){};
    constexpr explicit ARGB32( uint8_t x ) : data( store( 255, x, x, x ) ){};
    constexpr explicit ARGB32( uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255 )
        : data( store( a, r, g, b ) )
    {
    }
    constexpr explicit ARGB32( uint32_t x ) : data( x ) {}
    constexpr explicit ARGB32( RGBA32 x );
    constexpr uint8_t alpha() const { return ( data & 0xFF000000 ) >> 24; }
    constexpr uint8_t red() const { return ( data & 0xFF0000 ) >> 16; }
    constexpr uint8_t green() const { return ( data & 0xFF00 ) >> 8; }
    constexpr uint8_t blue() const { return data & 0xFF; }
    constexpr operator uint32_t() { return data; }
    constexpr operator uint8_t() { return red() * 0.299 + green() * 0.587 + blue() * 0.114; }
    constexpr operator uint16_t() { return red() * 76.843 + green() * 150.859 + blue() * 29.298; }
    constexpr operator float() { return operator double(); }
    constexpr operator double()
    {
        return red() * 0.001172549019608 + green() * 0.002301960784314 + blue() * 0.000447058823529;
    }
    constexpr bool operator==( const ARGB32 &rhs ) const { return data == rhs.data; }

private:
    uint32_t data;
};


//! Structure to store RGB data
using RGB = ARGB32;


//! Conversions
constexpr RGBA32::RGBA32( ARGB32 x ) : data( store( x.red(), x.green(), x.blue(), x.alpha() ) ) {}
constexpr ARGB32::ARGB32( RGBA32 x ) : data( store( x.alpha(), x.red(), x.green(), x.blue() ) ) {}


} // namespace AMP::Graphics

#endif
