#include "AMP/utils/extended_int.h"

#include <math.h>


using eint64   = AMP::extended::int64N<1>;
using eint128  = AMP::extended::int128_t;
using eint256  = AMP::extended::int256_t;
using eint512  = AMP::extended::int512_t;
using eint1024 = AMP::extended::int1024_t;
using eint2048 = AMP::extended::int2048_t;


/************************************************************************
 *  Check the sizes of the different types                               *
 ************************************************************************/
static_assert( sizeof( eint64 ) == 8, " eint64 is not 8 bytes" );
static_assert( sizeof( eint128 ) == 16, " eint128 is not 16 bytes" );
static_assert( sizeof( eint256 ) == 32, " eint256 is not 32 bytes" );
static_assert( sizeof( eint512 ) == 64, " eint512 is not 64 bytes" );
static_assert( sizeof( eint1024 ) == 128, " eint1024 is not 128 bytes" );
static_assert( sizeof( eint2048 ) == 256, " eint2048 is not 256 bytes" );


/************************************************************************
 *  Test numeric limits                                                  *
 ************************************************************************/
template<class TYPE>
static constexpr bool test_numeric_limits()
{
    constexpr TYPE zero( 0 );
    constexpr size_t bytes    = sizeof( TYPE );
    constexpr size_t digits   = 8 * bytes - 1;
    constexpr size_t digits10 = digits * 0.301029995663981;
    static_assert( bytes % 8 == 0 );
    static_assert( std::numeric_limits<TYPE>::is_specialized );
    static_assert( std::numeric_limits<TYPE>::is_specialized );
    static_assert( std::numeric_limits<TYPE>::is_signed );
    static_assert( std::numeric_limits<TYPE>::is_integer );
    static_assert( std::numeric_limits<TYPE>::is_exact );
    static_assert( !std::numeric_limits<TYPE>::has_infinity );
    static_assert( !std::numeric_limits<TYPE>::has_quiet_NaN );
    static_assert( !std::numeric_limits<TYPE>::has_signaling_NaN );
    static_assert( std::numeric_limits<TYPE>::has_denorm == std::denorm_absent );
    static_assert( !std::numeric_limits<TYPE>::has_denorm_loss );
    static_assert( std::numeric_limits<TYPE>::round_style == std::round_toward_zero );
    static_assert( !std::numeric_limits<TYPE>::is_iec559 );
    static_assert( std::numeric_limits<TYPE>::is_bounded );
    static_assert( !std::numeric_limits<TYPE>::is_modulo );
    static_assert( std::numeric_limits<TYPE>::digits == digits );
    static_assert( std::numeric_limits<TYPE>::digits10 == digits10 );
    static_assert( std::numeric_limits<TYPE>::max_digits10 == 0 );
    static_assert( std::numeric_limits<TYPE>::radix == 2 );
    static_assert( std::numeric_limits<TYPE>::min_exponent == 0 );
    static_assert( std::numeric_limits<TYPE>::min_exponent10 == 0 );
    static_assert( std::numeric_limits<TYPE>::max_exponent == 0 );
    static_assert( std::numeric_limits<TYPE>::max_exponent10 == 0 );
    static_assert( !std::numeric_limits<TYPE>::tinyness_before );
    static_assert( std::numeric_limits<TYPE>::min() == std::numeric_limits<TYPE>::lowest() );
    static_assert( std::numeric_limits<TYPE>::max() > zero );
    static_assert( zero < std::numeric_limits<TYPE>::max() );
    static_assert( std::numeric_limits<TYPE>::min() < zero );
    static_assert( zero > std::numeric_limits<TYPE>::min() );
    static_assert( zero == std::numeric_limits<TYPE>::epsilon() );
    return true;
}
static_assert( test_numeric_limits<int64_t>() );
static_assert( test_numeric_limits<eint64>() );
static_assert( test_numeric_limits<eint128>() );
static_assert( test_numeric_limits<eint256>() );
static_assert( test_numeric_limits<eint512>() );
static_assert( test_numeric_limits<eint1024>() );
static_assert( std::numeric_limits<eint128>::traps );


/************************************************************************
 *  Run basic tests                                                      *
 ************************************************************************/
template<class TYPE>
static constexpr bool compare( const TYPE &x, double y )
{
    auto x2    = static_cast<double>( x );
    double err = ( x2 - y ) / y;
    if ( err < 0 )
        err = -err;
    return err <= 1e-14;
}
template<class TYPE>
static constexpr TYPE createShift( int i0, int shift )
{
    TYPE tmp( i0 );
    if ( shift >= 0 )
        tmp <<= shift;
    else
        tmp >>= -shift;
    return tmp;
}
template<class TYPE>
static constexpr bool run_basic_tests()
{
    constexpr TYPE a( 0 );
    constexpr TYPE b( 12 );
    constexpr TYPE c( -13 );
    constexpr TYPE d( b );

    // Check some very basic math
    static_assert( static_cast<int>( a ) == 0 );
    static_assert( static_cast<int>( b ) == 12 );
    static_assert( static_cast<int>( c ) == -13 );
    static_assert( static_cast<int>( d ) == 12 );
    static_assert( static_cast<int>( -b ) == -12 );
    static_assert( static_cast<int>( a + b ) == 12 );
    static_assert( static_cast<int>( b + b ) == 24 );
    static_assert( static_cast<int>( b + c ) == -1 );
    static_assert( static_cast<int>( c + c ) == -26 );
    static_assert( static_cast<int>( b - c ) == 25 );
    static_assert( static_cast<int>( c - b ) == -25 );
    // static_assert( compare( b, 12 ) );
    // static_assert( compare( c, -13 ) );

    // Test more complex math
    constexpr TYPE t1( 18330084947 );
    constexpr TYPE t2( 11658926934 );
    static_assert( static_cast<int64_t>( t1 + t2 ) == 29989011881 );
    static_assert( static_cast<int64_t>( -t1 - t2 ) == -29989011881 );

    // Check some shifts
    constexpr TYPE tmp1 = createShift<TYPE>( 1, 6 );
    constexpr TYPE tmp2 = createShift<TYPE>( 512, -6 );
    constexpr TYPE tmp3 = tmp1 >> 1;
    constexpr TYPE tmp4 = tmp1 << 1;
    constexpr TYPE tmp5 = tmp2 >> 12;
    static_assert( static_cast<int>( tmp1 ) == 64 );
    static_assert( static_cast<int>( tmp2 ) == 8 );
    static_assert( static_cast<int>( tmp3 ) == 32 );
    static_assert( static_cast<int>( tmp4 ) == 128 );
    static_assert( static_cast<int>( tmp5 ) == 0 );
    if constexpr ( std::numeric_limits<TYPE>::digits > 251 ) {
        constexpr TYPE tmp6 = TYPE( 1 ) << 250;
        constexpr TYPE tmp7 = tmp6 >> 245;
        static_assert( static_cast<int>( tmp7 ) == 32 );
        constexpr double ans = 1.809251394333066e+75;
        static_assert( compare( tmp6, ans ) );
    }
    return true;
}
static_assert( run_basic_tests<int64_t>() );
static_assert( run_basic_tests<eint64>() );
static_assert( run_basic_tests<eint128>() );
static_assert( run_basic_tests<eint256>() );
static_assert( run_basic_tests<eint512>() );
static_assert( run_basic_tests<eint1024>() );
static_assert( run_basic_tests<AMP::extended::int64N<3>>() );
static_assert( std::numeric_limits<eint256>::digits > 251 );


/************************************************************************
 *  Test geting hex values                                               *
 ************************************************************************/
template<class TYPE>
static constexpr bool compare( const TYPE &x, const char *ans )
{
    auto hex  = x.hex( false );
    bool test = true;
    for ( size_t i = 0; i < hex.size() && hex[i] != 0; i++ )
        test = test && hex[i] == ans[i];
    return test;
}
template<class TYPE>
static constexpr TYPE compliment( TYPE x )
{
    auto y = x;
    y.compliment();
    return y;
}
static constexpr bool testHex()
{
    // Test geting some hex values
    constexpr AMP::extended::int64N<3> tmp1( "0x10000000000000000" ); // 2^64
    constexpr auto tmp2 = -tmp1;
    constexpr auto tmp3 = compliment( tmp1 );
    constexpr auto tmp4 = AMP::extended::int64N<3>( 7 ) - tmp1;
    static_assert( compare( tmp1, "0x10000000000000000" ) );
    static_assert( compare( tmp2, "0xffffffffffffffffffffffffffffffff0000000000000000" ) );
    static_assert( compare( tmp3, "0xffffffffffffffffffffffffffffffff0000000000000000" ) );
    static_assert( compare( tmp4, "0xffffffffffffffffffffffffffffffff0000000000000007" ) );
    // Test a known problem
    constexpr eint128 a1( "0x9040ac62a509a917168" );
    constexpr eint128 b1( "0x152CDCE1C94F7300127" );
    constexpr eint256 c1 = eint256( a1 ) * eint256( b1 );
    static_assert( static_cast<double>( eint256( a1 ) ) == static_cast<double>( a1 ) );
    static_assert( compare( c1, "0xbee95b886e9fd2b0d774eecc702b2b919aed8" ) );
    constexpr eint512 a2( "0xddcc1e79e6412673fadda4008b7b4a378a7797b254f0459736b80451d79f055e" );
    constexpr eint512 b2( "0x8a7797b254f0459736b80451d79f055ef6cd3476c54b5b80dd4315d3d439efaa" );
    constexpr eint512 c2    = a2 * b2;
    constexpr eint512 c3    = a2 + b2;
    constexpr eint512 c4    = a2 - b2;
    constexpr char c2_ans[] = "0x77f7a5bdc847643d84facc96e32f7eba5421923c9d882163b6066197eb493e8d19"
                              "920ea8431093c2d61f10cabacd6625dcc16a79aa0d0a7230201bfaaf8a526c";
    constexpr char c3_ans[] = "0x16843b62c3b316c0b3195a852631a4f968144cc291a3ba11813fb1a25abd8f508";
    constexpr char c4_ans[] = "0x535486c79150e0dcc4259faeb3dc44d893aa633b8fa4ea165974ee7e036515b4";
    static_assert( compare( c2, c2_ans ) );
    static_assert( compare( c3, c3_ans ) );
    static_assert( compare( c4, c4_ans ) );
    return true;
}
static_assert( testHex() );


/************************************************************************
 *  Test repeated multiplication of -1e8                                 *
 ************************************************************************/
static constexpr bool testMult()
{
    // Test multiply by repeated multiplication of -1e18
    eint2048 x( -1000000000000000000 ), x2( 1 );
    double ans[] = { -1e18, 1e36,   -1e54, 1e72,   -1e90, 1e108,  -1e126, 1e144, -1e162,
                     1e180, -1e198, 1e216, -1e234, 1e252, -1e270, 1e288,  -1e306 };
    bool test    = true;
    for ( double an : ans ) {
        x2   = x2 * x;
        test = test && compare( x2, an );
    }
    // Test multiplication of large integers
    constexpr auto y1 = createShift<eint2048>( 1, 76 );
    constexpr auto y2 = createShift<eint2048>( 1, 152 );
    constexpr auto y3 = y1 * y1;
    test              = test && y2 == y3;
    return test;
}
static constexpr bool testInf()
{
    // Create an integer that cannot be represented by double
    eint2048 x1( 1 );
    x1 <<= 1341;
    auto x2 = -x1;
    return static_cast<double>( x1 ) == std::numeric_limits<double>::infinity() &&
           static_cast<double>( x2 ) == -std::numeric_limits<double>::infinity();
}
static_assert( testMult() );
static_assert( testInf() );

//#endif
