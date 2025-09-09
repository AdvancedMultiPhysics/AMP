#include "AMP/graphics/RGBA.h"
#include "AMP/utils/Array.hpp"


namespace AMP {


/********************************************************
 *  RGBA32 / ARGB32                                      *
 ********************************************************/
static_assert( sizeof( ARGB32 ) == 4 );
static_assert( sizeof( RGBA32 ) == 4 );
static constexpr bool runTests()
{
    constexpr ARGB32 argb( (uint32_t) 0x01020304 );
    constexpr RGBA32 rgba( (uint32_t) 0x02030401 );
    constexpr ARGB32 argb2 = argb;
    static_assert( argb.red() == 2 );
    static_assert( argb.green() == 3 );
    static_assert( argb.blue() == 4 );
    static_assert( argb.alpha() == 1 );
    static_assert( argb.red() == rgba.red() );
    static_assert( argb.green() == rgba.green() );
    static_assert( argb.blue() == rgba.blue() );
    static_assert( argb.alpha() == rgba.alpha() );
    static_assert( argb2 == argb );
    return true;
}
static_assert( runTests() );


} // namespace AMP


/********************************************************
 *  Explicit instantiations of Array<RGB>                *
 ********************************************************/
// clang-format off
instantiateArrayConstructors( AMP::RGBA32 );
instantiateArrayConstructors( AMP::ARGB32 );
template AMP::Array<AMP::RGBA32> AMP::Array<AMP::RGBA32>::repmat( const AMP::ArraySize & ) const;
template AMP::Array<AMP::ARGB32> AMP::Array<AMP::ARGB32>::repmat( const AMP::ArraySize & ) const;
template AMP::Array<AMP::RGBA32> AMP::Array<AMP::RGBA32>::subset( const std::vector<size_t> & ) const;
template AMP::Array<AMP::ARGB32> AMP::Array<AMP::ARGB32>::subset( const std::vector<size_t> & ) const;
template void AMP::Array<AMP::RGBA32>::copySubset( const std::vector<size_t> &, const AMP::Array<AMP::RGBA32> & );
template void AMP::Array<AMP::ARGB32>::copySubset( const std::vector<size_t> &, const AMP::Array<AMP::ARGB32> & );
// clang-format on
