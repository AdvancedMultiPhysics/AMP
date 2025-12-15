#include "AMP/vectors/Scalar.hpp"
#include "AMP/utils/AMP_MPI.h"

#include <cmath>
#include <math.h>


namespace AMP {


/********************************************************************
 * boolean operator overloading                                      *
 ********************************************************************/
bool Scalar::operator<( const Scalar &rhs ) const
{
    if ( !has_value() || !rhs.has_value() )
        AMP_ERROR( "Comparing empty scalar" );
    if ( is_complex() || rhs.is_complex() ) {
        AMP_ERROR( "No operator < for std::complex" );
    } else if ( is_floating_point() || rhs.is_floating_point() ) {
        return get<double>() < rhs.get<double>();
    } else if ( is_integral() && rhs.is_integral() ) {
        return get<int64_t>() < rhs.get<int64_t>();
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return false;
}
bool Scalar::operator==( const Scalar &rhs ) const
{
    try {
        if ( has_value() != rhs.has_value() )
            return false;
        if ( is_floating_point() ) {
            return get<double>() == rhs.get<double>();
        } else if ( is_integral() ) {
            return get<int64_t>() == rhs.get<int64_t>();
        } else if ( is_complex() ) {
            return get<std::complex<double>>() == rhs.get<std::complex<double>>();
        } else {
            AMP_ERROR( "Unable to get types for Scalar" );
        }
    } catch ( ... ) {
    }
    return false;
}
bool Scalar::operator>( const Scalar &rhs ) const { return rhs < *this; }
bool Scalar::operator<=( const Scalar &rhs ) const { return !( *this > rhs ); }
bool Scalar::operator>=( const Scalar &rhs ) const { return !( *this < rhs ); }
bool Scalar::operator!=( const Scalar &rhs ) const { return !( *this == rhs ); }


/********************************************************************
 * arithmetic operator overloading                                   *
 ********************************************************************/
Scalar operator-( const Scalar &x )
{
    if ( !x.has_value() )
        return x;
    if ( x.is_complex() ) {
        return ( -x.get<std::complex<double>>() );
    } else if ( x.is_floating_point() ) {
        return ( -x.get<double>() );
    } else if ( x.is_integral() ) {
        return ( -x.get<int64_t>() );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
Scalar operator+( const Scalar &x, const Scalar &y )
{
    if ( !x.has_value() )
        return y;
    if ( !y.has_value() )
        return x;
    if ( x.is_complex() || y.is_complex() ) {
        return ( x.get<std::complex<double>>() + y.get<std::complex<double>>() );
    } else if ( x.is_floating_point() || y.is_floating_point() ) {
        return ( x.get<double>() + y.get<double>() );
    } else if ( x.is_integral() && y.is_integral() ) {
        return ( x.get<int64_t>() + y.get<int64_t>() );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
Scalar operator-( const Scalar &x, const Scalar &y )
{
    if ( !x.has_value() )
        return -y;
    if ( !y.has_value() )
        return x;
    if ( x.is_complex() || y.is_complex() ) {
        return ( x.get<std::complex<double>>() - y.get<std::complex<double>>() );
    } else if ( x.is_floating_point() || y.is_floating_point() ) {
        return ( x.get<double>() - y.get<double>() );
    } else if ( x.is_integral() && y.is_integral() ) {
        return ( x.get<int64_t>() - y.get<int64_t>() );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
Scalar operator*( const Scalar &x, const Scalar &y )
{
    if ( !x.has_value() || !y.has_value() )
        return 0.0;
    if ( x.is_complex() || y.is_complex() ) {
        return ( x.get<std::complex<double>>() * y.get<std::complex<double>>() );
    } else if ( x.is_floating_point() || y.is_floating_point() ) {
        return ( x.get<double>() * y.get<double>() );
    } else if ( x.is_integral() && y.is_integral() ) {
        return ( x.get<int64_t>() * y.get<int64_t>() );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
Scalar operator/( const Scalar &x, const Scalar &y )
{
    if ( !x.has_value() || !y.has_value() )
        return 0.0;
    if ( x.is_complex() || y.is_complex() ) {
        return ( x.get<std::complex<double>>() / y.get<std::complex<double>>() );
    } else if ( x.is_floating_point() || y.is_floating_point() ) {
        return ( x.get<double>() / y.get<double>() );
    } else if ( x.is_integral() && y.is_integral() ) {
        return ( x.get<int64_t>() / y.get<int64_t>() );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}


/********************************************************************
 * Math functions                                                    *
 ********************************************************************/
Scalar Scalar::sqrt() const
{
    if ( !has_value() )
        return 0.0;
    if ( is_floating_point() ) {
        return ( std::sqrt( get<double>() ) );
    } else if ( is_integral() ) {
        return ( std::sqrt( get<int64_t>() ) );
    } else if ( is_complex() ) {
        return ( std::sqrt( get<std::complex<double>>() ) );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
Scalar Scalar::abs() const
{
    if ( !has_value() )
        return Scalar();
    if ( is_floating_point() ) {
        return ( std::abs( get<double>() ) );
    } else if ( is_integral() ) {
        return ( std::abs( get<int64_t>() ) );
    } else if ( is_complex() ) {
        return ( std::abs( get<std::complex<double>>() ) );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}


/********************************************************************
 * Reductions                                                        *
 ********************************************************************/
template<>
Scalar AMP_MPI::minReduce( const Scalar &x ) const
{
    if ( d_size <= 1 )
        return x;
    if ( x.is_floating_point() ) {
        return minReduce( x.get<double>() );
    } else if ( x.is_integral() ) {
        return minReduce( x.get<int64_t>() );
    } else if ( x.is_complex() ) {
        return minReduce( x.get<std::complex<double>>() );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
template<>
Scalar AMP_MPI::maxReduce( const Scalar &x ) const
{
    if ( d_size <= 1 )
        return x;
    if ( x.is_floating_point() ) {
        return maxReduce( x.get<double>() );
    } else if ( x.is_integral() ) {
        return maxReduce( x.get<int64_t>() );
    } else if ( x.is_complex() ) {
        return maxReduce( x.get<std::complex<double>>() );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
template<>
Scalar AMP_MPI::sumReduce( const Scalar &x ) const
{
    if ( d_size <= 1 )
        return x;
    if ( x.is_floating_point() ) {
        return sumReduce( x.get<double>() );
    } else if ( x.is_integral() ) {
        return sumReduce( x.get<int64_t>() );
    } else if ( x.is_complex() ) {
        return sumReduce( x.get<std::complex<double>>() );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}


/********************************************************
 *  Limits                                               *
 ********************************************************/
static constexpr auto hash_double  = AMP::getTypeID<double>().hash;
static constexpr auto hash_float   = AMP::getTypeID<float>().hash;
static constexpr auto hash_complex = AMP::getTypeID<std::complex<double>>().hash;
static constexpr auto hash_int32   = AMP::getTypeID<int32_t>().hash;
static constexpr auto hash_int64   = AMP::getTypeID<int64_t>().hash;
Scalar Scalar::limitsMax() const
{
    if ( d_hash == hash_double ) {
        return std::numeric_limits<double>::max();
    } else if ( d_hash == hash_float ) {
        return std::numeric_limits<float>::max();
    } else if ( d_hash == hash_complex ) {
        return std::numeric_limits<std::complex<double>>::max();
    } else if ( d_hash == hash_int32 ) {
        return std::numeric_limits<int32_t>::max();
    } else if ( d_hash == hash_int64 ) {
        return std::numeric_limits<int32_t>::max();
    } else {
        AMP_ERROR( "Unknown type for limitsMax()" );
    }
}
Scalar Scalar::limitsMin() const
{
    if ( d_hash == hash_double ) {
        return std::numeric_limits<double>::min();
    } else if ( d_hash == hash_float ) {
        return std::numeric_limits<float>::min();
    } else if ( d_hash == hash_complex ) {
        return std::numeric_limits<std::complex<double>>::min();
    } else if ( d_hash == hash_int32 ) {
        return std::numeric_limits<int32_t>::min();
    } else if ( d_hash == hash_int64 ) {
        return std::numeric_limits<int64_t>::min();
    } else {
        AMP_ERROR( "Unknown type for limitsMin()" );
    }
}
Scalar Scalar::zero() const { return create( 0 ); }


/********************************************************
 *  ostream operator                                     *
 ********************************************************/
void Scalar::print( std::ostream &out ) const
{
    if ( !has_value() )
        return;
    if ( is_floating_point() ) {
        out << get<double>();
    } else if ( is_integral() ) {
        out << get<int64_t>();
    } else if ( is_complex() ) {
        out << get<std::complex<double>>();
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
}


/********************************************************
 * Explicit instantiation                                *
 ********************************************************/
INSTANTIATE_SCALAR( char );
INSTANTIATE_SCALAR( int8_t );
INSTANTIATE_SCALAR( int16_t );
INSTANTIATE_SCALAR( int32_t );
INSTANTIATE_SCALAR( int64_t );
INSTANTIATE_SCALAR( uint8_t );
INSTANTIATE_SCALAR( uint16_t );
INSTANTIATE_SCALAR( uint32_t );
INSTANTIATE_SCALAR( uint64_t );
INSTANTIATE_SCALAR( float );
INSTANTIATE_SCALAR( double );
INSTANTIATE_SCALAR( long double );
INSTANTIATE_SCALAR( std::complex<float> );
INSTANTIATE_SCALAR( std::complex<double> );

template AMP::Scalar AMP::Scalar::create<AMP::Scalar>( const AMP::Scalar & ) const;
// INSTANTIATE_SCALAR( Scalar );


} // namespace AMP
