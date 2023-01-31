#include "AMP/vectors/Scalar.h"

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
 * Special functions                                                 *
 ********************************************************************/
Scalar minReduce( const AMP::AMP_MPI &comm, const Scalar &x )
{
    if ( comm.getSize() <= 1 )
        return x;
    if ( x.is_floating_point() ) {
        return ( comm.minReduce( x.get<double>() ) );
    } else if ( x.is_integral() ) {
        return ( comm.minReduce( x.get<int64_t>() ) );
    } else if ( x.is_complex() ) {
        return ( comm.minReduce( x.get<std::complex<double>>() ) );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
Scalar maxReduce( const AMP::AMP_MPI &comm, const Scalar &x )
{
    if ( comm.getSize() <= 1 )
        return x;
    if ( x.is_floating_point() ) {
        return ( comm.maxReduce( x.get<double>() ) );
    } else if ( x.is_integral() ) {
        return ( comm.maxReduce( x.get<int64_t>() ) );
    } else if ( x.is_complex() ) {
        return ( comm.maxReduce( x.get<std::complex<double>>() ) );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
Scalar sumReduce( const AMP::AMP_MPI &comm, const Scalar &x )
{
    if ( comm.getSize() <= 1 )
        return x;
    if ( x.is_floating_point() ) {
        return ( comm.sumReduce( x.get<double>() ) );
    } else if ( x.is_integral() ) {
        return ( comm.sumReduce( x.get<int64_t>() ) );
    } else if ( x.is_complex() ) {
        return ( comm.sumReduce( x.get<std::complex<double>>() ) );
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return Scalar();
}
Scalar Scalar::sqrt() const
{
    if ( !has_value() )
        return 0.0;
    if ( is_floating_point() ) {
        return ( ::sqrt( get<double>() ) );
    } else if ( is_integral() ) {
        return ( ::sqrt( get<int64_t>() ) );
    } else if ( is_complex() ) {
        return ( ::sqrt( get<std::complex<double>>() ) );
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

/********************************************************
 *  ostream operator                                     *
 ********************************************************/
template<>
std::enable_if_t<std::is_same_v<AMP::Scalar, AMP::Scalar>, std::ostream &>
operator<<<AMP::Scalar>( std::ostream &out, const AMP::Scalar &x )
{
    if ( !x.has_value() )
        return out;
    if ( x.is_floating_point() ) {
        out << x.get<double>();
    } else if ( x.is_integral() ) {
        out << x.get<int64_t>();
    } else if ( x.is_complex() ) {
        out << x.get<std::complex<double>>();
    } else {
        AMP_ERROR( "Unable to get types for Scalar" );
    }
    return out;
}

} // namespace AMP
