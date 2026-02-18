#include "AMP/materials/ScalarProperty.h"
#include "AMP/utils/MathExpr.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Materials {


/*******************************************************************
 *  StringProperty                                                  *
 *******************************************************************/
StringProperty::StringProperty( std::string_view name, std::string value, std::string_view source )
    : Property( name, { 1 }, {}, source ), d_value( std::move( value ) )
{
}
void StringProperty::eval( AMP::Array<double> &, const AMP::Array<double> & ) const
{
    AMP_ERROR( "numerically evaluating StringProperty is not supported" );
}


/*******************************************************************
 *  ScalarProperty                                                  *
 *******************************************************************/
ScalarProperty::ScalarProperty( std::string_view name,
                                double value,
                                const AMP::Units &unit,
                                std::string_view source )
    : Property( std::move( name ), { 1 }, unit, std::move( source ) )
{
    d_value.resize( 1 );
    d_value( 0 ) = value;
}
ScalarProperty::ScalarProperty( std::string_view name,
                                AMP::Array<double> value,
                                const AMP::Units &unit,
                                std::string_view source )
    : Property( std::move( name ), value.size(), unit, std::move( source ) ),
      d_value( std::move( value ) )
{
}
void ScalarProperty::eval( AMP::Array<double> &result, const AMP::Array<double> & ) const
{
    size_t N1 = d_value.length();
    size_t N2 = result.size( d_value.ndim() );
    AMP_ASSERT( N1 * N2 == result.length() );
    for ( size_t i = 0; i < N2; i++ )
        memcpy( &result( 0, i ), d_value.data(), N1 * sizeof( double ) );
}


/*******************************************************************
 *  PolynomialProperty                                              *
 *******************************************************************/
PolynomialProperty::PolynomialProperty( std::string_view name,
                                        std::string_view source,
                                        const AMP::Units &unit,
                                        std::vector<double> params,
                                        std::vector<std::string> args,
                                        std::vector<std::array<double, 2>> ranges,
                                        std::vector<AMP::Units> argUnits )
    : Property( std::move( name ),
                { 1 },
                unit,
                std::move( source ),
                std::move( args ),
                std::move( ranges ),
                std::move( argUnits ) ),
      d_p( std::move( params ) )
{
    if ( d_p.size() > 1 )
        AMP_ASSERT( d_arguments.size() == 1 );
    else
        AMP_ASSERT( d_arguments.empty() );
}
void PolynomialProperty::eval( AMP::Array<double> &result, const AMP::Array<double> &args ) const
{
    if ( d_p.size() == 1 )
        return result.fill( d_p[0] );
    for ( size_t i = 0; i < result.length(); i++ ) {
        double x  = args( i );
        double y  = 0;
        double x2 = 1.0;
        for ( size_t j = 0; j < d_p.size(); j++ ) {
            y += d_p[j] * x2;
            x2 *= x;
        }
        result( i ) = y;
    }
}


/*******************************************************************
 *  ScalarPropInterpolatedPropertyerty                              *
 *******************************************************************/
InterpolatedProperty::InterpolatedProperty( std::string_view name,
                                            const AMP::Units &unit,
                                            const std::string &var_name,
                                            std::vector<double> x,
                                            std::vector<double> y,
                                            const std::array<double, 2> range,
                                            const AMP::Units &argUnit,
                                            double default_value,
                                            std::string_view source,
                                            std::string_view method )
    : Property( std::move( name ),
                { 1 },
                unit,
                std::move( source ),
                { var_name },
                { range },
                { argUnit } ),
      d_x( std::move( x ) ),
      d_y( std::move( y ) ),
      d_method( 0 )
{
    set_defaults( { default_value } );
    AMP_ASSERT( d_x.size() == d_y.size() );
    AMP::Utilities::quicksort( d_x, d_y );
    for ( size_t i = 1; i < d_x.size(); i++ )
        AMP_INSIST( d_x[i] != d_x[i - 1], "Values of interpolant must be unique" );
    if ( method == "linear" )
        d_method = 1;
    else if ( method == "cubic" )
        d_method = 3;
    else
        AMP_ERROR( "Unknown interpolation method" );
}
static double pchip( size_t N, const double *xi, const double *yi, double x )
{
    if ( x <= xi[0] || N <= 2 ) {
        double dx = ( x - xi[0] ) / ( xi[1] - xi[0] );
        return ( 1.0 - dx ) * yi[0] + dx * yi[1];
    } else if ( x >= xi[N - 1] ) {
        double dx = ( x - xi[N - 2] ) / ( xi[N - 1] - xi[N - 2] );
        return ( 1.0 - dx ) * yi[N - 2] + dx * yi[N - 1];
    }
    size_t i  = AMP::Utilities::findfirst( N, xi, x );
    double f1 = yi[i - 1];
    double f2 = yi[i];
    double dx = ( x - xi[i - 1] ) / ( xi[i] - xi[i - 1] );
    // Compute the gradient in normalized coordinates [0,1]
    double g1 = 0, g2 = 0;
    if ( i <= 1 ) {
        g1 = f2 - f1;
    } else if ( ( f1 < f2 && f1 > yi[i - 2] ) || ( f1 > f2 && f1 < yi[i - 2] ) ) {
        // Compute the gradient by using a 3-point finite difference to f'(x)
        // Note: the real gradient is g1/(xi[i]-xi[i-1])
        double f0    = yi[i - 2];
        double dx1   = xi[i - 1] - xi[i - 2];
        double dx2   = xi[i] - xi[i - 1];
        double a1    = ( dx2 - dx1 ) / dx1;
        double a2    = dx1 / ( dx1 + dx2 );
        g1           = a1 * ( f1 - f0 ) + a2 * ( f2 - f0 );
        double g_max = 2 * dx2 * std::min( fabs( f1 - f0 ) / dx1, fabs( f2 - f1 ) / dx2 );
        g1           = ( ( g1 >= 0 ) ? 1 : -1 ) * std::min( fabs( g1 ), g_max );
    }
    if ( i >= N - 1 ) {
        g2 = f2 - f1;
    } else if ( ( f2 < f1 && f2 > yi[i + 1] ) || ( f2 > f1 && f2 < yi[i + 1] ) ) {
        // Compute the gradient by using a 3-point finite difference to f'(x)
        // Note: the real gradient is g2/(xi[i]-xi[i-1])
        double f0    = yi[i + 1];
        double dx1   = xi[i] - xi[i - 1];
        double dx2   = xi[i + 1] - xi[i];
        double a1    = -dx2 / ( dx1 + dx2 );
        double a2    = ( dx2 - dx1 ) / dx2;
        g2           = a1 * ( f1 - f0 ) + a2 * ( f2 - f0 );
        double g_max = 2 * dx1 * std::min( fabs( f2 - f1 ) / dx1, fabs( f0 - f2 ) / dx2 );
        g2           = ( ( g2 >= 0 ) ? 1 : -1 ) * std::min( fabs( g2 ), g_max );
    }
    // Perform the interpolation
    double dx2 = dx * dx;
    return f1 + dx2 * ( 2 * dx - 3 ) * ( f1 - f2 ) + dx * g1 -
           dx2 * ( g1 + ( 1 - dx ) * ( g1 + g2 ) );
}
void InterpolatedProperty::eval( AMP::Array<double> &result, const AMP::Array<double> &args ) const
{
    if ( d_method == 1 ) {
        for ( size_t i = 0; i < result.length(); i++ )
            result( i ) = AMP::Utilities::linear( d_x, d_y, args( i ) );
    } else if ( d_method == 3 ) {
        for ( size_t i = 0; i < result.length(); i++ )
            result( i ) = pchip( d_x.size(), d_x.data(), d_y.data(), args( i ) );
    }
}


/*******************************************************************
 *  EquationProperty                                                *
 *******************************************************************/
std::vector<std::array<double, 2>> getDefaultRanges( std::vector<std::array<double, 2>> ranges,
                                                     const std::vector<std::string> &vars )
{
    if ( ranges.empty() )
        ranges.resize( vars.size(), { { -1e100, 1e100 } } );
    return ranges;
}
EquationProperty::EquationProperty( std::string_view name,
                                    std::shared_ptr<const MathExpr> eq,
                                    const AMP::Units &unit,
                                    std::vector<std::array<double, 2>> ranges,
                                    std::vector<AMP::Units> argUnits,
                                    std::string_view source )
    : Property( std::move( name ),
                { 1 },
                unit,
                std::move( source ),
                eq->getVars(),
                getDefaultRanges( std::move( ranges ), eq->getVars() ),
                std::move( argUnits ) ),
      d_eq( eq )
{
    AMP_ASSERT( d_eq );
}
EquationProperty::EquationProperty( std::string_view name,
                                    const std::string &expression,
                                    const AMP::Units &unit,
                                    std::vector<std::string> args,
                                    std::vector<std::array<double, 2>> ranges,
                                    std::vector<AMP::Units> argUnits,
                                    std::string_view source )
    : Property( std::move( name ),
                { 1 },
                unit,
                std::move( source ),
                std::move( args ),
                getDefaultRanges( std::move( ranges ), args ),
                std::move( argUnits ) ),
      d_eq( std::make_shared<MathExpr>( expression, args ) )
{
    AMP_ASSERT( d_eq );
}
void EquationProperty::eval( AMP::Array<double> &result, const AMP::Array<double> &args ) const
{
    AMP_ASSERT( d_eq );
    for ( size_t i = 0; i < result.length(); i++ ) {
        if ( args.empty() )
            result( i ) = ( *d_eq )();
        else
            result( i ) = ( *d_eq )( &args( 0, i ) );
    }
}


} // namespace AMP::Materials
