#ifndef included_AMP_Property_hpp
#define included_AMP_Property_hpp

#include "AMP/materials/Property.hpp"

#include <algorithm>
#include <sstream>


namespace AMP::Materials {


/************************************************************************
 *  evalv interfaces                                                     *
 ************************************************************************/
template<class... Args>
double Property::eval( const Units &unit, const Args &...args ) const
{
    double value = 0;
    auto result  = AMP::Array<double>::staticView( { 1 }, &value );
    // Get the arguments
    double tmp[128];
    memcpy( tmp, d_defaults.data(), d_defaults.size() * sizeof( double ) );
    auto args2 = AMP::Array<double>::staticView( { d_arguments.size() }, tmp );
    evalArgs( args2, args... );
    checkArgs( args2 );
    // Evaluate the property
    eval( result, args2 );
    // Convert units if required
    if ( !unit.isNull() )
        value *= d_units.convert( unit );
    return value;
}
template<class... Args>
void Property::evalv( std::vector<double> &r, const Units &unit, const Args &...args ) const
{
    AMP::Array<std::vector<double> *> r2( 1 );
    r2( 0 ) = &r;
    evalv( r2, unit, args... );
}
template<class... Args>
void Property::evalv( AMP::LinearAlgebra::Vector &r, const Args &...args ) const
{
    AMP::Array<AMP::LinearAlgebra::Vector *> r2( 1 );
    r2( 0 ) = &r;
    evalv( r2, args... );
}
template<class... Args>
void Property::evalv( std::vector<std::shared_ptr<std::vector<double>>> &r,
                      const Units &unit,
                      const Args &...args ) const
{
    AMP::Array<std::vector<double> *> r2( r.size() );
    for ( size_t i = 0; i < r2.length(); i++ )
        r2( i ) = r[i].get();
    evalv( r2, unit, args... );
}
template<class... Args>
void Property::evalv( std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> &r,
                      const Args &...args ) const
{
    AMP::Array<AMP::LinearAlgebra::Vector *> r2( r.size() );
    for ( size_t i = 0; i < r2.length(); i++ )
        r2( i ) = r[i].get();
    evalv( r2, args... );
}
template<class... Args>
void Property::evalv( AMP::Array<std::shared_ptr<std::vector<double>>> &r,
                      const Units &unit,
                      const Args &...args ) const
{
    AMP::Array<std::vector<double> *> r2( r.size() );
    for ( size_t i = 0; i < r2.length(); i++ )
        r2( i ) = r( i ).get();
    evalv( r2, unit, args... );
}
template<class... Args>
void Property::evalv( AMP::Array<std::shared_ptr<AMP::LinearAlgebra::Vector>> &r,
                      const Args &...args ) const
{
    AMP::Array<AMP::LinearAlgebra::Vector *> r2( r.size() );
    for ( size_t i = 0; i < r2.length(); i++ )
        r2( i ) = r( i ).get();
    evalv( r2, args... );
}
template<class... Args>
void Property::evalv( AMP::Array<std::vector<double> *> &r,
                      const Units &unit,
                      const Args &...args ) const
{
    // Load the arguments
    auto args2 = defaultArgs( r( 0 )->size() );
    evalArgs( args2, args... );
    checkArgs( args2 );

    // Call eval (and convert units)
    evalv( args2, r, unit );
}
template<class... Args>
void Property::evalv( AMP::Array<AMP::LinearAlgebra::Vector *> &r, const Args &...args ) const
{
    // Load the arguments
    auto args2 = defaultArgs( getSize( *r( 0 ) ) );
    evalArgs( args2, args... );
    checkArgs( args2 );

    // Call eval (and convert units)
    evalv( args2, r );
}


/************************************************************************
 *  Evaluate the input arguments                                         *
 ************************************************************************/
template<class VEC>
void Property::evalArgs( AMP::Array<double> &args2, const std::map<std::string, VEC> &args ) const
{
    for ( const auto &arg : args )
        evalArgs( args2, arg.first, arg.second );
}
template<class... Args>
void Property::evalArgs( AMP::Array<double> &args2,
                         std::string_view name,
                         double v,
                         const Args &...args ) const
{
    size_t N = args2.size( 1 );
    int i    = get_arg_index( name );
    if ( i >= 0 ) {
        for ( size_t j = 0; j < N; j++ )
            args2( i, j ) = v;
    }
    evalArgs( args2, args... );
}
template<class... Args>
void Property::evalArgs( AMP::Array<double> &args2,
                         std::string_view name,
                         const Units &unit,
                         double v,
                         const Args &...args ) const
{
    size_t N = args2.size( 1 );
    int i    = get_arg_index( name );
    if ( i >= 0 ) {
        if ( !unit.isNull() )
            v *= unit.convert( d_argUnits[i] );
        for ( size_t j = 0; j < N; j++ )
            args2( i, j ) = v;
    }
    evalArgs( args2, args... );
}
template<class... Args>
void Property::evalArgs( AMP::Array<double> &args2,
                         std::string_view name,
                         const Units &unit,
                         const std::vector<double> &v,
                         const Args &...args ) const
{
    evalArg( args2, name, unit, v );
    evalArgs( args2, args... );
}
template<class... Args>
void Property::evalArgs( AMP::Array<double> &args2,
                         std::string_view name,
                         const Units &unit,
                         const AMP::LinearAlgebra::Vector &v,
                         const Args &...args ) const
{
    evalArg( args2, name, unit, v );
    evalArgs( args2, args... );
}
template<class... Args>
void Property::evalArgs( AMP::Array<double> &args2,
                         std::string_view name,
                         const std::vector<double> &v,
                         const Args &...args ) const
{
    evalArgs( args2, name, Units(), v, args... );
}
template<class... Args>
void Property::evalArgs( AMP::Array<double> &args2,
                         std::string_view name,
                         const AMP::LinearAlgebra::Vector &v,
                         const Args &...args ) const
{
    evalArgs( args2, name, getUnits( v ), v, args... );
}
template<class VEC, class... Args>
void Property::evalArgs( AMP::Array<double> &args2,
                         std::string_view name,
                         const Units &unit,
                         const std::shared_ptr<VEC> &v,
                         const Args &...args ) const
{
    evalArgs( args2, name, unit, *v, args... );
}
template<class VEC, class... Args>
void Property::evalArgs( AMP::Array<double> &args2,
                         std::string_view name,
                         const std::shared_ptr<VEC> &v,
                         const Args &...args ) const
{
    evalArgs( args2, name, *v, args... );
}


/************************************************************************
 *  Determine if a set of values are all within range or not             *
 ************************************************************************/
inline bool
Property::in_range( std::string_view name, double value, const Units &unit, bool throwError ) const
{
    auto index = get_arg_index( name );
    if ( index == -1 )
        return true;
    double scale = 1.0;
    if ( !unit.isNull() && !d_argUnits[index].isNull() )
        scale = unit.convert( d_argUnits[index] );
    auto range = d_ranges[index];
    bool pass  = scale * value >= range[0] && scale * value <= range[1];
    if ( throwError && !pass ) {
        std::stringstream ss;
        ss << "Property '" + name + "' out of range in function '" + d_name + "'\n";
        ss << "Value is " << value << " ";
        ss << std::endl << "Valid range is [" << range[0] << "," << range[1] << "]" << std::endl;
        AMP_ERROR( ss.str() );
    }
    return pass;
}
template<class INPUT_VTYPE>
inline bool Property::in_range( std::string_view name,
                                const INPUT_VTYPE &values,
                                const Units &unit,
                                bool throwError ) const
{
    auto index = get_arg_index( name );
    if ( index == -1 )
        return true;
    double scale = 1.0;
    if ( !unit.isNull() && !d_argUnits[index].isNull() )
        scale = unit.convert( d_argUnits[index] );
    auto range = d_ranges[index];
    bool pass  = true;
    for ( auto value : values )
        pass = pass && scale * value >= range[0] && scale * value <= range[1];
    if ( throwError && !pass ) {
        std::stringstream ss;
        ss << "Property '" + name + "' out of range in function '" + d_name + "'\n";
        ss << "Values are ";
        for ( auto &value : values )
            ss << value << " ";
        ss << std::endl << "Valid range is [" << range[0] << "," << range[1] << "]" << std::endl;
        AMP_ERROR( ss.str() );
    }
    return pass;
}


/************************************************************************
 *  Get/Set auxillary data                                               *
 ************************************************************************/
template<class TYPE>
TYPE Property::getAuxiliaryData( const std::string &key ) const
{
    return d_auxiliaryData.getScalar<TYPE>( key );
}
template<class TYPE>
void Property::setAuxiliaryData( const std::string &key, const TYPE &data )
{
    return d_auxiliaryData.putScalar( key, data, {}, AMP::Database::Check::Overwrite );
}


} // namespace AMP::Materials


#endif
