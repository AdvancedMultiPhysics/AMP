#ifndef included_AMP_ScalarProperty
#define included_AMP_ScalarProperty

#include "AMP/materials/Property.h"

#include <cstring>


// Forward declares
namespace AMP {
class MathExpr;
}


namespace AMP::Materials {


// Helper function
std::vector<std::array<double, 2>> getDefaultRanges( std::vector<std::array<double, 2>> ranges,
                                                     const std::vector<std::string> &vars );

//! String property class
class StringProperty final : public Property
{
public:
    StringProperty( std::string_view name, std::string value, std::string_view source = "" );
    bool isString() const override { return true; }
    std::string evalString() const override { return d_value; }
    void eval( AMP::Array<double> &, const AMP::Array<double> & ) const override;

private:
    std::string d_value;
};


//! Scalar property class (fixed value)
class ScalarProperty final : public Property
{
public:
    ScalarProperty( std::string_view name,
                    double value,
                    const AMP::Units &unit  = AMP::Units(),
                    std::string_view source = "" );
    ScalarProperty( std::string_view name,
                    AMP::Array<double> value,
                    const AMP::Units &unit  = AMP::Units(),
                    std::string_view source = "" );
    void eval( AMP::Array<double> &result, const AMP::Array<double> & ) const override;
    inline const AMP::Array<double> &getValue() const { return d_value; }

private:
    AMP::Array<double> d_value;
};


//! Polynomial based property class
class PolynomialProperty : public Property
{
public:
    PolynomialProperty() {}
    PolynomialProperty( std::string_view name,
                        std::string_view source,
                        const AMP::Units &unit                    = {},
                        std::vector<double> params                = {},
                        std::vector<std::string> args             = {},
                        std::vector<std::array<double, 2>> ranges = {},
                        std::vector<AMP::Units> argUnits          = {} );
    void eval( AMP::Array<double> &result, const AMP::Array<double> &args ) const override;

private:
    std::vector<double> d_p;
};


//! Interpolated property class
class InterpolatedProperty final : public Property
{
public:
    InterpolatedProperty( std::string_view name,
                          const AMP::Units &unit,
                          const std::string &var_name,
                          std::vector<double> x,
                          std::vector<double> y,
                          const std::array<double, 2> range,
                          const AMP::Units &argUnit,
                          double default_value,
                          std::string_view source = "",
                          std::string_view method = "linear" );
    void eval( AMP::Array<double> &result, const AMP::Array<double> & ) const override;

private:
    std::vector<double> d_x;
    std::vector<double> d_y;
    int d_method;
};


//! Equation based property class
class EquationProperty : public Property
{
public:
    EquationProperty() {}
    EquationProperty( std::string_view name,
                      std::shared_ptr<const MathExpr> eq,
                      const AMP::Units &unit                    = {},
                      std::vector<std::array<double, 2>> ranges = {},
                      std::vector<AMP::Units> argUnits          = {},
                      std::string_view source                   = "" );
    EquationProperty( std::string_view name,
                      const std::string &expression,
                      const AMP::Units &unit                    = {},
                      std::vector<std::string> args             = {},
                      std::vector<std::array<double, 2>> ranges = {},
                      std::vector<AMP::Units> argUnits          = {},
                      std::string_view source                   = "" );
    void eval( AMP::Array<double> &result, const AMP::Array<double> &args ) const override;

private:
    std::shared_ptr<const MathExpr> d_eq;
};


//! Function based property class
template<class... Args>
class FunctionProperty : public Property
{
public:
    FunctionProperty( std::string_view name,
                      std::function<double( Args... )> fun,
                      const AMP::Units &unit                    = {},
                      std::vector<std::string> args             = {},
                      std::vector<std::array<double, 2>> ranges = {},
                      std::vector<AMP::Units> argUnits          = {},
                      const std::vector<double> &default_values = {},
                      std::string_view source                   = "" )
        : Property( std::move( name ),
                    { 1 },
                    unit,
                    std::move( source ),
                    std::move( args ),
                    getDefaultRanges( std::move( ranges ), args ),
                    std::move( argUnits ) ),
          d_fun( fun )
    {
        AMP_ASSERT( get_number_arguments() == sizeof...( Args ) );
        set_defaults( default_values );
    }
    void eval( AMP::Array<double> &result, const AMP::Array<double> &args ) const override
    {
        constexpr std::size_t N = sizeof...( Args );
        static_assert( N <= 5, "More than 5 arguments is not supported yet" );
        if constexpr ( N == 0 ) {
            result.fill( d_fun() );
            return;
        }
        for ( size_t i = 0; i < result.length(); i++ ) {
            const double *x = &args( 0, i );
            if constexpr ( N == 1 )
                result( i ) = d_fun( x[0] );
            else if constexpr ( N == 2 )
                result( i ) = d_fun( x[0], x[1] );
            else if constexpr ( N == 3 )
                result( i ) = d_fun( x[0], x[1], x[2] );
            else if constexpr ( N == 4 )
                result( i ) = d_fun( x[0], x[1], x[2], x[3] );
            else if constexpr ( N == 5 )
                result( i ) = d_fun( x[0], x[1], x[2], x[3], x[4] );
        }
    }

private:
    std::function<double( Args... )> d_fun;
};


} // namespace AMP::Materials

#endif
