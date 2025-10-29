#ifndef included_AMP_Property
    #define included_AMP_Property

    #include "AMP/utils/ArraySize.h"
    #include "AMP/utils/Database.h"
    #include "AMP/utils/Units.h"
    #include "AMP/utils/UtilityMacros.h"

    #include <algorithm>
    #include <array>
    #include <limits>
    #include <map>
    #include <memory>
    #include <string>
    #include <string_view>
    #include <vector>


// Foward declare classes
namespace AMP::LinearAlgebra {
class Vector;
class MultiVector;
} // namespace AMP::LinearAlgebra


namespace AMP::Materials {


/**
 * \namespace Materials
 * The materials design envisions a material property as a function.
 * These classes encapsulate the notion of functions that may
 * or may not have parameters that are remembered by the function
 * so you don't have to pass them in at every function call.
 * It also includes fields for sources such as a journal reference,
 * and a name.
 */


/**
 * \class          Property
 * \brief          Provides material properties of scalar type.
 */
class Property
{
public:
    /**
     * Constructor
     * \param name      name of property (required)
     * \param size      size of returned evaluation
     * \param unit      Units of property
     * \param source    literature reference for model and notes
     * \param args      names of arguments
     * \param ranges    valid ranges for arguments
     * \param argUnits  Optional units for each argument
     */
    Property( std::string_view name,
              const ArraySize &size                     = { 1 },
              const Units &unit                         = Units(),
              std::string_view source                   = "",
              std::vector<std::string> args             = {},
              std::vector<std::array<double, 2>> ranges = {},
              std::vector<Units> argUnits               = {} );

    //! Destructor
    virtual ~Property() {}

    //! get dimensions of evalv return value
    inline const ArraySize &size() const { return d_dim; }

    //! Return name of property
    inline const std::string &get_name() const { return d_name; }

    //! Get the source of the information
    inline const std::string &get_source() const { return d_source; }

    //! Return source reference
    inline const Units &get_units() const { return d_units; }

    //! Return the names of the arguments to eval
    inline const std::vector<std::string> &get_arguments() const { return d_arguments; }

    //! Return the number of arguments to eval
    inline size_t get_number_arguments() const { return d_arguments.size(); }

    //! Return the argument index
    inline int get_argument_index( std::string_view name ) const
    {
        int index = -1;
        for ( size_t i = 0; i < d_arguments.size(); i++ ) {
            if ( name == d_arguments[i] )
                index = i;
        }
        return index;
    }

    //! Get the default for the given argument (NaN if it is an invalid argument)
    double get_default( std::string_view name ) const;

    //! Get the defaults
    inline const std::vector<double> &get_defaults() const { return d_defaults; }

    //! Set the default
    inline void
    set_default( std::string_view name, double value, const AMP::Units &unit = AMP::Units() )
    {
        int i = get_argument_index( name );
        if ( i != -1 ) {
            if ( !unit.isNull() )
                value *= unit.convert( d_argUnits[i] );
            d_defaults[i] = value;
        }
    }

    //! Set the defaults
    inline void set_defaults( std::vector<double> defaults )
    {
        AMP_INSIST( defaults.size() == d_arguments.size(),
                    "incorrect number of defaults specified" );
        d_defaults = std::move( defaults );
    }

    //! Determine if a string is an argument
    bool is_argument( std::string_view argname ) const;

    //! Indicator for scalar evaluator
    virtual bool isString() const { return false; }

    //! Indicator for scalar evaluator
    bool isScalar() const { return d_dim.length() == 1; }

    //! Indicator for vector evaluator
    bool isVector() const { return d_dim.ndim() == 1 && d_dim.length() > 0; }

    //! Indicator for tensor evaluator
    bool isTensor() const { return d_dim.ndim() == 2; }

    // converts AMP::MultiVector to a map of pointers to AMP::Vectors based on argument names
    std::map<std::string, std::shared_ptr<AMP::LinearAlgebra::Vector>>
    make_map( const std::shared_ptr<AMP::LinearAlgebra::MultiVector> &args,
              const std::map<std::string, std::string> &translator ) const;


public: // Functions dealing with the ranges of the arguments
    //! Get units for all arguments used in this material
    const std::vector<Units> &get_arg_units() const { return d_argUnits; }

    //! Get ranges for all arguments used in this material
    std::vector<std::array<double, 2>> get_arg_ranges() const { return d_ranges; }

    //! Get range for a specific argument
    std::array<double, 2> get_arg_range( std::string_view argname ) const;

    //! Determine if a value is within range or not
    inline bool in_range( std::string_view argname,
                          double value,
                          const Units &unit = Units(),
                          bool throwError   = false ) const;

    //! Determine if a set of values are all within range or not
    template<class INPUT_VTYPE>
    inline bool in_range( std::string_view argname,
                          const INPUT_VTYPE &values,
                          const Units &unit = Units(),
                          bool throwError   = false ) const;


    //! Get auxiliary data
    inline const Database &getAuxiliaryData() const { return d_auxiliaryData; }

    //! Get auxiliary data
    template<class TYPE>
    TYPE getAuxiliaryData( const std::string &key ) const;

    //! Set auxiliary data
    template<class TYPE>
    void setAuxiliaryData( const std::string &key, const TYPE &data );


public: // Evaluators
    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions
     * \param unit      The desired units of the result.  If this is not specified,
     *                  the native units of the property are use (see get_units())
     * \param args      Optional arguments specifying input arguments to the eval() function.
     *                  In general arguments are of the form:
     *                     evalv( r, unit, "arg1", unit1, vec1, "arg2", unit2, vec2, ... ).
     * \return scalar value of property
     */
    template<class... Args>
    double eval( const Units &unit = Units(), const Args &...args ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions
     * \param unit      The desired units of the result.  If this is not specified,
     *                  the native units of the property are use (see get_units())
     * \param args      List of arguments for the property
     * \param units     List of units for each argument
     * \param vec       List of values for each argument
     * \return scalar value of property
     */
    double eval( const Units &unit,
                 const std::vector<std::string> &args,
                 const std::vector<double> &vec,
                 const std::vector<Units> &units = {} ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions for multiple points.
     * \param r         std::vector of return values
     * \param unit      The desired units of the result.  If this is not specified,
     *                  the native units of the property are use (see get_units())
     * \param args      Optional arguments specifying input arguments to the eval() function.
     *                  In general arguments are of the form:
     *                     evalv( r, unit, "arg1", unit1, vec1, "arg2", unit2, vec2, ... ).
     */
    template<class... Args>
    void evalv( std::vector<double> &r, const Units &unit, const Args &...args ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions for multiple points.
     * \param r         std::vector of return values
     * \param args      Optional arguments specifying input arguments to the eval() function.
     *                  In general arguments are of the form:
     *                     evalv( r, unit, "arg1", unit1, vec1, "arg2", unit2, vec2, ... ).
     */
    template<class... Args>
    void evalv( AMP::LinearAlgebra::Vector &r, const Args &...args ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions for multiple points.
     * \param r         std::vector of return values
     * \param unit      The desired units of the result.  If this is not specified,
     *                  the native units of the property are use (see get_units())
     * \param args      Optional arguments specifying input arguments to the eval() function.
     *                  In general arguments are of the form:
     *                     evalv( r, unit, "arg1", unit1, vec1, "arg2", unit2, vec2, ... ).
     */
    template<class... Args>
    void evalv( std::vector<std::shared_ptr<std::vector<double>>> &r,
                const Units &unit,
                const Args &...args ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions for multiple points.
     * \param r         Vector of vector of return values
     * \param args      Optional arguments specifying input arguments to the eval() function.
     *                  In general arguments are of the form:
     *                     evalv( r, unit, "arg1", unit1, vec1, "arg2", unit2, vec2, ... ).
     */
    template<class... Args>
    void evalv( std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> &r,
                const Args &...args ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions for multiple points.
     * \param r            std::vector of return values
     * \param unit         Units to use for return values
     * \param args         Optional arguments specifying input arguments to the eval() function
     */
    template<class... Args>
    void evalv( AMP::Array<std::shared_ptr<std::vector<double>>> &r,
                const Units &unit,
                const Args &...args ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions for multiple points.
     * \param r         Vector of return values
     * \param args      Optional arguments specifying input arguments to the eval() function.
     *                  In general arguments are of the form:
     *                     evalv( r, unit, "arg1", unit1, vec1, "arg2", unit2, vec2, ... ).
     */
    template<class... Args>
    void evalv( AMP::Array<std::shared_ptr<AMP::LinearAlgebra::Vector>> &r,
                const Args &...args ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions for multiple points.
     * \param r         std::vector of return values
     * \param unit      The desired units of the result.  If this is not specified,
     *                  the native units of the property are use (see get_units())
     * \param args      Optional arguments specifying input arguments to the eval() function.
     *                  In general arguments are of the form:
     *                     evalv( r, unit, "arg1", unit1, vec1, "arg2", unit2, vec2, ... ).
     */
    template<class... Args>
    void
    evalv( AMP::Array<std::vector<double> *> &r, const Units &unit, const Args &...args ) const;

    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions for multiple points.
     * \param r         std::vector of return values
     * \param args      Optional arguments specifying input arguments to the eval() function.
     *                  In general arguments are of the form:
     *                     evalv( r, unit, "arg1", unit1, vec1, "arg2", unit2, vec2, ... ).
     */
    template<class... Args>
    void evalv( AMP::Array<AMP::LinearAlgebra::Vector *> &r, const Args &...args ) const;

    //! Get the string value of the property
    virtual std::string evalString() const;


protected: // Virtual function to override to load the property
    /**
     * scalar evaluation function for a single argument set
     *  \param result       Output result (N)
     *  \param args         Input arguments (MxN)
     */
    virtual void eval( AMP::Array<double> &result, const AMP::Array<double> &args ) const = 0;


protected: // Internal eval/evalv/evalArg
    // clang-format off
    static size_t getSize( const AMP::LinearAlgebra::Vector& );
    static Units getUnits( const AMP::LinearAlgebra::Vector& );
    void evalv( const AMP::Array<double>&, AMP::Array<std::vector<double> *> &, const Units & ) const;
    void evalv( const AMP::Array<double>&, AMP::Array<AMP::LinearAlgebra::Vector *> & ) const;
    void evalArg( AMP::Array<double>&, std::string_view, const Units&, double ) const;
    void evalArg( AMP::Array<double>&, std::string_view, const Units&, const std::vector<double>& ) const;
    void evalArg( AMP::Array<double>&, std::string_view, const Units&, const AMP::LinearAlgebra::Vector& ) const;
    // clang-format on


protected: // Functions to load the arguments
    // clang-format off
    void evalArgs( AMP::Array<double>& ) const {}
    void evalArgs( AMP::Array<double>&, const std::shared_ptr<AMP::LinearAlgebra::MultiVector>&, const std::map<std::string, std::string>& = {} ) const;
    template<class VEC>
    void evalArgs( AMP::Array<double>&, const std::map<std::string, VEC>& ) const;
    template<class... Args>
    void evalArgs( AMP::Array<double>&, std::string_view, double, const Args&... ) const;
    template<class... Args>
    void evalArgs( AMP::Array<double>&, std::string_view, const Units&, double, const Args&... ) const;
    template<class... Args>
    void evalArgs( AMP::Array<double>&, std::string_view, const std::vector<double>&, const Args&... ) const;
    template<class... Args>
    void evalArgs( AMP::Array<double>&, std::string_view, const AMP::LinearAlgebra::Vector&, const Args&... ) const;
    template<class... Args>
    void evalArgs( AMP::Array<double>&, std::string_view, const Units&, const std::vector<double>&, const Args&... ) const;
    template<class... Args>
    void evalArgs( AMP::Array<double>&, std::string_view, const Units&, const AMP::LinearAlgebra::Vector&, const Args&... ) const;
    template<class VEC, class... Args>
    void evalArgs( AMP::Array<double>&, std::string_view, const std::shared_ptr<VEC>&, const Args&... ) const;
    template<class VEC, class... Args>
    void evalArgs( AMP::Array<double>&, std::string_view, const Units&, const std::shared_ptr<VEC>&, const Args&... ) const;
    void evalArgs( AMP::Array<double>&, const std::vector<double> &args, const std::vector<std::string> &names, const std::vector<Units> &argUnits = {} ) const;
    // clang-format on


protected:
    std::string d_name;                                 //!< should be unique
    AMP::ArraySize d_dim;                               //!< size of the result
    AMP::Units d_units;                                 //!< default units to return
    std::string d_source;                               //!< reference for source data
    std::vector<std::string> d_arguments;               //!< names of the arguments
    std::vector<Units> d_argUnits;                      //!< default units for the arguments
    std::vector<double> d_defaults;                     //!< default values of arguments
    std::vector<std::array<double, 2>> d_ranges;        //!< allowed ranges of arguments
    std::map<std::string_view, size_t> d_argToIndexMap; //!< map argument names to indices
    AMP::Database d_auxiliaryData;                      //!< Database containing auxiliary data


protected:
    Property() = default;

    //! Create the default argument array
    AMP::Array<double> defaultArgs( size_t ) const;

    //! Check the argument values
    void checkArgs( const AMP::Array<double> &args ) const;

    // Get the index for the desired argument
    inline int get_arg_index( std::string_view name ) const
    {
        auto it = d_argToIndexMap.find( name );
        if ( it == d_argToIndexMap.end() )
            return -1;
        return it->second;
    }
};


//! Create a property from a database data
std::unique_ptr<Property> createProperty( std::string_view key, const Database &db );


} // namespace AMP::Materials

#endif


#include "AMP/materials/Property.hpp"
