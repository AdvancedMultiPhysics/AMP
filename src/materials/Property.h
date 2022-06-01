#ifndef included_AMP_Property
#define included_AMP_Property

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
 *
 * A Property class may provide only one of eval(), evalVector() or evalTensor(). It can not
 * provide both scalar and tensor
 */
class Property
{
public:
    /**
     * Constructor
     * \param name      name of property (required)
     * \param source    literature reference for model and notes
     * \param params    default parameter values
     * \param args      names of arguments
     * \param units     Optional units for each argument
     */
    Property( std::string name,
              Units unit                                = Units(),
              std::string source                        = "None",
              std::vector<std::string> args             = {},
              std::vector<std::array<double, 2>> ranges = {},
              std::vector<Units> argUnits               = {} );

    //! Destructor
    virtual ~Property() {}

    //! Return name of property
    inline const std::string &get_name() const { return d_name; }

    //! Return source reference
    inline const std::string &get_source() const { return d_source; }

    //! Return source reference
    inline const Units &get_units() const { return d_units; }

    //! Return the names of the arguments to eval
    inline const std::vector<std::string> &get_arguments() const { return d_arguments; }

    //! Return the number of arguments to eval
    inline size_t get_number_arguments() const { return d_arguments.size(); }

    //! Get the defaults
    inline const std::vector<double> &get_defaults() const { return d_defaults; }

    //! Set the defaults
    inline void set_defaults( std::vector<double> defaults )
    {
        AMP_INSIST( defaults.size() == d_arguments.size(),
                    "incorrect number of defaults specified" );
        d_defaults = defaults;
    }

    //! Determine if a string is an argument
    bool is_argument( const std::string &argname ) const;

    //! Indicator for scalar evaluator
    virtual bool isScalar() const { return true; }

    //! Indicator for vector evaluator
    virtual bool isVector() const { return false; }

    //! Indicator for tensor evaluator
    virtual bool isTensor() const { return false; }

    // converts AMP::MultiVector to a map of pointers to AMP::Vectors based on argument names
    std::map<std::string, std::shared_ptr<AMP::LinearAlgebra::Vector>>
    make_map( const std::shared_ptr<AMP::LinearAlgebra::MultiVector> &args,
              const std::map<std::string, std::string> &translator ) const;


public: // Functions dealing with the ranges of the arguments
    //! Get units for all arguments used in this material
    std::vector<Units> get_arg_units() const { return d_argUnits; }

    //! Get ranges for all arguments used in this material
    std::vector<std::array<double, 2>> get_arg_ranges() const { return d_ranges; }

    //! Get range for a specific argument
    std::array<double, 2> get_arg_range( const std::string &argname ) const;

    //! Determine if a value is within range or not
    inline bool in_range( const std::string &argname,
                          double value,
                          Units unit      = Units(),
                          bool throwError = false ) const;

    //! Determine if a set of values are all within range or not
    template<class INPUT_VTYPE>
    inline bool in_range( const std::string &argname,
                          const INPUT_VTYPE &values,
                          Units unit      = Units(),
                          bool throwError = false ) const;


public: // Functions dealing with auxilliary data
    //! Set auxiliary data
    void setAuxiliaryData( const std::string &key, const double val );

    //! Set auxiliary data
    void setAuxiliaryData( const std::string &key, const int val );

    //! Set auxiliary data
    void setAuxiliaryData( const std::string &key, const std::string &val );

    //! Get auxiliary data
    void getAuxiliaryData( const std::string &key, double &val ) const;

    //! Get auxiliary data
    void getAuxiliaryData( const std::string &key, int &val ) const;

    //! Get auxiliary data
    void getAuxiliaryData( const std::string &key, std::string &val ) const;


public: // Evaluators
    /**
     * \brief    Evaluate the property
     * \details  This function evaluates the property at the desired conditions
     * \param unit      The desired units of the result.  If this is not specified,
     *                  the native units of the property are use (see get_units())
     * \param args      The values for the optional arguments.  If names is not specified,
     *                  this must match the values from get_parameters().
     * \param names     The names for the optional arguments.
     * \param argUnits  The units for the given arguments.  If not specified then the
     *                  native units are used (see get_arg_units())
     * \return scalar value of property
     */
    double eval( const Units &unit                     = Units(),
                 const std::vector<double> &args       = {},
                 const std::vector<std::string> &names = {},
                 const std::vector<Units> &argUnits    = {} ) const;

    /** Wrapper function that calls evalvActual for each argument set
     *  \param r vector of return values
     *  \param args map of vectors of arguments, indexed by strings which are members of
     *     get_arguments()
     *
     *  The  \a args  parameter need not contain all the members of  get_arguments()  as
     * indices. Arguments left out will have values supplied by the entries in  get_defaults() .
     *  Sizes of  \a r  and \a args["name"] must match. Members of
     *  \a args  indexed by names other than those in  get_arguments()  are ignored.
     */
    // template<class... Args>
    // void evalv( std::vector<double> &r, Units u, Args... args ) const;

    /** Wrapper function that calls evalvActual for each argument set
     *  \param r AMP vector of return values
     *  \param args map of AMP vectors of arguments, indexed by strings which are members of
     * get_arguments()
     *
     *  The  \a args  parameter need not contain all the members of  get_arguments()  as
     * indices. Arguments left out will have values supplied by the entries in  get_defaults() .
     *  Sizes of  \a r  and \a args["name"] must match. Members of
     *  \a args  indexed by names other than those in  get_arguments()  are ignored.
     *  The list {args["name-1"][i], ..., args["name-n"][i]} will be passed to eval() and the
     * k-j-th result returned in (*r[k][j])[i].
     */
    void evalv(
        std::shared_ptr<AMP::LinearAlgebra::Vector> &r,
        const std::map<std::string, std::shared_ptr<AMP::LinearAlgebra::Vector>> &args = {} ) const;

    /** Wrapper function that calls evalvActual for each argument set
     *  Upon invocation, the \a args parameter is converted to a map of AMP vectors via
     *     make_map() and passed to another version of evalv.
     *  \param r            AMP vector of return values
     *  \param args         AMP multivector of arguments
     *  \param translator   Optional translator between property arguments and AMP::Multivector
     * entries
     */
    void evalv( std::shared_ptr<AMP::LinearAlgebra::Vector> &r,
                const std::shared_ptr<AMP::LinearAlgebra::MultiVector> &args,
                const std::map<std::string, std::string> &translator = {} ) const;


public: // Advanced interfaces
    template<class VEC>
    struct argumentDataStruct {
        argumentDataStruct( std::string_view s, const VEC &v ) : str( s ), vec( v ) {}
        argumentDataStruct( std::string_view s, const VEC &v, const Units &u )
            : str( s ), vec( v ), units( u )
        {
        }
        std::string_view str;
        const VEC &vec;
        Units units;
    };

    // Convert the argument data
    template<class VEC, class... Args>
    static std::vector<argumentDataStruct<VEC>> convertArgs( Args... args );

    // Loops through input vectors, calling the child eval function, returning scalar
    template<class OUT, class IN = OUT>
    void evalv( OUT &r, const Units &units, const std::vector<argumentDataStruct<IN>> &args ) const;

    //[[deprecated]]
    void evalv( std::vector<double> &r,
                const std::map<std::string, std::shared_ptr<std::vector<double>>> &args ) const;

    // [[deprecated]]
    inline double evalDirect( const std::vector<double> &args ) const { return eval( args ); }

protected:
    Property() = default;

    /**
     * scalar evaluation function for a single argument set
     * \param args list of argument values, in correct order, in the correct units, given by
     *    get_arguments()
     * \return scalar value of property
     */
    virtual double eval( const std::vector<double> &args ) const = 0;


protected:
    std::string d_name;                                 //!< should be unique
    Units d_units;                                      //!< default units to return
    std::string d_source;                               //!< reference for source data
    std::vector<std::string> d_arguments;               //!< names of the arguments
    std::vector<Units> d_argUnits;                      //!< default units for the arguments
    std::vector<double> d_defaults;                     //!< default values of arguments
    std::vector<std::array<double, 2>> d_ranges;        //!< allowed ranges of arguments
    std::map<std::string_view, size_t> d_argToIndexMap; //!< map argument names to indices

    std::map<std::string, double> d_AuxiliaryDataDouble;
    std::map<std::string, int> d_AuxiliaryDataInteger;
    std::map<std::string, std::string> d_AuxiliaryDataString;


protected:
    // Get the index for the desired argument
    inline int get_arg_index( const std::string &name ) const
    {
        auto it = d_argToIndexMap.find( name );
        if ( it == d_argToIndexMap.end() )
            return -1;
        return it->second;
    }

    template<class VEC, class... Args>
    static void convertArgs1( std::vector<argumentDataStruct<VEC>> &,
                              const std::string &,
                              const VEC &,
                              Args... args );
    template<class VEC, class... Args>
    static void convertArgs2( std::vector<argumentDataStruct<VEC>> &,
                              const std::string &,
                              const VEC &,
                              const Units &u,
                              Args... args );

protected: // Friend classes (need to clean this up)
    friend class ThermalDiffusionCoefficientProp;
    friend class DxThermalDiffusionCoefficientProp;
    friend class DTThermalDiffusionCoefficientProp;
};


} // namespace AMP::Materials

#include "Property.i.h"

#endif
