// This file contains useful macros including AMP_ERROR, AMP_WARNING, AMP_INSIST, AMP_ASSERT, etc.
#ifndef included_AMP_UtilityMacros
#define included_AMP_UtilityMacros

#include <sstream>
#include <string_view>

#include "StackTrace/source_location.h"


// Forward declare StackTrace abort
namespace StackTrace::Utilities {
[[noreturn]] void abort( const std::string &message, const source_location &source );
}


// Forward declare stream operators
namespace AMP {
extern std::ostream pout;
extern std::ostream perr;
extern std::ostream plog;
} // namespace AMP


template<class T>
inline constexpr bool failed_assert_v = !std::is_same<T, T>::value;


// Overload some operators for string_view for convience
namespace AMP {
template<std::size_t N>
static inline std::string operator+( char x[N], std::string_view y )
{
    return std::string( x ) + std::string( y );
}
template<std::size_t N>
static inline std::string operator+( std::string_view y, char x[N] )
{
    return std::string( x ) + std::string( y );
}
static inline std::string operator+( std::string_view x, std::string_view y )
{
    return std::string( x ) + std::string( y );
}
} // namespace AMP


/*! \defgroup Macros Set of utility macro functions used in AMP
 *  \details  These functions are a list of C++ macros that are used within AMP
 *     for common operations, including checking for errors and profiling.
 *  \addtogroup Macros
 *  @{
 */


/*! \def NULL_USE(variable)
 *  \brief    A null use of a variable
 *  \details  A null use of a variable, use to avoid GNU compiler warnings about unused variables.
 *  \param variable  Variable to pretend to use
 */
#undef NULL_USE
#define NULL_USE( variable )                                     \
    do {                                                         \
        if ( 0 ) {                                               \
            [[maybe_unused]] auto static t = (char *) &variable; \
            t++;                                                 \
        }                                                        \
    } while ( 0 )


/*! \def STATIC_ERROR(variable)
 *  \brief      Fail assert
 *  \details    Fail assert at compile time
 *  \param MSG  Error message to print
 */
#undef STATIC_ERROR
#define STATIC_ERROR( MSG ) static_assert( failed_assert_v<TYPE>, MSG )


/*! \def AMP_ERROR(MSG)
 *  \brief      Throw error
 *  \details    Throw an error exception from within any C++ source code.  The
 *     macro argument may be any standard ostream expression.  The file and
 *     line number of the abort are also printed.
 *  \param MSG  Error message to print
 */
#define AMP_ERROR( MSG ) StackTrace::Utilities::abort( MSG, SOURCE_LOCATION_CURRENT() )


/*! \def AMP_WARNING(MSG)
 *  \brief   Print a warning
 *  \details Print a warning without exit.  Print file and line number of the warning.
 *  \param MSG  Warning message to print
 */
#define AMP_WARNING( MSG )                                                                        \
    do {                                                                                          \
        AMP::pout << "WARNING: " << MSG << std::endl;                                             \
        AMP::plog << "WARNING: " << MSG << std::endl;                                             \
        AMP::pout << "   Warning called in " << __FILE__ << " on line " << __LINE__ << std::endl; \
        AMP::plog << "   Warning called in " << __FILE__ << " on line " << __LINE__ << std::endl; \
        AMP::plog << std::flush;                                                                  \
    } while ( 0 )


/*! \def AMP_WARN_ONCE(MSG)
 *  \brief   Print a warning
 *  \details Print a warning without exit.  Print file and line number of the warning.
 *           The message will only be printed once per execution
 *  \param MSG  Warning message to print
 */
#define AMP_WARN_ONCE( MSG )         \
    do {                             \
        static bool printed = false; \
        if ( !printed ) {            \
            AMP_WARNING( MSG );      \
            printed = true;          \
        }                            \
    } while ( 0 )


/*! \def AMP_ASSERT(EXP)
 *  \brief Assert error
 *  \details Throw an error exception from within any C++ source code if the
 *     given expression is not true.  This is a parallel-friendly version
 *     of assert.
 *     The file and line number of the abort are printed along with the stack trace (if available).
 *  \param EXP  Expression to evaluate
 */
#define AMP_ASSERT( EXP )                                                                         \
    do {                                                                                          \
        if ( !( EXP ) ) {                                                                         \
            StackTrace::Utilities::abort( "Failed assertion: " #EXP, SOURCE_LOCATION_CURRENT() ); \
        }                                                                                         \
    } while ( 0 )


/*! \def AMP_INSIST(EXP,MSG)
 *  \brief Insist error
 *  \details Throw an error exception from within any C++ source code if the
 *     given expression is not true.  This will also print the given message.
 *     This is a parallel-friendly version of assert.
 *     The file and line number of the abort are printed along with the stack trace (if available).
 *  \param EXP  Expression to evaluate
 *  \param MSG  Debug message to print
 */
#define AMP_INSIST( EXP, MSG )                                                   \
    do {                                                                         \
        if ( !( EXP ) ) {                                                        \
            StackTrace::Utilities::abort( "Failed insist: " #EXP "\nMessage: " + \
                                              std::string( MSG ),                \
                                          SOURCE_LOCATION_CURRENT() );           \
        }                                                                        \
    } while ( 0 )


// Macros for use when assertions are to be included only when debugging.
/*! \def AMP_DEBUG_ASSERT(EXP)
 *  \brief Assert error (debug only)
 *  \details Throw an error exception from within any C++ source code if the
 *     given expression is not true.  This only runs if compiled with debug
 *     is enabled.  If enabled, this is the same as a call to AMP_ASSERT.
 *  \param EXP  Expression to evaluate
 */
/*! \def AMP_DEBUG_INSIST(EXP,MSG)
 *  \brief Insist error
 *  \details Throw an error exception from within any C++ source code if the
 *     given expression is not true.  This will also print the given message.
 *     This is a parallel-friendly version of assert.
 *     The file and line number of the abort are printed along with the stack trace (if available).
 *     This only runs if compiled with debug is enabled.
 *     If enabled, this is the same as a call to AMP_INSIST.
 *  \param EXP  Expression to evaluate
 *  \param MSG  Debug message to print
 */
#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    #define AMP_DEBUG_ASSERT( EXP ) AMP_ASSERT( EXP )
    #define AMP_DEBUG_INSIST( EXP, MSG ) AMP_INSIST( EXP, MSG )
#else
    #define AMP_DEBUG_ASSERT( EXP ) \
        do {                        \
        } while ( 0 )
    #define AMP_DEBUG_INSIST( EXP, MSG ) \
        do {                             \
        } while ( 0 )
#endif


// clang-format off

/*! \def DISABLE_WARNINGS
 *  \brief Re-enable warnings
 *  \details This will re-enable warnings after a call to DIASABLE_WARNINGS
 */
/*! \def ENABLE_WARNINGS
 *  \brief Suppress all warnings
 *  \details This will start to supress all compile warnings.
 *      Be sure to follow with ENABLE_WARNINGS
 */
#ifndef DISABLE_WARNINGS
    #if defined( _MSC_VER )
        #define DISABLE_WARNINGS __pragma( warning( push, 0 ) )
        #define ENABLE_WARNINGS __pragma( warning( pop ) )
    #elif defined( __clang__ )
        #define DISABLE_WARNINGS                                                      \
            _Pragma( "clang diagnostic push" )                                        \
            _Pragma( "clang diagnostic ignored \"-Wall\"" )                           \
            _Pragma( "clang diagnostic ignored \"-Wextra\"" )                         \
            _Pragma( "clang diagnostic ignored \"-Wunused-private-field\"" )          \
            _Pragma( "clang diagnostic ignored \"-Wdeprecated-declarations\"" )       \
            _Pragma( "clang diagnostic ignored \"-Winteger-overflow\"" )              \
            _Pragma( "clang diagnostic ignored \"-Winconsistent-missing-override\"" ) \
            _Pragma( "clang diagnostic ignored \"-Wimplicit-int-float-conversion\"" )
        #define ENABLE_WARNINGS _Pragma( "clang diagnostic pop" )
    #elif defined( __GNUC__ )
        #define DISABLE_WARNINGS                                                \
            _Pragma( "GCC diagnostic push" )                                    \
            _Pragma( "GCC diagnostic ignored \"-Wpragmas\"" )                   \
            _Pragma( "GCC diagnostic ignored \"-Wall\"" )                       \
            _Pragma( "GCC diagnostic ignored \"-Wextra\"" )                     \
            _Pragma( "GCC diagnostic ignored \"-Wpedantic\"" )                  \
            _Pragma( "GCC diagnostic ignored \"-Wunused-local-typedefs\"" )     \
            _Pragma( "GCC diagnostic ignored \"-Woverloaded-virtual\"" )        \
            _Pragma( "GCC diagnostic ignored \"-Wunused-parameter\"" )          \
            _Pragma( "GCC diagnostic ignored \"-Wdeprecated-copy\"" )           \
            _Pragma( "GCC diagnostic ignored \"-Wdeprecated-declarations\"" )   \
            _Pragma( "GCC diagnostic ignored \"-Wvirtual-move-assign\"" )       \
            _Pragma( "GCC diagnostic ignored \"-Wunused-function\"" )           \
            _Pragma( "GCC diagnostic ignored \"-Woverflow\"" )                  \
            _Pragma( "GCC diagnostic ignored \"-Wunused-variable\"" )           \
            _Pragma( "GCC diagnostic ignored \"-Wignored-qualifiers\"" )        \
            _Pragma( "GCC diagnostic ignored \"-Wenum-compare\"" )              \
            _Pragma( "GCC diagnostic ignored \"-Wsign-compare\"" )              \
            _Pragma( "GCC diagnostic ignored \"-Wterminate\"" )                 \
            _Pragma( "GCC diagnostic ignored \"-Wimplicit-fallthrough\"" )      \
            _Pragma( "GCC diagnostic ignored \"-Wmaybe-uninitialized\"" )       \
            _Pragma( "GCC diagnostic ignored \"-Winaccessible-base\"" )         \
            _Pragma( "GCC diagnostic ignored \"-Wclass-memaccess\"" )           \
            _Pragma( "GCC diagnostic ignored \"-Waggressive-loop-optimizations\"" )
        #define ENABLE_WARNINGS _Pragma( "GCC diagnostic pop" )
    #elif defined( __INTEL_COMPILER )
        #if defined ( __INTEL_LLVM_COMPILER )
            // have to figure these warnings out
            #define DISABLE_WARNINGS                \
                _Pragma( "warning (push)" )
            #define ENABLE_WARNINGS _Pragma( "warning(pop)" )
        #else
           #define DISABLE_WARNINGS                \
               _Pragma( "warning (push)" )         \
               _Pragma( "warning disable 488" )    \
               _Pragma( "warning disable 1011" )   \
               _Pragma( "warning disable 61" )     \
               _Pragma( "warning disable 1478" )   \
               _Pragma( "warning disable 488" )    \
               _Pragma( "warning disable 2651" )
           #define ENABLE_WARNINGS _Pragma( "warning(pop)" )
        #endif
    #else
        #define DISABLE_WARNINGS
        #define ENABLE_WARNINGS
    #endif
#endif
// clang-format on

/*! @} */

#endif
