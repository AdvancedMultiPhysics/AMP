// This file contains helper functions to convert a simple struct to a tuple
#ifndef included_AMP_to_tuple
#define included_AMP_to_tuple

#include "AMP/utils/MacroFunctions.h"


#include <tuple>
#include <type_traits>


namespace AMP {


// Helper functions
struct any_type {
    template<class T>
    constexpr operator T();
};
template<class T, typename... Args>
decltype( void( T{ std::declval<Args>()... } ), std::true_type() ) test( int );
template<class T, typename... Args>
std::false_type test( ... );
template<class T, typename... Args>
struct is_braces_constructible : decltype( test<T, Args...>( 0 ) ) {
};
template<class T, typename... Args>
inline constexpr bool is_braces_constructible_v = is_braces_constructible<T, Args...>{};


// Helper macros
#define AMP_PRINT( X ) _##X
#define AMP_COMMA() ,
#define AMP_ANY( X ) any_type
#define AMP_TO_TUPLE_BLOCK2( N )                                                           \
    if constexpr ( is_braces_constructible_v<type, AMP_SEQ_U( N, AMP_ANY, AMP_COMMA )> ) { \
        auto &&[AMP_SEQ_U( N, AMP_PRINT, AMP_COMMA )] = object;                            \
        return std::make_tuple( AMP_SEQ_U( N, AMP_PRINT, AMP_COMMA ) );                    \
    } else
#define AMP_TO_TUPLE_BLOCK( N ) AMP_SEQ_D( N, AMP_TO_TUPLE_BLOCK2, AMP_NULL )


// Convert an object (struct) to a tuple
// Note to view the code below run:
//    g++ -E to_tuple.h -I /projects/AMP/build/debug/AMP/include | clang-format | tail -n 50
template<class T>
auto to_tuple( T &&object ) noexcept
{
#ifndef DISABLE_TO_TUPLE
    using type = std::decay_t<T>;
    AMP_TO_TUPLE_BLOCK( 64 ) { return std::make_tuple(); }
#else
    AMP_ERROR( "to_tuple is not supported by compiler" )
#endif
}


} // namespace AMP


#endif
