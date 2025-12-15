// clang-format off
#include <cassert>
#include <type_traits>


struct any_type { template<class T> constexpr operator T(); };
template<class T, typename... Args> decltype( void( T{ std::declval<Args>()... } ), std::true_type() ) test( int );
template<class T, typename... Args> std::false_type test( ... );
template<class T, typename... Args> struct is_braces_constructible : decltype( test<T, Args...>( 0 ) ) {};


int main()
{
    struct myClass {
        int a;
        double b;
        float c;
    };

    [[maybe_unused]] myClass tmp = { 0, 0, 0 };

    static_assert( is_braces_constructible<myClass, any_type, any_type>::value );
    static_assert( is_braces_constructible<myClass, any_type, any_type, any_type>::value );
    static_assert( !is_braces_constructible<myClass, any_type, any_type, any_type, any_type>::value );
}

