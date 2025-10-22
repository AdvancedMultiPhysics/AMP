// This file contains helper functions and interfaces for reading/writing HDF5
#ifndef included_AMP_TypeTraits
#define included_AMP_TypeTraits

#include <memory>
#include <utility>
#include <vector>

#include "AMP/utils/ArraySize.h" // Forward declare Array


namespace AMP {


//! Checks whether T is a shared_ptr
template<typename T>
struct is_shared_ptr : std::false_type {
};
template<typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {
};

//! Checks whether T is a unique_ptr
template<typename T>
struct is_unique_ptr : std::false_type {
};
template<typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {
};

//! Checks whether T is a std::vector
template<typename T>
struct is_vector : std::false_type {
};
template<typename T>
struct is_vector<std::vector<T>> : std::true_type {
};


//! Checks whether T is a std::vector
template<typename T>
struct is_array : std::false_type {
};
template<typename T, std::size_t N>
struct is_array<std::array<T, N>> : std::true_type {
};


// Function to test if a type is a std::pair
template<typename>
struct is_pair : std::false_type {
};
template<typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {
};


//! Checks whether T is an AMP::Array
template<typename T>
struct is_Array : std::false_type {
};
template<typename T>
struct is_Array<Array<T>> : std::true_type {
};


//! Checks whether T is convertible to a string
template<typename T>
struct is_string : std::false_type {
};
template<>
struct is_string<std::string> : std::true_type {
};
template<>
struct is_string<std::string_view> : std::true_type {
};
template<>
struct is_string<char *> : std::true_type {
};
template<std::size_t N>
struct is_string<char[N]> : std::true_type {
};


//! Checks whether T has a size() function
template<typename T>
struct has_size {
private:
    typedef std::true_type yes;
    typedef std::false_type no;
    template<typename U>
    static auto test( int ) -> decltype( std::declval<U>().size() == 1, yes() );
    template<typename>
    static no test( ... );

public:
    static constexpr bool value = std::is_same<decltype( test<T>( 0 ) ), yes>::value;
};


//! Checks whether T has a begin()/end() function
using std::begin;
template<typename T, typename = void>
struct has_begin : std::false_type {
};
template<typename T>
struct has_begin<T, decltype( void( std::begin( std::declval<T &>() ) ) )> : std::true_type {
};
template<typename T, typename = void>
struct has_end : std::false_type {
};
template<typename T>
struct has_end<T, decltype( void( std::end( std::declval<T &>() ) ) )> : std::true_type {
};
template<typename T, typename = void>
struct has_empty : std::false_type {
};
template<typename T>
struct has_empty<T, decltype( void( std::empty( std::declval<T &>() ) ) )> : std::true_type {
};


//! Checks whether T is an initializer_list
template<typename T>
struct is_initializer_list : std::false_type {
};
template<typename T>
struct is_initializer_list<std::initializer_list<T>> : std::true_type {
};

//! Checks whether T is complex
template<class T>
struct is_complex : public std::false_type {
};
template<class T>
struct is_complex<const T> : public is_complex<T> {
};
template<class T>
struct is_complex<volatile const T> : public is_complex<T> {
};
template<class T>
struct is_complex<volatile T> : public is_complex<T> {
};
template<class T>
struct is_complex<std::complex<T>> : public std::true_type {
};


// Helper functions
template<class T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;
template<class T>
inline constexpr bool is_unique_ptr_v = is_unique_ptr<T>::value;
template<class T>
inline constexpr bool is_vector_v = is_vector<T>::value;
template<class T>
inline constexpr bool is_array_v = is_array<T>::value;
template<class T>
inline constexpr bool is_Array_v = is_Array<T>::value;
template<class T>
inline constexpr bool is_pair_v = is_pair<T>::value;
template<class T>
inline constexpr bool is_string_v = is_string<T>::value;
template<class T>
inline constexpr bool has_size_v = has_size<T>::value;
template<class T>
inline constexpr bool has_begin_v = has_begin<T>::value;
template<class T>
inline constexpr bool has_end_v = has_end<T>::value;
template<class T>
inline constexpr bool has_empty_v = has_empty<T>::value;
template<class T>
inline constexpr bool is_container_v = has_begin_v<T> &&has_end_v<T> &&has_empty_v<T>;
template<class T>
inline constexpr bool is_initializer_list_v = is_initializer_list<T>::value;
template<class T>
inline constexpr bool is_complex_v = is_complex<T>::value;


//! Checks whether two types are both integers of the same size and sign
template<class T1, class T2>
inline constexpr bool is_same_int_v = std::is_integral_v<T1> &&std::is_integral_v<T2> &&
                                      (std::is_signed_v<T1> ==
                                       std::is_signed_v<T2>) &&( sizeof( T1 ) == sizeof( T2 ) );


// Remove const and reference
template<typename T>
using remove_cvref_t = typename std::remove_cv_t<typename std::remove_reference_t<T>>;


// Return the number of arguments in a given function
template<typename T>
struct arg_count;
template<typename T, typename... Args>
struct arg_count<T( Args... )> {
    static constexpr std::size_t value = sizeof...( Args );
};
template<class T>
inline constexpr bool arg_count_v = arg_count<T>::value;


} // namespace AMP


#endif
