#ifndef included_AMP_Scalar
    #define included_AMP_Scalar

    #include <any>
    #include <cstddef>
    #include <cstdint>

    #include "AMP/utils/typeid.h"

namespace AMP {


/**
 *  \class  Scalar
 *  \brief  Scalar is a class used to store a scalar variable that may be different types/precision
 */
class Scalar
{
public:
    //! Empty constructor
    inline Scalar();

    /**
     * \brief Construct a Scalar value
     * \details Default constructor allowing implicit conversion to double
     * \param[in] x         Input scalar
     */
    template<class TYPE>
    inline Scalar( const TYPE &x );

    //! Copy constructor
    Scalar( const Scalar & ) = default;

    //! Move constructor
    Scalar( Scalar && ) = default;

    //! Assignment operator
    Scalar &operator=( const Scalar & ) = default;

    //! Move operator
    Scalar &operator=( Scalar && ) = default;

    /**
     * \brief Construct a scalar value
     * \param[in] tol       Tolerance to allow for the conversion (absolute error)
     * \return              Returns the scalar value
     */
    template<class TYPE>
    inline TYPE get( double tol = Scalar::getTol<TYPE>() ) const;

    //! Return true if the type is a floating point type
    inline bool is_floating_point() const { return d_type == 'f'; }

    //! Return true if the type is a integer point type
    inline bool is_integral() const { return d_type == 'i'; }

    //! Return true if the type is a complex type
    inline bool is_complex() const { return d_type == 'c'; }

    //! Return the storage type
    inline const auto &type() const { return d_data.type(); }

    //! Check if we are storing a value
    inline bool has_value() const noexcept { return d_data.has_value(); }

    //! Get default tolerance
    template<class TYPE>
    static constexpr double getTol();

public: // Comparison operators
    bool operator==( const Scalar &rhs ) const;
    bool operator!=( const Scalar &rhs ) const;
    bool operator>( const Scalar &rhs ) const;
    bool operator>=( const Scalar &rhs ) const;
    bool operator<( const Scalar &rhs ) const;
    bool operator<=( const Scalar &rhs ) const;

public: // Overload some typecast operators
    template<class TYPE>
    inline explicit operator TYPE() const
    {
        return get<TYPE>();
    }

public: // Math functions
    Scalar abs() const;
    Scalar sqrt() const;

private: // Helper functions
    template<class TYPE>
    inline void store( const TYPE &x );

private: // Internal data
    char d_type;
    uint32_t d_hash;
    std::any d_data;
};


} // namespace AMP

#endif


#include "AMP/vectors/Scalar.hpp"
