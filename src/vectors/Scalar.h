#ifndef included_AMP_Scalar
#define included_AMP_Scalar

#include <any>


namespace AMP {


/**
 *  \class  Scalar
 *  \brief  Scalar is a class used to store a scalar variable that may be different types/precision
 */
class Scalar
{
public:
    /**
     * \brief Construct a sclar value
     * \param[in] x         Input scalar
     */
    template<class TYPE>
    explicit Scalar( TYPE x );

    /**
     * \brief Construct a sclar value
     * \param[in] tol       Tolerance to allow for the conversion (absolute error)
     * \return              Returns the scalar value
     */
    template<class TYPE>
    TYPE get( double tol = 0 ) const;

    //! Return true if the type is a floating point type
    inline bool is_floating_point() const { return d_type == 'f'; }

    //! Return true if the type is a integer point type
    inline bool is_integral() const { return d_type == 'i'; }

    //! Return true if the type is a complex type
    inline bool is_complex() const { return d_type == 'c'; }

    //! Return the storage type
    inline const auto &type() const { return d_data.type(); }

private:
    char d_type;
    std::any d_data;
};


} // namespace AMP

#endif


#include "AMP/vectors/Scalar.hpp"