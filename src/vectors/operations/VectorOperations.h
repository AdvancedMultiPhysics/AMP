#ifndef included_AMP_VectorOperations
#define included_AMP_VectorOperations

#include "AMP/vectors/Scalar.h"

#include <memory>
#include <vector>


namespace AMP::IO {
class RestartManager;
}


namespace AMP::LinearAlgebra {


class VectorData;


/**
 * \brief  A class used to hold vector operations
 * \details  VectorOperations is a class used to perform vector operations
 *    such as dot product, norms, etc.
 */
class VectorOperations
{
public:
    //! Destructor
    virtual ~VectorOperations() {}

    //! Get the type name
    virtual std::string VectorOpName() const = 0;

    //! Clone the operations
    virtual std::shared_ptr<VectorOperations> cloneOperations() const = 0;

    // VectorData versions
    /**
     * \brief  Set vector equal to x
     *      For Vectors, \f$z_i = x_i\f$.
     * \param[in] x         a vector
     * \param[out] z        a vector
     */
    virtual void copy( const VectorData &x, VectorData &z ) = 0;

    /**
     * \brief  Set vector equal to x (with different precision)
     *      For Vectors, \f$z_i = x_i\f$.
     * \param[in] x         a vector
     * \param[out] z        a vector
     */
    virtual void copyCast( const VectorData &x, VectorData &z ) = 0;

    /**
     *\brief Set vector entries (including ghosts) to zero
     *\details This is equivalent (but more efficient) to calling setToScalar ( 0.0 ) followed by a
     *     makeConsistent(SET)
     */
    virtual void zero( VectorData &x ) = 0;

    /**
     * \brief  Set all compenents of a vector to a scalar.
     *      For Vectors, the components of <em>z</em> are set to \f$\alpha\f$.
     * \param[in] alpha     scalar value
     * \param[out] z        a vector
     */
    virtual void setToScalar( const Scalar &alpha, VectorData &z ) = 0;

    /**
     * \brief Set data in this vector to random values
     * \param[out] x        a vector
     */
    virtual void setRandomValues( VectorData &x ) = 0;

    /**
     * \brief  Set vector equal to scaled input.
     *      For Vectors, \f$y_i = \alpha x_i\f$.
     * \param[in] alpha     a scalar
     * \param[in] x         a vector
     * \param[out] y        a vector
     */
    virtual void scale( const Scalar &alpha, const VectorData &x, VectorData &y ) = 0;

    /**
     * \brief  Scale a vector.
     *     For Vectors, \f$x_i = \alpha x_i\f$.
     * \param[in] alpha     a scalar
     * \param[in,out] x        a vector
     */
    virtual void scale( const Scalar &alpha, VectorData &x ) = 0;

    /**
     * \brief  Adds two vectors.
     *      For Vectors, \f$\mathit{this}_i = x_i + y_i\f$.
     * \param[in] x         Input vector x
     * \param[in] y         Input vector y
     * \param[out] z        Output vector z
     */
    virtual void add( const VectorData &x, const VectorData &y, VectorData &z ) = 0;

    /**
     * \brief Subtracts one vector from another.
     *     For Vectors, \f$z_i = x_i - y_i\f$
     * \param[in] x         Input vector x
     * \param[in] y         Input vector y
     * \param[out] z        Output vector z
     */
    virtual void subtract( const VectorData &x, const VectorData &y, VectorData &z ) = 0;

    /**
     * \brief Component-wise multiply one vector with another.
     *    For Vectors, \f$z_i = x_i  y_i\f$
     * \param[in] x         Input vector x
     * \param[in] y         Input vector y
     * \param[out] z        Output vector z
     */
    virtual void multiply( const VectorData &x, const VectorData &y, VectorData &z ) = 0;

    /**
     * \brief Component-wise divide one vector by another.
     *    For Vectors, \f$z_i = x_i / y_i\f$
     * \param[in] x         Input vector x
     * \param[in] y         Input vector y
     * \param[out] z        Output vector z
     */
    virtual void divide( const VectorData &x, const VectorData &y, VectorData &z ) = 0;

    /**
     * \brief Set this to the component-wise reciprocal of a vector.  \f$y_i = 1/x_i\f$.
     * \param x  a vector
     * \param y  a vector
     */
    virtual void reciprocal( const VectorData &x, VectorData &y ) = 0;

    /**
     * \brief Set a vector to be a linear combination of two vectors.
     *      \f$z_i = \alpha x_i + \beta y_i\f$.
     * \param[in] alpha     a scalar
     * \param[in] x         a vector
     * \param[in] beta      a scalar
     * \param[in] y         a vector
     * \param[out] z        Output vector z
     */
    virtual void linearSum( const Scalar &alpha,
                            const VectorData &x,
                            const Scalar &beta,
                            const VectorData &y,
                            VectorData &z ) = 0;

    /**
     * \brief Set this vector to alpha * x + y.  \f$z_i = \alpha x_i + y_i\f$.
     * \param[in] alpha    a scalar
     * \param[in] x        a vector
     * \param[in] y        a vector
     * \param[out] z       Output vector z
     */
    virtual void
    axpy( const Scalar &alpha, const VectorData &x, const VectorData &y, VectorData &z ) = 0;

    /**
     * \brief Set this vector alpha * x + this.
     *     \f$y_i = \alpha x_i + \beta y_i \f$
     * \param[in] alpha    a scalar
     * \param[in] beta     a scalar
     * \param[in] x        a vector
     * \param[in,out] y    Output vector z
     */
    virtual void
    axpby( const Scalar &alpha, const Scalar &beta, const VectorData &x, VectorData &y ) = 0;

    /**
     * \brief Set this to the component-wise absolute value of a vector.
     *     \f$z_i = |x_i|\f$.
     * \param[in] x        a vector
     * \param[out] z       Output vector z
     */
    virtual void abs( const VectorData &x, VectorData &z ) = 0;

    /**
     * \brief set vector to \f$x + \alpha \bar{1}\f$.
     * \details  for vectors, \f$y_i = x_i + \alpha\f$.
     * \param[in] x a vector
     * \param[in] alpha a scalar
     * \param[out] y       Output vector y
     */
    virtual void addScalar( const VectorData &x, const Scalar &alpha, VectorData &y ) = 0;

    /**
     * \brief modify vector to set \f$x_i = max(x_i, val)\f$.
     * \param[in] x a vector
     * \param[in] val a scalar
     */
    virtual void setMax( const Scalar &val, VectorData &x ) = 0;

    /**
     * \brief modify vector to set \f$x_i = min(x_i, val)\f$.
     * \param[in] x a vector
     * \param[in] val a scalar
     */
    virtual void setMin( const Scalar &val, VectorData &x ) = 0;

    /**
     * \brief Return the minimum value of the vector.  \f$\min_i x_i\f$.
     * \param[in] x        a vector
     */
    virtual Scalar min( const VectorData &x ) const;

    /**
     * \brief Return the maximum value of the vector.  \f$\max_i x_i\f$.
     * \param[in] x        a vector
     */
    virtual Scalar max( const VectorData &x ) const;

    /**
     * \brief Return the sum of the values of the vector.
     * \param[in] x        a vector
     */
    virtual Scalar sum( const VectorData &x ) const;

    /**
     * \brief Return the mean of the values of the vector.
     * \param[in] x        a vector
     */
    Scalar mean( const VectorData &x ) const;

    /**
     * \brief Return discrete @f$ L_1 @f$ -norm of this vector.
     * \details Returns \f[\sum_i |x_i|\f]
     * \param[in] x        a vector
     */
    virtual Scalar L1Norm( const VectorData &x ) const;

    /**
     * \brief Return discrete @f$ L_2 @f$ -norm of this vector.
     * \details Returns \f[\sqrt{\sum_i x_i^2}\f]
     * \param[in] x        a vector
     */
    virtual Scalar L2Norm( const VectorData &x ) const;

    /**
     * \brief Return the @f$ L_\infty @f$ -norm of this vector.
     * \details Returns \f[\max_i |x_i|\f]
     * \param[in] x        a vector
     */
    virtual Scalar maxNorm( const VectorData &x ) const;
    /**
     * \brief Returns the minimum of the quotient of two vectors:
     *    \f[\min_{i,y_i\neq0} x_i/x_i\f]
     * \param[in] x a vector
     * \param[in] y a vector
     * \return \f[\min_{i,y_i\neq0} x_i/y_i\f]
     */
    virtual Scalar minQuotient( const VectorData &x, const VectorData &y ) const;

    /**
     * \brief Return a weighted norm of a vector
     * \param[in] x a vector
     * \param[in] y a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_i x^2_iy^2_i}{n}}\f]
     */
    virtual Scalar wrmsNorm( const VectorData &x, const VectorData &y ) const;

    /**
     * \brief Return a weighted norm of a subset of a vector
     * \param[in] x a vector
     * \param[in] y a vector
     * \param[in] mask a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_{i,\mathit{mask}_i>0} x^2_iy^2_i}{n}}\f]
     */
    virtual Scalar
    wrmsNormMask( const VectorData &x, const VectorData &mask, const VectorData &y ) const;

    /**
     * \brief Return the dot product of this vector with the argument vector.
     * \details Returns \f[\sum_i x_i y_i\f]
     * \param[in] x         a vector
     * \param[in,out] y     a vector
     */
    virtual Scalar dot( const VectorData &x, const VectorData &y ) const;

    /**
     * \brief Check if two vectors are equal
     * \details Returns true if all values are equal within tolerance
     * \param[in] x        a vector
     * \param[in] y        a vector
     * \param[in] tol      tolerance to use
     */
    virtual bool equals( const VectorData &x, const VectorData &y, const Scalar &tol ) const;

    /**
     * \brief Return the local minimum value of the vector.  \f$\min_i x_i\f$.
     * \param[in] x a vector
     */
    virtual Scalar localMin( const VectorData &x ) const = 0;

    /**
     * \brief Return the local maximum value of the vector.  \f$\max_i x_i\f$.
     * \param[in] x a vector
     */
    virtual Scalar localMax( const VectorData &x ) const = 0;

    /**
     * \brief Return the local sum of the vector.
     */
    virtual Scalar localSum( const VectorData &x ) const = 0;

    /**
     * \brief Return local discrete @f$ L_1 @f$ -norm of this vector.
     * \details Returns \f[\sum_i |x_i|\f]
     * \param[in] x     a vector
     */
    virtual Scalar localL1Norm( const VectorData &x ) const = 0;

    /**
     * \brief Return local discrete @f$ L_2 @f$ -norm of this vector.
     * \details Returns \f[\sqrt{\sum_i x_i^2}\f]
     * \param[in] x     a vector
     */
    virtual Scalar localL2Norm( const VectorData &x ) const = 0;

    /**
     * \brief Return the local @f$ L_\infty @f$ -norm of this vector.
     * \details Returns \f[\max_i |x_i|\f]
     * \param[in] x     a vector
     */
    virtual Scalar localMaxNorm( const VectorData &x ) const = 0;

    /**
     * \brief Return the local dot product of this vector with the argument vector.
     * \details Returns \f[\sum_i x_i y_i\f]
     * \param[in] x     a vector
     * \param[in] y     a vector
     */
    virtual Scalar localDot( const VectorData &x, const VectorData &y ) const = 0;

    /**
     * \brief Returns the local minimum of the quotient of two vectors:
     *    \f[\min_{i,y_i\neq0} x_i/y_i\f]
     * \param[in] x a vector
     * \param[in] y a vector
     * \return \f[\min_{i,y_i\neq0} x_i/y_i\f]
     */
    virtual Scalar localMinQuotient( const VectorData &x, const VectorData &y ) const = 0;

    /**
     * \brief Return a weighted norm of a vector
     * \param[in] x a vector
     * \param[in] y a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_i x^2_i y^2_i}{n}}\f]
     */
    virtual Scalar localWrmsNorm( const VectorData &x, const VectorData &y ) const = 0;

    /**
     * \brief Return a weighted norm of a subset of a vector
     * \param[in] x a vector
     * \param[in] y a vector
     * \param[in] mask a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_{i,\mathit{mask}_i>0} y^2_iy^2_i}{n}}\f]
     */
    virtual Scalar
    localWrmsNormMask( const VectorData &x, const VectorData &mask, const VectorData &y ) const = 0;

    /**
     * \brief  Determine if the local portion of two vectors are equal using an absolute tolerance
     * \param[in] x a vector
     * \param[in] y a vector
     * \param[in] tol      Tolerance of comparison
     * \return  True iff \f$||y - x||_\infty < \mathit{tol}\f$
     */
    virtual bool
    localEquals( const VectorData &x, const VectorData &y, const Scalar &tol = 1e-6 ) const = 0;


    //! Get a unique id hash for the vector
    uint64_t getID() const;


public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    virtual void writeRestart( int64_t fid ) const;


protected:
    VectorOperations();

protected:
    uint64_t d_hash = 0;
};


} // namespace AMP::LinearAlgebra

#endif
