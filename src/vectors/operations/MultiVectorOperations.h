#ifndef included_AMP_MultiVectorOperations
#define included_AMP_MultiVectorOperations


#include "AMP/vectors/operations/VectorOperations.h"


namespace AMP {
namespace LinearAlgebra {

class MultiVectorData;
  
/**
 * \brief  A set of vector operations for multivectors
 * \details MultiVectorOperations impliments a default set of
 *    vector operations for multivectors.
 */
class MultiVectorOperations : virtual public VectorOperations
{
public:
    // Constructor
    MultiVectorOperations() {}

    //! Destructor
    virtual ~MultiVectorOperations() {}

    //! Clone the operations
    std::shared_ptr<VectorOperations> cloneOperations() const override;

    /**
     * \brief  Set vector equal to x
     *      For Vectors, \f$\mathit{this}_i = x_i\f$.
     * \param[in] x         a vector
     */
    void copy( const VectorOperations &x ) override;

    /**
     *\brief Set vector entries (including ghosts) to zero
     *\details This is equivalent (but more efficient) to calling setToScalar ( 0.0 ) followed by a
     *makeConsistent(SET)
     */
    void zero() override;

    /**
     * \param  alpha a scalar double
     * \brief  Set all compenents of a vector to a scalar.
     * For Vectors, the components of <em>this</em> are set to \f$\alpha\f$.
     */
    void setToScalar( double alpha ) override;

    /**
     * \brief Set data in this vector to random values on [0,1).
     */
    void setRandomValues( void ) override;

    /**
     * \brief Set data in this vector to random values using
     * a particular generator
     * \param[in]  rng  The generator to use.
     */
    void setRandomValues( RNG::shared_ptr rng ) override;

    /**
     * \param  alpha  a scalar double
     * \param  x  a vector
     * \brief  Set vector equal to scaled input.
     * For Vectors, \f$\mathit{this}_i = \alpha x_i\f$.
     */
    void scale( double alpha, const VectorOperations &x ) override;

    /**
     * \param  alpha  a scalar double
     *
     * \brief  Scale a vector.
     * For Vectors, \f$\mathit{this}_i = \alpha\mathit{this}_i\f$.
     */
    void scale( double alpha ) override;

    /**
     * \param  x  a vector
     * \param  y  a vector
     * \brief  Adds two vectors.
     * For Vectors, \f$\mathit{this}_i = x_i + y_i\f$.
     */
    void add( const VectorOperations &x, const VectorOperations &y ) override;

    /**
     * \param x  a vector
     * \param y  a vector
     * \brief Subtracts one vector from another.
     * For Vectors, \f$\mathit{this}_i = x_i - y_i\f$
     */
    void subtract( const VectorOperations &x, const VectorOperations &y ) override;

    /**
     * \param x  a vector
     * \param y  a vector
     * \brief Component-wise multiply one vector with another.
     * For Vectors, \f$\mathit{this}_i = x_i  y_i\f$
     */
    void multiply( const VectorOperations &x, const VectorOperations &y ) override;

    /**
     * \param x  a vector
     * \param y  a vector
     * \brief Component-wise divide one vector by another.
     * For Vectors, \f$\mathit{this}_i = x_i / y_i\f$
     */
    void divide( const VectorOperations &x, const VectorOperations &y ) override;

    /**
     * \param x  a vector
     * \brief Set this to the component-wise reciprocal of a vector.  \f$\mathit{this}_i =
     * 1/x_i\f$.
     */
    void reciprocal( const VectorOperations &x ) override;


    /**
     * \param alpha a scalar
     * \param x a vector
     * \param beta a scalar
     * \param y a vector
     * \brief Set a vector to be a linear combination of two vectors.
     *  \f$\mathit{this}_i = \alpha x_i + \beta y_i\f$.
     */
    void linearSum( double alpha,
                    const VectorOperations &x,
                    double beta,
                    const VectorOperations &y ) override;

    /**
     * \param alpha a scalar
     * \param x a vector
     * \param y a vector
     * \brief Set this vector to alpha * x + y.  \f$\mathit{this}_i = \alpha x_i + y_i\f$.
     */
    void axpy( double alpha, const VectorOperations &x, const VectorOperations &y ) override;

    /**
     * \param alpha a scalar
     * \param beta a scalar
     * \param x  a vector
     * \brief Set this vector alpha * x + this.
     * \f$\mathit{this}_i = \alpha x_i + \beta \mathit{this}_i \f$
     */
    void axpby( double alpha, double beta, const VectorOperations &x ) override;

    /**
     * \param x a vector
     * \brief Set this to the component-wise absolute value of a vector.
     * \f$\mathit{this}_i = |x_i|\f$.
     */
    void abs( const VectorOperations &x ) override;

    /**
     * \brief Return the local minimum value of the vector.  \f$\min_i \mathit{this}_i\f$.
     */
    double localMin( void ) const override;

    /**
     * \brief Return the local maximum value of the vector.  \f$\max_i \mathit{this}_i\f$.
     */
    double localMax( void ) const override;

    /**
     * \brief Return local discrete @f$ L_1 @f$ -norm of this vector.
     * \details Returns \f[\sum_i |\mathit{this}_i|\f]
     */
    double localL1Norm( void ) const override;

    /**
     * \brief Return local discrete @f$ L_2 @f$ -norm of this vector.
     * \details Returns \f[\sqrt{\sum_i \mathit{this}_i^2}\f]
     */
    double localL2Norm( void ) const override;

    /**
     * \brief Return the local @f$ L_\infty @f$ -norm of this vector.
     * \details Returns \f[\max_i |\mathit{this}_i|\f]
     */
    double localMaxNorm( void ) const override;

    /**
     * \param[in] x a vector
     * \brief Return the local dot product of this vector with the argument vector.
     * \details Returns \f[\sum_i x_i\mathit{this}_i\f]
     */
    double localDot( const VectorOperations &x ) const override;

    /**
     * \brief  Determine if the local portion of two vectors are equal using an absolute tolerance
     * \param[in] rhs      Vector to compare to
     * \param[in] tol      Tolerance of comparison
     * \return  True iff \f$||\mathit{rhs} - x||_\infty < \mathit{tol}\f$
     */
    bool localEquals( const VectorOperations &rhs, double tol = 0.000001 ) const override;

    /**
     * \brief set vector to \f$x + \alpha \bar{1}\f$.
     * \param[in] x a vector
     * \param[in] alpha a scalar
     * \details  for vectors, \f$\mathit{this}_i = x_i + \alpha\f$.
     */
    void addScalar( const VectorOperations &x, double alpha ) override;


private:
    /**
     * \brief Returns the local minimum of the quotient of two vectors:
     *    \f[\min_{i,y_i\neq0} x_i/\mathit{this}_i\f]
     * \param[in] x a vector
     * \return \f[\min_{i,y_i\neq0} x_i/\mathit{this}_i\f]
     */
    double localMinQuotient( const VectorOperations &x ) const override;

    /**
     * \brief Return a weighted norm of a vector
     * \param[in] x a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_i x^2_i \mathit{this}^2_i}{n}}\f]
     */
    double localWrmsNorm( const VectorOperations &x ) const override;

    /**
     * \brief Return a weighted norm of a subset of a vector
     * \param[in] x a vector
     * \param[in] mask a vector
     * \return \f[\sqrt{\frac{\displaystyle \sum_{i,\mathit{mask}_i>0}
     * \mathit{this}^2_iy^2_i}{n}}\f]
     */
    double localWrmsNormMask( const VectorOperations &x,
                              const VectorOperations &mask ) const override;

protected:
//  static function that operate on VectorData
    static VectorData *getVectorDataComponent( VectorData &x, size_t i );
    static MultiVectorData *getMultiVectorData( VectorData &x );
    static const MultiVectorData *getMultiVectorData( const VectorData &x );
#if 0
    static void zero( VectorData &z );
    static void setToScalar( double alpha, VectorData &z );
    static void setRandomValues( VectorData &x );    
    static void setRandomValues( RNG::shared_ptr rng, VectorData &x );    
    static void copy( const VectorData &x, VectorData &z );
    static void scale( double alpha, const VectorData &x, VectorData &y );
    static void scale( double alpha, VectorData &x );
    static void add( const VectorData &x, const VectorData &y, VectorData &z );
    static void subtract( const VectorData &x, const VectorData &y, VectorData &z );
    static void multiply( const VectorData &x, const VectorData &y, VectorData &z );
    static void divide( const VectorData &x, const VectorData &y, VectorData &z );
    static void reciprocal( const VectorData &x, VectorData &y );
    static void linearSum( double alpha,
			   const VectorData &x,
			   double beta,
			   const VectorData &y,
			   VectorData &z);
    static void axpy( double alpha, const VectorData &x, const VectorData &y, VectorData &z );
    static void axpby( double alpha, double beta, const VectorData &x, VectorData &y );
    static void abs( const VectorData &x, VectorData &z );
    static void addScalar( const VectorData &x, double alpha_in, VectorData &y );

    static double localMin( const VectorData &x );
    static double localMax( const VectorData &x );
    static double localL1Norm( const VectorData &x );
    static double localL2Norm( const VectorData &x  );
    static double localMaxNorm( const VectorData &x );
    static double localDot( const VectorData &x, const VectorData &y );
    static double localMinQuotient( const VectorData &x, const VectorData &y );
    static double localWrmsNorm( const VectorData &x, const VectorData &y );
    static double localWrmsNormMask( const VectorData &x, const VectorData &mask, const VectorData &y );
    static bool   localEquals( const VectorData &x, const VectorData &y, double tol = 0.000001 );
#endif
protected:
    // Internal data
    std::vector<VectorOperations *> d_operations;


public: // Pull VectorOperations into the current scope
    using VectorOperations::abs;
    using VectorOperations::add;
    using VectorOperations::addScalar;
    using VectorOperations::axpby;
    using VectorOperations::axpy;
    using VectorOperations::divide;
    using VectorOperations::dot;
    using VectorOperations::equals;
    using VectorOperations::linearSum;
    using VectorOperations::minQuotient;
    using VectorOperations::multiply;
    using VectorOperations::reciprocal;
    using VectorOperations::scale;
    using VectorOperations::setRandomValues;
    using VectorOperations::subtract;
    using VectorOperations::wrmsNorm;
    using VectorOperations::wrmsNormMask;
};


} // namespace LinearAlgebra
} // namespace AMP

#endif
