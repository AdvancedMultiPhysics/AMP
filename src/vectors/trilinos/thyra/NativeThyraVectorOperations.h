#ifndef included_AMP_NativeThyraVectorOperations
#define included_AMP_NativeThyraVectorOperations

#include "AMP/vectors/Vector.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"
#include "AMP/vectors/trilinos/thyra/ThyraVector.h"

namespace AMP::LinearAlgebra {


/** \class NativeThyraVectorOperations
 * \brief An AMP Vector that uses Thyra for parallel data management, linear algebra,
 * etc.
 * \details  This is an AMP wrapper to Thyra.  This is different from ManagedThyraVector
 * in that this class does not replace calls to Vec*.  Rather, it wraps these calls.
 * This class is used when Thyra is chosen as the default linear algebra engine.
 *
 * This class is not to be used directly, just through base class interfaces.
 * \see ThyraVector
 * \see ManagedThyraVector
 */
class NativeThyraVectorOperations : public VectorOperationsDefault<double>
{
public:
    NativeThyraVectorOperations() : VectorOperationsDefault<double>(){};
    virtual ~NativeThyraVectorOperations();
    //  function that operate on VectorData
    void setToScalar( const Scalar &alpha, VectorData &z ) override;
    void setRandomValues( VectorData &x ) override;
    void copy( const VectorData &x, VectorData &z ) override;
    void scale( const Scalar &alpha, const VectorData &x, VectorData &y ) override;
    void scale( const Scalar &alpha, VectorData &x ) override;
    void add( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void subtract( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void multiply( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void divide( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void reciprocal( const VectorData &x, VectorData &y ) override;
    void linearSum( const Scalar &alpha,
                    const VectorData &x,
                    const Scalar &beta,
                    const VectorData &y,
                    VectorData &z ) override;
    void
    axpy( const Scalar &alpha, const VectorData &x, const VectorData &y, VectorData &z ) override;
    void
    axpby( const Scalar &alpha, const Scalar &beta, const VectorData &x, VectorData &y ) override;
    void abs( const VectorData &x, VectorData &z ) override;

    Scalar min( const VectorData &x ) const override;
    Scalar max( const VectorData &x ) const override;
    Scalar L1Norm( const VectorData &x ) const override;
    Scalar L2Norm( const VectorData &x ) const override;
    Scalar maxNorm( const VectorData &x ) const override;
    Scalar dot( const VectorData &x, const VectorData &y ) const override;

private:
    static Teuchos::RCP<const Thyra::VectorBase<double>> getThyraVec( const VectorData &v );
    static Teuchos::RCP<Thyra::VectorBase<double>> getThyraVec( VectorData &v );

public: // Pull VectorOperations into the current scope
    using VectorOperationsDefault::abs;
    using VectorOperationsDefault::add;
    using VectorOperationsDefault::axpby;
    using VectorOperationsDefault::axpy;
    using VectorOperationsDefault::divide;
    using VectorOperationsDefault::dot;
    using VectorOperationsDefault::L1Norm;
    using VectorOperationsDefault::L2Norm;
    using VectorOperationsDefault::linearSum;
    using VectorOperationsDefault::max;
    using VectorOperationsDefault::maxNorm;
    using VectorOperationsDefault::min;
    using VectorOperationsDefault::minQuotient;
    using VectorOperationsDefault::multiply;
    using VectorOperationsDefault::reciprocal;
    using VectorOperationsDefault::scale;
    using VectorOperationsDefault::setRandomValues;
    using VectorOperationsDefault::subtract;
    using VectorOperationsDefault::wrmsNorm;
    using VectorOperationsDefault::wrmsNormMask;
};


} // namespace AMP::LinearAlgebra

#endif
