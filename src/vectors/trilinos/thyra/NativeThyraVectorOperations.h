#ifndef included_AMP_NativeThyraVectorOperations
#define included_AMP_NativeThyraVectorOperations

#include "AMP/vectors/Vector.h"
#include "AMP/vectors/operations/VectorOperationsDefault.h"
#include "AMP/vectors/trilinos/thyra/ThyraVector.h"

namespace AMP {
namespace LinearAlgebra {


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
    void setToScalar( double alpha, VectorData &z ) override;
    void setRandomValues( VectorData &x ) override;
    void setRandomValues( RNG::shared_ptr rng, VectorData &x ) override;
    void copy( const VectorData &x, VectorData &z ) override;
    void scale( double alpha, const VectorData &x, VectorData &y ) override;
    void scale( double alpha, VectorData &x ) override;
    void add( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void subtract( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void multiply( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void divide( const VectorData &x, const VectorData &y, VectorData &z ) override;
    void reciprocal( const VectorData &x, VectorData &y ) override;
    void linearSum( double alpha,
                    const VectorData &x,
                    double beta,
                    const VectorData &y,
                    VectorData &z ) override;
    void axpy( double alpha, const VectorData &x, const VectorData &y, VectorData &z ) override;
    void axpby( double alpha, double beta, const VectorData &x, VectorData &y ) override;
    void abs( const VectorData &x, VectorData &z ) override;
    //    void addScalar( const VectorData &x, double alpha_in, VectorData &y ) override;

    double min( const VectorData &x ) const override;
    double max( const VectorData &x ) const override;
    double L1Norm( const VectorData &x ) const override;
    double L2Norm( const VectorData &x ) const override;
    double maxNorm( const VectorData &x ) const override;
    double dot( const VectorData &x, const VectorData &y ) const override;

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


} // namespace LinearAlgebra
} // namespace AMP

#endif
