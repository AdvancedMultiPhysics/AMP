#ifndef included_AMP_EpetraVectorOperations
#define included_AMP_EpetraVectorOperations


#include "AMP/vectors/operations/default/VectorOperationsDefault.h"

DISABLE_WARNINGS
#include <Epetra_Vector.h>
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/** \class EpetraVectorOperations
 * \brief A linear algebra engine that uses Epetra
 * \details  Use the Epetra implementation of the L1 BLAS routines.  Unlike other
 * libraries, it is very difficult to separate the data from the engine.  For this
 * reason, the EpetraVectorEngine contains the Epetra_Vector to operate on.
 */
class EpetraVectorOperations : public VectorOperationsDefault<double>
{
public:
    // Constructor
    EpetraVectorOperations() {}

public:
    void setToScalar( const Scalar &alpha, VectorData &z ) override;
    void setRandomValues( VectorData &x ) override;
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
    Scalar dot( const VectorData &x, const VectorData &y ) const override;
    Scalar L1Norm( const VectorData &x ) const override;
    Scalar L2Norm( const VectorData &x ) const override;
    Scalar maxNorm( const VectorData &x ) const override;

public: // Pull VectorOperations into the current scope
    using VectorOperationsDefault::abs;
    using VectorOperationsDefault::add;
    using VectorOperationsDefault::axpby;
    using VectorOperationsDefault::axpy;
    using VectorOperationsDefault::divide;
    using VectorOperationsDefault::dot;
    using VectorOperationsDefault::linearSum;
    using VectorOperationsDefault::minQuotient;
    using VectorOperationsDefault::multiply;
    using VectorOperationsDefault::reciprocal;
    using VectorOperationsDefault::scale;
    using VectorOperationsDefault::setRandomValues;
    using VectorOperationsDefault::subtract;
    using VectorOperationsDefault::wrmsNorm;
    using VectorOperationsDefault::wrmsNormMask;

    using VectorOperationsDefault::localDot;
    using VectorOperationsDefault::localL1Norm;
    using VectorOperationsDefault::localL2Norm;
    using VectorOperationsDefault::localMax;
    using VectorOperationsDefault::localMaxNorm;
    using VectorOperationsDefault::localMin;

    using VectorOperationsDefault::L1Norm;
    using VectorOperationsDefault::L2Norm;
    using VectorOperationsDefault::max;
    using VectorOperationsDefault::maxNorm;
    using VectorOperationsDefault::min;

protected:
};


} // namespace AMP::LinearAlgebra


#endif
