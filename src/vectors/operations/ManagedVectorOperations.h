#ifndef included_AMP_ManagedVectorOperations
#define included_AMP_ManagedVectorOperations

#include "AMP/vectors/operations/default/VectorOperationsDefault.h"


namespace AMP::LinearAlgebra {

/**
   \brief Class used to control data and kernels of various vector libraries
   \details  ManagedVectorOperations wraps the vector operations to call the
      native vector operations with the native data
*/
class ManagedVectorOperations : public VectorOperationsDefault<double>
{

public:
    ManagedVectorOperations(){};

public:
    //**********************************************************************
    // functions that operate on VectorData
    std::string VectorOpName() const override { return "ManagedVectorOperations"; }
    void copy( const VectorData &src, VectorData &dst ) override;
    void copyCast( const VectorData &src, VectorData &dst ) override;
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
    using VectorOperationsDefault::addScalar;
    using VectorOperationsDefault::axpby;
    using VectorOperationsDefault::axpy;
    using VectorOperationsDefault::divide;
    using VectorOperationsDefault::dot;
    using VectorOperationsDefault::equals;
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
    using VectorOperationsDefault::zero;
};


} // namespace AMP::LinearAlgebra


#endif
