#ifndef included_AMP_ManagedVectorOperations
#define included_AMP_ManagedVectorOperations

#include "AMP/vectors/operations/VectorOperationsDefault.h"


namespace AMP {
namespace LinearAlgebra {

/**
   \brief Class used to control data and kernels of various vector libraries
   \details  A ManagedVector will take an engine and create a buffer, if
   necessary.

   A ManagedVector has two pointers: data and engine.  If the data pointer
   is null, then the engine is assumed to have the data.
*/
class ManagedVectorOperations : virtual public VectorOperationsDefault<double>
{

public:
    ManagedVectorOperations(){};

public:
    //**********************************************************************
    // functions that operate on VectorData
    void copy( const VectorData &src, VectorData &dst ) override;
    void setToScalar( double alpha, VectorData &z ) override;
    void setRandomValues( VectorData &x ) override;
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

    double min( const VectorData &x ) const override;
    double max( const VectorData &x ) const override;
    double dot( const VectorData &x, const VectorData &y ) const override;
    double L1Norm( const VectorData &x ) const override;
    double L2Norm( const VectorData &x ) const override;
    double maxNorm( const VectorData &x ) const override;

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


} // namespace LinearAlgebra
} // namespace AMP


#endif