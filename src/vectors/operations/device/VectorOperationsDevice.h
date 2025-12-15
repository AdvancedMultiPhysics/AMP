#ifndef included_AMP_VectorOperationsDevice
#define included_AMP_VectorOperationsDevice


#include "AMP/vectors/operations/default/VectorOperationsDefault.h"


namespace AMP {
namespace LinearAlgebra {


/**
 * \brief  A default set of vector operations
 * \details VectorOperationsDevice impliments a default set of
 *    vector operations on the CPU.
 */
template<typename TYPE = double>
class VectorOperationsDevice : public VectorOperations
{
public:
    // Constructor
    VectorOperationsDevice() {}

    //! Destructor
    virtual ~VectorOperationsDevice();

    //! Clone the operations
    virtual std::shared_ptr<VectorOperations> cloneOperations() const override;

public:
    //  functions that operate on VectorData
    std::string VectorOpName() const override { return "VectorOperationsDevice"; }
    void zero( VectorData &z ) override;
    void setToScalar( const Scalar &alpha, VectorData &z ) override;
    void setRandomValues( VectorData &x ) override;
    void copy( const VectorData &x, VectorData &z ) override;
    void copyCast( const VectorData &x, VectorData &z ) override;
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
    void addScalar( const VectorData &x, const Scalar &alpha_in, VectorData &y ) override;

    void setMax( const Scalar &val, VectorData &x ) override;
    void setMin( const Scalar &val, VectorData &x ) override;

    Scalar localMin( const VectorData & ) const override;
    Scalar localMax( const VectorData & ) const override;
    Scalar localSum( const VectorData & ) const override;
    Scalar localL1Norm( const VectorData & ) const override;
    Scalar localL2Norm( const VectorData & ) const override;
    Scalar localMaxNorm( const VectorData & ) const override;
    Scalar localDot( const VectorData &x, const VectorData &y ) const override;
    Scalar localMinQuotient( const VectorData &x, const VectorData &y ) const override;
    Scalar localWrmsNorm( const VectorData &x, const VectorData &y ) const override;
    Scalar localWrmsNormMask( const VectorData &x,
                              const VectorData &mask,
                              const VectorData &y ) const override;
    bool localEquals( const VectorData &x,
                      const VectorData &y,
                      const Scalar &tol = 1e-6 ) const override;

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

private:
    VectorOperationsDefault<TYPE> *getDefaultOps( void );
    const VectorOperationsDefault<TYPE> *getDefaultOps( void ) const;
    mutable VectorOperationsDefault<TYPE> *d_default_ops = nullptr;
};


} // namespace LinearAlgebra
} // namespace AMP


#endif
