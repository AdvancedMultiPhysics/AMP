#ifndef included_AMP_VectorOperationsOpenMP
#define included_AMP_VectorOperationsOpenMP


#include "AMP/vectors/operations/VectorOperations.h"


namespace AMP::LinearAlgebra {


/**
 * \brief  A default set of vector operations
 * \details VectorOperationsOpenMP impliments a default set of
 *    vector operations on the CPU.
 */
template<typename TYPE = double>
class VectorOperationsOpenMP : public VectorOperations
{
public:
    // Constructor
    VectorOperationsOpenMP() {}

    //! Destructor
    virtual ~VectorOperationsOpenMP() {}

    //! Clone the operations
    std::shared_ptr<VectorOperations> cloneOperations() const override;

public:
    //  function that operate on VectorData
    std::string VectorOpName() const override { return "VectorOperationsOpenMP"; }
    void zero( VectorData & ) override;
    void setToScalar( const Scalar &, VectorData & ) override;
    void setRandomValues( VectorData & ) override;
    void copy( const VectorData &, VectorData & ) override;
    void copyCast( const VectorData &x, VectorData &z ) override;
    void scale( const Scalar &, const VectorData &, VectorData & ) override;
    void scale( const Scalar &, VectorData & ) override;
    void add( const VectorData &, const VectorData &, VectorData & ) override;
    void subtract( const VectorData &, const VectorData &, VectorData & ) override;
    void multiply( const VectorData &, const VectorData &, VectorData & ) override;
    void divide( const VectorData &, const VectorData &, VectorData & ) override;
    void reciprocal( const VectorData &, VectorData & ) override;
    void linearSum( const Scalar &,
                    const VectorData &,
                    const Scalar &,
                    const VectorData &,
                    VectorData & ) override;
    void axpy( const Scalar &, const VectorData &, const VectorData &, VectorData & ) override;
    void axpby( const Scalar &, const Scalar &, const VectorData &, VectorData & ) override;
    void abs( const VectorData &, VectorData & ) override;
    void addScalar( const VectorData &, const Scalar &_in, VectorData & ) override;

    void setMax( const Scalar &val, VectorData &x ) override;
    void setMin( const Scalar &val, VectorData &x ) override;

    Scalar localMin( const VectorData & ) const override;
    Scalar localMax( const VectorData & ) const override;
    Scalar localSum( const VectorData & ) const override;
    Scalar localL1Norm( const VectorData & ) const override;
    Scalar localL2Norm( const VectorData & ) const override;
    Scalar localMaxNorm( const VectorData & ) const override;
    Scalar localDot( const VectorData &, const VectorData & ) const override;
    Scalar localMinQuotient( const VectorData &, const VectorData & ) const override;
    Scalar localWrmsNorm( const VectorData &, const VectorData & ) const override;
    Scalar localWrmsNormMask( const VectorData &,
                              const VectorData &mask,
                              const VectorData & ) const override;
    bool
    localEquals( const VectorData &, const VectorData &, const Scalar &tol = 1e-6 ) const override;

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
    using VectorOperations::setMax;
    using VectorOperations::setMin;
    using VectorOperations::setRandomValues;
    using VectorOperations::subtract;
    using VectorOperations::wrmsNorm;
    using VectorOperations::wrmsNormMask;
};


} // namespace AMP::LinearAlgebra


#endif
