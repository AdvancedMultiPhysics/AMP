#ifndef included_AMP_MultiVectorOperations
#define included_AMP_MultiVectorOperations


#include "AMP/vectors/operations/VectorOperations.h"


namespace AMP::LinearAlgebra {

class MultiVectorData;

/**
 * \brief  A set of vector operations for multivectors
 * \details MultiVectorOperations impliments a default set of
 *    vector operations for multivectors.
 */
class MultiVectorOperations : public VectorOperations
{
public:
    // Constructor
    MultiVectorOperations() : VectorOperations() {}

    // Create from a single vector operations
    MultiVectorOperations( std::shared_ptr<VectorOperations> op );

    //! Destructor
    virtual ~MultiVectorOperations() {}

    //! Clone the operations
    std::shared_ptr<VectorOperations> cloneOperations() const override;

private:
    //  static function that operate on VectorData
    static VectorData *getVectorDataComponent( VectorData &, size_t );
    static const VectorData *getVectorDataComponent( const VectorData &, size_t );
    static MultiVectorData *getMultiVectorData( VectorData & );
    static const MultiVectorData *getMultiVectorData( const VectorData & );
    static size_t getVectorDataSize( const VectorData & );

public:
    std::string VectorOpName() const override { return "MultiVectorOperations"; }
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
    void addScalar( const VectorData &, const Scalar &, VectorData & ) override;

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
    Scalar
    localWrmsNormMask( const VectorData &, const VectorData &, const VectorData & ) const override;
    bool
    localEquals( const VectorData &, const VectorData &, const Scalar &tol = 1e-6 ) const override;

    void resetVectorOperations( std::vector<std::shared_ptr<VectorOperations>> ops );

public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t fid ) const override;
    MultiVectorOperations( int64_t fid, AMP::IO::RestartManager *manager );

protected:
    // Internal data
    std::vector<std::shared_ptr<VectorOperations>> d_operations;


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
