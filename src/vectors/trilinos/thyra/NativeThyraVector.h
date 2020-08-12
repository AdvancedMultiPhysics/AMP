#ifndef included_AMP_NativeThyraVector
#define included_AMP_NativeThyraVector

#include "AMP/vectors/Vector.h"
#include "AMP/vectors/operations/VectorOperationsDefault.h"
#include "AMP/vectors/trilinos/thyra/ThyraVector.h"

namespace AMP {
namespace LinearAlgebra {


/** \class NativeThyraVectorParameters
 * \brief Parameters to set when creating a NativeThyraVector
 */
class NativeThyraVectorParameters : public VectorParameters
{
public:
    //! The vector to wrap
    Teuchos::RCP<Thyra::VectorBase<double>> d_InVec;

    //! The local size of the vector
    size_t d_local;

    //! The comm of the vector
    AMP_MPI d_comm;

    //! The variable to use with the vector
    Variable::shared_ptr d_var;
};


/** \class NativeThyraVector
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
class NativeThyraVector : public Vector, public ThyraVector
{
public:
    /** \brief Construct a wrapper for a Thyra Vec from a set of parameters
     * \param[in] params The parameters describing the Vec
     */
    explicit NativeThyraVector( VectorParameters::shared_ptr params );

    //! Destructor
    virtual ~NativeThyraVector();


    //! Overloaded functions
    std::string type() const override { return "Native Thyra Vector"; }
    std::string VectorDataName() const override { return "NativeThyraVector"; }
    Vector::shared_ptr cloneVector( const Variable::shared_ptr ) const override;
    void swapVectors( Vector &other ) override;
    void aliasVector( Vector & ) override;
    size_t numberOfDataBlocks() const override;
    size_t sizeOfDataBlock( size_t i ) const override;

    void setValuesByLocalID( int, size_t *, const double * ) override;
    void setLocalValuesByGlobalID( int, size_t *, const double * ) override;
    void addValuesByLocalID( int, size_t *, const double * ) override;
    void addLocalValuesByGlobalID( int, size_t *, const double * ) override;
    void getLocalValuesByGlobalID( int numVals, size_t *ndx, double *vals ) const override;
    void getValuesByLocalID( int numVals, size_t *ndx, double *vals ) const override;
    void assemble() override;
    size_t getLocalSize() const override;
    size_t getGlobalSize() const override;
    void putRawData( const double * ) override;
    void copyOutRawData( double *out ) const override;
    uint64_t getDataID() const override
    {
        return reinterpret_cast<uint64_t>( getRawDataBlockAsVoid( 0 ) );
    }
    bool isTypeId( size_t hash, size_t ) const override
    {
        return hash == typeid( double ).hash_code();
    }
    void swapData( VectorData & ) override { AMP_ERROR( "Not finished" ); }

protected:
    //! Empty constructor.
    NativeThyraVector();

    void *getRawDataBlockAsVoid( size_t i ) override;
    const void *getRawDataBlockAsVoid( size_t i ) const override;
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( double ); }

private:
    size_t d_local;

    static Teuchos::RCP<const Thyra::VectorBase<double>> getThyraVec( const VectorOperations &v );
    static Teuchos::RCP<const Thyra::VectorBase<double>>
    getThyraVec( const Vector::const_shared_ptr &v );

    static Teuchos::RCP<const Thyra::VectorBase<double>> getThyraVec( const VectorData &v );
    static Teuchos::RCP<Thyra::VectorBase<double>> getThyraVec( VectorData &v );
};


} // namespace LinearAlgebra
} // namespace AMP

#endif
