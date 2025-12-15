#ifndef included_AMP_NativeThyraVectorData
#define included_AMP_NativeThyraVectorData

#include "AMP/vectors/data/GhostDataHelper.hpp"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/trilinos/thyra/ThyraVector.h"

namespace AMP::LinearAlgebra {


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
class NativeThyraVectorData : public GhostDataHelper<double>
{
public:
    /** \brief Construct a wrapper for a Thyra Vec from a set of parameters
     * \param[in] vec           The Thyra vector
     * \param[in] localsize     The local vector size
     * \param[in] comm          The communicator
     */
    explicit NativeThyraVectorData( Teuchos::RCP<Thyra::VectorBase<double>> vec,
                                    size_t localsize,
                                    AMP_MPI comm );

    //! Destructor
    virtual ~NativeThyraVectorData();

    std::string VectorDataName() const override { return "NativeThyraVector"; }
    size_t numberOfDataBlocks() const override;
    size_t sizeOfDataBlock( size_t i ) const override;

    void addValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void setValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getValuesByLocalID( size_t, const size_t *, void *, const typeID & ) const override;
    void putRawData( const void *, const typeID & ) override;
    void getRawData( void *, const typeID & ) const override;
    uint64_t getDataID() const override
    {
        return reinterpret_cast<uint64_t>( getRawDataBlockAsVoid( 0 ) );
    }
    void *getRawDataBlockAsVoid( size_t i ) override;
    const void *getRawDataBlockAsVoid( size_t i ) const override;
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( double ); }
    typeID getType( size_t ) const override { return getTypeID<double>(); }
    void swapData( VectorData & ) override;
    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;

    Teuchos::RCP<Thyra::VectorBase<double>> getVec() { return d_thyraVec; }
    Teuchos::RCP<const Thyra::VectorBase<double>> getVec() const { return d_thyraVec; }

protected:
    //! Empty constructor.
    NativeThyraVectorData();


private:
    Teuchos::RCP<Thyra::VectorBase<double>> d_thyraVec;

    static Teuchos::RCP<const Thyra::VectorBase<double>>
    getThyraVec( std::shared_ptr<const VectorData> vec );

    static Teuchos::RCP<const Thyra::VectorBase<double>> getThyraVec( const VectorData &v );
    static Teuchos::RCP<Thyra::VectorBase<double>> getThyraVec( VectorData &v );
};


} // namespace AMP::LinearAlgebra

#endif
