#ifndef included_AMP_ManagedVectorData
#define included_AMP_ManagedVectorData

#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/DataChangeListener.h"
#include "AMP/vectors/data/GhostDataHelper.hpp"
#include "AMP/vectors/data/VectorData.h"

#include <stdexcept>
#include <vector>


namespace AMP::LinearAlgebra {


/**
   \brief Class used to control data and kernels of various vector libraries
   \details  A ManagedVector will take an engine and create a buffer, if
   necessary.

   A ManagedVector has two pointers: data and engine.  If the data pointer
   is null, then the engine is assumed to have the data.
*/
class ManagedVectorData : public GhostDataHelper<double>, public DataChangeListener
{

public:
    /** \brief Construct a ManagedVector from a set of parameters
     * \param[in] vec  The description of the ManagedVector
     */
    explicit ManagedVectorData( std::shared_ptr<Vector> vec );

    /** \brief Construct a view of an AMP vector
     * \param[in] alias  Vector to view
     */
    explicit ManagedVectorData( std::shared_ptr<VectorData> alias );

    //! Destructor
    virtual ~ManagedVectorData();

    /** \brief  Return the engine associated with this ManagedVector
     * \return The engine
     */
    std::shared_ptr<Vector> getVectorEngine();
    std::shared_ptr<const Vector> getVectorEngine() const;

    bool isAnAliasOf( const VectorData &rhs ) const override;

    void receiveDataChanged() override { fireDataChange(); }

protected:
    //! The underlying vector
    std::shared_ptr<Vector> d_Engine;


public: // Derived from VectorData
    size_t numberOfDataBlocks() const override;
    size_t sizeOfDataBlock( size_t ) const override;
    void setValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void addValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getValuesByLocalID( size_t, const size_t *, void *, const typeID & ) const override;
    void setGhostValuesByGlobalID( size_t, const size_t *, const void *, const typeID & ) override;
    void addGhostValuesByGlobalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getGhostValuesByGlobalID( size_t, const size_t *, void *, const typeID & ) const override;
    size_t getAllGhostValues( void *, const typeID & ) const override;
    void putRawData( const void *, const typeID & ) override;
    void getRawData( void *, const typeID & ) const override;
    UpdateState getLocalUpdateStatus() const override;
    void setUpdateStatus( UpdateState state ) override;
    typeID getType( size_t ) const override;
    uint64_t getDataID() const override
    {
        return reinterpret_cast<uint64_t>( getRawDataBlockAsVoid( 0 ) );
    }
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( double ); }
    std::string VectorDataName() const override;
    void swapData( VectorData & ) override;
    void assemble() override;
    void dataChanged() override;

    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;
    bool hasContiguousData() const override;
    void makeConsistent( ScatterType t ) override;
    using VectorData::makeConsistent;

protected: // Derived from VectorData
    void *getRawDataBlockAsVoid( size_t i ) override;
    const void *getRawDataBlockAsVoid( size_t i ) const override;

private:
    ManagedVectorData();
};


} // namespace AMP::LinearAlgebra


#endif
