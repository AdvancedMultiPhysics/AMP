#ifndef included_AMP_VectorDataNull
#define included_AMP_VectorDataNull

#include "AMP/utils/typeid.h"
#include "AMP/vectors/data/VectorData.h"


namespace AMP::LinearAlgebra {


/**
  \brief  A class used to hold vector data
  \details  VectorDataNull is a default implementation of VectorData that stores
      the local values as a single block of data on the CPU.
  */
class VectorDataNull : public VectorData
{
public: // Functions inherited from VectorData
    VectorDataNull() = delete;
    VectorDataNull( const typeID &type ) : d_type( type ) {}
    std::string VectorDataName() const override { return "VectorDataNull"; }
    inline size_t numberOfDataBlocks() const override { return 0; }
    inline size_t sizeOfDataBlock( size_t = 0 ) const override { return 0; }
    inline void putRawData( const void *, const typeID & ) override {}
    inline void getRawData( void *, const typeID & ) const override {}
    bool hasGhosts() const override { return false; }
    void fillGhosts( const Scalar & ) override {}
    void getValuesByLocalID( size_t N, const size_t *, void *, const typeID & ) const override;
    void setValuesByLocalID( size_t N, const size_t *, const void *, const typeID & ) override;
    void addValuesByLocalID( size_t N, const size_t *, const void *, const typeID & ) override;
    void setGhostValuesByGlobalID( size_t, const size_t *, const void *, const typeID & ) override;
    void addGhostValuesByGlobalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getGhostValuesByGlobalID( size_t, const size_t *, void *, const typeID & ) const override;
    void
    getGhostAddValuesByGlobalID( size_t, const size_t *, void *, const typeID & ) const override;
    size_t getAllGhostValues( void *, const typeID & ) const override;
    typeID getType( size_t ) const override { return d_type; }
    UpdateState getLocalUpdateStatus() const override { return UpdateState::UNCHANGED; }
    void setUpdateStatus( UpdateState ) override {}
    void setUpdateStatusPtr( std::shared_ptr<UpdateState> ) override {}
    std::shared_ptr<UpdateState> getUpdateStatusPtr() const override { return nullptr; }
    bool containsGlobalElement( size_t ) const override { return false; }
    void dataChanged() override {}
    void dumpGhostedData( std::ostream &, size_t ) const override {}
    void copyGhostValues( const VectorData & ) override {}
    std::shared_ptr<CommunicationList> getCommunicationList() const override { return nullptr; }
    void setCommunicationList( std::shared_ptr<CommunicationList> ) override {}
    const AMP_MPI &getComm() const override;
    size_t getGhostSize() const override { return 0; }
    uint64_t getDataID() const override { return 0; }
    void *getRawDataBlockAsVoid( size_t ) override { return nullptr; }
    const void *getRawDataBlockAsVoid( size_t ) const override { return nullptr; }
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( d_type.bytes ); }
    void swapData( VectorData & ) override { AMP_ERROR( "Not finished" ); }
    std::shared_ptr<VectorData> cloneData( const std::string & = "" ) const override;
    void makeConsistent( ScatterType ) override {}
    using VectorData::makeConsistent;

private:
    typeID d_type;
};


} // namespace AMP::LinearAlgebra


#endif
