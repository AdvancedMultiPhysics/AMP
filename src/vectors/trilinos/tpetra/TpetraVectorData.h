#ifndef included_TpetraVectorData_H_
#define included_TpetraVectorData_H_

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/data/GhostDataHelper.hpp"
#include "AMP/vectors/data/VectorData.h"

#include <Teuchos_Comm.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>


namespace AMP::LinearAlgebra {


template<typename ST = double,
         typename LO = int32_t,
         typename GO = int64_t,
         typename NT = Tpetra::Vector<>::node_type>
class TpetraVectorData : public GhostDataHelper<ST>
{
public:
    TpetraVectorData( std::shared_ptr<CommunicationList> commList,
                      std::shared_ptr<AMP::Discretization::DOFManager> dofManager );
    std::string VectorDataName() const override { return "TpetraVectorData"; }
    size_t numberOfDataBlocks() const override { return 1; }
    size_t sizeOfDataBlock( size_t i ) const override { return i == 0 ? this->d_localSize : 0; }
    void setValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void addValuesByLocalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getValuesByLocalID( size_t, const size_t *, void *, const typeID & ) const override;
    void putRawData( const void *in, const typeID &id ) override;
    void getRawData( void *out, const typeID &id ) const override;
    uint64_t getDataID() const override
    {
        return reinterpret_cast<uint64_t>( this->getRawDataBlockAsVoid( 0 ) );
    }
    void *getRawDataBlockAsVoid( size_t i ) override;
    const void *getRawDataBlockAsVoid( size_t i ) const override;
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( ST ); }
    typeID getType( size_t ) const override { return getTypeID<ST>(); }
    void swapData( VectorData & ) override;
    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;

    Teuchos::RCP<Tpetra::Vector<ST, LO, GO, NT>> getTpetraVector() { return this->d_pTpetraVector; }
    Teuchos::RCP<const Tpetra::Vector<ST, LO, GO, NT>> getTpetraVector() const
    {
        return this->d_pTpetraVector;
    }

protected:
    std::shared_ptr<AMP::Discretization::DOFManager> d_pDOFManager;
    Teuchos::RCP<Tpetra::Vector<ST, LO, GO, NT>> d_pTpetraVector;
};

} // namespace AMP::LinearAlgebra
#endif
