#ifndef included_TpetraVectorData_HPP_
#define included_TpetraVectorData_HPP_

#include "AMP/AMP_TPLs.h"
#ifdef AMP_USE_MPI
    #include "Teuchos_DefaultMpiComm.hpp"
#else
    #include "Teuchos_DefaultSerialComm.hpp"
#endif

#include "AMP/vectors/trilinos/tpetra/TpetraVectorData.h"

namespace AMP::LinearAlgebra {


template<typename ST, typename LO, typename GO, typename NT>
TpetraVectorData<ST, LO, GO, NT>::TpetraVectorData(
    std::shared_ptr<CommunicationList> commList,
    std::shared_ptr<AMP::Discretization::DOFManager> dofManager )
    : GhostDataHelper<ST>( commList ), d_pDOFManager( dofManager )
{
    AMP_DEBUG_ASSERT( dofManager );
#ifdef AMP_USE_MPI
    const auto &mpiComm = dofManager->getComm().getCommunicator();
    auto comm           = Teuchos::rcp( new Teuchos::MpiComm<int>( mpiComm ) );
#else
    auto comm = Tpetra::getDefaultComm();
#endif

    auto map        = Teuchos::rcp( new Tpetra::Map<LO, GO, NT>(
        dofManager->numGlobalDOF(), dofManager->numLocalDOF(), comm ) );
    d_pTpetraVector = Teuchos::rcp( new Tpetra::Vector<ST, LO, GO, NT>( map, true ) );
}

template<typename ST, typename LO, typename GO, typename NT>
TpetraVectorData<ST, LO, GO, NT>::~TpetraVectorData()
{
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorData<ST, LO, GO, NT>::setValuesByLocalID( size_t,
                                                           const size_t *,
                                                           const void *,
                                                           const typeID & )
{
    AMP_ERROR( "Not implemented" );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorData<ST, LO, GO, NT>::addValuesByLocalID( size_t,
                                                           const size_t *,
                                                           const void *,
                                                           const typeID & )
{
    AMP_ERROR( "Not implemented" );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorData<ST, LO, GO, NT>::getValuesByLocalID( size_t N,
                                                           const size_t *indices,
                                                           void *out,
                                                           const typeID &id ) const
{
    if ( N == 0 )
        return;
    AMP_INSIST( id == getTypeID<ST>(), "Tpetra only supports native type at this time" );
    auto vals = reinterpret_cast<ST *>( out );
    auto tVec = this->getTpetraVector();
    AMP_INSIST( tVec->getNumVectors() == 1, "Only single TpetraVectors supported" );
    auto data = tVec->getData();
    for ( size_t i = 0; i != N; i++ )
        vals[i] = data[indices[i]];
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorData<ST, LO, GO, NT>::putRawData( const void *in, const typeID &id )
{
    AMP_INSIST( id == getTypeID<ST>(), "Tpetra only supports putRawData for native type" );
    const auto srcData = reinterpret_cast<const ST *>( in );
    auto dstData       = this->getTpetraVector()->getDataNonConst( 0 );
    AMP::Utilities::Algorithms<ST>::copy_n( srcData, dstData.size(), dstData.get() );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorData<ST, LO, GO, NT>::getRawData( void *out, const typeID &id ) const
{
    AMP_INSIST( id == getTypeID<ST>(), "Tpetra only supports getRawData for native type" );
    auto dstData = reinterpret_cast<ST *>( out );
    auto tVec    = this->getTpetraVector();
    AMP_INSIST( tVec && tVec->getNumVectors() == 1,
                "Only implemented for single data block vectors" );
    const auto srcData = tVec->getData( 0 );
    AMP::Utilities::Algorithms<ST>::copy_n( srcData.get(), srcData.size(), dstData );
}

template<typename ST, typename LO, typename GO, typename NT>
void *TpetraVectorData<ST, LO, GO, NT>::getRawDataBlockAsVoid( size_t )
{
    auto tVec = this->getTpetraVector();
    AMP_INSIST( tVec && tVec->getNumVectors() == 1,
                "Only implemented for single data block vectors" );
    return tVec->getDataNonConst( 0 ).get();
}

template<typename ST, typename LO, typename GO, typename NT>
const void *TpetraVectorData<ST, LO, GO, NT>::getRawDataBlockAsVoid( size_t ) const
{
    auto tVec = this->getTpetraVector();
    AMP_INSIST( tVec && tVec->getNumVectors() == 1,
                "Only implemented for single data block vectors" );
    return tVec->getData( 0 ).get();
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorData<ST, LO, GO, NT>::swapData( VectorData &other )
{
    auto otherData = dynamic_cast<TpetraVectorData<ST, LO, GO, NT> *>( &other );
    AMP_INSIST( otherData, "Not TpetraVectorData" );
    this->getTpetraVector()->swap( *( otherData->getTpetraVector() ) );
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<VectorData> TpetraVectorData<ST, LO, GO, NT>::cloneData( const std::string & ) const
{
    return std::make_shared<TpetraVectorData<ST, LO, GO, NT>>( this->getCommunicationList(),
                                                               d_pDOFManager );
}


} // namespace AMP::LinearAlgebra
#endif
