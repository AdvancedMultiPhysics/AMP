#include "AMP/vectors/data/SubsetCommSelfVectorData.h"
#include "AMP/discretization/subsetDOFManager.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <algorithm>


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
SubsetCommSelfVectorData::SubsetCommSelfVectorData( std::shared_ptr<SubsetVectorParameters> params )
    : VectorData()
{
    d_parentData = params->d_ViewVector->getVectorData();
    d_DOFManager = std::dynamic_pointer_cast<AMP::Discretization::subsetCommSelfDOFManager>(
        params->d_DOFManager );
    AMP_ASSERT( d_parentData );
    AMP_ASSERT( d_DOFManager );
    size_t N = d_DOFManager->numLocalDOF();
    setCommunicationList( std::make_shared<CommunicationList>( N, AMP_COMM_SELF ) );
    d_localSize  = N;
    d_globalSize = N;
    d_localStart = 0;
}

/****************************************************************
 * Functions to access the raw data blocks                       *
 ****************************************************************/
size_t SubsetCommSelfVectorData::numberOfDataBlocks() const
{
    return d_parentData->numberOfDataBlocks();
}
size_t SubsetCommSelfVectorData::sizeOfDataBlock( size_t i ) const
{
    return d_parentData->sizeOfDataBlock( i );
}
void *SubsetCommSelfVectorData::getRawDataBlockAsVoid( size_t i )
{
    return d_parentData->getRawDataBlockAsVoid( i );
}
const void *SubsetCommSelfVectorData::getRawDataBlockAsVoid( size_t i ) const
{
    return d_parentData->getRawDataBlockAsVoid( i );
}
void SubsetCommSelfVectorData::putRawData( const void *in, const typeID &id )
{
    d_parentData->putRawData( in, id );
}
void SubsetCommSelfVectorData::getRawData( void *out, const typeID &id ) const
{
    d_parentData->putRawData( out, id );
}


/****************************************************************
 * Functions get/set/add values                                  *
 ****************************************************************/
void SubsetCommSelfVectorData::addValuesByLocalID( size_t N,
                                                   const size_t *ndx,
                                                   const void *vals,
                                                   const typeID &id )
{
    return d_parentData->addValuesByLocalID( N, ndx, vals, id );
}
void SubsetCommSelfVectorData::setValuesByLocalID( size_t N,
                                                   const size_t *ndx,
                                                   const void *vals,
                                                   const typeID &id )
{
    return d_parentData->setValuesByLocalID( N, ndx, vals, id );
}
void SubsetCommSelfVectorData::getValuesByLocalID( size_t N,
                                                   const size_t *ndx,
                                                   void *vals,
                                                   const typeID &id ) const
{
    return d_parentData->getValuesByLocalID( N, ndx, vals, id );
}
void SubsetCommSelfVectorData::swapData( VectorData &rhs )
{
    auto s = dynamic_cast<SubsetCommSelfVectorData *>( &rhs );
    AMP_ASSERT( s != nullptr );
    std::swap( d_parentData, s->d_parentData );
    std::swap( d_DOFManager, s->d_DOFManager );
    std::swap( d_localSize, s->d_localSize );
    std::swap( d_globalSize, s->d_globalSize );
    std::swap( d_localStart, s->d_localStart );
    std::swap( d_CommList, s->d_CommList );
    std::swap( d_UpdateState, s->d_UpdateState );
}

std::shared_ptr<VectorData> SubsetCommSelfVectorData::cloneData( const std::string & ) const
{
    AMP_ERROR( "Not finished" );
    return std::shared_ptr<VectorData>();
}


} // namespace AMP::LinearAlgebra
