#include "AMP/vectors/data/SubsetCommSelfVectorData.h"
#include "AMP/discretization/subsetDOFManager.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <algorithm>


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
SubsetCommSelfVectorData::SubsetCommSelfVectorData( std::shared_ptr<VectorData> data )
{
    d_parentData = data;
    AMP_ASSERT( d_parentData );
    size_t N     = d_parentData->getLocalSize();
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
 * makeConsistent / UpdateState                                  *
 ****************************************************************/
UpdateState SubsetCommSelfVectorData::getLocalUpdateStatus() const
{
    return UpdateState::UNCHANGED;
}
void SubsetCommSelfVectorData::setUpdateStatus( UpdateState ) {}
void SubsetCommSelfVectorData::setUpdateStatusPtr( std::shared_ptr<UpdateState> ) {}
std::shared_ptr<UpdateState> SubsetCommSelfVectorData::getUpdateStatusPtr() const
{
    return nullptr;
}
void SubsetCommSelfVectorData::makeConsistent( ScatterType ) {}
void SubsetCommSelfVectorData::makeConsistent() {}
void SubsetCommSelfVectorData::dataChanged() {}

/****************************************************************
 * Functions get/set/add values                                  *
 ****************************************************************/
bool SubsetCommSelfVectorData::containsGlobalElement( size_t i ) const { return i < d_localSize; }
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
void SubsetCommSelfVectorData::setGhostValuesByGlobalID( size_t,
                                                         const size_t *,
                                                         const void *,
                                                         const typeID & )
{
}
void SubsetCommSelfVectorData::addGhostValuesByGlobalID( size_t,
                                                         const size_t *,
                                                         const void *,
                                                         const typeID & )
{
}
void SubsetCommSelfVectorData::getGhostValuesByGlobalID( size_t,
                                                         const size_t *,
                                                         void *,
                                                         const typeID & ) const
{
}
void SubsetCommSelfVectorData::getGhostAddValuesByGlobalID( size_t,
                                                            const size_t *,
                                                            void *,
                                                            const typeID & ) const
{
}
size_t SubsetCommSelfVectorData::getAllGhostValues( void *, const typeID & ) const { return 0; }
void SubsetCommSelfVectorData::swapData( VectorData &rhs )
{
    auto s = dynamic_cast<SubsetCommSelfVectorData *>( &rhs );
    AMP_ASSERT( s );
    std::swap( d_parentData, s->d_parentData );
    std::swap( d_localSize, s->d_localSize );
    std::swap( d_globalSize, s->d_globalSize );
    std::swap( d_localStart, s->d_localStart );
}

std::shared_ptr<VectorData> SubsetCommSelfVectorData::cloneData( const std::string & ) const
{
    AMP_ERROR( "Not finished" );
    return std::shared_ptr<VectorData>();
}


const AMP_MPI &SubsetCommSelfVectorData::getComm() const
{
    static AMP_MPI comm( AMP_COMM_SELF );
    return comm;
}

} // namespace AMP::LinearAlgebra
