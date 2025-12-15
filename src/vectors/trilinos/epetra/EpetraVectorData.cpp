#include "AMP/vectors/trilinos/epetra/EpetraVectorData.h"
#include "AMP/vectors/data/VectorDataDefault.h"


namespace AMP::LinearAlgebra {

static inline double *getBufferPtr( std::shared_ptr<VectorData> buf )
{
    size_t N_blocks = buf->numberOfDataBlocks();
    if ( N_blocks == 0 )
        return nullptr;
    if ( N_blocks > 1 )
        AMP_ERROR( "More than 1 data block detected" );
    return buf->getRawDataBlock<double>( 0 );
}


EpetraVectorData::EpetraVectorData( std::shared_ptr<EpetraVectorEngineParameters> alias,
                                    Epetra_DataAccess method,
                                    const Epetra_BlockMap &map,
                                    std::shared_ptr<VectorData> bufData,
                                    int localStart,
                                    int localSize,
                                    int globalSize )
    : GhostDataHelper<double>( alias->d_CommList ),
      d_epetraVector( method, map, getBufferPtr( bufData ) ),
      d_buf_scope{ bufData }
{
    d_localStart = localStart;
    d_localSize  = localSize;
    d_globalSize = globalSize;
    AMP_ASSERT( d_globalSize < 0x80000000 );
}

std::shared_ptr<EpetraVectorData>
EpetraVectorData::create( std::shared_ptr<EpetraVectorEngineParameters> alias,
                          std::shared_ptr<VectorData> buf )
{
    auto params = std::dynamic_pointer_cast<EpetraVectorEngineParameters>( alias );
    return std::make_shared<EpetraVectorData>( alias,
                                               View,
                                               params->getEpetraMap(),
                                               buf,
                                               params->beginDOF(),
                                               params->getLocalSize(),
                                               params->getGlobalSize() );
}

void *EpetraVectorData::getRawDataBlockAsVoid( size_t i )
{
    if ( i > 1 )
        return nullptr;
    double *p;
    d_epetraVector.ExtractView( &p );
    return p;
}

const void *EpetraVectorData::getRawDataBlockAsVoid( size_t i ) const
{
    if ( i > 1 )
        return nullptr;
    double *p;
    d_epetraVector.ExtractView( &p );
    return p;
}


/****************************************************************
 * Functions get/set/add values                                  *
 ****************************************************************/
void EpetraVectorData::setValuesByLocalID( size_t N,
                                           const size_t *indices,
                                           const void *in,
                                           const typeID &id )
{
    AMP_INSIST( id == getTypeID<double>(), "Epetra only supports double" );
    auto vals = reinterpret_cast<const double *>( in );
    for ( size_t i = 0; i != N; i++ )
        d_epetraVector[indices[i]] = vals[i];
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
}
void EpetraVectorData::addValuesByLocalID( size_t N,
                                           const size_t *indices,
                                           const void *in,
                                           const typeID &id )
{
    if ( N == 0 )
        return;
    AMP_INSIST( id == getTypeID<double>(), "Epetra only supports double" );
    auto vals = reinterpret_cast<const double *>( in );
    for ( size_t i = 0; i != N; i++ )
        d_epetraVector[indices[i]] += vals[i];
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
}
void EpetraVectorData::getValuesByLocalID( size_t N,
                                           const size_t *indices,
                                           void *out,
                                           const typeID &id ) const
{
    if ( N == 0 )
        return;
    AMP_INSIST( id == getTypeID<double>(), "Epetra only supports double" );
    auto vals = reinterpret_cast<double *>( out );
    double *data;
    d_epetraVector.ExtractView( &data );
    for ( size_t i = 0; i != N; i++ )
        vals[i] = data[indices[i]];
}


void EpetraVectorData::putRawData( const void *in, const typeID &id )
{
    AMP_INSIST( id == getTypeID<double>(), "Epetra only supports double" );
    auto data = reinterpret_cast<const double *>( in );
    double *p;
    d_epetraVector.ExtractView( &p );
    memcpy( p, data, d_localSize * sizeof( double ) );
}

void EpetraVectorData::getRawData( void *out, const typeID &id ) const
{
    AMP_INSIST( id == getTypeID<double>(), "Epetra only supports double" );
    auto data = reinterpret_cast<double *>( out );
    d_epetraVector.ExtractCopy( data );
}

std::shared_ptr<VectorData> EpetraVectorData::cloneData( const std::string & ) const
{
    auto buffer =
        std::make_shared<VectorDataDefault<double>>( d_localStart, d_localSize, d_globalSize );
    auto params = std::make_shared<EpetraVectorEngineParameters>(
        d_localSize, getComm(), getCommunicationList() );
    auto data = EpetraVectorData::create( params, buffer );
    return data;
}

void EpetraVectorData::swapData( VectorData &rhs )
{
    auto rhs2 = dynamic_cast<EpetraVectorData *>( &rhs );
    AMP_INSIST( rhs2, "Cannot swap with arbitrary VectorData" );
    std::swap( d_epetraVector, rhs2->d_epetraVector );
    std::swap( d_buf_scope, rhs2->d_buf_scope );
    std::swap( d_localStart, rhs2->d_localStart );
    std::swap( d_localSize, rhs2->d_localSize );
    std::swap( d_globalSize, rhs2->d_globalSize );
}


} // namespace AMP::LinearAlgebra
