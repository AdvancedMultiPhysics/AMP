#include "AMP/vectors/data/SubsetVectorData.h"
#include "AMP/discretization/subsetDOFManager.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <algorithm>


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
template<class T>
static std::tuple<std::vector<void *>, std::vector<size_t>>
getDataBlocks( const std::vector<T *> &data )
{
    PROFILE( "SubsetVectorData::getDataBlocks" );

    if ( data.empty() )
        return std::tuple<std::vector<void *>, std::vector<size_t>>();
    std::vector<void *> ptr;
    std::vector<size_t> size;
    ptr.reserve( data.size() );
    size.reserve( data.size() );
    ptr.push_back( data[0] );
    size.push_back( 1 );
    T *last = data[0];
    for ( size_t i = 1; i < data.size(); i++ ) {
        if ( data[i] == ( ++last ) ) {
            size.back()++;
        } else {
            last = data[i];
            ptr.push_back( last );
            size.push_back( 1 );
        }
    }
    return std::tie( ptr, size );
}

SubsetVectorData::SubsetVectorData( std::shared_ptr<SubsetVectorParameters> params )
    : GhostDataHelper<double>( params->d_CommList ),
      d_ViewVector( params->d_ViewVector ),
      d_DOFManager{ params->d_DOFManager }
{
    PROFILE( "SubsetVectorData::constructor" );

    auto parentDOF       = d_ViewVector->getDOFManager();
    d_parentLocalStartID = parentDOF->beginDOF();
    auto tmp = std::dynamic_pointer_cast<AMP::Discretization::subsetDOFManager>( d_DOFManager );
    if ( tmp != nullptr ) {
        d_SubsetLocalIDToViewGlobalID = tmp->getLocalParentDOFs();
    } else if ( d_DOFManager->numLocalDOF() == parentDOF->numLocalDOF() ) {
        d_SubsetLocalIDToViewGlobalID.resize( parentDOF->numLocalDOF() );
        size_t beginDOF = parentDOF->beginDOF();
        size_t endDOF   = parentDOF->endDOF();
        for ( size_t i = beginDOF; i < endDOF; i++ )
            d_SubsetLocalIDToViewGlobalID[i - beginDOF] = i;
    } else {
        AMP_ERROR( "Internal error with SubsetVector" );
    }
    // Get a pointer to every value in the subset
    if ( d_ViewVector->getVectorData()->isType<double>() ) {
        d_typeID = getTypeID<double>();
        std::vector<double *> data_ptr( d_SubsetLocalIDToViewGlobalID.size(), nullptr );
        auto iterator   = d_ViewVector->constBegin();
        size_t last_pos = d_ViewVector->getCommunicationList()->getStartGID();
        for ( size_t i = 0; i < data_ptr.size(); i++ ) {
            iterator += (int) ( d_SubsetLocalIDToViewGlobalID[i] - last_pos );
            last_pos    = d_SubsetLocalIDToViewGlobalID[i];
            data_ptr[i] = const_cast<double *>( std::addressof( *iterator ) );
        }
        std::tie( d_dataBlockPtr, d_dataBlockSize ) = getDataBlocks<double>( data_ptr );
    } else if ( d_ViewVector->getVectorData()->isType<float>() ) {
        d_typeID = getTypeID<float>();
        std::vector<float *> data_ptr( d_SubsetLocalIDToViewGlobalID.size(), nullptr );
        auto iterator   = d_ViewVector->constBegin<float>();
        size_t last_pos = d_ViewVector->getCommunicationList()->getStartGID();
        for ( size_t i = 0; i < data_ptr.size(); i++ ) {
            iterator += (int) ( d_SubsetLocalIDToViewGlobalID[i] - last_pos );
            last_pos    = d_SubsetLocalIDToViewGlobalID[i];
            data_ptr[i] = const_cast<float *>( std::addressof( *iterator ) );
        }
        std::tie( d_dataBlockPtr, d_dataBlockSize ) = getDataBlocks<float>( data_ptr );
    } else {
        AMP_ERROR( "scalar data type no handled at present" );
    }
    // Cache the local/global size
    AMP_ASSERT( d_ViewVector );
    d_localSize  = d_CommList->numLocalRows();
    d_globalSize = d_CommList->getTotalSize();
    d_localStart = d_CommList->getStartGID();
}

/****************************************************************
 * Functions to access the raw data blocks                       *
 ****************************************************************/
size_t SubsetVectorData::numberOfDataBlocks() const { return d_dataBlockSize.size(); }
size_t SubsetVectorData::sizeOfDataBlock( size_t i ) const { return d_dataBlockSize[i]; }
void *SubsetVectorData::getRawDataBlockAsVoid( size_t i ) { return d_dataBlockPtr[i]; }
const void *SubsetVectorData::getRawDataBlockAsVoid( size_t i ) const { return d_dataBlockPtr[i]; }

template<typename T>
static void putData( const void *in,
                     std::vector<void *> &dataBlockPtr,
                     const std::vector<size_t> &dataBlockSize )
{
    PROFILE( "SubsetVectorData::putData" );

    auto data = reinterpret_cast<const T *>( in );
    size_t k  = 0;
    for ( size_t i = 0; i < dataBlockPtr.size(); i++ ) {
        auto block_i = reinterpret_cast<T *>( dataBlockPtr[i] );
        for ( size_t j = 0; j < dataBlockSize[i]; j++, k++ )
            block_i[j] = data[k];
    }
}

template<typename T>
static void getData( void *out,
                     const std::vector<void *> &dataBlockPtr,
                     const std::vector<size_t> &dataBlockSize )
{
    PROFILE( "SubsetVectorData::getData" );

    auto data = reinterpret_cast<T *>( out );
    size_t k  = 0;
    for ( size_t i = 0; i < dataBlockPtr.size(); i++ ) {
        auto block_i = reinterpret_cast<const T *>( dataBlockPtr[i] );
        for ( size_t j = 0; j < dataBlockSize[i]; j++, k++ )
            data[k] = block_i[j];
    }
}

void SubsetVectorData::putRawData( const void *in, const typeID &id )
{
    AMP_ASSERT( id == d_typeID );
    if ( id == getTypeID<double>() ) {
        putData<double>( in, d_dataBlockPtr, d_dataBlockSize );
    } else if ( id == getTypeID<float>() ) {
        putData<float>( in, d_dataBlockPtr, d_dataBlockSize );
    } else {
        AMP_ERROR( "scalar data type no handled at present" );
    }
}

void SubsetVectorData::getRawData( void *out, const typeID &id ) const
{
    AMP_ASSERT( id == getTypeID<double>() );
    if ( id == getTypeID<double>() ) {
        getData<double>( out, d_dataBlockPtr, d_dataBlockSize );
    } else if ( id == getTypeID<float>() ) {
        getData<float>( out, d_dataBlockPtr, d_dataBlockSize );
    } else {
        AMP_ERROR( "scalar data type no handled at present" );
    }
}


/****************************************************************
 * Functions get/set/add values                                  *
 ****************************************************************/
void SubsetVectorData::addValuesByLocalID( size_t N,
                                           const size_t *ndx,
                                           const void *vals,
                                           const typeID &id )
{
    PROFILE( "SubsetVectorData::addValuesByLocalID" );

    AMP_ASSERT( d_ViewVector );
    std::vector<size_t> index( N );
    for ( size_t i = 0; i != N; i++ )
        index[i] = d_SubsetLocalIDToViewGlobalID[ndx[i]] - d_parentLocalStartID;
    d_ViewVector->getVectorData()->addValuesByLocalID( N, index.data(), vals, id );
}

void SubsetVectorData::setValuesByLocalID( size_t N,
                                           const size_t *ndx,
                                           const void *vals,
                                           const typeID &id )
{
    PROFILE( "SubsetVectorData::setValuesByLocalID" );

    AMP_ASSERT( d_ViewVector );
    std::vector<size_t> index( N );
    for ( size_t i = 0; i != N; i++ )
        index[i] = d_SubsetLocalIDToViewGlobalID[ndx[i]] - d_parentLocalStartID;
    d_ViewVector->getVectorData()->setValuesByLocalID( N, index.data(), vals, id );
}

void SubsetVectorData::getValuesByLocalID( size_t N,
                                           const size_t *ndx,
                                           void *vals,
                                           const typeID &id ) const
{
    PROFILE( "SubsetVectorData::getValuesByLocalID" );

    AMP_ASSERT( d_ViewVector );
    std::vector<size_t> index( N );
    for ( size_t i = 0; i != N; i++ )
        index[i] = d_SubsetLocalIDToViewGlobalID[ndx[i]] - d_parentLocalStartID;
    d_ViewVector->getVectorData()->getValuesByLocalID( N, index.data(), vals, id );
}

void SubsetVectorData::swapData( VectorData &rhs )
{
    PROFILE( "SubsetVectorData::" );

    auto s = dynamic_cast<SubsetVectorData *>( &rhs );
    AMP_ASSERT( s );
    std::swap( d_ViewVector, s->d_ViewVector );
    std::swap( d_SubsetLocalIDToViewGlobalID, s->d_SubsetLocalIDToViewGlobalID );
    std::swap( d_dataBlockSize, s->d_dataBlockSize );
    std::swap( d_dataBlockPtr, s->d_dataBlockPtr );
}

std::shared_ptr<VectorData> SubsetVectorData::cloneData( const std::string & ) const
{
    PROFILE( "SubsetVectorData::cloneData" );

    AMP_ERROR( "Not finished" );
    return std::shared_ptr<VectorData>();
}


const AMP_MPI &SubsetVectorData::getComm() const { return d_DOFManager->getComm(); }


} // namespace AMP::LinearAlgebra
