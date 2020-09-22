#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/ManagedVector.h"
#include "AMP/vectors/Vector.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>


namespace AMP {
namespace LinearAlgebra {


// Helper functions
static inline ManagedVectorData *getManaged( VectorData *x )
{
    auto y = dynamic_cast<ManagedVectorData *>( x );
    AMP_INSIST( y != nullptr, "x is not a ManagedVectorData" );
    return y;
}
static inline std::shared_ptr<ManagedVectorData> getManaged( std::shared_ptr<VectorData> x )
{
    auto y = std::dynamic_pointer_cast<ManagedVectorData>( x );
    AMP_INSIST( y != nullptr, "x is not a ManagedVectorData" );
    return y;
}
static inline std::shared_ptr<VectorData> getEngineData( VectorData &x )
{
    auto y = dynamic_cast<ManagedVectorData *>( &x );
    AMP_INSIST( y != nullptr, "x is not a ManagedVector" );
    auto engine = y->getVectorEngine();
    AMP_INSIST( engine, "ManagedVector Engine is Null" );
    auto vecEngine = std::dynamic_pointer_cast<Vector>( engine );
    if ( vecEngine )
        return vecEngine->getVectorData();
    else {
        AMP_ERROR( "Not programmed for as yet" );
    }
    return nullptr;
}
static inline std::shared_ptr<const VectorData> getEngineData( const VectorData &x )
{
    auto y = dynamic_cast<const ManagedVectorData *>( &x );
    AMP_INSIST( y != nullptr, "x is not a ManagedVector" );
    auto engine = y->getVectorEngine();
    AMP_INSIST( engine, "ManagedVector Engine is Null" );
    auto vecEngine = std::dynamic_pointer_cast<const Vector>( engine );
    if ( vecEngine )
        return vecEngine->getVectorData();
    else {
        AMP_ERROR( "Not programmed for as yet" );
    }
    return nullptr;
}


/********************************************************
 * Constructors                                          *
 ********************************************************/
ManagedVectorData::ManagedVectorData( std::shared_ptr<ManagedVectorParameters> params )
    : VectorData( params->d_CommList ), d_pParameters( params )
{
    d_vBuffer = d_pParameters->d_Buffer;
    d_Engine  = d_pParameters->d_Engine;
    AMP_ASSERT( d_Engine );
    if ( d_vBuffer )
        d_vBuffer->setUpdateStatusPtr( getUpdateStatusPtr() );
    if ( d_Engine )
        d_Engine->getVectorData()->setUpdateStatusPtr( getUpdateStatusPtr() );

    // this object will listen for changes from the d_Engine
    // and will fire off a change to any objects that are listening
    auto listener = dynamic_cast<DataChangeListener *>( this );
    d_Engine->getVectorData()->registerListener( listener );
}

ManagedVectorData::ManagedVectorData( const std::shared_ptr<VectorData> alias )
    : VectorData( getManaged( alias )->getParameters()->d_CommList )
{
    auto vec      = getManaged( alias );
    d_vBuffer     = vec->d_vBuffer;
    d_Engine      = vec->d_Engine;
    d_pParameters = vec->d_pParameters;
    aliasGhostBuffer( vec );

    auto vec2 = getVectorEngine();
    AMP_ASSERT( vec2 );

    if ( d_vBuffer )
        setUpdateStatusPtr( d_vBuffer->getUpdateStatusPtr() );
    else {
        if ( vec2 )
            setUpdateStatusPtr( vec2->getVectorData()->getUpdateStatusPtr() );
    }

    // this object will listen for changes from the d_Engine
    // and will fire off a change to any objects that are listening
    auto listener = dynamic_cast<DataChangeListener *>( this );
    vec2->getVectorData()->registerListener( listener );
}

ManagedVectorData::~ManagedVectorData() {}

/********************************************************
 * Subset                                                *
 ********************************************************/

bool ManagedVectorData::isAnAliasOf( VectorData &rhs )
{
    bool retVal = false;
    auto other  = getManaged( &rhs );
    if ( other != nullptr ) {
        if ( d_vBuffer && ( other->d_vBuffer == d_vBuffer ) ) {
            retVal = true;
        }
    }
    return retVal;
}

VectorData::UpdateState ManagedVectorData::getUpdateStatus() const
{
    VectorData::UpdateState state     = *d_UpdateState;
    std::shared_ptr<const Vector> vec = getVectorEngine();
    if ( vec.get() != nullptr ) {
        VectorData::UpdateState sub_state = vec->getUpdateStatus();
        if ( sub_state == UpdateState::UNCHANGED ) {
            // No change in state
        } else if ( sub_state == UpdateState::LOCAL_CHANGED && state == UpdateState::UNCHANGED ) {
            state = UpdateState::LOCAL_CHANGED;
        } else if ( sub_state == UpdateState::LOCAL_CHANGED ) {
            // No change in state
        } else if ( sub_state == UpdateState::ADDING &&
                    ( state == UpdateState::UNCHANGED || state == UpdateState::LOCAL_CHANGED ||
                      state == UpdateState::ADDING ) ) {
            state = UpdateState::ADDING;
        } else if ( sub_state == UpdateState::SETTING &&
                    ( state == UpdateState::UNCHANGED || state == UpdateState::LOCAL_CHANGED ||
                      state == UpdateState::SETTING ) ) {
            state = UpdateState::SETTING;
        } else {
            state = UpdateState::MIXED;
        }
    }
    return state;
}


void ManagedVectorData::setUpdateStatus( UpdateState state )
{
    *d_UpdateState = state;
    auto vec       = getVectorEngine();
    if ( vec.get() != nullptr )
        vec->setUpdateStatus( state );
}

void ManagedVectorData::swapData( VectorData &other )
{
    auto in = getManaged( &other );
    std::swap( d_vBuffer, in->d_vBuffer );
    std::swap( d_Engine, in->d_Engine );

    if ( d_vBuffer )
        d_vBuffer->setUpdateStatusPtr( getUpdateStatusPtr() );
    auto vec = getVectorEngine();
    if ( vec )
        vec->getVectorData()->setUpdateStatusPtr( getUpdateStatusPtr() );

    if ( in->d_vBuffer )
        in->d_vBuffer->setUpdateStatusPtr( in->getUpdateStatusPtr() );
    vec = in->getVectorEngine();
    if ( vec )
        vec->getVectorData()->setUpdateStatusPtr( in->getUpdateStatusPtr() );
}

void ManagedVectorData::aliasData( VectorData &other )
{
    auto in       = getManaged( &other );
    d_pParameters = in->d_pParameters;
    d_vBuffer     = in->d_vBuffer;
}

void ManagedVectorData::getValuesByGlobalID( int numVals, size_t *ndx, double *vals ) const
{
    auto const vec = getVectorEngine();
    if ( vec.get() == nullptr ) {
        VectorData::getValuesByGlobalID( numVals, ndx, vals );
    } else {
        vec->getValuesByGlobalID( numVals, ndx, vals );
    }
}

void ManagedVectorData::getLocalValuesByGlobalID( int numVals, size_t *ndx, double *vals ) const
{
    d_Engine->getLocalValuesByGlobalID( numVals, ndx, vals );
}

void ManagedVectorData::getGhostValuesByGlobalID( int numVals, size_t *ndx, double *vals ) const
{
    auto vec = getVectorEngine();
    if ( vec.get() == nullptr ) {
        VectorData::getGhostValuesByGlobalID( numVals, ndx, vals );
    } else {
        vec->getGhostValuesByGlobalID( numVals, ndx, vals );
    }
}

void ManagedVectorData::setValuesByLocalID( int i, size_t *id, const double *val )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::ADDING );
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
    getEngineData( *this )->setValuesByLocalID( i, id, val );
    fireDataChange();
}

void ManagedVectorData::setLocalValuesByGlobalID( int numVals, size_t *ndx, const double *vals )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::ADDING );
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
    getEngineData( *this )->setLocalValuesByGlobalID( numVals, ndx, vals );
    fireDataChange();
}

void ManagedVectorData::setGhostValuesByGlobalID( int numVals, size_t *ndx, const double *vals )
{
    auto vec = getVectorEngine();
    if ( vec.get() == nullptr ) {
        VectorData::setGhostValuesByGlobalID( numVals, ndx, vals );
    } else {
        vec->setGhostValuesByGlobalID( numVals, ndx, vals );
    }
}

void ManagedVectorData::setValuesByGlobalID( int numVals, size_t *ndx, const double *vals )
{
    auto vec = getVectorEngine();
    if ( vec.get() != nullptr ) {
        AMP_ASSERT( *d_UpdateState != UpdateState::ADDING );
        *d_UpdateState = UpdateState::SETTING;
        vec->setValuesByGlobalID( numVals, ndx, vals );
        fireDataChange();
    } else {
        std::vector<size_t> local_ndx;
        local_ndx.reserve( numVals );
        std::vector<double> local_val;
        local_val.reserve( numVals );
        std::vector<size_t> ghost_ndx;
        ghost_ndx.reserve( numVals );
        std::vector<double> ghost_val;
        ghost_val.reserve( numVals );
        for ( int i = 0; i < numVals; i++ ) {
            if ( ( ndx[i] < getLocalStartID() ) ||
                 ( ndx[i] >= ( getLocalStartID() + getLocalMaxID() ) ) ) {
                ghost_ndx.push_back( ndx[i] );
                ghost_val.push_back( vals[i] );
            } else {
                local_ndx.push_back( ndx[i] );
                local_val.push_back( vals[i] );
            }
        }
        if ( !ghost_ndx.empty() )
            setGhostValuesByGlobalID( ghost_ndx.size(), &ghost_ndx[0], &ghost_val[0] );
        if ( !local_ndx.empty() )
            setLocalValuesByGlobalID( local_ndx.size(), &local_ndx[0], &local_val[0] );
    }
}

void ManagedVectorData::addValuesByLocalID( int i, size_t *id, const double *val )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::SETTING );
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
    getEngineData( *this )->addValuesByLocalID( i, id, val );
    fireDataChange();
}

void ManagedVectorData::addLocalValuesByGlobalID( int i, size_t *id, const double *val )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::SETTING );
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;

    getEngineData( *this )->addLocalValuesByGlobalID( i, id, val );
    fireDataChange();
}

void ManagedVectorData::putRawData( const double *in ) { getEngineData( *this )->putRawData( in ); }

void ManagedVectorData::copyOutRawData( double *in ) const
{
    getEngineData( *this )->copyOutRawData( in );
}

std::shared_ptr<VectorData> ManagedVectorData::cloneData( void ) const
{
    auto params        = std::make_shared<ManagedVectorParameters>();
    params->d_CommList = d_pParameters->d_CommList;

    auto vec = getVectorEngine();
    if ( vec ) {
        auto vec2        = vec->cloneVector( "ManagedVectorClone" );
        params->d_Engine = std::dynamic_pointer_cast<Vector>( vec2 );
        params->d_Buffer = std::dynamic_pointer_cast<VectorData>( vec2 );
    } else {
        AMP_ERROR( "ManagedVectorData::cloneVector() should not have reached here!" );
    }

    auto retVal = std::make_shared<ManagedVectorData>( params );
    return retVal;
}

std::string ManagedVectorData::VectorDataName() const
{
    std::string retVal = " ( managed view of ";
    auto vec           = getVectorEngine();
    retVal += vec->type();
    retVal += " )";
    return retVal;
}

void ManagedVectorData::dataChanged()
{
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
}

void *ManagedVectorData::getRawDataBlockAsVoid( size_t i )
{
    return getEngineData( *this )->getRawDataBlockAsVoid( i );
}

const void *ManagedVectorData::getRawDataBlockAsVoid( size_t i ) const
{
    return getEngineData( *this )->getRawDataBlockAsVoid( i );
}

size_t ManagedVectorData::numberOfDataBlocks() const
{
    return getEngineData( *this )->numberOfDataBlocks();
}

size_t ManagedVectorData::sizeOfDataBlock( size_t i ) const
{
    return getEngineData( *this )->sizeOfDataBlock( i );
}

std::shared_ptr<ManagedVectorParameters> ManagedVectorData::getParameters()
{
    return d_pParameters;
}

size_t ManagedVectorData::getLocalSize() const { return getEngineData( *this )->getLocalSize(); }

size_t ManagedVectorData::getGlobalSize() const { return getEngineData( *this )->getGlobalSize(); }

Vector::shared_ptr ManagedVectorData::getVectorEngine( void ) { return d_Engine; }

Vector::const_shared_ptr ManagedVectorData::getVectorEngine( void ) const { return d_Engine; }

ManagedVectorParameters::ManagedVectorParameters() : d_Buffer( nullptr ) {}

} // namespace LinearAlgebra
} // namespace AMP