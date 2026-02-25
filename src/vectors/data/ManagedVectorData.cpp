#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/operations/ManagedVectorOperations.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>


namespace AMP::LinearAlgebra {


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
ManagedVectorData::ManagedVectorData( std::shared_ptr<Vector> vec )
    : GhostDataHelper( vec->getVectorData()->getCommunicationList() ), d_Engine( vec )
{
    d_Engine->getVectorData()->setUpdateStatusPtr( getUpdateStatusPtr() );

    // this object will listen for changes from the d_Engine
    // and will fire off a change to any objects that are listening
    auto listener = dynamic_cast<DataChangeListener *>( this );
    d_Engine->getVectorData()->registerListener( listener );

    d_localSize  = getEngineData( *this )->getLocalSize();
    d_globalSize = getEngineData( *this )->getGlobalSize();
    d_localStart = d_CommList->getStartGID();
}

ManagedVectorData::ManagedVectorData( std::shared_ptr<VectorData> alias )
    : GhostDataHelper( alias->getCommunicationList() )
{
    auto vec = getManaged( alias );
    d_Engine = vec->d_Engine;

    auto vec2 = getVectorEngine();
    AMP_ASSERT( vec2 );

    setUpdateStatusPtr( d_Engine->getVectorData()->getUpdateStatusPtr() );

    // this object will listen for changes from the d_Engine
    // and will fire off a change to any objects that are listening
    auto listener = dynamic_cast<DataChangeListener *>( this );
    vec2->getVectorData()->registerListener( listener );

    d_localSize  = getEngineData( *this )->getLocalSize();
    d_globalSize = getEngineData( *this )->getGlobalSize();
    d_localStart = d_CommList->getStartGID();
}
ManagedVectorData::~ManagedVectorData() = default;


/********************************************************
 * Subset                                                *
 ********************************************************/
bool ManagedVectorData::isAnAliasOf( const VectorData &rhs ) const
{
    auto other = dynamic_cast<const ManagedVectorData *>( &rhs );
    if ( other ) {
        if ( other->d_Engine == d_Engine )
            return true;
    }
    return false;
}
UpdateState ManagedVectorData::getLocalUpdateStatus() const
{
    UpdateState state                 = *d_UpdateState;
    std::shared_ptr<const Vector> vec = getVectorEngine();
    if ( vec ) {
        UpdateState sub_state = vec->getUpdateStatus();
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
    if ( vec )
        vec->setUpdateStatus( state );
}

void ManagedVectorData::swapData( VectorData &other )
{
    auto in = getManaged( &other );
    d_Engine->swapVectors( in->d_Engine );

    d_Engine->getVectorData()->setUpdateStatusPtr( getUpdateStatusPtr() );
    auto vec = getVectorEngine();
    if ( vec )
        vec->getVectorData()->setUpdateStatusPtr( getUpdateStatusPtr() );

    in->d_Engine->getVectorData()->setUpdateStatusPtr( in->getUpdateStatusPtr() );
    vec = in->getVectorEngine();
    if ( vec )
        vec->getVectorData()->setUpdateStatusPtr( in->getUpdateStatusPtr() );
}

void ManagedVectorData::assemble() { d_Engine->getVectorData()->assemble(); }


/****************************************************************
 * Get/Set values                                                *
 ****************************************************************/
void ManagedVectorData::getValuesByLocalID( size_t N,
                                            const size_t *ndx,
                                            void *vals,
                                            const typeID &id ) const
{
    getEngineData( *this )->getValuesByLocalID( N, ndx, vals, id );
}
void ManagedVectorData::getGhostValuesByGlobalID( size_t N,
                                                  const size_t *ndx,
                                                  void *vals,
                                                  const typeID &id ) const
{
    auto vec = getVectorEngine();
    if ( !vec ) {
        GhostDataHelper<double>::getGhostValuesByGlobalID( N, ndx, vals, id );
    } else {
        vec->getVectorData()->getGhostValuesByGlobalID( N, ndx, vals, id );
    }
}
void ManagedVectorData::setValuesByLocalID( size_t N,
                                            const size_t *ndx,
                                            const void *val,
                                            const typeID &id )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::ADDING );
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
    getEngineData( *this )->setValuesByLocalID( N, ndx, val, id );
    fireDataChange();
}
void ManagedVectorData::setGhostValuesByGlobalID( size_t N,
                                                  const size_t *ndx,
                                                  const void *vals,
                                                  const typeID &id )
{
    auto vec = getVectorEngine();
    if ( !vec ) {
        GhostDataHelper<double>::setGhostValuesByGlobalID( N, ndx, vals, id );
    } else {
        vec->getVectorData()->setGhostValuesByGlobalID( N, ndx, vals, id );
    }
}
void ManagedVectorData::addValuesByLocalID( size_t N,
                                            const size_t *ndx,
                                            const void *val,
                                            const typeID &id )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::SETTING );
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
    getEngineData( *this )->addValuesByLocalID( N, ndx, val, id );
    fireDataChange();
}
void ManagedVectorData::addGhostValuesByGlobalID( size_t N,
                                                  const size_t *ndx,
                                                  const void *vals,
                                                  const typeID &id )
{
    auto vec = getVectorEngine();
    if ( !vec ) {
        GhostDataHelper<double>::addGhostValuesByGlobalID( N, ndx, vals, id );
    } else {
        vec->getVectorData()->addGhostValuesByGlobalID( N, ndx, vals, id );
    }
}
size_t ManagedVectorData::getAllGhostValues( void *vals, const typeID &id ) const
{
    auto vec = getVectorEngine();
    if ( !vec ) {
        return GhostDataHelper<double>::getAllGhostValues( vals, id );
    } else {
        return vec->getVectorData()->getAllGhostValues( vals, id );
    }
}
void ManagedVectorData::makeConsistent( ScatterType t )
{
    auto vec = getVectorEngine();
    if ( !vec ) {
        GhostDataHelper<double>::makeConsistent( t );
    } else {
        vec->makeConsistent( t );
    }
    *d_UpdateState = UpdateState::UNCHANGED;
}

void ManagedVectorData::putRawData( const void *in, const typeID &id )
{
    getEngineData( *this )->putRawData( in, id );
}

void ManagedVectorData::getRawData( void *in, const typeID &id ) const
{
    getEngineData( *this )->getRawData( in, id );
}

typeID ManagedVectorData::getType( size_t i ) const { return getEngineData( *this )->getType( i ); }

std::shared_ptr<VectorData> ManagedVectorData::cloneData( const std::string &name ) const
{
    auto vec = getVectorEngine();
    AMP_ASSERT( vec );
    auto cname  = ( name == "" ) ? "ManagedVectorClone" : name;
    auto vec2   = vec->clone( cname );
    auto retVal = std::make_shared<ManagedVectorData>( vec2 );
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

bool ManagedVectorData::hasContiguousData() const
{
    return getEngineData( *this )->hasContiguousData();
}

Vector::shared_ptr ManagedVectorData::getVectorEngine() { return d_Engine; }

Vector::const_shared_ptr ManagedVectorData::getVectorEngine() const { return d_Engine; }


} // namespace AMP::LinearAlgebra
