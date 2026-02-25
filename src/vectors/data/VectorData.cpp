#include "AMP/vectors/data/VectorData.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/data/DataChangeListener.h"
#include "AMP/vectors/data/MultiVectorData.h"
#include "AMP/vectors/data/VectorDataDefault.h"
#include "AMP/vectors/data/VectorDataFactory.h"

#include "ProfilerApp.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * makeConsistent / UpdateState                                  *
 ****************************************************************/
UpdateState VectorData::getGlobalUpdateStatus() const
{
    PROFILE( "VectorData::getGlobalUpdateStatus" );

    auto local = getComm().allGather<uint8_t>( static_cast<int8_t>( getLocalUpdateStatus() ) );
    auto state = UpdateState::UNCHANGED;
    for ( auto s : local ) {
        auto s2 = static_cast<UpdateState>( s );
        if ( s2 == UpdateState::UNCHANGED ) {
            continue;
        } else if ( s2 == UpdateState::LOCAL_CHANGED && state == UpdateState::UNCHANGED ) {
            state = UpdateState::LOCAL_CHANGED;
        } else if ( s2 == UpdateState::LOCAL_CHANGED ) {
            continue;
        } else if ( s2 == UpdateState::ADDING &&
                    ( state == UpdateState::UNCHANGED || state == UpdateState::LOCAL_CHANGED ||
                      state == UpdateState::ADDING ) ) {
            state = UpdateState::ADDING;
        } else if ( s2 == UpdateState::SETTING &&
                    ( state == UpdateState::UNCHANGED || state == UpdateState::LOCAL_CHANGED ||
                      state == UpdateState::SETTING ) ) {
            state = UpdateState::SETTING;
        } else {
            state = UpdateState::MIXED;
        }
    }
    return state;
}

void VectorData::makeConsistent()
{
    PROFILE( "VectorData::makeConsistent" );

    auto state = getGlobalUpdateStatus();
    if ( state == UpdateState::UNCHANGED ) {
        return;
    } else if ( state == UpdateState::LOCAL_CHANGED || state == UpdateState::SETTING ) {
        makeConsistent( ScatterType::CONSISTENT_SET );
    } else if ( state == UpdateState::ADDING || state == UpdateState::MIXED ) {
        makeConsistent( ScatterType::CONSISTENT_ADD );
    } else {
        AMP_ERROR( "Unknown update status" );
    }
}


/****************************************************************
 * Check if two data blocks are alias to each other              *
 ****************************************************************/
bool VectorData::isAnAliasOf( const VectorData &rhs ) const
{
    if ( numberOfDataBlocks() != rhs.numberOfDataBlocks() )
        return false;
    for ( size_t i = 0; i < numberOfDataBlocks(); i++ ) {
        if ( sizeOfDataBlock( i ) != rhs.sizeOfDataBlock( i ) ||
             getRawDataBlockAsVoid( i ) != rhs.getRawDataBlockAsVoid( i ) )
            return false;
    }
    return true;
}


/****************************************************************
 * Get the local sizes on each rank                              *
 ****************************************************************/
std::vector<size_t> VectorData::getLocalSizes() const
{
    PROFILE( "VectorData::getLocalSizes" );

    auto commList = getCommunicationList();
    if ( commList ) {
        auto x = commList->getPartition();
        for ( int i = ( (int) x.size() ) - 1; i > 0; i-- )
            x[i] -= x[i - 1];
        return x;
    } else {
        return getComm().allGather( d_localSize );
    }
}


/****************************************************************
 * dump data to ostream                                          *
 ****************************************************************/
void VectorData::dumpOwnedData( std::ostream &out, size_t GIDoffset, size_t LIDoffset ) const
{
    auto curElement = begin();
    size_t gid      = GIDoffset;
    if ( getCommunicationList() )
        gid += getCommunicationList()->getStartGID();
    size_t lid = LIDoffset;
    while ( curElement != end() ) {
        out << "  GID: " << gid << "  LID: " << lid << "  Value: " << *curElement << "\n";
        ++curElement;
        ++gid;
        ++lid;
    }
}


/****************************************************************
 * Component data                                                *
 ****************************************************************/
size_t VectorData::getNumberOfComponents() const { return 1; }
std::shared_ptr<VectorData> VectorData::getComponent( size_t i )
{
    AMP_ASSERT( getNumberOfComponents() == 1 );
    AMP_ASSERT( i == 0 );
    return shared_from_this();
}
std::shared_ptr<const VectorData> VectorData::getComponent( size_t i ) const
{
    AMP_ASSERT( getNumberOfComponents() == 1 );
    AMP_ASSERT( i == 0 );
    return shared_from_this();
}


/****************************************************************
 * reset vector data                                             *
 ****************************************************************/
void VectorData::reset() {}
void VectorData::setNoGhosts() {}
void VectorData::print( std::ostream &, const std::string &, const std::string & ) const
{
    AMP_ERROR( "Not implemented" );
}


/****************************************************************
 * Get an id                                                     *
 ****************************************************************/
uint64_t VectorData::getID() const
{
    return getComm().bcast( reinterpret_cast<uint64_t>( this ), 0 );
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void VectorData::registerChildObjects( AMP::IO::RestartManager * ) const {}
void VectorData::writeRestart( int64_t fid ) const
{
    IO::writeHDF5( fid, "localSize", d_localSize );
    IO::writeHDF5( fid, "globalSize", d_globalSize );
    IO::writeHDF5( fid, "localStart", d_localStart );
}
VectorData::VectorData( int64_t fid, AMP::IO::RestartManager * )
{
    IO::readHDF5( fid, "localSize", d_localSize );
    IO::readHDF5( fid, "globalSize", d_globalSize );
    IO::readHDF5( fid, "localStart", d_localStart );
}

} // namespace AMP::LinearAlgebra


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::VectorData>::DataStoreType(
    std::shared_ptr<const AMP::LinearAlgebra::VectorData> data, RestartManager *manager )
    : d_data( data )
{
    d_hash = data->getID();
    d_data->registerChildObjects( manager );
    // Register the communication list
    auto commList = data->getCommunicationList();
    if ( commList )
        manager->registerObject( commList );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::VectorData>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid         = createGroup( fid, name );
    auto commList     = d_data->getCommunicationList();
    auto commListHash = commList ? commList->getID() : 0;
    writeHDF5( gid, "ClassType", d_data->VectorDataName() );
    writeHDF5( gid, "CommListHash", commListHash );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::LinearAlgebra::VectorData>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::VectorData>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid    = openGroup( fid, name );
    auto vecData = AMP::LinearAlgebra::VectorDataFactory::create( gid, manager );
    closeGroup( gid );
    return vecData;
}
template<>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::UpdateState>::DataStoreType(
    std::shared_ptr<const AMP::LinearAlgebra::UpdateState> data, RestartManager * )
    : d_data( data )
{
    d_hash = reinterpret_cast<uint64_t>( d_data.get() );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::UpdateState>::write(
    hid_t fid, const std::string &name ) const
{
    int value = static_cast<int>( *d_data );
    writeHDF5( fid, name, value );
}
template<>
std::shared_ptr<AMP::LinearAlgebra::UpdateState>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::UpdateState>::read(
    hid_t fid, const std::string &name, RestartManager * ) const
{
    int value;
    readHDF5( fid, name, value );
    auto state = static_cast<AMP::LinearAlgebra::UpdateState>( value );
    return std::make_shared<AMP::LinearAlgebra::UpdateState>( state );
}
