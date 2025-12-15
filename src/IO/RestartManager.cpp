#include "AMP/IO/RestartManager.h"
#include "AMP/IO/RestartManager.hpp"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <complex>
#include <cstddef>
#include <set>
#include <string>
#include <vector>


namespace AMP::IO {


/********************************************************
 *  Constructor/destructor                               *
 ********************************************************/
RestartManager::RestartManager() : d_fid( -1 ) {}
RestartManager::RestartManager( const std::string &name ) : d_fid( -1 ) { load( name ); }
RestartManager::RestartManager( RestartManager &&rhs ) : d_fid( -1 )
{
    std::swap( d_fid, rhs.d_fid );
    std::swap( d_data, rhs.d_data );
    std::swap( d_names, rhs.d_names );
    std::swap( d_comms, rhs.d_comms );
}
RestartManager &RestartManager::operator=( RestartManager &&rhs )
{
    if ( this == &rhs )
        return *this;
    std::swap( d_fid, rhs.d_fid );
    std::swap( d_data, rhs.d_data );
    std::swap( d_names, rhs.d_names );
    std::swap( d_comms, rhs.d_comms );
    return *this;
}
RestartManager::~RestartManager() { reset(); }
void RestartManager::reset()
{
    if ( d_fid != hid_t( -1 ) )
        closeHDF5( d_fid );
    d_fid   = hid_t( -1 );
    d_data  = {};
    d_names = {};
    d_comms = {};
}
void RestartManager::load( const std::string &name )
{
    PROFILE( "load" );
    reset(); // Clear existing data
    int rank = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    d_data.clear();
    d_names.clear();
    auto file = name + "." + AMP::Utilities::nodeToString( rank ) + ".h5";
    d_fid     = openHDF5( file, "r" );
    std::vector<std::string> names;
    std::vector<uint64_t> ids;
    readHDF5( d_fid, "RestartDataIDs", ids );
    readHDF5( d_fid, "RestartDataNames", names );
    for ( size_t i = 0; i < names.size(); i++ )
        d_names[names[i]] = ids[i];
    readCommData( name );
}


/********************************************************
 *  Write the data                                       *
 ********************************************************/
void RestartManager::write( const std::string &name, Compression compress )
{
    PROFILE( "write" );
    int rank  = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    auto file = name + "." + AMP::Utilities::nodeToString( rank ) + ".h5";
    auto fid  = openHDF5( file, "w", compress );
    std::vector<std::string> names;
    std::vector<uint64_t> ids;
    for ( const auto &[name, id] : d_names ) {
        names.push_back( name );
        ids.push_back( id );
    }
    writeHDF5( fid, "RestartDataIDs", ids );
    writeHDF5( fid, "RestartDataNames", names );
    for ( const auto &[id, data] : d_data ) {
        auto name = hash2String( id );
        data->write( fid, name );
    }
    closeHDF5( fid );
    writeCommData( name, compress );
}


/********************************************************
 *  Register/load communicators                          *
 ********************************************************/
uint64_t RestartManager::registerComm( const AMP::AMP_MPI &comm )
{
    AMP_INSIST( d_fid == hid_t( -1 ),
                "We cannot register new items while a restart file is being read" );
    auto hash = comm.hash();
    // No need to register known comms
    if ( hash == AMP_MPI::hashNull || hash == AMP_MPI::hashSelf || hash == AMP_MPI::hashWorld ||
         hash == AMP_MPI::hashMPI )
        return hash;
    // Check if we have previously registered the comm and do so
    if ( d_comms.find( hash ) != d_comms.end() )
        return hash;
    d_comms[hash] = comm;
    return hash;
}
AMP_MPI RestartManager::getComm( uint64_t hash )
{
    // Check if it is a known comm
    if ( hash == AMP_MPI::hashNull )
        return AMP_COMM_NULL;
    if ( hash == AMP_MPI::hashSelf )
        return AMP_COMM_SELF;
    if ( hash == AMP_MPI::hashWorld )
        return AMP_COMM_WORLD;
#ifdef AMP_USE_MPI
    if ( hash == AMP_MPI::hashMPI )
        return MPI_COMM_WORLD;
#else
    if ( hash == AMP_MPI::hashMPI )
        return AMP_COMM_WORLD;
#endif
    // Find the appropriate comm
    auto it = d_comms.find( hash );
    if ( it == d_comms.end() )
        AMP_ERROR( "Unable to find comm: " + std::to_string( hash ) );
    return it->second;
}
void RestartManager::writeCommData( const std::string &name, Compression compress )
{
    PROFILE( "writeCommData" );
    // Collect the comm data
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    int rank = globalComm.getRank();
    int size = globalComm.getSize();
    // Get the list of comm ids and their hash ranks
    std::map<uint64_t, uint64_t> hashMap;
    for ( auto &[id, comm] : d_comms )
        hashMap[id] = comm.hashRanks();
    globalComm.mapGather( hashMap );
    // Get the list of hashRanks values
    std::set<uint64_t> tmp;
    for ( auto [id, value] : hashMap )
        tmp.insert( value );
    std::vector<uint64_t> hashRanks( tmp.begin(), tmp.end() );
    // Create the id list and index to the appropriate hashRanks entry
    std::vector<uint64_t> ids;
    std::vector<int> index;
    for ( auto [id, key] : hashMap ) {
        ids.push_back( id );
        index.push_back( AMP::Utilities::findfirst( hashRanks, key ) );
    }
    size_t N = hashRanks.size();
    // Create the global rank map
    AMP::Array<int> data( size, N );
    data.fill( -1 );
    for ( auto &[id, comm] : d_comms ) {
        size_t i   = AMP::Utilities::findfirst( ids, id );
        size_t j   = index[i];
        auto ranks = comm.globalRanks();
        for ( size_t k = 0; k < ranks.size(); k++ )
            data( ranks[k], j ) = k;
    }
    globalComm.maxReduce( data.data(), data.length() );
    // Write the comm data
    if ( rank == 0 ) {
        auto file = name + ".comms.h5";
        auto fid  = openHDF5( file, "w", compress );
        writeHDF5( fid, "ids", ids );
        writeHDF5( fid, "index", index );
        writeHDF5( fid, "data", data );
        closeHDF5( fid );
    }
}
void RestartManager::readCommData( const std::string &name )
{
    PROFILE( "readCommData" );
    // Load the comm data
    auto file = name + ".comms.h5";
    std::vector<uint64_t> ids;
    std::vector<int> index;
    AMP::Array<int> data;
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto rank = globalComm.getRank();
    if ( rank == 0 ) {
        auto fid = openHDF5( file, "r" );
        readHDF5( fid, "ids", ids );
        readHDF5( fid, "index", index );
        readHDF5( fid, "data", data );
        closeHDF5( fid );
    }
    ids   = globalComm.bcast( ids, 0 );
    index = globalComm.bcast( index, 0 );
    data  = globalComm.bcast( data, 0 );
    // Check that the global size did not change
    AMP_INSIST( (int) data.size( 0 ) == globalComm.getSize(),
                "The global communicator size changed" );
    // Create the comms
    d_comms.clear();
    std::vector<int> size( data.size( 1 ), 0 );
    for ( size_t i = 0; i < data.size( 1 ); i++ ) {
        for ( size_t j = 0; j < data.size( 0 ); j++ )
            if ( data( j, i ) != -1 )
                size[i]++;
        AMP_ASSERT( size[i] > 0 );
    }
    for ( size_t i = 0; i < ids.size(); i++ ) {
        int idx = index[i];
        int key = data( rank, idx );
        if ( size[idx] > 1 ) {
            auto comm = globalComm.split( key == -1 ? -1 : 0, key );
            if ( key != -1 )
                d_comms[ids[i]] = comm;
        } else if ( size[idx] == 1 && key != -1 ) {
            d_comms[ids[i]] = AMP::AMP_MPI( AMP_COMM_SELF ).dup();
        }
    }
}


/********************************************************
 *  Register data with the manager                       *
 ********************************************************/
std::string AMP::IO::RestartManager::hash2String( uint64_t id )
{
    char id_chars[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$";
    std::string name;
    while ( id > 0 ) {
        name += id_chars[id & 0x3F];
        id >>= 6;
    }
    return name;
}
bool AMP::IO::RestartManager::isRegistered( uint64_t hash )
{
    bool test1 = d_data.find( hash ) != d_data.end();
    auto test2 = d_comms.find( hash ) != d_comms.end();
    return test1 || test2;
}


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
template<class TYPE>
AMP::IO::RestartManager::DataStoreType<TYPE>::DataStoreType( std::shared_ptr<const TYPE> data,
                                                             RestartManager * )
    : d_data( data )
{
    d_hash = reinterpret_cast<uint64_t>( data.get() );
}
template<class TYPE>
void RestartManager::DataStoreType<TYPE>::write( hid_t fid, const std::string &name ) const
{
    writeHDF5( fid, name, *d_data );
}
template<class TYPE>
std::shared_ptr<TYPE> RestartManager::DataStoreType<TYPE>::read( hid_t fid,
                                                                 const std::string &name,
                                                                 RestartManager * ) const
{
    auto data = std::make_shared<TYPE>();
    readHDF5( fid, name, *data );
    return data;
}
#define INSTANTIATE( TYPE )                                                          \
    template class RestartManager::DataStoreType<TYPE>;                              \
    template void RestartManager::registerData( const TYPE &, const std::string & ); \
    template std::shared_ptr<TYPE> RestartManager::getData( uint64_t );              \
    template std::shared_ptr<TYPE> RestartManager::getData( const std::string & )
#define INSTANTIATE2( TYPE )          \
    INSTANTIATE( TYPE );              \
    INSTANTIATE( std::vector<TYPE> ); \
    INSTANTIATE( AMP::Array<TYPE> )
INSTANTIATE( bool );
INSTANTIATE( char );
INSTANTIATE( uint8_t );
INSTANTIATE( uint16_t );
INSTANTIATE( uint32_t );
INSTANTIATE( uint64_t );
INSTANTIATE( int8_t );
INSTANTIATE( int16_t );
INSTANTIATE( int32_t );
INSTANTIATE( int64_t );
INSTANTIATE( float );
INSTANTIATE( double );
INSTANTIATE( std::complex<float> );
INSTANTIATE( std::complex<double> );
INSTANTIATE( std::byte );
INSTANTIATE( std::string );
INSTANTIATE( std::string_view );
INSTANTIATE( AMP::Database );
INSTANTIATE( std::vector<double> );


} // namespace AMP::IO
