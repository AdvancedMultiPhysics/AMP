#include "AMP/discretization/MultiDOFHelper.h"
#include "AMP/IO/HDF.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/data/VectorData.h"


namespace AMP::Discretization {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
multiDOFHelper::multiDOFHelper( const DOFManager &manager )
{
    initialize( manager.getComm().getRank(), manager.getLocalSizes() );
}
multiDOFHelper::multiDOFHelper( const AMP::LinearAlgebra::VectorData &data )
{
    initialize( data.getComm().getRank(), data.getLocalSizes() );
}
multiDOFHelper::multiDOFHelper( const std::vector<std::shared_ptr<DOFManager>> &managers,
                                const AMP::AMP_MPI &comm )
{
    PROFILE( "multiDOFHelper" );
    AMP::Array<size_t> data( 3, managers.size() );
    for ( size_t i = 0; i < managers.size(); i++ ) {
        data( 0, i ) = managers[i]->getComm().rand();
        data( 1, i ) = managers[i]->numLocalDOF();
        data( 2, i ) = managers[i]->beginDOF();
    }
    initialize( comm, data );
    AMP_ASSERT( d_index.size() == managers.size() );
}
multiDOFHelper::multiDOFHelper( const std::vector<AMP::LinearAlgebra::VectorData *> &vectorData,
                                const AMP::AMP_MPI &comm )
{
    PROFILE( "multiDOFHelper" );
    AMP::Array<size_t> data( 3, vectorData.size() );
    for ( size_t i = 0; i < vectorData.size(); i++ ) {
        data( 0, i ) = vectorData[i]->getComm().rand();
        data( 1, i ) = vectorData[i]->getLocalSize();
        data( 2, i ) = vectorData[i]->getLocalStartID();
    }
    initialize( comm, data );
    AMP_ASSERT( d_index.size() == vectorData.size() );
}
static std::tuple<std::vector<int>, AMP::Array<size_t>>
gatherData( const AMP::AMP_MPI &comm, const AMP::Array<size_t> &sendData )
{
    if ( comm.getSize() <= 1 ) {
        std::vector<int> count( 1, sendData.size( 1 ) );
        return std::tie( count, sendData );
    }
    PROFILE( "gatherData" );
    // Get the data of vectors on each rank
    auto count = comm.allGather<int>( sendData.length() );
    size_t N   = count[0];
    for ( auto size : count )
        N += size;
    if ( N == 0 ) {
        AMP::Array<size_t> recvData( 3, 0 );
        return std::tie( count, recvData );
    }
    // Send the data to all ranks
    AMP::Array<size_t> recvData( 3, N / 3 );
    std::vector<int> disp( comm.getSize(), 0 );
    for ( size_t i = 1; i < count.size(); i++ )
        disp[i] = disp[i - 1] + count[i - 1];
    comm.allGather<size_t>(
        sendData.data(), sendData.length(), recvData.data(), count.data(), disp.data(), true );
    // Return the results
    for ( auto &c : count )
        c /= 3;
    return std::tie( count, recvData );
}
void multiDOFHelper::initialize( const AMP::AMP_MPI &comm, const AMP::Array<size_t> &sendData )
{
    d_rank       = comm.getRank();
    int commSize = comm.getSize();
    // Gather the data
    auto [count, recvData] = gatherData( comm, sendData );
    if ( recvData.empty() ) {
        d_local = std::vector<size_t>( commSize, 0 );
        d_begin = std::vector<size_t>( commSize, 0 );
        return;
    }
    // Get the unique ids
    std::set<size_t> id_list;
    for ( size_t i = 0; i < recvData.size( 1 ); i++ )
        id_list.insert( recvData( 0, i ) );
    // Get the index for each object
    d_index.resize( sendData.size( 1 ) );
    for ( size_t i = 0; i < d_index.size(); i++ ) {
        d_index[i] = std::distance( id_list.begin(), id_list.find( sendData( 0, i ) ) );
        AMP_ASSERT( d_index[i] < id_list.size() );
    }
    // Build the arrays
    d_localSize.resize( id_list.size(), commSize );
    d_localOffset.resize( id_list.size(), commSize );
    d_globalOffset.resize( id_list.size(), commSize );
    d_localSize.fill( 0 );
    d_localOffset.fill( 0 );
    d_globalOffset.fill( 0 );
    d_local.resize( commSize, 0 );
    d_begin.resize( commSize, 0 );
    size_t i     = 0;
    size_t start = 0;
    for ( int rank = 0; rank < commSize; rank++ ) {
        d_begin[rank] = start;
        for ( int j = 0; j < count[rank]; j++, i++ ) {
            auto id                       = recvData( 0, i );
            auto size                     = recvData( 1, i );
            auto offset                   = recvData( 2, i );
            auto index                    = std::distance( id_list.begin(), id_list.find( id ) );
            d_localSize( index, rank )    = size;
            d_localOffset( index, rank )  = offset;
            d_globalOffset( index, rank ) = start;
            start += size;
        }
        d_local[rank] = start - d_begin[rank];
    }
    // Fill any remaining data
    d_globalSize.resize( d_index.size(), 0 );
    for ( size_t i = 0; i < d_index.size(); i++ ) {
        for ( size_t j = 0; j < d_localSize.size( 1 ); j++ )
            d_globalSize[i] += d_localSize( d_index[i], j );
    }
}
void multiDOFHelper::initialize( int rank, std::vector<size_t> &&data )
{
    d_rank  = rank;
    d_index = { 0 };
    d_local = std::move( data );
    d_begin.resize( d_local.size(), 0 );
    d_localSize.resize( 1, d_local.size() );
    d_localOffset.resize( 1, d_local.size() );
    d_globalOffset.resize( 1, d_local.size() );
    d_localSize( 0 ) = d_local[0];
    d_localOffset.fill( 0 );
    for ( size_t i = 1; i < d_local.size(); i++ ) {
        d_begin[i]         = d_begin[i - 1] + d_local[i - 1];
        d_localSize( i )   = d_local[i];
        d_localOffset( i ) = d_begin[i];
    }
    d_globalOffset = d_localOffset.view();
    d_globalSize   = { d_localSize.sum() };
}
multiDOFHelper::multiDOFHelper( const multiDOFHelper &rhs )
    : d_rank( rhs.d_rank ),
      d_index( rhs.d_index ),
      d_local( rhs.d_local ),
      d_begin( rhs.d_begin ),
      d_globalSize( rhs.d_globalSize ),
      d_localSize( const_cast<Array<size_t> &>( rhs.d_localSize ).view() ),
      d_localOffset( const_cast<Array<size_t> &>( rhs.d_localOffset ).view() ),
      d_globalOffset( const_cast<Array<size_t> &>( rhs.d_globalOffset ).view() )
{
}
multiDOFHelper &multiDOFHelper::operator=( const multiDOFHelper &rhs )
{
    if ( &rhs == this )
        return *this;
    d_rank         = rhs.d_rank;
    d_index        = rhs.d_index;
    d_local        = rhs.d_local;
    d_begin        = rhs.d_begin;
    d_globalSize   = rhs.d_globalSize;
    d_localSize    = const_cast<Array<size_t> &>( rhs.d_localSize ).view();
    d_localOffset  = const_cast<Array<size_t> &>( rhs.d_localOffset ).view();
    d_globalOffset = const_cast<Array<size_t> &>( rhs.d_globalOffset ).view();
    return *this;
}


/****************************************************************
 * Convert between local and global ids                          *
 ****************************************************************/
size_t multiDOFHelper::subToGlobal( int manager, size_t dof ) const
{
    AMP_ASSERT( manager < (int) d_index.size() );
    size_t index = d_index[manager];
    int rank     = -1;
    for ( size_t i = 0; i < d_localSize.size( 1 ); i++ ) {
        size_t offset = d_localOffset( index, i );
        size_t size   = d_localSize( index, i );
        if ( dof >= offset && dof < offset + size )
            rank = i;
    }
    AMP_ASSERT( rank != -1 );
    size_t dof2 = d_globalOffset( index, rank ) - d_localOffset( index, rank ) + dof;
    AMP_DEBUG_ASSERT( dof2 < numGlobal() );
    return dof2;
}
std::pair<size_t, int> multiDOFHelper::globalToSub( size_t dof ) const
{
    constexpr size_t neg_one = ~( (size_t) 0 );
    AMP_DEBUG_ASSERT( dof < numGlobal() );
    size_t dof2 = neg_one;
    size_t id   = neg_one;
    for ( size_t i = 0; i < d_localSize.length(); i++ ) {
        size_t offset = d_globalOffset( i );
        size_t size   = d_localSize( i );
        if ( dof >= offset && dof < offset + size ) {
            dof2 = dof - d_globalOffset( i ) + d_localOffset( i );
            id   = i % d_globalOffset.size( 0 );
            break;
        }
    }
    for ( size_t j = 0; j < d_index.size(); j++ ) {
        if ( id == d_index[j] ) {
            AMP_ASSERT( dof2 < d_globalSize[j] );
            return std::pair<size_t, int>( dof2, j );
        }
    }
    // DOF is not present on the current processor
    return std::pair<size_t, int>( neg_one, -1 );
}
std::vector<size_t> multiDOFHelper::getGlobalDOF( const int manager,
                                                  const std::vector<size_t> &subDOFs ) const
{
    std::vector<size_t> dofs( subDOFs.size() );
    for ( size_t i = 0; i < dofs.size(); i++ )
        dofs[i] = subToGlobal( manager, subDOFs[i] );
    return dofs;
}
std::vector<size_t> multiDOFHelper::getSubDOF( const int manager,
                                               const std::vector<size_t> &globalDOFs ) const
{
    constexpr size_t neg_one = ~( (size_t) 0 );
    std::vector<size_t> dofs( globalDOFs.size() );
    for ( size_t i = 0; i < dofs.size(); i++ ) {
        auto map = globalToSub( globalDOFs[i] );
        dofs[i]  = map.second == manager ? map.first : neg_one;
    }
    return dofs;
}


/********************************************************
 *  Write/Read from HDF5                                 *
 ********************************************************/
void multiDOFHelper::writeHDF5( size_t gid ) const
{
    PROFILE( "writeHDF5", 1 );
    AMP::IO::writeHDF5( gid, "rank", d_rank );
    AMP::IO::writeHDF5( gid, "index", d_index );
    AMP::IO::writeHDF5( gid, "local", d_local );
    AMP::IO::writeHDF5( gid, "begin", d_begin );
    AMP::IO::writeHDF5( gid, "globalSize", d_globalSize );
    AMP::IO::writeHDF5( gid, "localSize", d_localSize );
    AMP::IO::writeHDF5( gid, "localOffset", d_localOffset );
    AMP::IO::writeHDF5( gid, "globalOffset", d_globalOffset );
}
multiDOFHelper::multiDOFHelper( size_t gid )
{
    PROFILE( "readHDF5", 1 );
    AMP::IO::readHDF5( gid, "rank", d_rank );
    AMP::IO::readHDF5( gid, "index", d_index );
    AMP::IO::readHDF5( gid, "local", d_local );
    AMP::IO::readHDF5( gid, "begin", d_begin );
    AMP::IO::readHDF5( gid, "globalSize", d_globalSize );
    AMP::IO::readHDF5( gid, "localSize", d_localSize );
    AMP::IO::readHDF5( gid, "localOffset", d_localOffset );
    AMP::IO::readHDF5( gid, "globalOffset", d_globalOffset );
}


} // namespace AMP::Discretization


/********************************************************
 *  Write/Read from HDF5                                 *
 ********************************************************/
#include "AMP/IO/HDF.hpp"
template<>
void AMP::IO::writeHDF5<AMP::Discretization::multiDOFHelper>(
    hid_t fid, const std::string &name, const AMP::Discretization::multiDOFHelper &data )
{
    auto gid = AMP::IO::createGroup( fid, name );
    data.writeHDF5( gid );
    AMP::IO::closeGroup( fid );
}
template<>
void AMP::IO::readHDF5<AMP::Discretization::multiDOFHelper>(
    hid_t fid, const std::string &name, AMP::Discretization::multiDOFHelper &data )
{
    auto gid = AMP::IO::openGroup( fid, name );
    data     = AMP::Discretization::multiDOFHelper( gid );
    AMP::IO::closeGroup( fid );
}
