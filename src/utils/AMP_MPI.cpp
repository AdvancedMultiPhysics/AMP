// This file implements a wrapper class for MPI functions

// Include AMP headers
#include "AMP/AMP_TPLs.h"

#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/threadpool/ThreadHelpers.h"

#include "ProfilerApp.h"
#include "StackTrace/ErrorHandlers.h"
#include "StackTrace/StackTrace.h"

// Include all other headers
#include <algorithm>
#include <chrono>
#include <climits>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>


// Include mpi.h (or define MPI objects)
#ifdef AMP_USE_MPI
    #include "mpi.h"
#elif defined( AMP_USE_PETSC )
    #include "petsc/mpiuni/mpi.h"
#elif defined( AMP_USE_SAMRAI )
    #include "SAMRAI/tbox/SAMRAI_MPI.h"
#elif defined( AMP_USE_TRILINOS )
    #include "mpi.h"
#elif defined( __has_include )
    #if __has_include( "mpi.h" )
        #include "mpi.h"
    #else
        #define SET_MPI_TYPES
    #endif
#else
    #define SET_MPI_TYPES
#endif
#if defined( SET_MPI_TYPES )
typedef uint32_t MPI_Comm;
typedef uint32_t MPI_Request;
typedef uint32_t MPI_Status;
typedef void *MPI_Errhandler;
    #define MPI_COMM_NULL ( (MPI_Comm) 0xF4000010 )
    #define MPI_COMM_WORLD ( (MPI_Comm) 0xF4000012 )
    #define MPI_COMM_SELF ( (MPI_Comm) 0xF4000011 )
#endif


namespace std {
template<class TYPE>
static inline bool operator<( const std::complex<TYPE> &a, const std::complex<TYPE> &b )
{
    if ( a.real() < b.real() )
        return true;
    if ( a.real() > b.real() )
        return false;
    return a.imag() < b.imag();
}
} // namespace std


namespace AMP {


// Check the alignment
static_assert( sizeof( AMP_MPI ) % 8 == 0 );


// Initialized static member variables
short AMP_MPI::profile_level                      = 255;
AMP_MPI::atomic_int AMP_MPI::N_MPI_Comm_created   = 0;
AMP_MPI::atomic_int AMP_MPI::N_MPI_Comm_destroyed = 0;
AMP_MPI::Comm AMP_MPI::commNull                   = ( (AMP_MPI::Comm) 0xF400F001 );
AMP_MPI::Comm AMP_MPI::commSelf                   = ( (AMP_MPI::Comm) 0xF400F003 );
AMP_MPI::Comm AMP_MPI::commWorld                  = ( (AMP_MPI::Comm) 0xF400F007 );
int AMP_MPI::d_maxTag                             = 0x3FFFFFFF;


// Static data for asyncronous communication without MPI
// Note: these routines may not be thread-safe yet
#ifndef AMP_USE_MPI
struct Isendrecv_struct {
    const void *data;   // Pointer to data
    int bytes;          // Number of bytes in the message
    int status;         // Status: 1-sending, 2-recieving
    AMP_MPI::Comm comm; // Communicator
    int tag;            // Tag
};
std::map<AMP_MPI::Request2, Isendrecv_struct> global_isendrecv_list;
static AMP_MPI::Request2 getRequest( AMP_MPI::Comm comm, int tag )
{
    AMP_ASSERT( tag >= 0 );
    // Use hashing function: 2^64*0.5*(sqrt(5)-1)
    uint64_t a    = static_cast<uint8_t>( comm ) * 0x9E3779B97F4A7C15;
    uint64_t b    = static_cast<uint8_t>( tag ) * 0x9E3779B97F4A7C15;
    uint64_t hash = a ^ b;
    AMP_MPI::Request2 request;
    memcpy( &request, &hash, sizeof( AMP_MPI::Request2 ) );
    return request;
}
#endif


/************************************************************************
 *  Get the MPI version                                                  *
 ************************************************************************/
std::array<int, 2> AMP_MPI::version()
{
#ifdef AMP_USE_MPI
    int MPI_version;
    int MPI_subversion;
    MPI_Get_version( &MPI_version, &MPI_subversion );
    return { MPI_version, MPI_subversion };
#else
    return { 0, 0 };
#endif
}
std::string AMP_MPI::info()
{
    std::string MPI_info;
#ifdef AMP_USE_MPI
    #if MPI_VERSION >= 3
    int MPI_version_length = 0;
    char MPI_version_string[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Get_library_version( MPI_version_string, &MPI_version_length );
    if ( MPI_version_length > 0 ) {
        MPI_info   = std::string( MPI_version_string, MPI_version_length );
        size_t pos = MPI_info.find( '\n' );
        while ( pos != std::string::npos ) {
            MPI_info.insert( pos + 1, "   " );
            pos = MPI_info.find( '\n', pos + 1 );
        }
    }
    #else
    auto tmp = version();
    MPI_info = std::to_string( tmp[0] ) + "." + std::to_string( tmp[0] );
    #endif
    size_t pos = MPI_info.find( "\n\n" );
    while ( pos != std::string::npos ) {
        MPI_info.erase( pos + 1, 1 );
        pos = MPI_info.find( '\n', pos + 1 );
    }
    while ( MPI_info.back() == '\n' || MPI_info.back() == '\r' || MPI_info.back() == ' ' )
        MPI_info.pop_back();
    MPI_info += '\n';
#endif
    return MPI_info;
}


/************************************************************************
 *  Functions to get/set the process affinities                          *
 ************************************************************************/
int AMP_MPI::getNumberOfProcessors() { return std::thread::hardware_concurrency(); }
std::vector<int> AMP_MPI::getProcessAffinity() { return AMP::Thread::getProcessAffinity(); }
void AMP_MPI::setProcessAffinity( const std::vector<int> &procs )
{
    AMP::Thread::setProcessAffinity( procs );
}


/************************************************************************
 *  Function to check if MPI is active                                   *
 ************************************************************************/
bool AMP_MPI::MPI_Active()
{
#ifdef AMP_USE_MPI
    int initialized = 0, finalized = 0;
    MPI_Initialized( &initialized );
    MPI_Finalized( &finalized );
    return initialized != 0 && finalized == 0;
#else
    return true;
#endif
}
AMP_MPI::ThreadSupport AMP_MPI::queryThreadSupport()
{
#ifdef AMP_USE_MPI
    int provided = 0;
    MPI_Query_thread( &provided );
    if ( provided == MPI_THREAD_SINGLE )
        return ThreadSupport::SINGLE;
    if ( provided == MPI_THREAD_FUNNELED )
        return ThreadSupport::FUNNELED;
    if ( provided == MPI_THREAD_SERIALIZED )
        return ThreadSupport::SERIALIZED;
    if ( provided == MPI_THREAD_MULTIPLE )
        return ThreadSupport::MULTIPLE;
    return ThreadSupport::SINGLE;
#else
    return ThreadSupport::MULTIPLE;
#endif
}


/************************************************************************
 *  Function to perform a load balance of the given processes            *
 ************************************************************************/
void AMP_MPI::balanceProcesses( const AMP_MPI &globalComm,
                                const int method,
                                const std::vector<int> &procs,
                                const int N_min_in,
                                const int N_max_in )
{
    // Build the list of processors to use
    std::vector<int> cpus = procs;
    if ( cpus.empty() ) {
        for ( int i = 0; i < getNumberOfProcessors(); i++ )
            cpus.push_back( i );
    }
    // Handle the "easy cases"
    if ( method == 1 ) {
        // Trivial case where we do not need any communication
        setProcessAffinity( cpus );
        return;
    }
    // Get the sub-communicator for the current node
    AMP_MPI nodeComm = globalComm.splitByNode();
    int N_min        = std::min<int>( std::max<int>( N_min_in, 1 ), cpus.size() );
    int N_max        = N_max_in;
    if ( N_max == -1 )
        N_max = cpus.size();
    N_max = std::min<int>( N_max, cpus.size() );
    AMP_ASSERT( N_max >= N_min );
    // Perform the load balance within the node
    if ( method == 2 ) {
        int N_proc = cpus.size() / nodeComm.getSize();
        N_proc     = std::max<int>( N_proc, N_min );
        N_proc     = std::min<int>( N_proc, N_max );
        std::vector<int> cpus2( N_proc, -1 );
        for ( int i = 0; i < N_proc; i++ )
            cpus2[i] = cpus[( nodeComm.getRank() * N_proc + i ) % cpus.size()];
        setProcessAffinity( cpus2 );
    } else {
        AMP_ERROR( "Unknown method for load balance" );
    }
}


/************************************************************************
 *  Empty constructor                                                    *
 ************************************************************************/
AMP_MPI::AMP_MPI() : d_comm( MPI_COMM_NULL ), d_size( 0 ) {}


/************************************************************************
 *  Empty deconstructor                                                  *
 ************************************************************************/
AMP_MPI::~AMP_MPI() { reset(); }
void AMP_MPI::reset()
{
    // Decrement the count if used
    int count = -1;
    if ( d_count )
        count = --( *d_count );
    if ( count == 0 ) {
        // We are holding that last reference to the MPI_Comm object, we need to free it
        if ( d_manage ) {
#if defined( AMP_USE_MPI ) || defined( AMP_USE_PETSC )
            MPI_Comm_set_errhandler( d_comm, MPI_ERRORS_ARE_FATAL );
            int err = MPI_Comm_free( (MPI_Comm *) &d_comm );
            if ( err != MPI_SUCCESS )
                AMP_ERROR( "Problem free'ing MPI_Comm object" );
            d_comm = commNull;
#endif
            ++N_MPI_Comm_destroyed;
        }
        if ( d_ranks )
            delete[] d_ranks;
        delete d_count;
    }
    if ( d_currentTag == nullptr ) {
        // No tag index
    } else if ( d_currentTag[1] > 1 ) {
        --( d_currentTag[1] );
    } else {
        delete[] d_currentTag;
        delete d_rand;
    }
    d_manage     = false;
    d_count      = nullptr;
    d_ranks      = nullptr;
    d_rank       = 0;
    d_size       = 1;
    d_hash       = hashNull;
    d_isNull     = true;
    d_currentTag = nullptr;
    d_call_abort = true;
}


/************************************************************************
 *  Copy constructors                                                    *
 ************************************************************************/
AMP_MPI::AMP_MPI( const AMP_MPI &comm )
    : d_comm( comm.d_comm ),
      d_isNull( comm.d_isNull ),
      d_manage( comm.d_manage ),
      d_call_abort( comm.d_call_abort ),
      d_rank( comm.d_rank ),
      d_size( comm.d_size ),
      d_hash( comm.d_hash ),
      d_currentTag( comm.d_currentTag ),
      d_ranks( comm.d_ranks ),
      d_count( comm.d_count ),
      d_rand( comm.d_rand )
{
    if ( d_currentTag )
        ++d_currentTag[1];
    if ( d_count )
        ++( *d_count );
}
AMP_MPI::AMP_MPI( AMP_MPI &&rhs ) : AMP_MPI()
{
    std::swap( d_comm, rhs.d_comm );
    std::swap( d_isNull, rhs.d_isNull );
    std::swap( d_manage, rhs.d_manage );
    std::swap( d_call_abort, rhs.d_call_abort );
    std::swap( profile_level, rhs.profile_level );
    std::swap( d_rank, rhs.d_rank );
    std::swap( d_size, rhs.d_size );
    std::swap( d_ranks, rhs.d_ranks );
    std::swap( d_hash, rhs.d_hash );
    std::swap( d_currentTag, rhs.d_currentTag );
    std::swap( d_count, rhs.d_count );
    std::swap( d_rand, rhs.d_rand );
}


/************************************************************************
 *  Assignment operators                                                 *
 ************************************************************************/
AMP_MPI &AMP_MPI::operator=( const AMP_MPI &comm )
{
    if ( this == &comm ) // protect against invalid self-assignment
        return *this;
    // Destroy the previous object
    this->reset();
    // Initialize the data members to the existing object
    this->d_comm       = comm.d_comm;
    this->d_rank       = comm.d_rank;
    this->d_size       = comm.d_size;
    this->d_ranks      = comm.d_ranks;
    this->d_isNull     = comm.d_isNull;
    this->d_manage     = comm.d_manage;
    this->d_hash       = comm.d_hash;
    this->d_rand       = comm.d_rand;
    this->d_call_abort = comm.d_call_abort;
    this->d_currentTag = comm.d_currentTag;
    if ( this->d_currentTag != nullptr )
        ++( this->d_currentTag[1] );
    // Set and increment the count
    this->d_count = comm.d_count;
    if ( this->d_count != nullptr )
        ++( *d_count );
    return *this;
}
AMP_MPI &AMP_MPI::operator=( AMP_MPI &&rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    std::swap( d_comm, rhs.d_comm );
    std::swap( d_isNull, rhs.d_isNull );
    std::swap( d_manage, rhs.d_manage );
    std::swap( d_call_abort, rhs.d_call_abort );
    std::swap( profile_level, rhs.profile_level );
    std::swap( d_rank, rhs.d_rank );
    std::swap( d_size, rhs.d_size );
    std::swap( d_ranks, rhs.d_ranks );
    std::swap( d_hash, rhs.d_hash );
    std::swap( d_currentTag, rhs.d_currentTag );
    std::swap( d_count, rhs.d_count );
    std::swap( d_rand, rhs.d_rand );
    return *this;
}


/************************************************************************
 *  Simple functions                                                     *
 *  Note: These are not in-lined to prevent symbols from being defined   *
 *     in translation units that may not otherwise be needed.            *
 ************************************************************************/
bool AMP_MPI::isNull() const { return d_isNull; }
int AMP_MPI::getRank() const { return d_rank; }
int AMP_MPI::getSize() const { return d_size; }
int AMP_MPI::maxTag() const { return d_maxTag; }
size_t AMP_MPI::MPI_Comm_created() { return N_MPI_Comm_created.load(); }
size_t AMP_MPI::MPI_Comm_destroyed() { return N_MPI_Comm_destroyed.load(); }
const AMP_MPI::Comm &AMP_MPI::getCommunicator() const { return d_comm; }


/************************************************************************
 *  Hash an MPI communicator                                             *
 ************************************************************************/
uint64_t AMP_MPI::hash() const { return d_hash; }
static inline uint64_t hashComm( AMP_MPI::Comm comm )
{
    if ( comm == MPI_COMM_NULL ) {
        return AMP_MPI::hashNull;
    } else if ( comm == MPI_COMM_SELF ) {
        return AMP_MPI::hashSelf;
    } else if ( comm == MPI_COMM_WORLD ) {
        return AMP_MPI::hashMPI;
    } else if ( comm == AMP_MPI::commWorld ) {
        return AMP_MPI::hashWorld;
    }
    uint64_t r = AMP::AMPManager::getCommWorld().getRank();
    uint64_t h = 0x6dac47f99495c90e + ( r << 32 ) + r;
    uint64_t x;
    if constexpr ( sizeof( comm ) == 4 ) {
        uint32_t y;
        memcpy( &y, &comm, sizeof( comm ) );
        x = y;
    } else if constexpr ( sizeof( comm ) == 8 ) {
        uint64_t y;
        memcpy( &y, &comm, sizeof( comm ) );
        x = y;
    } else {
        throw std::logic_error( "Not finished" );
    }
    uint64_t hash = h ^ x;
#ifdef AMP_USE_MPI
    MPI_Bcast( &hash, 1, MPI_UINT64_T, 0, comm );
#endif
    return hash;
}


/************************************************************************
 *  Constructor from existing MPI communicator                           *
 ************************************************************************/
static int d_global_currentTag_world1[2] = { 1, 1 };
static int d_global_currentTag_world2[2] = { 1, 1 };
static int d_global_currentTag_self[2]   = { 1, 1 };
static std::pair<int, int> getRankAndSize( [[maybe_unused]] AMP_MPI::Comm comm )
{
    int rank = 0;
    int size = 1;
#ifdef AMP_USE_MPI
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );
#endif
    return { rank, size };
}
AMP_MPI::AMP_MPI( Comm comm, bool manage )
{
    // Check if we are using our version of comm_world
    if ( comm == commWorld ) {
        // AMP_COMM_WORLD
        static auto rankSize = getRankAndSize( AMP::AMPManager::getCommWorld().d_comm );
        d_comm               = AMP::AMPManager::getCommWorld().d_comm;
        d_hash               = hashWorld;
        d_rank               = rankSize.first;
        d_size               = rankSize.second;
        d_currentTag         = d_global_currentTag_world1;
        ++( this->d_currentTag[1] );
    } else if ( comm == MPI_COMM_WORLD ) {
        // MPI_COMM_WORLD
        static auto rankSize = getRankAndSize( MPI_COMM_WORLD );
        d_comm               = MPI_COMM_WORLD;
        d_hash               = hashMPI;
        d_rank               = rankSize.first;
        d_size               = rankSize.second;
        d_currentTag         = d_global_currentTag_world2;
        ++( this->d_currentTag[1] );
    } else if ( comm == commSelf || comm == MPI_COMM_SELF ) {
        // MPI_COMM_SELF
        d_comm       = MPI_COMM_SELF;
        d_hash       = hashSelf;
        d_rank       = 0;
        d_size       = 1;
        d_currentTag = d_global_currentTag_self;
        ++( this->d_currentTag[1] );
    } else if ( comm == commNull || comm == MPI_COMM_NULL ) {
        // MPI_COMM_NULL
        d_comm       = MPI_COMM_NULL;
        d_hash       = hashNull;
        d_rank       = 0;
        d_size       = 0;
        d_currentTag = nullptr;
    } else {
        // Default
        auto rankSize              = getRankAndSize( comm );
        d_comm                     = comm;
        d_hash                     = hashComm( d_comm );
        d_rank                     = rankSize.first;
        d_size                     = rankSize.second;
        std::tie( d_rank, d_size ) = getRankAndSize( d_comm );
#ifdef AMP_USE_MPI
        // Attach the error handler
        StackTrace::setMPIErrorHandler( d_comm );
#endif
        d_manage = manage;
        d_count  = new std::atomic_int;
        *d_count = 1;
        if ( d_size > 1 ) {
            d_ranks    = new int[d_size];
            d_ranks[0] = -1;
        }
        if ( d_manage )
            ++N_MPI_Comm_created;
        d_currentTag    = new int[2];
        d_currentTag[0] = ( d_maxTag <= 0x10000 ) ? 1 : 0x1FFF;
        d_currentTag[1] = 1;
        if ( d_size > 1 ) {
            auto seed = bcast<size_t>( std::random_device()(), 0 );
            d_rand    = new std::mt19937_64( seed );
        }
    }
    d_isNull     = d_comm == MPI_COMM_NULL;
    d_call_abort = true;
}


/************************************************************************
 *  Return the ranks of the communicator in the global comm              *
 ************************************************************************/
static std::vector<int> MPI_COMM_WORLD_ranks = std::vector<int>();
static std::vector<int> AMP_COMM_WORLD_ranks = std::vector<int>();
std::vector<int> AMP_MPI::globalRanks() const
{
    // Get my global rank if it has not been set
    static int myGlobalRank = -1;
    if ( myGlobalRank == -1 ) {
        if ( MPI_Active() )
            myGlobalRank = AMP::AMPManager::getCommWorld().getRank();
    }
    // Check for special cases
    if ( d_size == 0 || d_comm == MPI_COMM_NULL ) {
        return std::vector<int>();
    } else if ( d_size == 1 ) {
        return std::vector<int>( 1, myGlobalRank );
    } else if ( d_comm == MPI_COMM_WORLD ) {
        if ( MPI_COMM_WORLD_ranks.empty() ) {
            MPI_COMM_WORLD_ranks.resize( d_size, -1 );
            this->allGather( myGlobalRank, MPI_COMM_WORLD_ranks.data() );
        }
        return MPI_COMM_WORLD_ranks;
    } else if ( d_comm == AMP::AMPManager::getCommWorld().d_comm ) {
        if ( AMP_COMM_WORLD_ranks.empty() ) {
            AMP_COMM_WORLD_ranks.resize( d_size, -1 );
            this->allGather( myGlobalRank, AMP_COMM_WORLD_ranks.data() );
        }
        return AMP_COMM_WORLD_ranks;
    }
    // Fill d_ranks if necessary
    AMP_ASSERT( d_ranks );
    if ( d_ranks[0] == -1 ) {
        AMP_ASSERT( myGlobalRank != -1 );
        this->allGather( myGlobalRank, d_ranks );
    }
    // Return d_ranks
    return std::vector<int>( d_ranks, d_ranks + d_size );
}
uint64_t AMP_MPI::hashRanks() const
{
    PROFILE( "hashRanks", profile_level );
    std::string str;
    for ( auto rank : globalRanks() )
        str += std::to_string( rank ) + '.';
    return std::hash<std::string>{}( str );
}


/************************************************************************
 *  Generate a random number                                             *
 ************************************************************************/
static auto randSerial = std::mt19937_64( std::random_device()() );
std::unique_ptr<std::mt19937_64> randWorld;
std::unique_ptr<std::mt19937_64> randMPI;
std::mt19937_64 *AMP_MPI::getRand() const
{
    if ( d_size < 2 ) {
        return &randSerial;
    } else if ( d_hash == hashWorld ) {
        if ( !randWorld ) {
            auto seed = bcast<size_t>( randSerial(), 0 );
            randWorld = std::make_unique<std::mt19937_64>( seed );
        }
        return randWorld.get();
    } else if ( d_hash == hashMPI ) {
        if ( !randMPI ) {
            auto seed = bcast<size_t>( randSerial(), 0 );
            randMPI   = std::make_unique<std::mt19937_64>( seed );
        }
        return randMPI.get();
    } else {
        AMP_DEBUG_ASSERT( d_rand );
        return d_rand;
    }
}
size_t AMP_MPI::rand() const
{
    static std::uniform_int_distribution<size_t> dist;
    auto randGen = getRand();
    size_t val   = dist( *randGen );
    AMP_DEBUG_ASSERT( val == bcast( val, 0 ) );
    return val;
}


/************************************************************************
 *  Intersect two communicators                                          *
 ************************************************************************/
#ifdef AMP_USE_MPI
static inline void MPI_Group_free2( MPI_Group *group )
{
    if ( *group != MPI_GROUP_EMPTY ) {
        // MPICH is fine with free'ing an empty group, OpenMPI crashes
        MPI_Group_free( group );
    }
}
AMP_MPI AMP_MPI::intersect( const AMP_MPI &comm1, const AMP_MPI &comm2 )
{
    // Deal with some special cases
    if ( comm1.d_hash == hashMPI )
        return comm2;
    if ( comm2.d_hash == hashMPI )
        return comm1;
    if ( comm1.d_hash == hashWorld )
        return comm2;
    if ( comm2.d_hash == hashWorld )
        return comm1;
    if ( comm1.d_size <= 1 ) {
        // We use comm1 for communication so this is safe
        if ( comm1.isNull() || comm2.isNull() )
            return {};
        return comm1;
    }
    // Compare the groups
    MPI_Group group1 = MPI_GROUP_EMPTY, group2 = MPI_GROUP_EMPTY;
    if ( !comm1.isNull() ) {
        MPI_Group_free2( &group1 );
        MPI_Comm_group( comm1.d_comm, &group1 );
    }
    if ( !comm2.isNull() ) {
        MPI_Group_free2( &group2 );
        MPI_Comm_group( comm2.d_comm, &group2 );
    }
    MPI_Group group12;
    MPI_Group_intersection( group1, group2, &group12 );
    int compare1, compare2;
    MPI_Group_compare( group1, group12, &compare1 );
    MPI_Group_compare( group2, group12, &compare2 );
    AMP_MPI new_comm( commNull );
    int size;
    MPI_Group_size( group12, &size );
    if ( compare1 != MPI_UNEQUAL && size != 0 ) {
        // The intersection matches comm1
        new_comm = comm1;
    } else if ( compare2 != MPI_UNEQUAL && size != 0 ) {
        // The intersection matches comm2
        new_comm = comm2;
    } else if ( comm1.isNull() ) {
        // comm1 is null, we can return safely (comm1 is needed for communication)
    } else {
        // The intersection is smaller than comm1 or comm2
        // Check if the new comm is nullptr for all processors
        int max_size = 0;
        MPI_Allreduce( &size, &max_size, 1, MPI_INT, MPI_MAX, comm1.d_comm );
        if ( max_size == 0 ) {
            // We are dealing with completely disjoint sets
            new_comm = AMP_MPI( commNull, false );
        } else {
            // Create the new comm
            // Note: OpenMPI crashes if the intersection group is EMPTY for any processors
            // We will set it to SELF for the EMPTY processors, then create a nullptr comm later
            if ( group12 == MPI_GROUP_EMPTY ) {
                MPI_Group_free2( &group12 );
                MPI_Comm_group( MPI_COMM_SELF, &group12 );
            }
            MPI_Comm new_MPI_comm;
            MPI_Comm_create( comm1.d_comm, group12, &new_MPI_comm );
            if ( size > 0 ) {
                // This is the valid case where we create a new intersection comm
                new_comm = AMP_MPI( new_MPI_comm, true );
            } else {
                // We actually want a null comm for this communicator
                new_comm = AMP_MPI( commNull, false );
                MPI_Comm_free( &new_MPI_comm );
            }
        }
    }
    MPI_Group_free2( &group1 );
    MPI_Group_free2( &group2 );
    MPI_Group_free2( &group12 );
    return new_comm;
}
#else
AMP_MPI AMP_MPI::intersect( const AMP_MPI &comm1, const AMP_MPI &comm2 )
{
    if ( comm1.isNull() || comm2.isNull() )
        return AMP_MPI( commNull, false );
    AMP_ASSERT( comm1.d_size == 1 && comm2.d_size == 1 );
    return comm1;
}
#endif


/************************************************************************
 *  Split a comm						                                 *
 ************************************************************************/
AMP_MPI AMP_MPI::split( int color, [[maybe_unused]] int key, bool manage ) const
{
    if ( d_isNull ) {
        return AMP_MPI( commNull );
    } else if ( d_size == 1 ) {
        if ( color == -1 )
            return AMP_MPI( commNull );
        return dup();
    }
    PROFILE( "split", profile_level );
    AMP_MPI::Comm new_MPI_comm = commNull;
#ifdef AMP_USE_MPI
    // USE MPI to split the communicator
    int error = 0;
    if ( color == -1 ) {
        error = MPI_Comm_split( d_comm, MPI_UNDEFINED, key, &new_MPI_comm );
    } else {
        error = MPI_Comm_split( d_comm, color, key, &new_MPI_comm );
    }
    AMP_INSIST( error == MPI_SUCCESS, "Error calling MPI routine" );
#endif
    // Create the new object
    AMP_MPI new_comm( new_MPI_comm, manage );
    new_comm.d_call_abort = d_call_abort;
    return new_comm;
}
AMP_MPI AMP_MPI::splitByNode( int key, bool manage ) const
{
    // Check if we are dealing with a single processor (trivial case)
    if ( d_size == 1 )
        return this->split( 0, 0, manage );
    PROFILE( "splitByNode", profile_level );
    // Get the node name
    std::string name = AMP_MPI::getNodeName();
    unsigned int id  = AMP::Utilities::hash_char( name.data() );
    // Gather the names from all ranks
    auto list = allGather( id );
    // Create the colors
    std::set<unsigned int> set( list.begin(), list.end() );
    int color = std::distance( set.begin(), set.find( id ) );
    return split( color, key, manage );
}


/************************************************************************
 *  Duplicate an exisiting comm object                                   *
 ************************************************************************/
AMP_MPI AMP_MPI::dup( bool manage ) const
{
    if ( d_isNull )
        return AMP_MPI( commNull );
    PROFILE( "dup", profile_level );
#if defined( AMP_USE_MPI ) || defined( AMP_USE_PETSC )
    // USE MPI to duplicate the communicator
    MPI_Comm new_MPI_comm;
    MPI_Comm_dup( d_comm, &new_MPI_comm );
#else
    static AMP_MPI::Comm uniqueGlobalComm = 11;
    auto new_MPI_comm                     = uniqueGlobalComm;
    uniqueGlobalComm++;
#endif
    // Create the new comm object
    AMP_MPI new_comm( new_MPI_comm, manage );
    new_comm.d_isNull     = d_isNull;
    new_comm.d_call_abort = d_call_abort;
    return new_comm;
}


/************************************************************************
 *  Get the node name                                                    *
 ************************************************************************/
std::string AMP_MPI::getNodeName()
{
#ifdef AMP_USE_MPI
    int length;
    char name[MPI_MAX_PROCESSOR_NAME + 1];
    memset( name, 0, MPI_MAX_PROCESSOR_NAME + 1 );
    MPI_Get_processor_name( name, &length );
    return std::string( name );
#else
    return "Node0";
#endif
}


/************************************************************************
 *  Overload operator ==                                                 *
 ************************************************************************/
bool AMP_MPI::operator==( const AMP_MPI &comm ) const { return d_comm == comm.d_comm; }


/************************************************************************
 *  Overload operator !=                                                 *
 ************************************************************************/
bool AMP_MPI::operator!=( const AMP_MPI &comm ) const { return d_comm != comm.d_comm; }


/************************************************************************
 *  Overload operator <                                                  *
 ************************************************************************/
bool AMP_MPI::operator<( const AMP_MPI &comm ) const
{
    AMP_ASSERT( !this->d_isNull && !comm.d_isNull );
    bool flag = true;
    // First check if either communicator is NULL
    if ( this->d_isNull )
        return false;
    if ( comm.d_isNull )
        flag = false;
    // Use compare to check if the comms are equal
    if ( compare( comm ) != 0 )
        return false;
    // Check that the size of the other communicator is > the current communicator size
    if ( d_size >= comm.d_size )
        flag = false;
// Check the union of the communicator groups
// this is < comm iff this group is a subgroup of comm's group
#ifdef AMP_USE_MPI
    MPI_Group group1 = MPI_GROUP_EMPTY, group2 = MPI_GROUP_EMPTY, group12 = MPI_GROUP_EMPTY;
    if ( !d_isNull )
        MPI_Comm_group( d_comm, &group1 );
    if ( !comm.d_isNull )
        MPI_Comm_group( comm.d_comm, &group2 );
    MPI_Group_union( group1, group2, &group12 );
    int compare;
    MPI_Group_compare( group2, group12, &compare );
    if ( compare == MPI_UNEQUAL )
        flag = false;
    MPI_Group_free( &group1 );
    MPI_Group_free( &group2 );
    MPI_Group_free( &group12 );
#endif
    // Perform a global reduce of the flag (equivalent to all operation)
    return allReduce( flag );
}


/************************************************************************
 *  Overload operator <=                                                 *
 ************************************************************************/
bool AMP_MPI::operator<=( const AMP_MPI &comm ) const
{
    AMP_ASSERT( !this->d_isNull && !comm.d_isNull );
    bool flag = true;
    // First check if either communicator is NULL
    if ( this->d_isNull )
        return false;
    if ( comm.d_isNull )
        flag = false;
#ifdef AMP_USE_MPI
    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    if ( comm.getSize() == world_size )
        return true;
    if ( getSize() == 1 && !comm.d_isNull )
        return true;
#endif
    // Use compare to check if the comms are equal
    if ( compare( comm ) != 0 )
        return true;
    // Check that the size of the other communicator is > the current communicator size
    // this is <= comm iff this group is a subgroup of comm's group
    if ( d_size > comm.d_size )
        flag = false;
// Check the unnion of the communicator groups
#ifdef AMP_USE_MPI
    MPI_Group group1, group2, group12;
    MPI_Comm_group( d_comm, &group1 );
    MPI_Comm_group( comm.d_comm, &group2 );
    MPI_Group_union( group1, group2, &group12 );
    int compare;
    MPI_Group_compare( group2, group12, &compare );
    if ( compare == MPI_UNEQUAL )
        flag = false;
    MPI_Group_free( &group1 );
    MPI_Group_free( &group2 );
    MPI_Group_free( &group12 );
#endif
    // Perform a global reduce of the flag (equivalent to all operation)
    return allReduce( flag );
}


/************************************************************************
 *  Overload operator >                                                  *
 ************************************************************************/
bool AMP_MPI::operator>( [[maybe_unused]] const AMP_MPI &comm ) const
{
    bool flag = true;
    // First check if either communicator is NULL
    if ( this->d_isNull )
        return false;
    if ( comm.d_isNull )
        flag = false;
    // Use compare to check if the comms are equal
    if ( compare( comm ) != 0 )
        return false;
    // Check that the size of the other communicator is > the current communicator size
    if ( d_size <= comm.d_size )
        flag = false;
// Check the unnion of the communicator groups
// this is > comm iff comm's group is a subgroup of this group
#ifdef AMP_USE_MPI
    MPI_Group group1 = MPI_GROUP_EMPTY, group2 = MPI_GROUP_EMPTY, group12 = MPI_GROUP_EMPTY;
    if ( !d_isNull )
        MPI_Comm_group( d_comm, &group1 );
    if ( !comm.d_isNull )
        MPI_Comm_group( comm.d_comm, &group2 );
    MPI_Group_union( group1, group2, &group12 );
    int compare;
    MPI_Group_compare( group1, group12, &compare );
    if ( compare == MPI_UNEQUAL )
        flag = false;
    MPI_Group_free( &group1 );
    MPI_Group_free( &group2 );
    MPI_Group_free( &group12 );
#endif
    // Perform a global reduce of the flag (equivalent to all operation)
    return allReduce( flag );
}


/************************************************************************
 *  Overload operator >=                                                 *
 ************************************************************************/
bool AMP_MPI::operator>=( const AMP_MPI &comm ) const
{
    bool flag = true;
    // First check if either communicator is NULL
    if ( this->d_isNull )
        return false;
    if ( comm.d_isNull )
        flag = false;
#ifdef AMP_USE_MPI
    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );
    if ( getSize() == world_size )
        return true;
    if ( comm.getSize() == 1 && !comm.d_isNull )
        return true;
#endif
    // Use compare to check if the comms are equal
    if ( compare( comm ) != 0 )
        return true;
    // Check that the size of the other communicator is > the current communicator size
    if ( d_size < comm.d_size )
        flag = false;
// Check the unnion of the communicator groups
// this is >= comm iff comm's group is a subgroup of this group
#ifdef AMP_USE_MPI
    MPI_Group group1 = MPI_GROUP_EMPTY, group2 = MPI_GROUP_EMPTY, group12 = MPI_GROUP_EMPTY;
    if ( !d_isNull )
        MPI_Comm_group( d_comm, &group1 );
    if ( !comm.d_isNull )
        MPI_Comm_group( comm.d_comm, &group2 );
    MPI_Group_union( group1, group2, &group12 );
    int compare;
    MPI_Group_compare( group1, group12, &compare );
    if ( compare == MPI_UNEQUAL )
        flag = false;
    MPI_Group_free( &group1 );
    MPI_Group_free( &group2 );
    MPI_Group_free( &group12 );
#endif
    // Perform a global reduce of the flag (equivalent to all operation)
    return allReduce( flag );
}


/************************************************************************
 *  Compare two comm objects                                             *
 ************************************************************************/
int AMP_MPI::compare( const AMP_MPI &comm ) const
{
    if ( d_hash == comm.d_hash )
        return 1;
#ifdef AMP_USE_MPI
    if ( d_isNull || comm.d_isNull )
        return 0;
    int result;
    int error = MPI_Comm_compare( d_comm, comm.d_comm, &result );
    AMP_INSIST( error == MPI_SUCCESS, "Error calling MPI routine" );
    if ( result == MPI_IDENT )
        return 2;
    else if ( result == MPI_CONGRUENT )
        return 3;
    else if ( result == MPI_SIMILAR )
        return 4;
    else if ( result == MPI_UNEQUAL )
        return 0;
    AMP_ERROR( "Unknown results from comm compare" );
    return 0;
#else
    if ( comm.d_comm == MPI_COMM_NULL || d_comm == MPI_COMM_NULL )
        return 0;
    else
        return 3;
#endif
}


/************************************************************************
 *  Abort the program.                                                   *
 ************************************************************************/
void AMP_MPI::setCallAbortInSerialInsteadOfExit( bool flag ) { d_call_abort = flag; }
void AMP_MPI::abort() const
{
#ifdef AMP_USE_MPI
    MPI_Comm comm = d_comm;
    if ( comm == MPI_COMM_NULL )
        comm = MPI_COMM_WORLD;
    if ( !MPI_Active() ) {
        // MPI is not available
        exit( -1 );
    } else if ( d_size > 1 ) {
        MPI_Abort( comm, -1 );
    } else if ( d_call_abort ) {
        MPI_Abort( comm, -1 );
    } else {
        exit( -1 );
    }
#else
    exit( -1 );
#endif
}


/************************************************************************
 *  newTag                                                               *
 ************************************************************************/
int AMP_MPI::newTag() const
{
#ifdef AMP_USE_MPI

    int tag = ( *d_currentTag )++;
    AMP_INSIST( tag <= d_maxTag, "Maximum number of tags exceeded\n" );
    AMP_DEBUG_ASSERT( tag == bcast( tag, 0 ) );
    return tag;
#else
    static int globalCurrentTag = 1;
    return globalCurrentTag++;
#endif
}


/************************************************************************
 *  allReduce                                                            *
 ************************************************************************/
[[maybe_unused]] static inline void setBit( uint64_t *x, size_t index )
{
    size_t i      = index >> 6;
    size_t j      = index & 0x3F;
    uint64_t mask = ( (uint64_t) 0x01 ) << j;
    x[i]          = x[i] | mask;
}
[[maybe_unused]] static inline bool readBit( const uint64_t *x, size_t index )
{
    size_t i      = index >> 6;
    size_t j      = index & 0x3F;
    uint64_t mask = ( (uint64_t) 0x01 ) << j;
    return ( x[i] & mask ) != 0;
}
bool AMP_MPI::allReduce( [[maybe_unused]] const bool value ) const
{
    bool ret = value;
    if ( d_size > 1 ) {
#ifdef AMP_USE_MPI
        MPI_Allreduce( (void *) &value, (void *) &ret, 1, MPI_UNSIGNED_CHAR, MPI_MIN, d_comm );
#else
        AMP_ERROR( "This shouldn't be possible" );
#endif
    }
    return ret;
}
void AMP_MPI::allReduce( [[maybe_unused]] std::vector<bool> &x ) const
{
    if ( d_size <= 1 )
        return;
#ifdef AMP_USE_MPI
    size_t N  = ( x.size() + 63 ) / 64;
    auto send = new uint64_t[N];
    auto recv = new uint64_t[N];
    memset( send, 0, N * sizeof( int64_t ) );
    memset( recv, 0, N * sizeof( int64_t ) );
    for ( size_t i = 0; i < x.size(); i++ ) {
        if ( x[i] )
            setBit( send, i );
    }
    MPI_Allreduce( send, recv, N, MPI_UINT64_T, MPI_BAND, d_comm );
    for ( size_t i = 0; i < x.size(); i++ )
        x[i] = readBit( recv, i );
    delete[] send;
    delete[] recv;
#else
    AMP_ERROR( "This shouldn't be possible" );
#endif
}


/************************************************************************
 *  anyReduce                                                            *
 ************************************************************************/
bool AMP_MPI::anyReduce( [[maybe_unused]] const bool value ) const
{
    bool ret = value;
    if ( d_size > 1 ) {
#ifdef AMP_USE_MPI
        MPI_Allreduce( (void *) &value, (void *) &ret, 1, MPI_UNSIGNED_CHAR, MPI_MAX, d_comm );
#else
        AMP_ERROR( "This shouldn't be possible" );
#endif
    }
    return ret;
}
void AMP_MPI::anyReduce( [[maybe_unused]] std::vector<bool> &x ) const
{
    if ( d_size <= 1 )
        return;
#ifdef AMP_USE_MPI
    size_t N  = ( x.size() + 63 ) / 64;
    auto send = new uint64_t[N];
    auto recv = new uint64_t[N];
    memset( send, 0, N * sizeof( int64_t ) );
    memset( recv, 0, N * sizeof( int64_t ) );
    for ( size_t i = 0; i < x.size(); i++ ) {
        if ( x[i] )
            setBit( send, i );
    }
    MPI_Allreduce( send, recv, N, MPI_UINT64_T, MPI_BOR, d_comm );
    for ( size_t i = 0; i < x.size(); i++ )
        x[i] = readBit( recv, i );
    delete[] send;
    delete[] recv;
#else
    AMP_ERROR( "This shouldn't be possible" );
#endif
}


/************************************************************************
 *  AllToAll                                                             *
 ************************************************************************/
int AMP_MPI::calcAllToAllDisp( const int *send_cnt,
                               int *send_disp,
                               int *recv_cnt,
                               int *recv_disp ) const
{
    size_t N = 0;
    for ( int i = 0; i < d_size; i++ )
        N += send_cnt[i];
    AMP_ASSERT( N < 0x7FFFFFFF );
    if ( send_disp ) {
        send_disp[0] = 0;
        for ( int i = 1; i < d_size; i++ )
            send_disp[i] = send_disp[i - 1] + send_cnt[i - 1];
    }
    AMP_ASSERT( recv_cnt && recv_disp );
    allToAll<int>( 1, &send_cnt[0], &recv_cnt[0] );
    N = 0;
    for ( int i = 0; i < d_size; i++ )
        N += recv_cnt[i];
    AMP_ASSERT( N < 0x7FFFFFFF );
    recv_disp[0] = 0;
    for ( int i = 1; i < d_size; i++ )
        recv_disp[i] = recv_disp[i - 1] + recv_cnt[i - 1];
    return N;
}
int AMP_MPI::calcAllToAllDisp( const std::vector<int> &send_cnt,
                               std::vector<int> &send_disp,
                               std::vector<int> &recv_cnt,
                               std::vector<int> &recv_disp ) const
{
    AMP_ASSERT( (int) send_cnt.size() == d_size );
    send_disp.resize( d_size );
    recv_cnt.resize( d_size );
    recv_disp.resize( d_size );
    return calcAllToAllDisp( send_cnt.data(), send_disp.data(), recv_cnt.data(), recv_disp.data() );
}


/************************************************************************
 *  Perform a global barrier across all processors.                      *
 ************************************************************************/
void AMP_MPI::barrier() const
{
#ifdef AMP_USE_MPI
    if ( d_size <= 1 )
        return;
    PROFILE( "barrier", profile_level );
    MPI_Barrier( d_comm );
#endif
}
void AMP_MPI::sleepBarrier( [[maybe_unused]] int ms ) const
{
#ifdef AMP_USE_MPI
    if ( d_size <= 1 )
        return;
    using namespace std::chrono_literals;
    PROFILE( "sleepBarrier", profile_level );
    int flag = 0;
    MPI_Request request;
    MPI_Ibarrier( d_comm, &request );
    int err = MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
    AMP_ASSERT( err == MPI_SUCCESS ); // Check that the first call is valid
    std::chrono::milliseconds time( ms );
    while ( !flag ) {
        // Put the current thread to sleep to allow other threads to run
        if ( ms > 0 )
            std::this_thread::sleep_for( time );
        else
            std::this_thread::yield();
        // Check if the request has finished
        MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
    }
#endif
}


/************************************************************************
 *  Send/Recv data                                                       *
 *  We need a concrete instantiation of send for use without MPI         *
 ************************************************************************/
#ifdef AMP_USE_MPI
void AMP_MPI::sendBytes( const void *buf, int bytes, int recv_proc, int tag ) const
{
    send<char>( (const char *) buf, bytes, recv_proc, tag );
}
void AMP_MPI::recvBytes( void *buf, int bytes, const int send_proc, int tag ) const
{
    int bytes2 = bytes;
    recv<char>( (char *) buf, bytes2, send_proc, false, tag );
}
AMP_MPI::Request AMP_MPI::IsendBytes( const void *buf, int bytes, int recv_proc, int tag ) const
{
    return Isend<char>( (const char *) buf, bytes, recv_proc, tag );
}
AMP_MPI::Request AMP_MPI::IrecvBytes( void *buf, int bytes, int send_proc, const int tag ) const
{
    return Irecv<char>( (char *) buf, bytes, send_proc, tag );
}
#else
void AMP_MPI::sendBytes( const void *buf, int bytes, int, int tag ) const
{
    AMP_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    AMP_INSIST( tag >= 0, "tag must be >= 0" );
    PROFILE( "sendBytes", profile_level );
    auto id = getRequest( d_comm, tag );
    auto it = global_isendrecv_list.find( id );
    AMP_INSIST( it != global_isendrecv_list.end(),
                "send must be paired with a previous call to irecv in serial" );
    AMP_ASSERT( it->second.status == 2 );
    memcpy( (char *) it->second.data, buf, bytes );
    global_isendrecv_list.erase( it );
}
void AMP_MPI::recvBytes( void *buf, int bytes, const int, int tag ) const
{
    AMP_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    AMP_INSIST( tag >= 0, "tag must be >= 0" );
    PROFILE( "recv<char>", profile_level );
    auto id = getRequest( d_comm, tag );
    auto it = global_isendrecv_list.find( id );
    AMP_INSIST( it != global_isendrecv_list.end(),
                "recv must be paired with a previous call to isend in serial" );
    AMP_ASSERT( it->second.status == 1 );
    AMP_ASSERT( it->second.bytes == bytes );
    memcpy( buf, it->second.data, bytes );
    global_isendrecv_list.erase( it );
}
AMP_MPI::Request AMP_MPI::IsendBytes( const void *buf, int bytes, int, int tag ) const
{
    AMP_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    AMP_INSIST( tag >= 0, "tag must be >= 0" );
    PROFILE( "IsendBytes", profile_level );
    auto id = getRequest( d_comm, tag );
    auto it = global_isendrecv_list.find( id );
    if ( it == global_isendrecv_list.end() ) {
        // We are calling isend first
        Isendrecv_struct data;
        data.bytes  = bytes;
        data.data   = buf;
        data.status = 1;
        data.comm   = d_comm;
        data.tag    = tag;
        global_isendrecv_list.insert( std::pair<AMP_MPI::Request2, Isendrecv_struct>( id, data ) );
    } else {
        // We called irecv first
        AMP_ASSERT( it->second.status == 2 );
        AMP_ASSERT( it->second.bytes == bytes );
        memcpy( (char *) it->second.data, buf, bytes );
        global_isendrecv_list.erase( it );
    }
    return id;
}
AMP_MPI::Request AMP_MPI::IrecvBytes( void *buf, const int bytes, const int, const int tag ) const
{
    AMP_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    AMP_INSIST( tag >= 0, "tag must be >= 0" );
    PROFILE( "Irecv<char>", profile_level );
    auto id = getRequest( d_comm, tag );
    auto it = global_isendrecv_list.find( id );
    if ( it == global_isendrecv_list.end() ) {
        // We are calling Irecv first
        Isendrecv_struct data;
        data.bytes  = bytes;
        data.data   = buf;
        data.status = 2;
        data.comm   = d_comm;
        data.tag    = tag;
        global_isendrecv_list.insert( std::pair<AMP_MPI::Request, Isendrecv_struct>( id, data ) );
    } else {
        // We called Isend first
        AMP_ASSERT( it->second.status == 1 );
        AMP_ASSERT( it->second.bytes == bytes );
        memcpy( buf, it->second.data, bytes );
        global_isendrecv_list.erase( it );
    }
    return id;
}
#endif


/************************************************************************
 *  Communicate ranks for communication                                  *
 ************************************************************************/
std::vector<int> AMP_MPI::commRanks( const std::vector<int> &ranks ) const
{
#ifdef AMP_USE_MPI
    // Get a byte array with the ranks to communicate
    auto data1 = new char[d_size];
    auto data2 = new char[d_size];
    memset( data1, 0, d_size );
    memset( data2, 0, d_size );
    for ( auto &rank : ranks )
        data1[rank] = 1;
    MPI_Alltoall( data1, 1, MPI_CHAR, data2, 1, MPI_CHAR, d_comm );
    int N = 0;
    for ( int i = 0; i < d_size; i++ )
        N += data2[i];
    std::vector<int> ranks_out;
    ranks_out.reserve( N );
    for ( int i = 0; i < d_size; i++ ) {
        if ( data2[i] )
            ranks_out.push_back( i );
    }
    delete[] data1;
    delete[] data2;
    return ranks_out;
#else
    return ranks;
#endif
}


/************************************************************************
 *  Wait functions                                                       *
 ************************************************************************/
#ifdef AMP_USE_MPI
void AMP_MPI::wait( Request2 request )
{
    PROFILE( "wait", profile_level );
    int flag = 0;
    int err  = MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
    AMP_ASSERT( err == MPI_SUCCESS ); // Check that the first call is valid
    while ( !flag ) {
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
        // Check if the request has finished
        MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
    }
}
int AMP_MPI::waitAny( int count, Request2 *request )
{
    if ( count == 0 )
        return -1;
    PROFILE( "waitAny", profile_level );
    int index = -1;
    int flag  = 0;
    int err   = MPI_Testany( count, request, &index, &flag, MPI_STATUS_IGNORE );
    AMP_ASSERT( err == MPI_SUCCESS ); // Check that the first call is valid
    while ( !flag ) {
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
        // Check if the request has finished
        MPI_Testany( count, request, &index, &flag, MPI_STATUS_IGNORE );
    }
    AMP_ASSERT( index >= 0 ); // Check that the index is valid
    return index;
}
void AMP_MPI::waitAll( int count, Request2 *request )
{
    if ( count == 0 )
        return;
    PROFILE( "waitAll", profile_level );
    int flag = 0;
    int err  = MPI_Testall( count, request, &flag, MPI_STATUS_IGNORE );
    AMP_ASSERT( err == MPI_SUCCESS ); // Check that the first call is valid
    while ( !flag ) {
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
        // Check if the request has finished
        MPI_Testall( count, request, &flag, MPI_STATUS_IGNORE );
    }
}
std::vector<int> AMP_MPI::waitSome( int count, Request2 *request )
{
    if ( count == 0 )
        return std::vector<int>();
    PROFILE( "waitSome", profile_level );
    std::vector<int> indicies( count, -1 );
    int outcount = 0;
    int err      = MPI_Testsome( count, request, &outcount, &indicies[0], MPI_STATUS_IGNORE );
    AMP_ASSERT( err == MPI_SUCCESS );        // Check that the first call is valid
    AMP_ASSERT( outcount != MPI_UNDEFINED ); // Check that the first call is valid
    while ( outcount == 0 ) {
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
        // Check if the request has finished
        MPI_Testsome( count, request, &outcount, &indicies[0], MPI_STATUS_IGNORE );
    }
    indicies.resize( outcount );
    return indicies;
}
#else
void AMP_MPI::wait( Request2 request )
{
    PROFILE( "wait", profile_level );
    while ( 1 ) {
        // Check if the request is in our list
        if ( global_isendrecv_list.find( request ) == global_isendrecv_list.end() )
            break;
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
    }
}
int AMP_MPI::waitAny( int count, Request2 *request )
{
    if ( count == 0 )
        return -1;
    PROFILE( "waitAny", profile_level );
    int index = 0;
    while ( 1 ) {
        // Check if the request is in our list
        bool found_any = false;
        for ( int i = 0; i < count; i++ ) {
            if ( global_isendrecv_list.find( request[i] ) == global_isendrecv_list.end() ) {
                found_any = true;
                index     = i;
            }
        }
        if ( found_any )
            break;
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
    }
    return index;
}
void AMP_MPI::waitAll( int count, Request2 *request )
{
    if ( count == 0 )
        return;
    PROFILE( "waitAll", profile_level );
    while ( 1 ) {
        // Check if the request is in our list
        bool found_all = true;
        for ( int i = 0; i < count; i++ ) {
            if ( global_isendrecv_list.find( request[i] ) != global_isendrecv_list.end() )
                found_all = false;
        }
        if ( found_all )
            break;
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
    }
}
std::vector<int> AMP_MPI::waitSome( int count, Request2 *request )
{
    if ( count == 0 )
        return std::vector<int>();
    PROFILE( "waitSome", profile_level );
    std::vector<int> indicies;
    while ( 1 ) {
        // Check if the request is in our list
        for ( int i = 0; i < count; i++ ) {
            if ( global_isendrecv_list.find( request[i] ) == global_isendrecv_list.end() )
                indicies.push_back( i );
        }
        if ( !indicies.empty() )
            break;
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
    }
    return indicies;
}
#endif
static std::vector<AMP_MPI::Request2> getRequests( int count, const AMP_MPI::Request *request )
{
    std::vector<AMP_MPI::Request2> request2( count );
    for ( int i = 0; i < count; i++ )
        request2[i] = request[i];
    return request2;
}
void AMP_MPI::wait( const Request &request ) { wait( static_cast<Request2>( request ) ); }
int AMP_MPI::waitAny( int count, const Request *request )
{
    return waitAny( count, getRequests( count, request ).data() );
}
void AMP_MPI::waitAll( int count, const Request *request )
{
    waitAll( count, getRequests( count, request ).data() );
}
std::vector<int> AMP_MPI::waitSome( int count, const Request *request )
{
    return waitSome( count, getRequests( count, request ).data() );
}


/************************************************************************
 *  Probe functions                                                      *
 ************************************************************************/
#ifdef AMP_USE_MPI
std::tuple<int, int, int> AMP_MPI::Iprobe( int source, int tag ) const
{
    AMP_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    // set tag and source to wildcards if appropriate
    source = source >= 0 ? source : MPI_ANY_SOURCE;
    tag    = tag >= 0 ? tag : MPI_ANY_TAG;
    MPI_Status status;
    int flag = 0;
    MPI_Iprobe( source, tag, d_comm, &flag, &status );
    if ( flag == 0 )
        return std::tuple<int, int, int>( -1, -1, -1 );
    int count;
    MPI_Get_count( &status, MPI_BYTE, &count );
    AMP_ASSERT( count >= 0 );
    return std::tuple<int, int, int>( status.MPI_SOURCE, status.MPI_TAG, count );
}
std::tuple<int, int, int> AMP_MPI::probe( int source, int tag ) const
{
    AMP_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    // set tag and source to wildcards if appropriate
    source = source >= 0 ? source : MPI_ANY_SOURCE;
    tag    = tag >= 0 ? tag : MPI_ANY_TAG;
    MPI_Status status;
    MPI_Probe( source, tag, d_comm, &status );
    int count;
    MPI_Get_count( &status, MPI_BYTE, &count );
    AMP_ASSERT( count >= 0 );
    return std::tuple<int, int, int>( status.MPI_SOURCE, status.MPI_TAG, count );
}
#else
std::tuple<int, int, int> AMP_MPI::Iprobe( int source, int tag ) const
{
    AMP_ASSERT( source == 0 || source == -1 );
    for ( const auto &tmp : global_isendrecv_list ) {
        const auto &data = tmp.second;
        if ( data.comm == d_comm && ( data.tag == tag || tag == -1 ) && data.status == 1 )
            return std::tuple<int, int, int>( 0, data.tag, data.bytes );
    }
    return std::tuple<int, int, int>( -1, -1, -1 );
}
std::tuple<int, int, int> AMP_MPI::probe( int source, int tag ) const
{
    auto tpl = Iprobe( source, tag );
    AMP_INSIST( std::get<2>( tpl ) >= 0, "probe called before message started in serial" );
    return tpl;
}
#endif


/************************************************************************
 *  Timer functions                                                      *
 ************************************************************************/
#ifdef AMP_USE_MPI
double AMP_MPI::time() { return MPI_Wtime(); }
double AMP_MPI::tick() { return MPI_Wtick(); }
#else
double AMP_MPI::time()
{
    auto t  = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>( t.time_since_epoch() );
    return 1e-9 * ns.count();
}
double AMP_MPI::tick()
{
    [[maybe_unused]] auto period = std::chrono::system_clock::period();
    return static_cast<double>( period.num ) / static_cast<double>( period.den );
}
#endif


/************************************************************************
 *  Serialize a block of code across MPI processes                       *
 ************************************************************************/
void AMP_MPI::serializeStart() const
{
#ifdef AMP_USE_MPI
    PROFILE( "serializeStart", profile_level );
    // Wait for a message from the previous rank
    if ( d_rank > 0 ) {
        MPI_Request request;
        int flag = false, buf = 0;
        MPI_Irecv( &buf, 1, MPI_INT, d_rank - 1, 5627, d_comm, &request );
        while ( !flag ) {
            MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
            std::this_thread::yield();
        }
    }
#endif
}
void AMP_MPI::serializeStop() const
{
#ifdef AMP_USE_MPI
    PROFILE( "serializeStop", profile_level );
    // Send flag to next rank
    if ( d_rank < d_size - 1 )
        MPI_Send( &d_rank, 1, MPI_INT, d_rank + 1, 5627, d_comm );
    // Final barrier to sync all threads
    sleepBarrier( 0 );
#endif
}


/****************************************************************************
 * Function to start/stop MPI                                                *
 ****************************************************************************/
#ifdef AMP_USE_MPI
static bool called_MPI_Init = false;
#endif
void AMP_MPI::start_MPI( [[maybe_unused]] int &argc,
                         [[maybe_unused]] char *argv[],
                         [[maybe_unused]] int profile_level )
{
    changeProfileLevel( profile_level );
#ifdef AMP_USE_MPI
    // Initialize MPI if needed
    if ( MPI_Active() ) {
        called_MPI_Init = false;
    } else {
        int provided;
        int result = MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
        if ( result != MPI_SUCCESS )
            AMP_ERROR( "AMP was unable to initialize MPI" );
        if ( provided < MPI_THREAD_MULTIPLE )
            AMP::perr << "Warning: Failed to start MPI with MPI_THREAD_MULTIPLE\n";
        called_MPI_Init = true;
        AMPManager::setCommWorld( MPI_COMM_WORLD );
    }
    // Set the maximum tag
    int flag, *val;
    int ierr = MPI_Comm_get_attr( MPI_COMM_WORLD, MPI_TAG_UB, &val, &flag );
    AMP_ASSERT( ierr == MPI_SUCCESS );
    if ( flag != 0 ) {
        d_maxTag = std::min<int>( *val, 0x7FFFFFFF );
        AMP_INSIST( d_maxTag >= 0x7FFF, "maximum tag size is < MPI standard" );
    }
#else
    AMPManager::setCommWorld( MPI_COMM_SELF );
#endif
}
void AMP_MPI::stop_MPI()
{
    MPI_COMM_WORLD_ranks = std::vector<int>();
    AMPManager::setCommWorld( AMP_COMM_NULL );
#ifdef AMP_USE_MPI
    int finalized;
    MPI_Finalized( &finalized );
    if ( called_MPI_Init && !finalized ) {
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_Finalize();
    }
#endif
}

/****************************************************************************
 * Request                                                                   *
 ****************************************************************************/
AMP_MPI::Request::Request( AMP_MPI::Request2 request, std::any data )
{
    using TYPE = typename AMP::remove_cvref_t<decltype( *( d_data.get() ) )>;
    if ( data.has_value() ) {
        auto deleter = []( TYPE *p ) {
            AMP_MPI::wait( p->first );
            delete p;
        };
        d_data.reset( new TYPE, deleter );
    } else {
        d_data.reset( new TYPE );
    }
    d_data->first  = request;
    d_data->second = std::move( data );
}
AMP_MPI::Request::~Request() {}


/****************************************************************************
 * getComm                                                                   *
 ****************************************************************************/
template<class TYPE>
AMP_MPI getComm( const TYPE & )
{
    return AMP_COMM_SELF;
}
#define INSTANTIATE_GET_COMM( TYPE )                                    \
    template AMP_MPI getComm<TYPE>( const TYPE & );                     \
    template AMP_MPI getComm<std::set<TYPE>>( const std::set<TYPE> & ); \
    template AMP_MPI getComm<std::vector<TYPE>>( const std::vector<TYPE> & )
INSTANTIATE_GET_COMM( bool );
INSTANTIATE_GET_COMM( char );
INSTANTIATE_GET_COMM( int8_t );
INSTANTIATE_GET_COMM( uint8_t );
INSTANTIATE_GET_COMM( int16_t );
INSTANTIATE_GET_COMM( uint16_t );
INSTANTIATE_GET_COMM( int32_t );
INSTANTIATE_GET_COMM( uint32_t );
INSTANTIATE_GET_COMM( int64_t );
INSTANTIATE_GET_COMM( uint64_t );
INSTANTIATE_GET_COMM( long long );
INSTANTIATE_GET_COMM( float );
INSTANTIATE_GET_COMM( double );
INSTANTIATE_GET_COMM( std::byte );
INSTANTIATE_GET_COMM( std::complex<float> );
INSTANTIATE_GET_COMM( std::complex<double> );
INSTANTIATE_GET_COMM( std::string );
INSTANTIATE_GET_COMM( std::string_view );


} // namespace AMP


/****************************************************************************
 * Explicit instantiation                                                    *
 ****************************************************************************/
typedef std::pair<int, int> PairInt;
typedef std::array<double, 1> ArrayDouble1;
typedef std::array<double, 2> ArrayDouble2;
typedef std::array<double, 3> ArrayDouble3;
typedef std::vector<size_t> VecSize;
typedef std::vector<std::array<int, 1>> VecArrayInt1;
typedef std::vector<std::array<int, 2>> VecArrayInt2;
typedef std::vector<std::array<int, 3>> VecArrayInt3;
INSTANTIATE_MPI_TYPE( char );
INSTANTIATE_MPI_TYPE( int8_t );
INSTANTIATE_MPI_TYPE( uint8_t );
INSTANTIATE_MPI_TYPE( int16_t );
INSTANTIATE_MPI_TYPE( uint16_t );
INSTANTIATE_MPI_TYPE( int32_t );
INSTANTIATE_MPI_TYPE( uint32_t );
INSTANTIATE_MPI_TYPE( int64_t );
INSTANTIATE_MPI_TYPE( uint64_t );
INSTANTIATE_MPI_TYPE( long long );
INSTANTIATE_MPI_TYPE( float );
INSTANTIATE_MPI_TYPE( double );
INSTANTIATE_MPI_TYPE( std::complex<float> );
INSTANTIATE_MPI_TYPE( std::complex<double> );
INSTANTIATE_MPI_BCAST( bool );
INSTANTIATE_MPI_SENDRECV( bool );
INSTANTIATE_MPI_BCAST( std::string );
INSTANTIATE_MPI_SENDRECV( std::string );
INSTANTIATE_MPI_GATHER( std::string );
template VecSize AMP::AMP_MPI::bcast<VecSize>( VecSize const &, int ) const;
template VecArrayInt1 AMP::AMP_MPI::bcast<VecArrayInt1>( VecArrayInt1 const &, int ) const;
template VecArrayInt2 AMP::AMP_MPI::bcast<VecArrayInt2>( VecArrayInt2 const &, int ) const;
template VecArrayInt3 AMP::AMP_MPI::bcast<VecArrayInt3>( VecArrayInt3 const &, int ) const;
template void AMP::AMP_MPI::mapGather<uint64_t, uint64_t>( std::map<uint64_t, uint64_t> & ) const;
template int
AMP::AMP_MPI::allGather<PairInt>( PairInt const *, int, PairInt *, int *, int *, bool ) const;
template void AMP::AMP_MPI::setGather<ArrayDouble1>( std::set<ArrayDouble1> & ) const;
template void AMP::AMP_MPI::setGather<ArrayDouble2>( std::set<ArrayDouble2> & ) const;
template void AMP::AMP_MPI::setGather<ArrayDouble3>( std::set<ArrayDouble3> & ) const;
