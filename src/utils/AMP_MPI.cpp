// This file implements a wrapper class for MPI functions

// Include AMP headers
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.I"
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>


// Make sure USE_MPI is set properly
#if !defined( USE_MPI ) && defined( AMP_USE_MPI )
    #define USE_MPI
#endif


// Convenience defines
#define MPI_CLASS_COMM_NULL AMP_COMM_NULL
#define MPI_CLASS_COMM_SELF AMP_COMM_SELF
#define MPI_CLASS_COMM_WORLD AMP_COMM_WORLD


// Set MPI_REQUEST_NULL
#if defined( USE_SAMRAI ) && defined( USE_PETSC ) && !defined( USE_MPI )
int MPI_REQUEST_NULL  = 3;
int MPI_ERR_IN_STATUS = 4;
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
static_assert( sizeof( MPI_CLASS ) % 8 == 0 );


// Initialized the static member variables
volatile uint32_t MPI_CLASS::N_MPI_Comm_created   = 0;
volatile uint32_t MPI_CLASS::N_MPI_Comm_destroyed = 0;
short MPI_CLASS::profile_level                    = 127;


// Default MPI max tag (will be overridden if MPI_TAG_UB attribute is valid)
static const int mpi_max_tag = 0x3FFFFFFF;


// Static data for asyncronous communication without MPI
// Note: these routines may not be thread-safe yet
#ifndef USE_MPI
struct Isendrecv_struct {
    const void *data;     // Pointer to data
    int bytes;            // Number of bytes in the message
    int status;           // Status: 1-sending, 2-recieving
    MPI_CLASS::Comm comm; // Communicator
    int tag;              // Tag
};
std::map<MPI_CLASS::Request2, Isendrecv_struct> global_isendrecv_list;
static MPI_CLASS::Request2 getRequest( MPI_CLASS::Comm comm, int tag )
{
    MPI_CLASS_ASSERT( tag >= 0 && tag <= mpi_max_tag );
    // Use hashing function: 2^64*0.5*(sqrt(5)-1)
    uint64_t a    = static_cast<uint8_t>( comm ) * 0x9E3779B97F4A7C15;
    uint64_t b    = static_cast<uint8_t>( tag ) * 0x9E3779B97F4A7C15;
    uint64_t hash = a ^ b;
    MPI_CLASS::Request2 request;
    memcpy( &request, &hash, sizeof( MPI_CLASS::Request2 ) );
    return request;
}
#endif


/************************************************************************
 *  Get the MPI version                                                  *
 ************************************************************************/
std::array<int, 2> MPI_CLASS::version()
{
#ifdef USE_MPI
    int MPI_version;
    int MPI_subversion;
    MPI_Get_version( &MPI_version, &MPI_subversion );
    return { MPI_version, MPI_subversion };
#else
    return { 0, 0 };
#endif
}
std::string MPI_CLASS::info()
{
    std::string MPI_info;
#ifdef USE_MPI
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
int MPI_CLASS::getNumberOfProcessors() { return std::thread::hardware_concurrency(); }
std::vector<int> MPI_CLASS::getProcessAffinity() { return AMP::Thread::getProcessAffinity(); }
void MPI_CLASS::setProcessAffinity( const std::vector<int> &procs )
{
    AMP::Thread::setProcessAffinity( procs );
}


/************************************************************************
 *  Function to check if MPI is active                                   *
 ************************************************************************/
bool MPI_CLASS::MPI_Active()
{
#ifdef USE_MPI
    int initialized = 0, finalized = 0;
    MPI_Initialized( &initialized );
    MPI_Finalized( &finalized );
    return initialized != 0 && finalized == 0;
#else
    return true;
#endif
}
MPI_CLASS::ThreadSupport MPI_CLASS::queryThreadSupport()
{
#ifdef USE_MPI
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
void MPI_CLASS::balanceProcesses( const MPI_CLASS &globalComm,
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
    MPI_CLASS nodeComm = globalComm.splitByNode();
    int N_min          = std::min<int>( std::max<int>( N_min_in, 1 ), cpus.size() );
    int N_max          = N_max_in;
    if ( N_max == -1 )
        N_max = cpus.size();
    N_max = std::min<int>( N_max, cpus.size() );
    MPI_CLASS_ASSERT( N_max >= N_min );
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
        MPI_CLASS_ERROR( "Unknown method for load balance" );
    }
}


/************************************************************************
 *  Empty constructor                                                    *
 ************************************************************************/
MPI_CLASS::MPI_CLASS()
    : d_comm( MPI_COMM_NULL ),
      d_isNull( true ),
      d_manage( false ),
      d_call_abort( true ),
      d_rank( 0 ),
      d_size( 1 ),
      d_maxTag( mpi_max_tag ),
      d_currentTag( nullptr ),
      d_ranks( nullptr ),
      d_count( nullptr )
{
}


/************************************************************************
 *  Empty deconstructor                                                  *
 ************************************************************************/
MPI_CLASS::~MPI_CLASS() { reset(); }
void MPI_CLASS::reset()
{
    // Decrement the count if used
    int count = -1;
    if ( d_count != nullptr )
        count = --( *d_count );
    if ( count == 0 ) {
        // We are holding that last reference to the MPI_Comm object, we need to free it
        if ( d_manage ) {
#if defined( USE_MPI ) || defined( USE_PETSC )
            MPI_Comm_set_errhandler( d_comm, MPI_ERRORS_ARE_FATAL );
            int err = MPI_Comm_free( (MPI_Comm *) &d_comm );
            if ( err != MPI_SUCCESS )
                MPI_CLASS_ERROR( "Problem free'ing MPI_Comm object" );
            d_comm = MPI_CLASS_COMM_NULL;
            ++N_MPI_Comm_destroyed;
#endif
        }
        if ( d_ranks != nullptr )
            delete[] d_ranks;
        delete d_count;
    }
    if ( d_currentTag == nullptr ) {
        // No tag index
    } else if ( d_currentTag[1] > 1 ) {
        --( d_currentTag[1] );
    } else {
        delete[] d_currentTag;
    }
    d_manage     = false;
    d_count      = nullptr;
    d_ranks      = nullptr;
    d_rank       = 0;
    d_size       = 1;
    d_maxTag     = 0;
    d_isNull     = true;
    d_currentTag = nullptr;
    d_call_abort = true;
}


/************************************************************************
 *  Copy constructors                                                    *
 ************************************************************************/
MPI_CLASS::MPI_CLASS( const MPI_CLASS &comm )
    : d_comm( comm.d_comm ),
      d_isNull( comm.d_isNull ),
      d_manage( comm.d_manage ),
      d_call_abort( comm.d_call_abort ),
      d_rank( comm.d_rank ),
      d_size( comm.d_size ),
      d_maxTag( comm.d_maxTag ),
      d_currentTag( comm.d_currentTag ),
      d_ranks( comm.d_ranks ),
      d_count( nullptr )
{
    // Initialize the data members to the existing comm object
    if ( d_currentTag != nullptr )
        ++d_currentTag[1];
    d_call_abort = comm.d_call_abort;
    // Set and increment the count
    d_count = comm.d_count;
    if ( d_count != nullptr )
        ++( *d_count );
}
MPI_CLASS::MPI_CLASS( MPI_CLASS &&rhs ) : MPI_CLASS()
{
    std::swap( d_comm, rhs.d_comm );
    std::swap( d_isNull, rhs.d_isNull );
    std::swap( d_manage, rhs.d_manage );
    std::swap( d_call_abort, rhs.d_call_abort );
    std::swap( profile_level, rhs.profile_level );
    std::swap( d_rank, rhs.d_rank );
    std::swap( d_size, rhs.d_size );
    std::swap( d_ranks, rhs.d_ranks );
    std::swap( d_maxTag, rhs.d_maxTag );
    std::swap( d_currentTag, rhs.d_currentTag );
    std::swap( d_count, rhs.d_count );
}


/************************************************************************
 *  Assignment operators                                                 *
 ************************************************************************/
MPI_CLASS &MPI_CLASS::operator=( const MPI_CLASS &comm )
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
    this->d_maxTag     = comm.d_maxTag;
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
MPI_CLASS &MPI_CLASS::operator=( MPI_CLASS &&rhs )
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
    std::swap( d_maxTag, rhs.d_maxTag );
    std::swap( d_currentTag, rhs.d_currentTag );
    std::swap( d_count, rhs.d_count );
    return *this;
}


/************************************************************************
 *  Constructor from existing MPI communicator                           *
 ************************************************************************/
int d_global_currentTag_world1[2] = { 1, 1 };
int d_global_currentTag_world2[2] = { 1, 1 };
int d_global_currentTag_self[2]   = { 1, 1 };
#ifdef USE_MPI
std::atomic_int d_global_count_world1 = { 1 };
std::atomic_int d_global_count_world2 = { 1 };
std::atomic_int d_global_count_self   = { 1 };
#endif
MPI_CLASS::MPI_CLASS( Comm comm, bool manage )
{
    d_count  = nullptr;
    d_ranks  = nullptr;
    d_manage = false;
    // Check if we are using our version of comm_world
    if ( comm == MPI_CLASS_COMM_WORLD ) {
        d_comm = AMP::AMPManager::getCommWorld().d_comm;
    } else if ( comm == MPI_CLASS_COMM_SELF ) {
        d_comm = MPI_COMM_SELF;
    } else if ( comm == MPI_CLASS_COMM_NULL ) {
        d_comm = MPI_COMM_NULL;
    } else {
        d_comm = comm;
    }
#ifdef USE_MPI
    // We are using MPI, use the MPI communicator to initialize the data
    if ( d_comm == MPI_COMM_NULL ) {
        d_rank   = 0;
        d_size   = 0;
        d_maxTag = mpi_max_tag;
    } else if ( d_comm == MPI_COMM_SELF && !MPI_Active() ) {
        d_rank   = 0;
        d_size   = 1;
        d_maxTag = mpi_max_tag;
    } else {
        // Attach the error handler
        StackTrace::setMPIErrorHandler( d_comm );
        // Get the communicator properties
        MPI_Comm_rank( d_comm, &d_rank );
        MPI_Comm_size( d_comm, &d_size );
        int flag, *val;
        int ierr = MPI_Comm_get_attr( d_comm, MPI_TAG_UB, &val, &flag );
        MPI_CLASS_ASSERT( ierr == MPI_SUCCESS );
        if ( flag == 0 ) {
            d_maxTag = mpi_max_tag; // The tag is not a valid attribute use default value
        } else {
            d_maxTag = *val;
            if ( d_maxTag < 0 ) {
                d_maxTag = 0x7FFFFFFF;
            } // The maximum tag is > a signed int (set to 2^31-1)
            MPI_CLASS_INSIST( d_maxTag >= 0x7FFF, "maximum tag size is < MPI standard" );
        }
    }
    d_isNull = d_comm == MPI_COMM_NULL;
    if ( manage && d_comm != MPI_COMM_NULL && d_comm != MPI_COMM_SELF && d_comm != MPI_COMM_WORLD )
        d_manage = true;
    // Create the count (Note: we do not need to worry about thread safety)
    if ( d_comm == AMP::AMPManager::getCommWorld().d_comm ) {
        d_count = &d_global_count_world1;
        ++( *d_count );
    } else if ( d_comm == MPI_COMM_WORLD ) {
        d_count = &d_global_count_world2;
        ++( *d_count );
    } else if ( d_comm == MPI_COMM_SELF ) {
        d_count = &d_global_count_self;
        ++( *d_count );
    } else if ( d_comm == MPI_COMM_NULL ) {
        d_count = nullptr;
    } else {
        d_count  = new std::atomic_int;
        *d_count = 1;
    }
    if ( d_manage ) {
        ++N_MPI_Comm_created;
        // StackTrace::multi_stack_info( StackTrace::getCallStack() ).print( std::cout );
    }
    // Create d_ranks
    if ( d_size > 1 ) {
        d_ranks    = new int[d_size];
        d_ranks[0] = -1;
    }
#else
    // We are not using MPI, intialize based on the communicator
    NULL_USE( manage );
    d_rank   = 0;
    d_size   = 1;
    d_maxTag = mpi_max_tag;
    d_isNull = d_comm == MPI_COMM_NULL;
    if ( d_isNull )
        d_size       = 0;
#endif
    if ( d_comm == AMP::AMPManager::getCommWorld().d_comm ) {
        d_currentTag = d_global_currentTag_world1;
        ++( this->d_currentTag[1] );
    } else if ( d_comm == MPI_COMM_WORLD ) {
        d_currentTag = d_global_currentTag_world2;
        ++( this->d_currentTag[1] );
    } else if ( d_comm == MPI_COMM_SELF ) {
        d_currentTag = d_global_currentTag_self;
        ++( this->d_currentTag[1] );
    } else if ( d_comm == MPI_COMM_NULL ) {
        d_currentTag = nullptr;
    } else {
        d_currentTag    = new int[2];
        d_currentTag[0] = ( d_maxTag <= 0x10000 ) ? 1 : 0x1FFF;
        d_currentTag[1] = 1;
    }
    d_call_abort = true;
}


/************************************************************************
 *  Return the ranks of the communicator in the global comm              *
 ************************************************************************/
std::vector<int> MPI_CLASS::globalRanks() const
{
    // Get my global rank if it has not been set
    static int myGlobalRank = -1;
    if ( myGlobalRank == -1 ) {
#ifdef USE_MPI
        if ( MPI_Active() )
            MPI_Comm_rank( AMP::AMPManager::getCommWorld().d_comm, &myGlobalRank );
#else
        myGlobalRank = 0;
#endif
    }
    // Check if we are dealing with a serial or null communicator
    if ( d_size == 1 )
        return std::vector<int>( 1, myGlobalRank );
    if ( d_ranks == nullptr || d_comm == MPI_COMM_NULL )
        return std::vector<int>();
    // Fill d_ranks if necessary
    if ( d_ranks[0] == -1 ) {
        if ( d_comm == AMP::AMPManager::getCommWorld().d_comm ) {
            for ( int i = 0; i < d_size; i++ )
                d_ranks[i] = i;
        } else {

            MPI_CLASS_ASSERT( myGlobalRank != -1 );
            this->allGather( myGlobalRank, d_ranks );
        }
    }
    // Return d_ranks
    return std::vector<int>( d_ranks, d_ranks + d_size );
}


/************************************************************************
 *  Generate a random number                                             *
 ************************************************************************/
size_t MPI_CLASS::rand() const
{
    size_t val = 0;
    if ( getRank() == 0 ) {
        static std::random_device rd;
        static std::mt19937 gen( rd() );
        static std::uniform_int_distribution<size_t> dist;
        val = dist( gen );
    }
    val = bcast( val, 0 );
    return val;
}


/************************************************************************
 *  Intersect two communicators                                          *
 ************************************************************************/
#ifdef USE_MPI
static inline void MPI_Group_free2( MPI_Group *group )
{
    if ( *group != MPI_GROUP_EMPTY ) {
        // MPICH is fine with free'ing an empty group, OpenMPI crashes
        MPI_Group_free( group );
    }
}
MPI_CLASS MPI_CLASS::intersect( const MPI_CLASS &comm1, const MPI_CLASS &comm2 )
{
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
    MPI_CLASS new_comm( MPI_CLASS_COMM_NULL );
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
            new_comm = MPI_CLASS( MPI_CLASS_COMM_NULL, false );
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
                new_comm = MPI_CLASS( new_MPI_comm, true );
            } else {
                // We actually want a null comm for this communicator
                new_comm = MPI_CLASS( MPI_CLASS_COMM_NULL, false );
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
MPI_CLASS MPI_CLASS::intersect( const MPI_CLASS &comm1, const MPI_CLASS &comm2 )
{
    if ( comm1.isNull() || comm2.isNull() )
        return MPI_CLASS( MPI_CLASS_COMM_NULL, false );
    MPI_CLASS_ASSERT( comm1.d_size == 1 && comm2.d_size == 1 );
    return comm1;
}
#endif


/************************************************************************
 *  Split a comm						                                 *
 ************************************************************************/
MPI_CLASS MPI_CLASS::split( int color, int key, bool manage ) const
{
    if ( d_isNull ) {
        return MPI_CLASS( MPI_CLASS_COMM_NULL );
    } else if ( d_size == 1 ) {
        if ( color == -1 )
            return MPI_CLASS( MPI_CLASS_COMM_NULL );
        return dup();
    }
    MPI_CLASS::Comm new_MPI_comm = MPI_CLASS_COMM_NULL;
#ifdef USE_MPI
    // USE MPI to split the communicator
    int error = 0;
    if ( color == -1 ) {
        error = MPI_Comm_split( d_comm, MPI_UNDEFINED, key, &new_MPI_comm );
    } else {
        error = MPI_Comm_split( d_comm, color, key, &new_MPI_comm );
    }
    MPI_CLASS_INSIST( error == MPI_SUCCESS, "Error calling MPI routine" );
#endif
    // Create the new object
    NULL_USE( key );
    MPI_CLASS new_comm( new_MPI_comm, manage );
    new_comm.d_call_abort = d_call_abort;
    return new_comm;
}
MPI_CLASS MPI_CLASS::splitByNode( int key, bool manage ) const
{
    // Check if we are dealing with a single processor (trivial case)
    if ( d_size == 1 )
        return this->split( 0, 0, manage );
    // Get the node name
    std::string name = MPI_CLASS::getNodeName();
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
MPI_CLASS MPI_CLASS::dup( bool manage ) const
{
    if ( d_isNull )
        return MPI_CLASS( MPI_CLASS_COMM_NULL );
    MPI_CLASS::Comm new_MPI_comm = d_comm;
#if defined( USE_MPI ) || defined( USE_PETSC )
    // USE MPI to duplicate the communicator
    MPI_Comm_dup( d_comm, &new_MPI_comm );
#else
    static MPI_CLASS::Comm uniqueGlobalComm = 11;
    new_MPI_comm = uniqueGlobalComm;
    uniqueGlobalComm++;
#endif
    // Create the new comm object
    MPI_CLASS new_comm( new_MPI_comm, manage );
    new_comm.d_isNull     = d_isNull;
    new_comm.d_call_abort = d_call_abort;
    return new_comm;
}


/************************************************************************
 *  Get the node name                                                    *
 ************************************************************************/
std::string MPI_CLASS::getNodeName()
{
#ifdef USE_MPI
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
bool MPI_CLASS::operator==( const MPI_CLASS &comm ) const { return d_comm == comm.d_comm; }


/************************************************************************
 *  Overload operator !=                                                 *
 ************************************************************************/
bool MPI_CLASS::operator!=( const MPI_CLASS &comm ) const { return d_comm != comm.d_comm; }


/************************************************************************
 *  Overload operator <                                                  *
 ************************************************************************/
bool MPI_CLASS::operator<( const MPI_CLASS &comm ) const
{
    MPI_CLASS_ASSERT( !this->d_isNull && !comm.d_isNull );
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
#ifdef USE_MPI
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
bool MPI_CLASS::operator<=( const MPI_CLASS &comm ) const
{
    MPI_CLASS_ASSERT( !this->d_isNull && !comm.d_isNull );
    bool flag = true;
    // First check if either communicator is NULL
    if ( this->d_isNull )
        return false;
    if ( comm.d_isNull )
        flag = false;
#ifdef USE_MPI
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
#ifdef USE_MPI
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
bool MPI_CLASS::operator>( const MPI_CLASS &comm ) const
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
#ifdef USE_MPI
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
#else
    NULL_USE( comm );
#endif
    // Perform a global reduce of the flag (equivalent to all operation)
    return allReduce( flag );
}


/************************************************************************
 *  Overload operator >=                                                 *
 ************************************************************************/
bool MPI_CLASS::operator>=( const MPI_CLASS &comm ) const
{
    bool flag = true;
    // First check if either communicator is NULL
    if ( this->d_isNull )
        return false;
    if ( comm.d_isNull )
        flag = false;
#ifdef USE_MPI
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
#ifdef USE_MPI
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
int MPI_CLASS::compare( const MPI_CLASS &comm ) const
{
    if ( d_comm == comm.d_comm )
        return 1;
#ifdef USE_MPI
    if ( d_isNull || comm.d_isNull )
        return 0;
    int result;
    int error = MPI_Comm_compare( d_comm, comm.d_comm, &result );
    MPI_CLASS_INSIST( error == MPI_SUCCESS, "Error calling MPI routine" );
    if ( result == MPI_IDENT )
        return 2;
    else if ( result == MPI_CONGRUENT )
        return 3;
    else if ( result == MPI_SIMILAR )
        return 4;
    else if ( result == MPI_UNEQUAL )
        return 0;
    MPI_CLASS_ERROR( "Unknown results from comm compare" );
#else
    if ( comm.d_comm == MPI_COMM_NULL || d_comm == MPI_COMM_NULL )
        return 0;
    else
        return 3;
#endif
    return 0;
}


/************************************************************************
 *  Abort the program.                                                   *
 ************************************************************************/
void MPI_CLASS::setCallAbortInSerialInsteadOfExit( bool flag ) { d_call_abort = flag; }
void MPI_CLASS::abort() const
{
#ifdef USE_MPI
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
int MPI_CLASS::newTag()
{
#ifdef USE_MPI
    // Syncronize the processes to ensure all ranks enter this call
    // Needed so the count will match
    barrier();
    // Return and increment the tag
    int tag = ( *d_currentTag )++;
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum number of tags exceeded\n" );
    return tag;
#else
    static int globalCurrentTag = 1;
    return globalCurrentTag++;
#endif
}


/************************************************************************
 *  allReduce                                                            *
 ************************************************************************/
static inline void setBit( uint64_t *x, size_t index )
{
    size_t i      = index >> 6;
    size_t j      = index & 0x3F;
    uint64_t mask = ( (uint64_t) 0x01 ) << j;
    x[i]          = x[i] | mask;
}
static inline bool readBit( const uint64_t *x, size_t index )
{
    size_t i      = index >> 6;
    size_t j      = index & 0x3F;
    uint64_t mask = ( (uint64_t) 0x01 ) << j;
    return ( x[i] & mask ) != 0;
}
bool MPI_CLASS::allReduce( const bool value ) const
{
    bool ret = value;
    if ( d_size > 1 ) {
#ifdef USE_MPI
        MPI_Allreduce( (void *) &value, (void *) &ret, 1, MPI_UNSIGNED_CHAR, MPI_MIN, d_comm );
#else
        NULL_USE( value );
        MPI_CLASS_ERROR( "This shouldn't be possible" );
#endif
    }
    return ret;
}
void MPI_CLASS::allReduce( std::vector<bool> &x ) const
{
    if ( d_size <= 1 )
        return;
#ifdef USE_MPI
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
    NULL_USE( x );
    MPI_CLASS_ERROR( "This shouldn't be possible" );
#endif
}


/************************************************************************
 *  anyReduce                                                            *
 ************************************************************************/
bool MPI_CLASS::anyReduce( const bool value ) const
{
    bool ret = value;
    if ( d_size > 1 ) {
#ifdef USE_MPI
        MPI_Allreduce( (void *) &value, (void *) &ret, 1, MPI_UNSIGNED_CHAR, MPI_MAX, d_comm );
#else
        NULL_USE( value );
        MPI_CLASS_ERROR( "This shouldn't be possible" );
#endif
    }
    return ret;
}
void MPI_CLASS::anyReduce( std::vector<bool> &x ) const
{
    if ( d_size <= 1 )
        return;
#ifdef USE_MPI
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
    NULL_USE( x );
    MPI_CLASS_ERROR( "This shouldn't be possible" );
#endif
}


/************************************************************************
 *  Perform a global barrier across all processors.                      *
 ************************************************************************/
void MPI_CLASS::barrier() const
{
#ifdef USE_MPI
    if ( d_size <= 1 )
        return;
    PROFILE_START( "barrier", profile_level );
    MPI_Barrier( d_comm );
    PROFILE_STOP( "barrier", profile_level );
#endif
}
void MPI_CLASS::sleepBarrier() const
{
#ifdef USE_MPI
    if ( d_size <= 1 )
        return;
    using namespace std::chrono_literals;
    PROFILE_START( "sleepBarrier", profile_level );
    int flag = 0;
    MPI_Request request;
    MPI_Ibarrier( d_comm, &request );
    int err = MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
    MPI_CLASS_ASSERT( err == MPI_SUCCESS ); // Check that the first call is valid
    while ( !flag ) {
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::sleep_for( 10ms );
        // Check if the request has finished
        MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
    }
    PROFILE_STOP( "sleepBarrier", profile_level );
#endif
}


/************************************************************************
 *  Send/Recv data                                                       *
 *  We need a concrete instantiation of send for use without MPI         *
 ************************************************************************/
#ifdef USE_MPI
void MPI_CLASS::sendBytes( const void *buf, int bytes, int recv_proc, int tag ) const
{
    send<char>( (const char *) buf, bytes, recv_proc, tag );
}
void MPI_CLASS::recvBytes( void *buf, int bytes, const int send_proc, int tag ) const
{
    int bytes2 = bytes;
    recv<char>( (char *) buf, bytes2, send_proc, false, tag );
}
MPI_CLASS::Request MPI_CLASS::IsendBytes( const void *buf, int bytes, int recv_proc, int tag ) const
{
    return Isend<char>( (const char *) buf, bytes, recv_proc, tag );
}
MPI_CLASS::Request MPI_CLASS::IrecvBytes( void *buf, int bytes, int send_proc, const int tag ) const
{
    return Irecv<char>( (char *) buf, bytes, send_proc, tag );
}
#else
void MPI_CLASS::sendBytes( const void *buf, int bytes, int, int tag ) const
{
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    MPI_CLASS_INSIST( tag >= 0, "tag must be >= 0" );
    PROFILE_START( "sendBytes", profile_level );
    auto id = getRequest( d_comm, tag );
    auto it = global_isendrecv_list.find( id );
    MPI_CLASS_INSIST( it != global_isendrecv_list.end(),
                      "send must be paired with a previous call to irecv in serial" );
    MPI_CLASS_ASSERT( it->second.status == 2 );
    memcpy( (char *) it->second.data, buf, bytes );
    global_isendrecv_list.erase( it );
    PROFILE_STOP( "sendBytes", profile_level );
}
void MPI_CLASS::recvBytes( void *buf, int bytes, const int, int tag ) const
{
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    MPI_CLASS_INSIST( tag >= 0, "tag must be >= 0" );
    PROFILE_START( "recv<char>", profile_level );
    auto id = getRequest( d_comm, tag );
    auto it = global_isendrecv_list.find( id );
    MPI_CLASS_INSIST( it != global_isendrecv_list.end(),
                      "recv must be paired with a previous call to isend in serial" );
    MPI_CLASS_ASSERT( it->second.status == 1 );
    MPI_CLASS_ASSERT( it->second.bytes == bytes );
    memcpy( buf, it->second.data, bytes );
    global_isendrecv_list.erase( it );
    PROFILE_STOP( "recv<char>", profile_level );
}
MPI_CLASS::Request MPI_CLASS::IsendBytes( const void *buf, int bytes, int, int tag ) const
{
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    MPI_CLASS_INSIST( tag >= 0, "tag must be >= 0" );
    PROFILE_START( "IsendBytes", profile_level );
    auto id = getRequest( d_comm, tag );
    auto it = global_isendrecv_list.find( id );
    if ( it == global_isendrecv_list.end() ) {
        // We are calling isend first
        Isendrecv_struct data;
        data.bytes = bytes;
        data.data = buf;
        data.status = 1;
        data.comm = d_comm;
        data.tag = tag;
        global_isendrecv_list.insert(
            std::pair<MPI_CLASS::Request2, Isendrecv_struct>( id, data ) );
    } else {
        // We called irecv first
        MPI_CLASS_ASSERT( it->second.status == 2 );
        MPI_CLASS_ASSERT( it->second.bytes == bytes );
        memcpy( (char *) it->second.data, buf, bytes );
        global_isendrecv_list.erase( it );
    }
    PROFILE_STOP( "IsendBytes", profile_level );
    return id;
}
MPI_CLASS::Request
MPI_CLASS::IrecvBytes( void *buf, const int bytes, const int, const int tag ) const
{
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    MPI_CLASS_INSIST( tag >= 0, "tag must be >= 0" );
    PROFILE_START( "Irecv<char>", profile_level );
    auto id = getRequest( d_comm, tag );
    auto it = global_isendrecv_list.find( id );
    if ( it == global_isendrecv_list.end() ) {
        // We are calling Irecv first
        Isendrecv_struct data;
        data.bytes = bytes;
        data.data = buf;
        data.status = 2;
        data.comm = d_comm;
        data.tag = tag;
        global_isendrecv_list.insert( std::pair<MPI_CLASS::Request, Isendrecv_struct>( id, data ) );
    } else {
        // We called Isend first
        MPI_CLASS_ASSERT( it->second.status == 1 );
        MPI_CLASS_ASSERT( it->second.bytes == bytes );
        memcpy( buf, it->second.data, bytes );
        global_isendrecv_list.erase( it );
    }
    PROFILE_STOP( "Irecv<char>", profile_level );
    return id;
}
#endif


/************************************************************************
 *  Communicate ranks for communication                                  *
 ************************************************************************/
std::vector<int> MPI_CLASS::commRanks( const std::vector<int> &ranks ) const
{
#ifdef USE_MPI
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
#ifdef USE_MPI
void MPI_CLASS::wait( Request2 request )
{
    PROFILE_START( "wait", profile_level );
    int flag = 0;
    int err  = MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
    MPI_CLASS_ASSERT( err == MPI_SUCCESS ); // Check that the first call is valid
    while ( !flag ) {
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
        // Check if the request has finished
        MPI_Test( &request, &flag, MPI_STATUS_IGNORE );
    }
    PROFILE_STOP( "wait", profile_level );
}
int MPI_CLASS::waitAny( int count, Request2 *request )
{
    if ( count == 0 )
        return -1;
    PROFILE_START( "waitAny", profile_level );
    int index = -1;
    int flag  = 0;
    int err   = MPI_Testany( count, request, &index, &flag, MPI_STATUS_IGNORE );
    MPI_CLASS_ASSERT( err == MPI_SUCCESS ); // Check that the first call is valid
    while ( !flag ) {
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
        // Check if the request has finished
        MPI_Testany( count, request, &index, &flag, MPI_STATUS_IGNORE );
    }
    MPI_CLASS_ASSERT( index >= 0 ); // Check that the index is valid
    PROFILE_STOP( "waitAny", profile_level );
    return index;
}
void MPI_CLASS::waitAll( int count, Request2 *request )
{
    if ( count == 0 )
        return;
    PROFILE_START( "waitAll", profile_level );
    int flag = 0;
    int err  = MPI_Testall( count, request, &flag, MPI_STATUS_IGNORE );
    MPI_CLASS_ASSERT( err == MPI_SUCCESS ); // Check that the first call is valid
    while ( !flag ) {
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
        // Check if the request has finished
        MPI_Testall( count, request, &flag, MPI_STATUS_IGNORE );
    }
    PROFILE_STOP( "waitAll", profile_level );
}
std::vector<int> MPI_CLASS::waitSome( int count, Request2 *request )
{
    if ( count == 0 )
        return std::vector<int>();
    PROFILE_START( "waitSome", profile_level );
    std::vector<int> indicies( count, -1 );
    int outcount = 0;
    int err      = MPI_Testsome( count, request, &outcount, &indicies[0], MPI_STATUS_IGNORE );
    MPI_CLASS_ASSERT( err == MPI_SUCCESS );        // Check that the first call is valid
    MPI_CLASS_ASSERT( outcount != MPI_UNDEFINED ); // Check that the first call is valid
    while ( outcount == 0 ) {
        // Put the current thread to sleep to allow other threads to run
        sched_yield();
        // Check if the request has finished
        MPI_Testsome( count, request, &outcount, &indicies[0], MPI_STATUS_IGNORE );
    }
    indicies.resize( outcount );
    PROFILE_STOP( "waitSome", profile_level );
    return indicies;
}
#else
void MPI_CLASS::wait( Request2 request )
{
    PROFILE_START( "wait", profile_level );
    while ( 1 ) {
        // Check if the request is in our list
        if ( global_isendrecv_list.find( request ) == global_isendrecv_list.end() )
            break;
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
    }
    PROFILE_STOP( "wait", profile_level );
}
int MPI_CLASS::waitAny( int count, Request2 *request )
{
    if ( count == 0 )
        return -1;
    PROFILE_START( "waitAny", profile_level );
    int index = 0;
    while ( 1 ) {
        // Check if the request is in our list
        bool found_any = false;
        for ( int i = 0; i < count; i++ ) {
            if ( global_isendrecv_list.find( request[i] ) == global_isendrecv_list.end() ) {
                found_any = true;
                index = i;
            }
        }
        if ( found_any )
            break;
        // Put the current thread to sleep to allow other threads to run
        std::this_thread::yield();
    }
    PROFILE_STOP( "waitAny", profile_level );
    return index;
}
void MPI_CLASS::waitAll( int count, Request2 *request )
{
    if ( count == 0 )
        return;
    PROFILE_START( "waitAll", profile_level );
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
    PROFILE_STOP( "waitAll", profile_level );
}
std::vector<int> MPI_CLASS::waitSome( int count, Request2 *request )
{
    if ( count == 0 )
        return std::vector<int>();
    PROFILE_START( "waitSome", profile_level );
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
    PROFILE_STOP( "waitSome", profile_level );
    return indicies;
}
#endif
static std::vector<MPI_CLASS::Request2> getRequests( int count, const MPI_CLASS::Request *request )
{
    std::vector<MPI_CLASS::Request2> request2( count );
    for ( int i = 0; i < count; i++ )
        request2[i] = request[i];
    return request2;
}
void MPI_CLASS::wait( const Request &request ) { wait( static_cast<Request2>( request ) ); }
int MPI_CLASS::waitAny( int count, const Request *request )
{
    return waitAny( count, getRequests( count, request ).data() );
}
void MPI_CLASS::waitAll( int count, const Request *request )
{
    waitAll( count, getRequests( count, request ).data() );
}
std::vector<int> MPI_CLASS::waitSome( int count, const Request *request )
{
    return waitSome( count, getRequests( count, request ).data() );
}


/************************************************************************
 *  Probe functions                                                      *
 ************************************************************************/
#ifdef USE_MPI
int MPI_CLASS::Iprobe( int source, int tag ) const
{
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    MPI_CLASS_INSIST( tag >= 0, "tag must be >= 0" );
    MPI_Status status;
    int flag = 0;
    MPI_Iprobe( source, tag, d_comm, &flag, &status );
    if ( flag == 0 )
        return -1;
    int count;
    MPI_Get_count( &status, MPI_BYTE, &count );
    MPI_CLASS_ASSERT( count >= 0 );
    return count;
}
int MPI_CLASS::probe( int source, int tag ) const
{
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    MPI_CLASS_INSIST( tag >= 0, "tag must be >= 0" );
    MPI_Status status;
    MPI_Probe( source, tag, d_comm, &status );
    int count;
    MPI_Get_count( &status, MPI_BYTE, &count );
    MPI_CLASS_ASSERT( count >= 0 );
    return count;
}
#else
int MPI_CLASS::Iprobe( int source, int tag ) const
{
    AMP_ASSERT( source == 0 || source < -1 );
    for ( const auto &tmp : global_isendrecv_list ) {
        const auto &data = tmp.second;
        if ( data.comm == d_comm && ( data.tag == tag || tag == -1 ) && data.status == 1 )
            return data.bytes;
    }
    return -1;
}
int MPI_CLASS::probe( int source, int tag ) const
{
    int bytes = Iprobe( source, tag );
    AMP_INSIST( bytes >= 0, "probe called before message started in serial" );
    return bytes;
}
#endif


/************************************************************************
 *  Timer functions                                                      *
 ************************************************************************/
#ifdef USE_MPI
double MPI_CLASS::time() { return MPI_Wtime(); }
double MPI_CLASS::tick() { return MPI_Wtick(); }
#else
double MPI_CLASS::time()
{
    auto t = std::chrono::system_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>( t.time_since_epoch() );
    return 1e-9 * ns.count();
}
double MPI_CLASS::tick()
{
    auto period = std::chrono::system_clock::period();
    return static_cast<double>( period.num ) / static_cast<double>( period.den );
}
#endif


/************************************************************************
 *  Serialize a block of code across MPI processes                       *
 ************************************************************************/
void MPI_CLASS::serializeStart()
{
#ifdef USE_MPI
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
void MPI_CLASS::serializeStop()
{
#ifdef USE_MPI
    // Send flag to next rank
    if ( d_rank < d_size - 1 )
        MPI_Send( &d_rank, 1, MPI_INT, d_rank + 1, 5627, d_comm );
    // Final barrier to sync all threads
    MPI_Barrier( d_comm );
#endif
}


/****************************************************************************
 * Function to start/stop MPI                                                *
 ****************************************************************************/
#ifdef USE_MPI
static bool called_MPI_Init = false;
#endif
void MPI_CLASS::start_MPI( int argc, char *argv[], int profile_level )
{
    changeProfileLevel( profile_level );
    NULL_USE( argc );
    NULL_USE( argv );
#ifdef USE_MPI
    if ( MPI_Active() ) {
        called_MPI_Init = false;
    } else {
        int provided;
        int result = MPI_Init_thread( &argc, &argv, MPI_THREAD_MULTIPLE, &provided );
        if ( result != MPI_SUCCESS )
            MPI_CLASS_ERROR( "AMP was unable to initialize MPI" );
        if ( provided < MPI_THREAD_MULTIPLE )
            AMP::perr << "Warning: Failed to start MPI with MPI_THREAD_MULTIPLE\n";
        called_MPI_Init = true;
        AMPManager::setCommWorld( MPI_COMM_WORLD );
    }
#else
    AMPManager::setCommWorld( MPI_COMM_SELF );
#endif
}
void MPI_CLASS::stop_MPI()
{
    AMPManager::setCommWorld( AMP_COMM_NULL );
#ifdef USE_MPI
    int finalized;
    MPI_Finalized( &finalized );
    if ( called_MPI_Init && !finalized ) {
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_Finalize();
        called_MPI_Init = true;
    }
#endif
}

/****************************************************************************
 * Request                                                                   *
 ****************************************************************************/
MPI_CLASS::Request::Request( MPI_CLASS::Request2 request, std::any data )
{
    using TYPE = typename std::remove_reference<decltype( *( d_data.get() ) )>::type;
    if ( data.has_value() ) {
        auto deleter = []( TYPE *p ) {
            MPI_CLASS::wait( p->first );
            delete p;
        };
        d_data.reset( new TYPE, deleter );
    } else {
        d_data.reset( new TYPE );
    }
    d_data->first  = request;
    d_data->second = std::move( data );
}
MPI_CLASS::Request::~Request() {}


/****************************************************************************
 * getComm                                                                   *
 ****************************************************************************/
template<class TYPE>
AMP_MPI getComm( const TYPE &obj )
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
INSTANTIATE_GET_COMM( float );
INSTANTIATE_GET_COMM( double );
INSTANTIATE_GET_COMM( std::complex<float> );
INSTANTIATE_GET_COMM( std::complex<double> );
INSTANTIATE_GET_COMM( std::string );


/****************************************************************************
 * pack/unpack routines                                                      *
 ****************************************************************************/
template<>
size_t packSize( const std::string &s )
{
    return s.size() + 1;
}
template<>
size_t pack( const std::string &s, std::byte *buf )
{
    memcpy( buf, s.data(), s.size() + 1 );
    return s.size() + 1;
}
template<>
size_t unpack( std::string &s, const std::byte *buf )
{
    s = std::string( reinterpret_cast<const char *>( buf ) );
    return s.size() + 1;
}
/*template<>
size_t packSize( const std::vector<bool>::reference &s )
{
    return s.size() + 1;
}
template<>
size_t pack( const std::vector<bool>::reference &s, std::byte *buf )
{
    memcpy( buf, s.data(), s.size() + 1 );
    return s.size() + 1;
}
template<>
size_t unpack( std::vector<bool>::reference &s, const std::byte *buf )
{
    s = std::vector<bool>::reference( reinterpret_cast<const char *>( buf ) );
    return s.size() + 1;
}*/


} // namespace AMP


/****************************************************************************
 * Explicit instantiation                                                    *
 ****************************************************************************/
INSTANTIATE_MPI_TYPE( char );
INSTANTIATE_MPI_TYPE( int8_t );
INSTANTIATE_MPI_TYPE( uint8_t );
INSTANTIATE_MPI_TYPE( int16_t );
INSTANTIATE_MPI_TYPE( uint16_t );
INSTANTIATE_MPI_TYPE( int32_t );
INSTANTIATE_MPI_TYPE( uint32_t );
INSTANTIATE_MPI_TYPE( int64_t );
INSTANTIATE_MPI_TYPE( uint64_t );
INSTANTIATE_MPI_TYPE( float );
INSTANTIATE_MPI_TYPE( double );
INSTANTIATE_MPI_TYPE( std::complex<float> );
INSTANTIATE_MPI_TYPE( std::complex<double> );
INSTANTIATE_MPI_BCAST( bool );
INSTANTIATE_MPI_SENDRECV( bool );
INSTANTIATE_MPI_BCAST( std::string );
INSTANTIATE_MPI_SENDRECV( std::string );
INSTANTIATE_MPI_GATHER( std::string );
