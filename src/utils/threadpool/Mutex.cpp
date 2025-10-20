#include "AMP/utils/threadpool/Mutex.h"
#include "AMP/utils/AMP_MPI.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <thread>

#include "ProfilerApp.h"


namespace AMP {


// Get the id for the current thread
static inline int getThreadId()
{
    auto id = std::this_thread::get_id();
    static std::hash<std::thread::id> hasher;
    int hash = static_cast<int>( hasher( id ) );
    hash     = std::abs( hash );
    return hash;
}


/******************************************************************
 * Mutex class                                                     *
 ******************************************************************/
Mutex::Mutex() : d_recursive( false ), d_count( 0 ), d_id( -1 ) {}
Mutex::Mutex( bool recursive ) : d_recursive( recursive ), d_count( 0 ), d_id( -1 ) {}
void Mutex::lock()
{
    // Check if we already own the lock
    int id = getThreadId();
    if ( d_count > 0 && d_id == id ) {
        if ( !d_recursive )
            throw std::logic_error( "Lock is already locked and non-recursive" );
        // Increment the lock count
        const_cast<int &>( d_count )++;
        return;
    }
    // Acquire the lock
    d_mutex.lock();
    if ( d_count != 0 ) // If we are getting the lock, the count must be 0
        throw std::logic_error( "Internal error" );
    d_count = 1; // Change lock count after acquiring mutex
    d_id    = id;
}
bool Mutex::tryLock()
{
    // Check if we already own the lock
    int id = getThreadId();
    if ( d_count > 0 && d_id == id ) {
        if ( !d_recursive )
            return false;
        // Increment the lock count and return
        const_cast<int &>( d_count )++;
        return true;
    }
    // Try and acquire the lock
    bool success = d_mutex.try_lock();
    if ( success ) {
        if ( d_count != 0 ) // If we are getting the lock, the count must be 0
            throw std::logic_error( "Internal error" );
        d_count = 1; // Change lock count after acquiring mutex
        d_id    = id;
    }
    return success;
}
void Mutex::unlock()
{
    // Check if we already own the lock
    int id = getThreadId();
    if ( d_count <= 0 )
        throw std::logic_error( "Trying to release a lock that has not been locked" );
    if ( d_id != id )
        throw std::logic_error( "Thread that does not own lock is attempting to release" );
    // Release the lock
    const_cast<int &>( d_count )--; // Change lock count before releasing mutex
    if ( d_count == 0 ) {
        d_id = -1;
        d_mutex.unlock();
    }
}
bool Mutex::ownLock() const
{
    int id = getThreadId();
    if ( d_count > 0 && d_id == id )
        return true;
    return false;
}


/****************************************************************************
 *  Function to lock a mutex across multiple threads                         *
 ****************************************************************************/
void lock_MPI_Mutex( Mutex &mutex, const AMP_MPI &comm )
{
#ifdef DISABLE_THREAD_CHANGES
    // Even without most of the thread changes we still need the lock to preserve
    // the lock of lock and free, but this is unlikely to affect performance
    mutex.lock();
#else
    PROFILE( "lock_MPI_Mutex", 5 );
    if ( mutex.ownLock() ) {
        // Special case when we already own the lock
        mutex.lock();
        return;
    }
    #if 1
    // First create a barrier to syncronize all processes
    comm.barrier();
    // Next, let rank 0 acquire the mutex
    if ( comm.getRank() == 0 )
        mutex.lock();
    comm.barrier();
    if ( comm.getRank() != 0 )
        mutex.lock();
    comm.barrier();
    /*// Let all other ranks try an acquire the lock
    bool success = true;
    if ( comm.getRank()!=0 )
        success = mutex.tryLock();
    // If all processes own the lock we are done, otherwise we can two or more communicators
    // with different ranks trying to acquire the same lock
    if ( comm.allReduce(success) )
        return;
    AMP_ERROR("Not finished");*/
    #else
    /*int result;
    MPI_Comm_compare( comm.getCommunicator(), MPI_COMM_WORLD, &result );
    if ( result==MPI_UNEQUAL )
        AMP_ERROR("Comm must match world");*/
    int rank = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    // Let rank 0 acquire the lock, then bcast when complete
    if ( rank == 0 )
        mutex.lock();
    int dummy = comm.bcast( rank, 0 );
    if ( rank != 0 )
        mutex.lock();
    // Syncronize the processes before exiting
    // Note: this should not be necessary, but simplifies debugging and should not create any
    // problems
    comm.barrier();
    #endif
#endif
}


} // namespace AMP
