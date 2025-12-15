#define _CRT_NONSTDC_NO_DEPRECATE
#include "AMP/utils/threadpool/ThreadPool.h"
#include "AMP/IO/PIO.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/threadpool/ThreadHelpers.h"

#include "StackTrace/StackTrace.h"
#include "StackTrace/Utilities.h"

#include <algorithm>
#include <bitset>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>
#include <typeinfo>


using AMP::perr;
using AMP::pout;


namespace AMP {


/******************************************************************
 * Run some basic compile-time checks                              *
 ******************************************************************/
static_assert( ThreadPool::MAX_THREADS % 64 == 0, "MAX_THREADS must be a multiple of 64" );
static_assert( ThreadPool::MAX_THREADS < 65535, "MAX_THREADS must < 65535" );
static_assert( sizeof( std::atomic_int32_t ) == 4, "atomic32 must be a 32-bit integer" );
static_assert( sizeof( std::atomic_int64_t ) == 8, "atomic64 must be a 64-bit integer" );
static_assert( sizeof( std::atomic_uint64_t ) == sizeof( uint64_t ), "atomic_uint64_t size" );


/******************************************************************
 * Get/Set a bit                                                   *
 * Note: The these int64_atomic versions are thread-safe           *
 ******************************************************************/
static inline void set_bit( volatile std::atomic_uint64_t *x, size_t index )
{
    uint64_t mask = 0x01;
    mask <<= index & 0x3F;
    size_t i  = index >> 6;
    bool test = false;
    while ( !test ) {
        uint64_t y = x[i].load();
        test       = x[i].compare_exchange_weak( y, ( y | mask ) );
    }
}
static inline void unset_bit( volatile std::atomic_uint64_t *x, size_t index )
{
    uint64_t mask = 0x01;
    mask <<= index & 0x3F;
    mask      = ~mask;
    size_t i  = index >> 6;
    bool test = false;
    while ( !test ) {
        uint64_t y = x[i].load();
        test       = x[i].compare_exchange_weak( y, ( y & mask ) );
    }
}
static inline bool get_bit( const volatile std::atomic_uint64_t *x, size_t index )
{
    uint64_t mask = 0x01;
    mask <<= index & 0x3F;
    uint64_t y = x[index >> 6].load();
    return ( y & mask ) != 0;
}


/******************************************************************
 * Simple function to check if the parity is odd (true) or even    *
 ******************************************************************/
static inline bool is_odd8( size_t x )
{
    static_assert( sizeof( size_t ) == 8, "This only works for 64-bit integers" );
    x ^= ( x >> 1 );
    x ^= ( x >> 2 );
    x ^= ( x >> 4 );
    x ^= ( x >> 8 );
    x ^= ( x >> 16 );
    x ^= ( x >> 32 );
    return ( x & 0x01 ) > 0;
}
template<class int_type>
static inline int count_bits( int_type x )
{
    int count = 0;
    for ( size_t i = 0; i < 8 * sizeof( int_type ); i++ ) {
        if ( ( x >> i ) & 0x01 )
            ++count;
    }
    return count;
}


/******************************************************************
 * Set the global constants                                        *
 ******************************************************************/
constexpr uint16_t ThreadPool::MAX_THREADS;


/******************************************************************
 * Set the behavior of OS warnings                                 *
 ******************************************************************/
static int global_OS_behavior = 0;
static std::mutex OS_warning_mutex;
void ThreadPool::set_OS_warnings( int behavior )
{
    if ( behavior < 0 || behavior > 2 )
        throw std::logic_error( "Unknown OS warning flag" );
    global_OS_behavior = behavior;
}
static void OS_warning( const std::string &message )
{
    OS_warning_mutex.lock();
    if ( global_OS_behavior == 0 ) {
        pout << "Warning: " << message << std::endl;
    } else if ( global_OS_behavior == 2 ) {
        perr << "Error: " << message << std::endl;
    }
    OS_warning_mutex.unlock();
}
void ThreadPool::setErrorHandler( std::function<void( const std::string & )> fun )
{
    d_errorHandler = fun;
}


/******************************************************************
 * Functions for getting/setting thread/process affinities         *
 ******************************************************************/
int ThreadPool::getNumberOfProcessors() { return AMP::Thread::getNumberOfProcessors(); }
int ThreadPool::getCurrentProcessor() { return AMP::Thread::getCurrentProcessor(); }
std::vector<int> ThreadPool::getProcessAffinity() { return AMP::Thread::getProcessAffinity(); }
void ThreadPool::setProcessAffinity( const std::vector<int> &procs )
{
    return AMP::Thread::setProcessAffinity( procs );
}
std::vector<int> ThreadPool::getThreadAffinity()
{
    return AMP::Thread::getThreadAffinity( AMP::Thread::getCurrentThread() );
}
std::vector<int> ThreadPool::getThreadAffinity( int thread ) const
{
    if ( thread >= getNumThreads() )
        std::logic_error( "Invalid thread number" );
    auto handle = const_cast<std::thread &>( d_thread[thread] ).native_handle();
    return AMP::Thread::getThreadAffinity( handle );
}
void ThreadPool::setThreadAffinity( const std::vector<int> &procs )
{
    return AMP::Thread::setThreadAffinity( AMP::Thread::getCurrentThread(), procs );
}
void ThreadPool::setThreadAffinity( int thread, const std::vector<int> &procs ) const
{
    if ( thread >= getNumThreads() )
        std::logic_error( "Invalid thread number" );
    auto handle = const_cast<std::thread &>( d_thread[thread] ).native_handle();
    return AMP::Thread::setThreadAffinity( handle, procs );
}


/******************************************************************
 * Function to perform some basic checks before we start           *
 ******************************************************************/
void ThreadPool::check_startup()
{
    // Check getting/setting a bit
    volatile std::atomic_uint64_t x[2] = { 0x0, 0x7 };
    set_bit( x, 2 );
    unset_bit( x, 66 );
    if ( x[0] != 4 || x[1] != 3 || !get_bit( x, 2 ) || get_bit( x, 66 ) )
        throw std::logic_error( "Getting/setting a bit failed" );
    // Check the thread id
    bool pass = true;
    ThreadPoolID id;
    if ( id.getPriority() != -128 )
        pass = false;
    id.reset( 3, 564, nullptr );
    if ( id.getPriority() != 3 || id.getLocalID() != 564 )
        pass = false;
    if ( count_bits( 0x0 ) != 0 || count_bits( 0x03 ) != 2 )
        pass = false;
    if ( count_bits( ~( (size_t) 0 ) ) != 8 * sizeof( size_t ) )
        pass = false;
    if ( sizeof( size_t ) == 8 ) {
        if ( is_odd8( 0x0 ) || !is_odd8( 0x02 ) || is_odd8( 0x03 ) )
            pass = false;
        if ( is_odd8( ~( (size_t) 0 ) ) || !is_odd8( ThreadPoolID::maxThreadID ) )
            pass = false;
        for ( size_t i = 0; i < 1024; i++ ) {
            if ( ( count_bits( ThreadPoolID::maxThreadID - i ) % 2 == 1 ) !=
                 is_odd8( ThreadPoolID::maxThreadID - i ) ) {
                printp( "%i %i %s\n",
                        count_bits( ThreadPoolID::maxThreadID - i ),
                        is_odd8( ThreadPoolID::maxThreadID - i ) ? 1 : 0,
                        std::bitset<64>( ThreadPoolID::maxThreadID - i ).to_string().c_str() );
                pass = false;
            }
        }
    }
    d_id_assign = ThreadPoolID::maxThreadID;
    --d_id_assign; // Advance the id
    --d_id_assign; // Advance the id
    ThreadPoolID id2;
    id2.reset( 3, d_id_assign, nullptr );
    if ( isValid( id ) || !isValid( id2 ) )
        pass = false;
    if ( !pass )
        throw std::logic_error( "thread id test failed" );
}


/******************************************************************
 * Constructors/destructor                                         *
 ******************************************************************/
ThreadPool::ThreadPool( const int N,
                        const std::string &affinity,
                        const std::vector<int> &procs,
                        int queueSize )
    : d_queue( queueSize )
{
    // Run some basic tests on startup
    check_startup();
    // Initialize the header/tail
    d_NULL_HEAD = 0xB968135D;
    d_NULL_TAIL = d_NULL_HEAD;
    // Initialize the variables to NULL values
    d_id_assign     = 0;
    d_signal_empty  = false;
    d_signal_count  = 0;
    d_N_threads     = 0;
    d_num_active    = 0;
    d_N_added       = 0;
    d_N_started     = 0;
    d_N_finished    = 0;
    d_max_wait_time = 600;
    memset( (void *) d_active, 0, MAX_THREADS / 8 );
    memset( (void *) d_cancel, 0, MAX_THREADS / 8 );
    // Initialize the id
    d_id_assign = ThreadPoolID::maxThreadID;
    // Create the threads
    d_thread = new std::thread[MAX_THREADS];
    setNumThreads( N, affinity, procs );
    // Verify that the threadpool is valid
    if ( !is_valid( this ) )
        throw std::logic_error( "Thread pool is not valid" );
}
ThreadPool::~ThreadPool()
{
    if ( !is_valid( this ) ) {
        std::cerr << "Thread pool is not valid, error calling destructor\n";
        return;
    }
    // Destroy the threads
    try {
        setNumThreads( 0 );
        delete[] d_thread;
    } catch ( std::exception &e ) {
        AMP::perr << "Error destroying thread pool, may be in a bad state:" << e.what()
                  << std::endl;
    } catch ( ... ) {
        AMP::perr << "Error destroying thread pool, may be in a bad state (unknown exception)\n";
    }
    // Delete all remaining data
    d_N_threads = ~0;
    d_NULL_HEAD = 0;
    d_NULL_TAIL = 0;
}


/******************************************************************
 * Check if the pointer points to a valid thread pool object       *
 ******************************************************************/
bool ThreadPool::is_valid( const ThreadPool *tpool )
{
    if ( tpool == nullptr )
        return false;
    if ( tpool->d_N_threads > MAX_THREADS )
        return false;
    if ( tpool->d_NULL_HEAD != 0xB968135D || tpool->d_NULL_TAIL != 0xB968135D )
        return false;
    return true;
}


/******************************************************************
 * This function creates the threads in the thread pool            *
 ******************************************************************/
void ThreadPool::setNumThreads( int num_worker_threads,
                                const std::string &affinity,
                                const std::vector<int> &procs )
{
    // Check if we are a member thread
    if ( isMemberThread() )
        throw std::logic_error(
            "Member threads are not allowed to change the number of threads in the pool" );
    // Determing the number of threads we need to create or destroy
    if ( num_worker_threads > MAX_THREADS ) {
        printp( "Warning: Maximum Number of Threads is %i\n", MAX_THREADS );
        printp( "         Only that number will be created\n" );
        num_worker_threads = MAX_THREADS;
    } else if ( num_worker_threads < 0 ) {
        printp( "Error: cannot have a negitive number of threads\n" );
        printp( "       Setting the number of threads to 0\n" );
        num_worker_threads = 0;
    }
    int d_N_threads_diff = num_worker_threads - d_N_threads;
    if ( d_N_threads_diff > 0 ) {
        // Check that no threads are in the process of being deleted
        for ( long i : d_cancel ) {
            if ( i != 0 )
                throw std::logic_error(
                    "Threads are being created and destroyed at the same time" );
        }
        // Create the threads
        for ( int i = 0; i < d_N_threads_diff; i++ ) {
            int j       = d_N_threads++;
            d_thread[j] = std::thread( create_new_thread, this, j );
        }
        // Wait for all of the threads to finish initialization
        while ( true ) {
            std::this_thread::sleep_for( std::chrono::milliseconds( 25 ) );
            bool wait = false;
            for ( long i : d_cancel ) {
                if ( i != 0 )
                    wait = true;
            }
            if ( !wait )
                break;
        }
        std::this_thread::sleep_for( std::chrono::milliseconds( 25 ) );
    } else if ( d_N_threads_diff < 0 ) {
        // Reduce the number of threads
        if ( num_worker_threads == 0 ) {
            // Special case if we want to delete all of the threads
            wait_pool_finished();
        }
        // Tell the threads to shutdown
        for ( int i = 0; i > d_N_threads_diff; i-- )
            set_bit( d_cancel, d_N_threads - 1 + i );
        // Wake all threads to process the shutdown
        d_wait_work.notify_all();
        std::this_thread::sleep_for( std::chrono::milliseconds( 25 ) );
        // Wait for the threads to close
        for ( int i = 0; i > d_N_threads_diff; i-- ) {
            d_thread[d_N_threads - 1 + i].join();
            d_thread[d_N_threads - 1 + i] = std::thread();
            unset_bit( d_cancel, d_N_threads - 1 + i );
        }
        d_N_threads += d_N_threads_diff;
    }
    if ( d_N_threads == 0 )
        return;
    // Get the default thread affinity to use
    std::vector<int> cpus;
    int tmp            = global_OS_behavior;
    global_OS_behavior = 1;
    OS_warning( "Dummy message (should not print)" );
    try {
        cpus = ThreadPool::getProcessAffinity();
    } catch ( ... ) {
        pout << "Warning: Unable to get default cpus for thread affinities\n";
    }
    if ( !cpus.empty() && !procs.empty() ) {
        cpus.resize( procs.size() );
        for ( size_t i = 0; i < procs.size(); i++ )
            cpus[i] = procs[i];
    }
    // Set the affinity model and the associated thread affinities
    // Note: not all OS's support setting the thread affinities
    std::vector<std::vector<int>> t_procs( d_N_threads );
    if ( cpus.empty() ) {
        // We do not have a list of cpus to use, do nothing (OS not supported)
    } else if ( affinity == "none" ) {
        // We are using the default thread affinities (all threads get all procs of the program)
        for ( int i = 0; i < d_N_threads; i++ )
            t_procs[i] = cpus;
    } else if ( affinity == "independent" ) {
        // We want to use an independent set of processors for each thread
        if ( cpus.size() == d_N_threads ) {
            // The number of cpus matches the number of threads
            for ( int i = 0; i < d_N_threads; i++ )
                t_procs[i] = std::vector<int>( 1, cpus[i] );
        } else if ( cpus.size() > d_N_threads ) {
            // There are more cpus than threads, threads will use more the one processor
            int N_procs_thread = ( cpus.size() + d_N_threads - 1 ) / d_N_threads;
            size_t k           = 0;
            for ( int i = 0; i < d_N_threads; i++ ) {
                for ( int j = 0; j < N_procs_thread && k < cpus.size(); j++ ) {
                    t_procs[i].push_back( cpus[k] );
                    k++;
                }
            }
        } else {
            // There are fewer cpus than threads, threads will share a processor
            auto N_threads_proc = ( cpus.size() + d_N_threads - 1 ) / cpus.size();
            for ( int i = 0; i < d_N_threads; i++ )
                t_procs[i].push_back( cpus[i / N_threads_proc] );
        }
    } else {
        global_OS_behavior = tmp;
        throw std::logic_error( "Unknown affinity model" );
    }
    try {
        for ( int i = 0; i < d_N_threads; i++ ) {
            ThreadPool::setThreadAffinity( i, t_procs[i] );
            auto cpus2 = getThreadAffinity( i );
            if ( cpus2 != t_procs[i] )
                pout << "Warning: error setting affinities (failed to set)\n";
        }
    } catch ( ... ) {
        pout << "Warning: error setting affinities (exception)\n";
    }
    global_OS_behavior = tmp;
}


/******************************************************************
 * This is the function that controls the individual thread and    *
 * allows it to do work.                                           *
 ******************************************************************/
void ThreadPool::tpool_thread( int thread_id )
{
    StackTrace::registerThread();
    if ( get_bit( d_active, thread_id ) )
        throw std::logic_error( "Thread cannot already be active" );
    ++d_num_active;
    set_bit( d_active, thread_id );
    unset_bit( d_cancel, thread_id );
    AMP::Utilities::setenv( "OMP_NUM_THREADS", "1" );
    AMP::Utilities::setenv( "MKL_NUM_THREADS", "1" );
    // Check for shutdown
    bool shutdown = false;
    using Utilities::stringf;
    while ( !shutdown ) {
        // Check if there is work to do
        if ( !d_queue.empty() ) {
            // Get next work item to process
            auto work_id = d_queue.pop();
            if ( work_id.isNull() ) {
                std::this_thread::yield();
                continue;
            }
            WorkItem *work = work_id.getWork();
            ++d_N_started;
            // Start work here
            work->d_state = ThreadPoolID::Status::started;
            try {
                work->run();
            } catch ( StackTrace::abort_error &e ) {
                auto msg = stringf( "Error, caught exception in thread %i:\n", thread_id );
                if ( d_errorHandler ) {
                    e.stackType = StackTrace::printStackType::local;
                    msg += std::string( e.what() ) + "/n";
                    d_errorHandler( msg );
                } else {
                    e.message = msg + e.message;
                    StackTrace::Utilities::terminate( e );
                }
            } catch ( std::exception &e ) {
                auto msg = stringf( "Error, caught exception in thread %i:\n", thread_id );
                msg += std::string( e.what() ) + "/n";
                if ( d_errorHandler )
                    d_errorHandler( msg );
                else
                    throw std::logic_error( msg );
            } catch ( ... ) {
                auto msg = stringf( "Error, caught unknown exception in thread %i:\n", thread_id );
                if ( d_errorHandler )
                    d_errorHandler( msg );
                else
                    throw std::logic_error( msg );
            }
            work->d_state = ThreadPoolID::Status::finished;
            ++d_N_finished;
            // Check the signal count and signal if desired
            if ( d_signal_count > 0 ) {
                int count = --d_signal_count;
                if ( count <= 0 ) {
                    d_wait_finished.notify_all();
                    d_signal_count.store( 0 );
                }
            }
            shutdown = get_bit( d_cancel, thread_id );
        } else {
            int N_active = --d_num_active;
            unset_bit( d_active, thread_id );
            // Alert main thread that a thread finished processing
            if ( ( N_active == 0 ) && d_signal_empty ) {
                d_wait_finished.notify_all();
                d_signal_empty = false;
            }
            // Wait for work
            double wait_time = thread_id <= 2 ? 0.01 : 0.1;
            while ( d_queue.empty() && !shutdown ) {
                d_wait_work.wait_for( wait_time );
                shutdown = get_bit( d_cancel, thread_id );
            }
            ++d_num_active;
            set_bit( d_active, thread_id );
        }
    }
    --d_num_active;
    unset_bit( d_active, thread_id );
    return;
}


/******************************************************************
 * This is the function that adds work to the thread pool          *
 * Note: this version uses a last in - first out work scheduling.  *
 ******************************************************************/
void ThreadPool::add_work( size_t N,
                           ThreadPool::WorkItem *work[],
                           const int *priority,
                           ThreadPoolID *ids )
{
    // If all items do not fit in the queue, add in blocks
    if ( N > d_queue.capacity() / 2 ) {
        size_t N2 = d_queue.capacity() / 2;
        for ( size_t i = 0; i < N; i += N2 )
            add_work( std::min( N2, N - i ), &work[i], &priority[i], &ids[i] );
    }
    // If we have a large list and the threads are not active, break the list
    if ( N > 256 && d_queue.size() < 128 ) {
        add_work( 256, work, priority, ids );
        add_work( N - 256, &work[256], &priority[256], &ids[256] );
        return;
    }
    // Create the thread ids (can be done without blocking)
    for ( size_t i = 0; i < N; i++ )
        ids[i].reset( priority[i], --d_id_assign, work[i] );
    // If there are no threads, perform the work immediately
    if ( d_N_threads < 1 ) {
        for ( size_t i = 0; i < N; i++ ) {
            work[i]->d_state = ThreadPoolID::Status::started;
            work[i]->run();
            work[i]->d_state = ThreadPoolID::Status::finished;
        }
        return;
    }
    // Check and change priorities of dependency ids
    std::vector<std::pair<uint64_t, int8_t>> tmp;
    for ( size_t i = 0; i < N; i++ ) {
        for ( int j = 0; j < work[i]->d_N_ids; j++ ) {
            const auto &id1 = work[i]->d_ids[j];
            if ( id1.status() == ThreadPoolID::Status::added && id1.getPriority() < priority[i] ) {
                auto id2   = id1.getLocalID();
                bool found = false;
                for ( auto &t : tmp ) {
                    if ( t.first == id2 ) {
                        found    = true;
                        t.second = std::max<int8_t>( t.second, priority[i] );
                    }
                }
                if ( !found )
                    tmp.emplace_back( id2, priority[i] );
            }
        }
    }
    if ( !tmp.empty() )
        d_queue.changePriorities( tmp );
    // Wait for enough room in the queue (doesn't need blocking since it isn't that precise)
    int N_wait = (int) N - (int) ( d_queue.capacity() - d_queue.size() );
    while ( N_wait > 0 ) {
        wait_N( std::min( N_wait, 255 ), 1e-4 );
        N_wait = (int) N - (int) ( d_queue.capacity() - d_queue.size() );
    }
    // Add the work items to the queue
    for ( size_t i = 0; i < N; i++ )
        work[i]->d_state = ThreadPoolID::Status::added;
    d_queue.insert( N, ids );
    d_N_added.fetch_add( N );
    // Activate sleeping threads
    if ( d_num_active == d_N_threads || d_queue.empty() ) {
        // All threads are active, no need to wake anybody
    } else if ( N == 1 ) {
        // Added 1 item to the queue, wake 1 worker
        d_wait_work.notify_one();
    } else {
        // Added multple items in the queue, wake all workers
        d_wait_work.notify_all();
    }
}


/******************************************************************
 * This function waits for a some of the work items to finish      *
 ******************************************************************/
void ThreadPool::wait_N( int N, double time ) const
{
    // Store the number of items to wait for
    uint32_t zero = 0;
    bool test     = d_signal_count.compare_exchange_strong( zero, N );
    if ( !test ) {
        uint32_t N0, N2;
        do {
            N0 = d_signal_count.load();
            N2 = std::min<uint32_t>( N, N0 );
        } while ( !d_signal_count.compare_exchange_weak( N0, N2 ) );
    }
    // Wait
    d_wait_finished.wait_for( time );
    d_signal_count.store( 0 );
}
static inline int elapsed( const std::chrono::time_point<std::chrono::high_resolution_clock> &t0 )
{
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>( t1 - t0 ).count();
}
std::vector<bool>
ThreadPool::wait_some( size_t N_work, const ThreadPoolID *ids, size_t N_wait, int max_wait ) const
{
    // Check the inputs
    if ( N_wait > N_work )
        throw std::logic_error( "Invalid arguments in thread pool wait" );
    for ( size_t i = 0; i < N_work; i++ ) {
        if ( ids[i].status() == AMP::ThreadPoolID::Status::none )
            throw std::logic_error( "Waiting on invalid id" );
    }
    // Wait for the ids
    size_t N_finished = 0;
    std::vector<bool> finished( N_work, false );
    auto t1 = std::chrono::high_resolution_clock::now();
    auto t2 = t1;
    do {
        for ( size_t i = 0; i < N_work; i++ ) {
            if ( !finished[i] ) {
                if ( ids[i].finished() ) {
                    finished[i] = true;
                    N_finished++;
                }
            }
        }
        if ( N_finished >= N_work )
            return finished;
        wait_N( N_work - N_finished, 0.01 );
        if ( elapsed( t2 ) >= d_max_wait_time ) {
            print_wait_warning();
            t2 = std::chrono::high_resolution_clock::now();
        }
    } while ( elapsed( t1 ) < max_wait );
    // Update any ids that have finished
    for ( size_t i = 0; i < N_work; i++ ) {
        if ( ids[i].finished() )
            finished[i] = true;
    }
    return finished;
}


/******************************************************************
 * This function waits for all of the threads to finish their work *
 ******************************************************************/
void ThreadPool::print_wait_warning() const
{
    pout << "Warning: Maximum wait time in ThreadPool exceeded, threads may be hung\n";
    pout << "N_active: " << d_num_active << std::endl;
    pout << "N_queued: " << d_queue.size() << std::endl;
    pout << "N_added: " << d_N_added << std::endl;
    pout << "N_started: " << d_N_started << std::endl;
    pout << "N_finished: " << d_N_finished << std::endl;
    pout << "Stack Trace:\n";
    auto call_stack = StackTrace::getAllCallStacks();
    StackTrace::cleanupStackTrace( call_stack );
    auto text = call_stack.print( "  " );
    for ( auto &line : text )
        pout << line << std::endl;
}
void ThreadPool::wait_pool_finished() const
{
    // First check that we are not one of the threads
    if ( isMemberThread() )
        throw std::logic_error( "Member thread attempted to call wait_pool_finished" );
    // Wait for all threads to finish their work
    auto t1 = std::chrono::high_resolution_clock::now();
    while ( d_num_active > 0 || !d_queue.empty() ) {
        // Wait for signal from last thread
        d_signal_empty = true;
        d_wait_finished.wait_for( 5e-4 );
        if ( d_num_active == 0 && d_queue.empty() )
            break;
        // Check that we have not exceeded the maximum time
        auto t2     = std::chrono::high_resolution_clock::now();
        int seconds = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
        if ( seconds > d_max_wait_time ) {
            print_wait_warning();
            t1 = t2;
        }
    }
    d_signal_empty = false;
}


/************************************************************************
 * Function to add dependencies to the work item                         *
 * Note: when expanding the size of d_ids, we need to allocate space for *
 * one extra entry for a spinlock.                                       *
 ************************************************************************/
void ThreadPool::WorkItem::add_dependencies( size_t N, const ThreadPoolID *ids )
{
    if ( d_state != ThreadPoolID::Status::none ) {
        // The item has already been added to the threadpool,
        // we are not allowed to add dependencies
        throw std::logic_error(
            "Cannot add dependency to work item once it has been added the the threadpool" );
    }
    if ( d_N_ids + N > 0xFFFF )
        throw std::logic_error( "Cannot add more than 65000 dependencies" );
    if ( d_N_ids + N + 1 > d_size ) {
        ThreadPoolID *tmp = d_ids;
        unsigned int N2   = d_size;
        if ( N2 == 0 ) {
            N2 = 8;
        }
        while ( N2 < d_N_ids + N + 1 )
            N2 *= 2;
        d_ids = new ThreadPoolID[N2];
        for ( size_t i = 0; i < d_N_ids; i++ )
            const_cast<ThreadPoolID &>( ids[i] ).swap( tmp[i] );
        delete[] tmp;
        d_size     = N2;
        auto *lock = reinterpret_cast<int *>( &d_ids[d_size - 1] );
        *lock      = 0;
    }
    const ThreadPoolID id0;
    for ( size_t i = 0; i < N; i++ ) {
        if ( ids[i] != id0 ) {
            if ( !ids[i].finished() ) {
                d_ids[d_N_ids] = ids[i];
                d_N_ids++;
            }
        }
    }
}


} // Namespace AMP
