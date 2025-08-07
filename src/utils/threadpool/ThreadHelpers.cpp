#include "AMP/utils/threadpool/ThreadHelpers.h"
#include "AMP/AMP_TPLs.h"

#include <climits>
#include <stdexcept>
#include <thread>


// OS specific includes / definitions
// clang-format off
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
    #define USE_WINDOWS
#elif defined( __APPLE__ )
    #define USE_MAC
#elif defined( __linux ) || defined( __linux__ ) || defined( __unix ) || defined( __posix )
    #define USE_LINUX
#else
    #error Unknown OS
#endif
#if defined( USE_WINDOWS )
    #include <process.h>
    #include <windows.h>
    // Disable warning: the inline specifier cannot be used when a friend
    // declaration refers to a specialization of a function template
    #pragma warning( disable : 4396 )
#endif
#if defined(USE_LINUX) || defined(USE_MAC)
    #include <pthread.h>
    #include <unistd.h>
#endif
#ifdef USE_MAC
    // https://developer.apple.com/library/mac/#releasenotes/Performance/RN-AffinityAPI
    // http://plugins.svn.wordpress.org/wp-xhprof-profiler/trunk/facebook-xhprof/extension/xhprof..c
    #include <mach/mach_init.h>
    #include <mach/thread_policy.h>
    #define cpu_set_t thread_affinity_policy_data_t
    #define CPU_SET( cpu_id, new_mask ) *new_mask.affinity_tag = ( cpu_id + 1 )
    #define CPU_ZERO( new_mask ) ( *( new_mask ) ).affinity_tag = THREAD_AFFINITY_TAG_NULL
    #define sched_setaffinity( pid, size, mask ) \
        thread_policy_set(                       \
            mach_thread_self(), THREAD_AFFINITY_POLICY, mask, THREAD_AFFINITY_POLICY_COUNT )
    #define sched_getaffinity( pid, size, mask ) \
        thread_policy_get(                       \
            mach_thread_self(), THREAD_AFFINITY_POLICY, mask, THREAD_AFFINITY_POLICY_COUNT )
#endif
// clang-format on


namespace AMP::Thread {


/******************************************************************
 * Return the number of processors available                       *
 ******************************************************************/
int getNumberOfProcessors() { return std::thread::hardware_concurrency(); }


/******************************************************************
 * Return the processor number of the current thread               *
 ******************************************************************/
int getCurrentProcessor()
{
#if defined( USE_LINUX )
    return sched_getcpu() + 1;
#elif defined( USE_MAC )
    OS_warning( "Warning: MAC does not support getCurrentProcessor" );
    return 0;
#elif defined( USE_WINDOWS )
    return GetCurrentProcessorNumber() + 1;
#else
    #error Unknown OS
#endif
}

/******************************************************************
 * Return the native handle to the current thread                  *
 ******************************************************************/
std::thread::native_handle_type getCurrentThread()
{
#if defined( USE_LINUX ) || defined( USE_MAC )
    return pthread_self();
#elif defined( USE_WINDOWS )
    return GetCurrentThread();
#else
    #error Unknown OS
#endif
}


/******************************************************************
 * Get/Set the affinity of the current process                     *
 ******************************************************************/
std::vector<int> getProcessAffinity()
{
    std::vector<int> procs;
#ifdef USE_LINUX
    #ifdef _GNU_SOURCE
    cpu_set_t mask;
    int error = sched_getaffinity( getpid(), sizeof( cpu_set_t ), &mask );
    if ( error != 0 )
        throw std::logic_error( "Error getting process affinity" );
    for ( size_t i = 0; i < sizeof( cpu_set_t ) * CHAR_BIT; i++ ) {
        if ( CPU_ISSET( i, &mask ) )
            procs.push_back( i );
    }
    #else
    OS_warning( "Warning: sched_getaffinity is not supported for this compiler/OS" );
    #endif
#elif defined( USE_MAC )
    // MAC does not support getting or setting the affinity
    OS_warning( "MAC does not support getting the process affinity" );
#elif defined( USE_WINDOWS )
    HANDLE hProc = GetCurrentProcess();
    size_t procMask;
    size_t sysMask;
    PDWORD_PTR procMaskPtr = reinterpret_cast<PDWORD_PTR>( &procMask );
    PDWORD_PTR sysMaskPtr  = reinterpret_cast<PDWORD_PTR>( &sysMask );
    GetProcessAffinityMask( hProc, procMaskPtr, sysMaskPtr );
    for ( size_t i = 0; i < sizeof( size_t ) * CHAR_BIT; i++ ) {
        if ( ( procMask & 0x1 ) != 0 )
            procs.push_back( i );
        procMask >>= 1;
    }
#else
    #error Unknown OS
#endif
    return procs;
}
void setProcessAffinity( const std::vector<int> &procs )
{
#ifdef USE_LINUX
    #ifdef _GNU_SOURCE
    cpu_set_t mask;
    CPU_ZERO( &mask );
    for ( size_t i = 0; i < procs.size(); i++ )
        CPU_SET( procs[i], &mask );
    int error = sched_setaffinity( getpid(), sizeof( cpu_set_t ), &mask );
    if ( error != 0 )
        throw std::logic_error( "Error setting process affinity" );
    #else
    OS_warning( "Warning: sched_setaffinity is not supported for this compiler/OS" );
    #endif
#elif defined( USE_MAC )
    // MAC does not support getting or setting the affinity
    OS_warning( "Warning: MAC does not support setting the process affinity" );
#elif defined( USE_WINDOWS )
    DWORD mask = 0;
    for ( size_t i = 0; i < procs.size(); i++ )
        mask |= ( (DWORD) 1 ) << procs[i];
    HANDLE hProc = GetCurrentProcess();
    SetProcessAffinityMask( hProc, mask );
#else
    #error Unknown OS
#endif
}


/******************************************************************
 * Function to get the thread affinities                           *
 ******************************************************************/
#ifdef USE_WINDOWS
static DWORD GetThreadAffinityMask( HANDLE thread )
{
    DWORD mask = 1;
    DWORD old  = 0;
    // try every CPU one by one until one works or none are left
    while ( mask ) {
        old = SetThreadAffinityMask( thread, mask );
        if ( old ) {                              // this one worked
            SetThreadAffinityMask( thread, old ); // restore original
            return old;
        } else {
            if ( GetLastError() != ERROR_INVALID_PARAMETER )
                return 0; // fatal error, might as well throw an exception
        }
        mask <<= 1;
    }
    return 0;
}
#endif
std::vector<int> getThreadAffinity( [[maybe_unused]] std::thread::native_handle_type handle )
{
    std::vector<int> procs;
#ifdef USE_LINUX
    #ifdef _GNU_SOURCE
    cpu_set_t mask;
    int error = pthread_getaffinity_np( handle, sizeof( cpu_set_t ), &mask );
    if ( error != 0 )
        throw std::logic_error( "Error getting thread affinity" );
    for ( size_t i = 0; i < sizeof( cpu_set_t ) * CHAR_BIT; i++ ) {
        if ( CPU_ISSET( i, &mask ) )
            procs.push_back( i );
    }
    #else
    OS_warning( "pthread does not support pthread_getaffinity_np" );
    #endif
#elif defined( USE_MAC )
    // MAC does not support getting or setting the affinity
    OS_warning( "MAC does not support getting the thread affinity" );
#elif defined( USE_WINDOWS )
    size_t procMask = GetThreadAffinityMask( handle );
    for ( size_t i = 0; i < sizeof( size_t ) * CHAR_BIT; i++ ) {
        if ( ( procMask & 0x1 ) != 0 )
            procs.push_back( i );
        procMask >>= 1;
    }
#else
    #error Unknown OS
#endif
    return procs;
}


/******************************************************************
 * Function to set the thread affinity                             *
 ******************************************************************/
void setThreadAffinity( [[maybe_unused]] std::thread::native_handle_type handle,
                        [[maybe_unused]] const std::vector<int> &procs )
{
#ifdef USE_LINUX
    #ifdef __USE_GNU
    cpu_set_t mask;
    CPU_ZERO( &mask );
    for ( size_t i = 0; i < procs.size(); i++ )
        CPU_SET( procs[i], &mask );
    int error = pthread_setaffinity_np( handle, sizeof( cpu_set_t ), &mask );
    if ( error != 0 )
        throw std::logic_error( "Error setting thread affinity" );
    #else
    OS_warning( "pthread does not support pthread_setaffinity_np" );
    #endif
#elif defined( USE_MAC )
    // MAC does not support getting or setting the affinity
    OS_warning( "MAC does not support getting the process affinity" );
#elif defined( USE_WINDOWS )
    DWORD mask = 0;
    for ( size_t i = 0; i < procs.size(); i++ )
        mask |= ( (DWORD) 1 ) << procs[i];
    SetThreadAffinityMask( handle, mask );
#else
    #error Unknown OS
#endif
}


} // namespace AMP::Thread
