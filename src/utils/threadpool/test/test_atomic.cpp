#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef __USE_GNU
    #include <unistd.h>
#endif


using AMP::Utilities::stringf;


// Function to increment/decrement a counter N times
static void modify_counter( int N, std::atomic_int64_t &counter )
{
    if ( N > 0 ) {
        for ( int i = 0; i < N; i++ )
            ++counter;
    } else if ( N < 0 ) {
        for ( int i = 0; i < -N; i++ )
            --counter;
    }
}


/******************************************************************
 * The main program                                                *
 ******************************************************************/
int main( int, char *[] )
{
    AMP::UnitTest ut;

    int N_threads = 64;      // Number of threads
    int N_count   = 1000000; // Number of work items

// Ensure we are using all processors
#ifdef __USE_GNU
    int N_procs = sysconf( _SC_NPROCESSORS_ONLN );
    cpu_set_t mask;
    CPU_ZERO( &mask );
    for ( int i = 0; i < N_procs; i++ )
        CPU_SET( i, &mask );
    sched_setaffinity( getpid(), sizeof( cpu_set_t ), &mask );
#endif

    // Create the counter we want to test
    std::atomic_int64_t count = 0;
    if ( ++count == 1 )
        ut.passes( "increment count" );
    else
        ut.failure( "increment count" );
    if ( --count == 0 )
        ut.passes( "decrement count" );
    else
        ut.failure( "decrement count" );
    count = 3;
    if ( count == 3 )
        ut.passes( "set count" );
    else
        ut.failure( "set count" );
    count = 0;

    // Increment the counter in serial
    auto start = std::chrono::high_resolution_clock::now();
    modify_counter( N_count, count );
    auto stop              = std::chrono::high_resolution_clock::now();
    double time_inc_serial = std::chrono::duration<double>( stop - start ).count() / N_count;
    int val                = count;
    if ( val != N_count ) {
        auto tmp = stringf( "Count of %i did not match expected count of %i", val, N_count );
        ut.failure( tmp );
    }
    printf( "Time to increment (serial) = %0.1f ns\n", 1e9 * time_inc_serial );

    // Decrement the counter in serial
    start = std::chrono::high_resolution_clock::now();
    modify_counter( -N_count, count );
    stop                   = std::chrono::high_resolution_clock::now();
    double time_dec_serial = std::chrono::duration<double>( stop - start ).count() / N_count;
    val                    = count;
    if ( val != 0 ) {
        auto tmp = stringf( "Count of %i did not match expected count of %i", val, 0 );
        ut.failure( tmp );
    }
    printf( "Time to decrement (serial) = %0.1f ns\n", 1e9 * time_dec_serial );

    // Increment the counter in parallel
    std::vector<std::thread> threads( N_threads );
    start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < N_threads; i++ )
        threads[i] = std::thread( modify_counter, N_count, std::ref( count ) );
    for ( int i = 0; i < N_threads; i++ )
        threads[i].join();
    stop = std::chrono::high_resolution_clock::now();
    double time_inc_parallel =
        std::chrono::duration<double>( stop - start ).count() / ( N_count * N_threads );
    val = count;
    if ( val != N_count * N_threads ) {
        auto tmp =
            stringf( "Count of %i did not match expected count of %i", val, N_count * N_threads );
        ut.failure( tmp );
    }
    printf( "Time to increment (parallel) = %0.1f ns\n", 1e9 * time_inc_parallel );

    // Decrement the counter in parallel
    start = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < N_threads; i++ )
        threads[i] = std::thread( modify_counter, -N_count, std::ref( count ) );
    for ( int i = 0; i < N_threads; i++ )
        threads[i].join();
    stop = std::chrono::high_resolution_clock::now();
    double time_dec_parallel =
        std::chrono::duration<double>( stop - start ).count() / ( N_count * N_threads );
    val = count;
    if ( val != 0 ) {
        auto tmp = stringf( "Count of %i did not match expected count of %i", val, 0 );
        ut.failure( tmp );
    }
    printf( "Time to decrement (parallel) = %0.1f ns\n", 1e9 * time_dec_parallel );

    // Check the time to increment/decrement
    if ( time_inc_serial > 100e-9 || time_dec_serial > 100e-9 || time_inc_parallel > 100e-9 ||
         time_dec_parallel > 100e-9 ) {
#if USE_GCOV
        ut.expected_failure( "Time to increment/decrement count is too expensive" );
#else
        ut.failure( "Time to increment/decrement count is too expensive" );
#endif
    } else {
        ut.passes( "Time to increment/decrement passed" );
    }

    // Finished
    ut.report();
    auto N_errors = static_cast<int>( ut.NumFailGlobal() );
    ut.reset();
    return N_errors;
}
