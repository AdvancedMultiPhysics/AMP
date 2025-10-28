#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/threadpool/Mutex.h"
#include "AMP/utils/threadpool/ThreadPool.h"

#include "ProfilerApp.h"

#include <chrono>
#include <random>
#include <thread>
#include <vector>


std::atomic<int> global_count  = 0;
std::atomic<bool> global_start = false;
AMP::Mutex global_lock( true );


void test_lock( AMP::AMP_MPI comm, int N, bool call_sleep )
{
    while ( !global_start )
        std::this_thread::yield();
    std::random_device rd;
    std::mt19937 gen( rd() );
    for ( int i = 0; i < N; i++ ) {
        // Acquire the lock
        AMP::lock_MPI_Mutex( global_lock, comm );
        {
            PROFILE( "work", 2 );
            comm.barrier();
            // Check and increment count
            int tmp = global_count++;
            if ( tmp != 0 )
                AMP_ERROR( "Invalid count" );
            // Acquire the lock a second time, then release
            global_lock.lock();
            global_lock.unlock();
            // Sleep for a while
            std::this_thread::yield();
            if ( call_sleep )
                AMP::Utilities::sleep_ms( 20 );
            // Check and decrement count
            tmp = global_count--;
            if ( tmp != 1 )
                AMP_ERROR( "Invalid count" );
        }
        // Release the mutex
        global_lock.unlock();
        // Try to add some random waits
        std::uniform_int_distribution<int> dist( 0, 10 );
        for ( int j = 0; j < dist( gen ); j++ ) {
            std::this_thread::yield();
            std::uniform_int_distribution<int> dist2( 0, 500000 );
            std::chrono::nanoseconds ns( dist2( gen ) );
            std::this_thread::sleep_for( ns );
        }
    }
}


int main( int argc, char *argv[] )
{
    // Initialize AMP
    AMP::AMPManager::startup( argc, argv );
    PROFILE_ENABLE( 2 );
    PROFILE_ENABLE_TRACE();
    PROFILE( "main" );

    {
        // Create the thread pool
        int N_threads = 8;
        AMP::ThreadPool tpool( N_threads );
        AMP::AMP_MPI comm_world( AMP_COMM_WORLD );
        comm_world.barrier();

        // Check the duration of the sleep functions
        double t0 = AMP::AMP_MPI::time();
        AMP::Utilities::sleep_ms( 1 );
        double t1 = AMP::AMP_MPI::time();
        std::cout << "AMP::Utilities::sleep_ms(1) = " << t1 - t0 << std::endl;

        // Run a single lock test
        AMP::pout << "Running single lock test\n";
        std::vector<AMP::ThreadPoolID> ids;
        {
            PROFILE( "single" );
            global_start = false;
            for ( int i = 0; i < N_threads; i++ )
                ids.push_back( TPOOL_ADD_WORK( &tpool, test_lock, ( comm_world.dup(), 1, true ) ) );
            global_start = true;
            tpool.wait_all( ids );
            ids.clear();
            comm_world.barrier();
        }

        // Run multiple lock tests
        AMP::pout << "Running multiple lock test\n";
        int N_it     = 100;
        double start = 0;
        double stop  = 0;
        {
            PROFILE( "multiple" );
            global_start = false;
            start        = AMP::AMP_MPI::time();
            for ( int i = 0; i < N_threads; i++ )
                ids.push_back(
                    TPOOL_ADD_WORK( &tpool, test_lock, ( comm_world.dup(), N_it, false ) ) );
            global_start = true;
            tpool.wait_all( ids );
            ids.clear();
            comm_world.barrier();
            stop = AMP::AMP_MPI::time();
        }
        AMP::pout << "   Time to acquire global MPI lock was " << ( stop - start ) / N_it
                  << " seconds/iteration\n";
    }

    // Finalize
    AMP::pout << "Test ran sucessfully\n";
    PROFILE_SAVE( "test_lock_MPI_Mutex" );
    AMP::AMPManager::shutdown();
    return 0;
}
