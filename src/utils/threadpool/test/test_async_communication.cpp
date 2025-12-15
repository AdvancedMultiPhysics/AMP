#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/threadpool/ThreadPool.h"


using namespace AMP;


void run_thread( size_t message_size,
                 int send_proc,
                 int recv_proc,
                 char *data_src,
                 char *data_dst,
                 int tag,
                 int ms_sleep_duration )
{
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    AMP::AMP_MPI::Request requests[2];
    requests[0] = globalComm.Isend( data_src, message_size, send_proc, tag );
    requests[1] = globalComm.Irecv( data_dst, message_size, recv_proc, tag );
    globalComm.waitAll( 2, requests );
    AMP::Utilities::sleep_ms( ms_sleep_duration ); // Mimic work
}


// Function to erase elements from an array
template<class T>
void erase( std::vector<T> &x, std::vector<int> index )
{
    std::sort( index.begin(), index.end() );
    for ( int i = index.size() - 1; i >= 0; i-- )
        x.erase( x.begin() + index[i] );
}


//  This will test the behavior of asynchronous communication under different behaviors
void run_test( size_t message_size, int N_messages, double sleep_duration, ThreadPool *tpool )
{
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    int rank      = globalComm.getRank();
    int size      = globalComm.getSize();
    int send_proc = ( size + rank - 1 ) % size;
    int recv_proc = ( size + rank + 1 ) % size;
    std::vector<AMP::AMP_MPI::Request> requests( 2 * N_messages );
    if ( rank == 0 ) {
        printf( "Running test with %i messages of size of %zu bytes, waiting %0.2e s\n",
                N_messages,
                message_size,
                sleep_duration );
    }
    auto ms_sleep_duration = static_cast<int>( 1000 * sleep_duration );
    // Create space for the src and dst messages
    auto data_src = new char[message_size];
    auto data_dst = new char *[N_messages];
    for ( int i = 0; i < N_messages; i++ )
        data_dst[i] = new char[message_size];
    memset( data_src, 0, message_size );
    // Initialize the communication pattern
    if ( N_messages > 0 )
        run_thread( message_size, send_proc, recv_proc, data_src, data_dst[0], 0, 0 );
    // Send the messages in sequence using isend / irecv
    globalComm.barrier();
    double start = globalComm.time();
    for ( int i = 0; i < N_messages; i++ ) {
        requests.clear();
        requests.push_back( globalComm.Isend( data_src, message_size, send_proc, i ) );
        requests.push_back( globalComm.Irecv( data_dst[i], message_size, recv_proc, i ) );
        globalComm.waitAll( requests.size(), requests.data() );
        AMP::Utilities::sleep_ms( ms_sleep_duration ); // Mimic work
    }
    double end = globalComm.time();
    if ( rank == 0 )
        printf( "   Sequential:   %e\n", end - start );
    // Send the messages in parallel using isend / irecv (1 thread)
    globalComm.barrier();
    start = globalComm.time();
    requests.clear();
    for ( int i = 0; i < N_messages; i++ ) {
        requests.push_back( globalComm.Isend( data_src, message_size, send_proc, i ) );
        requests.push_back( globalComm.Irecv( data_dst[i], message_size, recv_proc, i ) );
    }
    std::set<int> completed;
    while ( !requests.empty() ) {
        auto index = globalComm.waitSome( requests.size(), requests.data() );
        for ( int i : index ) {
            if ( i % 2 == 1 )
                AMP::Utilities::sleep_ms( ms_sleep_duration ); // Mimic work
        }
        erase( requests, index );
    }
    end = globalComm.time();
    if ( rank == 0 )
        printf( "   Asynchronous: %e\n", end - start );
    // Send the messages in parallel using isend / irecv (many threads)
    globalComm.barrier();
    start = globalComm.time();
    for ( int i = 0; i < N_messages; i++ ) {
        TPOOL_ADD_WORK(
            tpool,
            run_thread,
            ( message_size, send_proc, recv_proc, data_src, data_dst[i], i, ms_sleep_duration ) );
    }
    tpool->wait_pool_finished();
    end = globalComm.time();
    if ( rank == 0 )
        printf( "   Threaded:     %e\n", end - start );
    // Delete the temporary memory
    for ( int i = 0; i < N_messages; i++ )
        delete[] data_dst[i];
    delete[] data_dst;
    delete[] data_src;
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    int rank = globalComm.getRank();

    int N_messages = 1;
    if ( argc == 2 )
        N_messages = atoi( argv[1] );
    if ( rank == 0 )
        printf( "N_messages = %i\n", N_messages );

    // Create the thread pool
    std::vector<int> cpus( ThreadPool::getNumberOfProcessors(), 0 );
    for ( size_t i = 0; i < cpus.size(); i++ )
        cpus[i] = i;
    ThreadPool::setProcessAffinity( cpus );
    ThreadPool tpool( N_messages );

    // Choose problem sizes to run
    const size_t message_size[7] = { 0x01, 0x80, 0x400, 0x2000, 0x100000, 0x800000, 0x2000000 };

    // Run the tests without work
    if ( rank == 0 )
        printf( "\nRunning tests without work\n" );
    for ( unsigned long i : message_size )
        run_test( i, N_messages, 0, &tpool );

    // Run the tests with work
    if ( rank == 0 )
        printf( "\nRunning tests with work\n" );
    for ( unsigned long i : message_size )
        run_test( i, N_messages, 0.1, &tpool );

    // Finished testing
    globalComm.reset();
    AMP::AMPManager::shutdown();
    return 0;
}
