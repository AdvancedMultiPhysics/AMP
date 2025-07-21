#include "AMP/utils/UnitTest.h"
#include "AMP/IO/PIO.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/Utilities.h"

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


namespace AMP {


/********************************************************************
 *  Constructor/Destructor                                           *
 ********************************************************************/
UnitTest::UnitTest() : d_verbose( false )
{
    if ( AMP_MPI::MPI_Active() )
        d_comm = std::make_unique<AMP_MPI>( AMP_COMM_WORLD );
    else
        d_comm = std::make_unique<AMP_MPI>( AMP_COMM_SELF );
}
UnitTest::~UnitTest() { reset(); }
void UnitTest::reset()
{
    d_mutex.lock();
    // Clear the data forcing a reallocation
    std::vector<std::string>().swap( d_pass );
    std::vector<std::string>().swap( d_fail );
    std::vector<std::string>().swap( d_expected );
    d_mutex.unlock();
}


/********************************************************************
 *  Add a pass, fail, expected failure message in a thread-safe way  *
 ********************************************************************/
void UnitTest::passes( std::string in )
{
    d_mutex.lock();
    if ( d_verbose )
        printf( "UnitTest: %i passes: %s\n", d_comm->getRank(), in.data() );
    d_pass.emplace_back( std::move( in ) );
    d_mutex.unlock();
}
void UnitTest::failure( std::string in )
{
    d_mutex.lock();
    if ( d_verbose )
        printf( "UnitTest: %i failed: %s\n", d_comm->getRank(), in.data() );
    d_fail.emplace_back( std::move( in ) );
    d_mutex.unlock();
}
void UnitTest::expected_failure( std::string in )
{
    d_mutex.lock();
    if ( d_verbose )
        printf( "UnitTest: %i expected_failure: %s\n", d_comm->getRank(), in.data() );
    d_expected.emplace_back( std::move( in ) );
    d_mutex.unlock();
}


/********************************************************************
 *  Print a global report                                            *
 *  Note: only rank 0 will print, all messages will be aggregated    *
 ********************************************************************/
static inline void
print_messages( std::string_view indent, int N, const std::string *messages, bool removeDuplicates )
{
    for ( int i = 0; i < N; i++ ) {
        if ( removeDuplicates ) {
            bool found = false;
            for ( int j = 0; j < i; j++ )
                found = found || messages[i] == messages[j];
            if ( found )
                continue;
        }
        pout << indent << messages[i] << std::endl;
    }
}
static inline void print_messages( const std::vector<int> &N,
                                   const std::vector<std::string> &messages,
                                   bool removeDuplicates )
{
    if ( N.size() > 1 ) {
        int N1 = 0;
        for ( size_t rank = 0; rank < N.size(); rank++ ) {
            if ( N[rank] > 0 ) {
                printp( "   Proccessor %i:\n", static_cast<int>( rank ) );
                print_messages( "      ", N[rank], &messages[N1], removeDuplicates );
            }
            N1 += N[rank];
        }
    } else {
        print_messages( "   ", N[0], messages.data(), removeDuplicates );
    }
}
static void report2( const char *msg,
                     const std::vector<std::string> &results,
                     const AMP::AMP_MPI &comm,
                     int N_report,
                     bool removeDuplicates )
{
    // Gather the data
    int size = comm.getSize();
    int rank = comm.getRank();
    auto N   = comm.allGather<int>( results.size() );
    int Nt   = 0;
    for ( int i = 0; i < size; i++ )
        Nt += N[i];
    std::vector<std::string> results2;
    if ( Nt <= N_report && Nt > 0 ) {
        results2 = comm.allGather( results );
        AMP_ASSERT( (int) results2.size() == Nt );
    }
    if ( rank != 0 )
        return;
    // Print the results
    pout << "Tests " << msg << std::endl;
    if ( Nt == 0 ) {
        // No tests to print
    } else if ( results2.empty() ) {
        // We want to print a summary
        if ( size > 8 ) {
            // Print summary for all processors
            printp( "     %i tests %s\n", Nt, msg );
        } else {
            // Print a summary for each processor
            for ( int i = 0; i < size; i++ )
                printp( "     %i tests %s (proc %i)\n", N[i], msg, i );
        }
    } else {
        // We want to print all messages
        print_messages( N, results2, removeDuplicates );
    }
    pout << std::endl;
}
void UnitTest::report( const int level0, bool removeDuplicates ) const
{
    d_mutex.lock();
    // Give all processors a chance to print any remaining messages
    d_comm->barrier();
    Utilities::sleep_ms( 10 );
    // Broadcast the print level from rank 0
    int level = d_comm->bcast( level0, 0 );
    if ( level < 0 || level > 3 )
        AMP_ERROR( "Invalid print level" );
    // Report
    if ( d_comm->getRank() == 0 )
        pout << std::endl;
    if ( level == 0 ) {
        report2( "passed", d_pass, *d_comm, 0, removeDuplicates );
        report2( "failed", d_fail, *d_comm, 0, removeDuplicates );
        report2( "expected failed", d_expected, *d_comm, 0, removeDuplicates );
    } else if ( level == 1 ) {
        report2( "passed", d_pass, *d_comm, 20, removeDuplicates );
        report2( "failed", d_fail, *d_comm, 1000000, removeDuplicates );
        report2( "expected failed", d_expected, *d_comm, 50, removeDuplicates );
    } else if ( level == 2 ) {
        report2( "passed", d_pass, *d_comm, 50, removeDuplicates );
        report2( "failed", d_fail, *d_comm, 1000000, removeDuplicates );
        report2( "expected failed", d_expected, *d_comm, 1000000, removeDuplicates );
    } else {
        report2( "passed", d_pass, *d_comm, 1000000, removeDuplicates );
        report2( "failed", d_fail, *d_comm, 1000000, removeDuplicates );
        report2( "expected failed", d_expected, *d_comm, 1000000, removeDuplicates );
    }
    // Add a barrier to synchronize all processors (rank 0 is much slower)
    d_comm->barrier();
    AMP::Utilities::sleep_ms( 10 ); // Need a brief pause to allow any printing to finish
    d_mutex.unlock();
}


/********************************************************************
 *  Other functions                                                  *
 ********************************************************************/
size_t UnitTest::NumPassGlobal() const { return d_comm->sumReduce( d_pass.size() ); }
size_t UnitTest::NumFailGlobal() const { return d_comm->sumReduce( d_fail.size() ); }
size_t UnitTest::NumExpectedFailGlobal() const { return d_comm->sumReduce( d_expected.size() ); }


} // namespace AMP
