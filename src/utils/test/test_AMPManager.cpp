#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"


//  This test will start and shutdown AMP
int main( int argc, char *argv[] )
{
    // Get the maximum number of processors for AMP
    int procMax = -1;
    if ( argc > 1 ) {
        procMax = atoi( argv[1] );
        if ( procMax <= 0 )
            return -1;
    }

    // Create the comm used to initialize AMP
    auto AMP_comm = AMP_COMM_WORLD;
#ifdef AMP_USE_MPI
    if ( procMax > 0 ) {
        MPI_Init( &argc, &argv );
        // Create a new comm to initialize AMPManager
        int rank = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        int color = rank / procMax;
        MPI_Comm_split( MPI_COMM_WORLD, color, rank, &AMP_comm );
    }
#endif

    // Control the behavior of the startup
    AMP::AMPManagerProperties startup_properties;
    startup_properties.print_times   = true;
    startup_properties.use_MPI_Abort = false;
    startup_properties.COMM_WORLD    = AMP_comm;
    startup_properties.print_startup = true;

    // Start AMP
    AMP::AMPManager::startup( argc, argv, startup_properties );

    // Limit the scope of variables
    {
        // Create the global comm
        AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
        int globalSize = globalComm.getSize();
        if ( procMax > 0 && globalSize > procMax )
            AMP_ERROR( "AMP did not initialize on a sub-communicator" );
        // Introduce a memory leak to catch in valgrind later
        auto *x = new double[100];
        if ( x == nullptr )
            AMP_ERROR( "error" );
        // Test the abort
        auto start = std::chrono::steady_clock::now();
        try {
            AMP_ERROR( "Catch this 1" );
            std::cout << "Failed to catch AMP_ERROR\n";
            return -1;
        } catch ( ... ) {
            // This is correct
        }
        auto stop  = std::chrono::steady_clock::now();
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( stop - start ).count();
        std::cout << "Time to throw/catch AMP_ERROR: " << ns / 1000000 << " ms\n\n";
    }

    // Restart
    AMP::AMPManager::restart();

    // Shutdown
    int rank = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    AMP::AMPManager::shutdown();
    AMP_comm = AMP_COMM_NULL;

    // Test a reinitialization of AMP
    try {
        AMP::AMPManager::startup( argc, argv, startup_properties );
        std::cout << "AMP re-initialized (this is invalid)\n";
        return -1;
    } catch ( ... ) {
        // This is correct
    }

#ifdef AMP_USE_MPI
    // Close MPI
    if ( procMax > 0 ) {
        MPI_Barrier( MPI_COMM_WORLD );
        MPI_Finalize();
    }
#endif

    // Finished successfully
    if ( rank == 0 )
        std::cout << "Finished\n";
    return 0;
}
