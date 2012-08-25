#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>

#include "utils/AMPManager.h"
#include "utils/AMP_MPI.h"


//  This test will start and shutdown AMP
int main(int argc, char *argv[])
{
    // Get the maximum number of processors for AMP
    int procMax = -1;
    if ( argc > 1 ) {
        procMax = atoi(argv[1]);
        if ( procMax<=0 ) 
            return -1;
    }

    // Create the comm used to initialize AMP
    MPI_Comm AMP_comm = AMP_COMM_WORLD;
    #ifdef USES_MPI
        if ( procMax > 0 ) {
            MPI_Init(&argc, &argv);
            // Create a new comm to initialize AMPManager
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            int color = rank%procMax;
            MPI_Comm_split(MPI_COMM_WORLD,color,0,&AMP_comm);
        }
    #endif

    // Control the behavior of the startup
    AMP::AMPManagerProperties startup_properties;
    startup_properties.print_times = true;
    startup_properties.use_MPI_Abort = false;
    startup_properties.COMM_WORLD = AMP_comm;

    // Start AMP
    AMP::AMPManager::startup(argc,argv,startup_properties);
    int globalSize=0;
    // Limit the scope of variables
    { 
        // Create the global comm
        AMP::AMP_MPI globalComm(AMP_COMM_WORLD);
        globalSize = globalComm.getSize();
        if ( procMax>0 && globalSize>procMax )
            AMP_ERROR("AMP did not initialize on a sub-communicator");
        // Introduce a memory leak to catch in valgrind later
        double *x = new double[100];
        if ( x==NULL ) 
            AMP_ERROR("error");
        // Test the abort
        try {
            AMP_ERROR("Catch this");
            return -1;
        } catch (...) {
            // This is correct
        }
    }

    // Shutdown
    AMP::AMPManager::shutdown();

    // Test a reinitialization of AMP
    try {
        AMP::AMPManager::startup(argc,argv,startup_properties);
        AMP_ERROR("Catch this");
        return -1;
    } catch (...) {
        // This is correct
    }

    #ifdef USES_MPI
        // Close MPI
        if ( procMax > 0 ) {
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
        }
    #endif

    // Finished successfully
    return 0;
}   

