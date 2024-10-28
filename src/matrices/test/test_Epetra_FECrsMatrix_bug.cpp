/*
 *  This test tries to replicate a bug with the Epetra_FECrsMatrix using makeConsistent
 *  when the communication pattern.  Note: this test will fail (crash) if the
 *  -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC flags are set.
 *
 */
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"

#include <cstdio>

#include <Epetra_FECrsMatrix.h>
#include <Epetra_Map.h>

#ifdef AMP_USE_MPI
    #include <Epetra_MpiComm.h>
#else
    #include <Epetra_SerialComm.h>
#endif

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    int rank = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();
    int size = AMP::AMP_MPI( AMP_COMM_WORLD ).getSize();
    if ( size != 2 ) {
        std::cout << "Test is only designed for 2 processors\n";
        return -1;
    }

// Create the matrix
#ifdef AMP_USE_MPI
    Epetra_MpiComm comm = MPI_COMM_WORLD;
    #include <Epetra_MpiComm.h>
#else
    Epetra_SerialComm comm;
#endif
    Epetra_Map map( 4, 2, 0, comm );
    int entities[4] = { 4, 4, 4, 4 };
    Epetra_FECrsMatrix matrix( Copy, map, entities, false );

    // Zero out the matrix (using processor 0)
    if ( rank == 0 ) {
        int cols[4]      = { 1, 1, 1, 1 };
        double values[4] = { 0, 0, 0, 0 };
        for ( int i = 0; i < 4; i++ )
            matrix.ReplaceGlobalValues( i, 4, values, cols );
    }
    matrix.GlobalAssemble( true );

    // Modify the diagonal (by the local owner)
    double one = 1.0;
    if ( rank == 0 ) {
        int rows[2] = { 0, 1 };
        matrix.ReplaceGlobalValues( rows[0], 1, &one, &rows[0] );
        matrix.ReplaceGlobalValues( rows[1], 1, &one, &rows[1] );
    } else {
        int rows[2] = { 2, 3 };
        matrix.ReplaceGlobalValues( rows[0], 1, &one, &rows[0] );
        matrix.ReplaceGlobalValues( rows[1], 1, &one, &rows[1] );
    }
    matrix.GlobalAssemble( true );

    AMP::AMPManager::shutdown();
    return 0;
}
