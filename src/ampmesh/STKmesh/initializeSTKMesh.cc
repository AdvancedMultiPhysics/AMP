#include "ampmesh/STKmesh/initializeSTKMesh.h"
#include "ampmesh/STKmesh/STKMeshIterator.h"
#include "utils/MemoryDatabase.h"
#include "utils/AMPManager.h"

#include <string.h>

// STKMesh include
#include "ampmesh/Mesh.h"


namespace AMP {
namespace Mesh {


// Initialize static variables
namespace {
  unsigned N_copies(const int increment=0) {
    static unsigned copies(0);
    if ( copies + increment < 0 ) AMP_ERROR("Internal error");
    copies += increment;
    return copies;
  }
  AMP_MPI &d_comm(const AMP_MPI c=AMP_COMM_NULL) {
    static AMP_MPI comm(AMP_COMM_NULL);
    if (c!=AMP_COMM_NULL) comm = c;
    return comm;
  }
}

/************************************************************
* Constructor initilize STKmesh on the given comm           *
************************************************************/
initializeSTKMesh::initializeSTKMesh( AMP_MPI comm )
{
    if ( N_copies() ) {
        // STKmesh is already initialized, check if it is compatible with the current comm
        if ( canBeInitialized( comm ) ) {
            // Add 1 to the count and return
            N_copies(1);
        } else {
            // We can't initialize STKmesh
            AMP_ERROR("STKmesh was previously initialized with a different (incompatible) comm");
        }
    } else {
        // STKmesh is not initialized
        // Use a barrier to ensure all processors are at the same point
        N_copies(1);
        d_comm  (comm);
        d_comm().barrier();
    }
    return;
}


/************************************************************
* Deconstructor that will finalize STKmesh                  *
************************************************************/
initializeSTKMesh::~initializeSTKMesh( )
{
    // Use a barrier to ensure all processors are at the same point
    d_comm().barrier();
    N_copies(-1);
}


/************************************************************
* Function check if initiallize can be called successfully  *
************************************************************/
bool initializeSTKMesh::canBeInitialized( AMP_MPI comm )
{
    return ( !N_copies() || comm == d_comm() || d_comm().compare(comm) );
}


/************************************************************
* Function to check if STKmesh has been initialized         *
************************************************************/
bool initializeSTKMesh::isInitialized( )
{
    return !N_copies();
}


} // Mesh namespace
} // AMP namespace
