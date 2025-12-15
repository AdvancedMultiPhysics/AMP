#include "AMP/mesh/libmesh/initializeLibMesh.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"

#include "StackTrace/ErrorHandlers.h"

#include <atomic>
#include <cstring>

// LibMesh include
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"

// Petsc include (needed to fix PETSC_COMM_WORLD problem with libmesh)
#ifdef AMP_USE_PETSC
    #include "petscsys.h"
#endif

namespace AMP::Mesh {


// Initialize static member variables
std::atomic<int> initializeLibMesh::N_copies = 0;
void *initializeLibMesh::lminit              = nullptr;
AMP_MPI initializeLibMesh::d_comm            = AMP_COMM_NULL;


/************************************************************
 * Constructor initilize libmesh on the given comm           *
 ************************************************************/
initializeLibMesh::initializeLibMesh( const AMP_MPI &comm )
{
    if ( N_copies > 0 ) {
        // libmesh is already initialized, check if it is compatible with the current comm
        bool test = canBeInitialized( comm );
        if ( test ) {
            // Add 1 to the count and return
            N_copies++;
            return;
        } else {
            // We can't initialize libmesh
            AMP_ERROR( "libmesh was previously initialized with a different (incompatible) comm" );
        }
    } else {
        // libmesh is not initialized
        if ( lminit != nullptr )
            AMP_ERROR( "Internal error" );
        // Use a barrier to ensure all processors are at the same point
        N_copies = 1;
        d_comm   = comm.dup(); // Create a seperate duplicate comm for libmesh
        d_comm.barrier();
        // Reinitialize LibMesh with the new communicator
        auto [argc, argv0] = AMPManager::get_args();
        char *argv[1024]   = { nullptr };
        memcpy( argv, argv0, argc * sizeof( char * ) );
        char disableRefCount[] = "--disable-refcount-printing";
        char syncWithStdio[]   = "--sync-with-stdio";
        char sepOutput[]       = "--separate-libmeshout";
        argv[argc++]           = disableRefCount;
        argv[argc++]           = syncWithStdio;
        argv[argc++]           = sepOutput;
        auto terminate         = std::get_terminate();
#ifdef AMP_USE_MPI
    #ifdef AMP_USE_PETSC
        MPI_Comm petsc_comm = PETSC_COMM_WORLD;
    #endif
        lminit = new libMesh::LibMeshInit( argc, argv, d_comm.getCommunicator() );
    #ifdef AMP_USE_PETSC
        PETSC_COMM_WORLD = petsc_comm;
    #endif
#else
        lminit = new libMesh::LibMeshInit( argc, argv );
#endif
        // Reset the error handlers
        StackTrace::setMPIErrorHandler( d_comm.getCommunicator() );
        std::set_terminate( terminate );
    }
}


/************************************************************
 * Deconstructor that will finalize libmesh                  *
 ************************************************************/
initializeLibMesh::~initializeLibMesh()
{
    if ( N_copies <= 0 )
        AMP_ERROR( "Internal error" );
    // Use a barrier to ensure all processors are at the same point
    d_comm.barrier();
    if ( N_copies == 1 ) {
        // Shutdown libmesh
        if ( lminit == nullptr )
            AMP_ERROR( "Internal error" );
        // Free libmesh MPI types
        // type_hilbert.reset();
        // Delete libmesh
        delete (libMesh::LibMeshInit *) lminit;
        lminit   = nullptr;
        d_comm   = AMP_MPI( AMP_COMM_NULL );
        N_copies = 0;
    } else {
        N_copies--;
    }
}


/************************************************************
 * Function check if initiallize can be called successfully  *
 ************************************************************/
bool initializeLibMesh::canBeInitialized( const AMP_MPI &comm )
{
    if ( N_copies == 0 )
        return true;
    if ( comm == d_comm )
        return true;
    if ( d_comm.compare( comm ) != 0 )
        return true;
    return false;
}


/************************************************************
 * Function to check if libmesh has been initialized         *
 ************************************************************/
bool initializeLibMesh::isInitialized() { return N_copies > 0; }


} // namespace AMP::Mesh
