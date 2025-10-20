#ifndef included_AMP_initializeLibMesh
#define included_AMP_initializeLibMesh

#include "AMP/utils/AMP_MPI.h"

namespace AMP::Mesh {


/**
 * \class initializeLibMesh
 * \brief A class to initialize libMesh
 * \details  This class provides routines for initializing and finalizing libmesh.
 *    Note: this class is not thread safe yet.
 */
class initializeLibMesh final
{
public:
    /*!
     *  Constructor that inializes libmesh on the given communicator.
     *  Note: libmesh can only be initialized on one comm at a given time.
     *  This function can be called more than once.  If the the comms share
     *  the same comm, then this function will succeed.  It the comms are
     *  different, then this function will throw an error.  If a different comm
     *  is needed, then the user needs to call finalize first.
     */
    explicit initializeLibMesh( const AMP_MPI &comm );

    /*!
     *  Deconstructor that finalizes libmesh.  This allows libmesh to be reinitialized on
     *  a new comm, and frees any memory in use by libmesh.
     */
    ~initializeLibMesh();

    /*!
     *  Function to check if libmesh can be initialized with the given comm.
     *  If this function returns true, then initializeLibMesh::initialize
     *  can be safely called with the given comm.  Otherwise, initializeLibMesh::finalize
     *  must be called first.
     */
    static bool canBeInitialized( const AMP_MPI &comm );

    /*!
     *  Function to check if libmesh has been initialized for any communicator
     *  (are there any copies of the class that have not been destroyed)
     */
    static bool isInitialized();

private:
    // Constructor
    initializeLibMesh(){};

    // Internal data
    static std::atomic<int> N_copies;
    static AMP_MPI d_comm;
    static void *lminit;
};


} // namespace AMP::Mesh

#endif
