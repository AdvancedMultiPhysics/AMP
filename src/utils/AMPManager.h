#ifndef included_AMP_AMPManager
#define included_AMP_AMPManager

#include "AMP/utils/AMP_MPI.h"

#include <array>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>


namespace StackTrace {
class abort_error;
}


namespace AMP {


/*!
 * @brief Class AMPManagerProperties is a class that contains the various startup options for AMP
 */
class AMPManagerProperties
{
public:
    //! Empty constructor.  This creates the default startup parameters.
    AMPManagerProperties();

    /*!
     * Use MPI_Abort.  This determines if we want to use MPI_Abort (default) or
     * throw an exception when we abort.  Note that using MPI_Abort will safely terminate
     * all processes, but is incompatible with try, catch.  A standard exception can be
     * caught, but can easily lead to errors where different processors get stuck when
     * a single process encounters an error.  It is strongly recommended to use MPI_Abort.
     */
    bool use_MPI_Abort = true;

    //! Print the time required to initialize or shutdown each package.  Default is false.
    bool print_times = false;

    //! Print version information upon startup (default is false)
    bool print_startup = false;

    //! Print memory statistics (default is only if leaks detected)
    int print_memory = -1;

    //! Initialize CUDA/HIP device (default is true)
    bool initialize_device = true;

    //! Associate each accelerator with a single MPI rank (default is true)
    bool bind_process_to_accelerator = true;

    //! The default level for the MPI timers.  Default is 2
    int profile_MPI_level = 2;

    /*!
     * Stack trace to print on error:
     *    0 - Don't print the call stack
     *    1 - Current process/thread
     *    2 - Current process, all threads
     *    3 - Global call stack
     */
    int stack_trace_type = 3;

    //! The set of unhandled signals to set (will be initialized to a default set)
    std::set<int> catch_signals;

    //! Set error handlers for MPI routines
    bool catch_MPI = true;

    //! Set error handlers for PETSc routines
    bool catch_PETSc = true;

    //! Set error handlers for SAMRAI routines
    bool catch_SAMRAI = true;

    //! Set error handlers for HDF5 routines
    bool catch_HDF5 = true;

    //! Catch early exit
    bool catch_exit = true;

    //! Set a user-provided function for handling all errors
    std::function<void( StackTrace::abort_error & )> error_handler = nullptr;

    /*!
     *  MPI communicator to use for AMP_COMM_WORLD.  By default this should be set to
     *  AMP_COMM_WORLD if MPI is not initialized.  If MPI is initialized, this can be
     *  set to AMP_COMM_WORLD, MPI_COMM_WORLD, or any valid MPI communicator.  If it
     *  is set to AMP_COMM_WORLD then is will default internally to MPI_COMM_WORLD.
     *  Note:  Currently AMP_COMM_WORLD cannot be changed once AMPManager::startup
     *  has been called.
     */
    AMP_MPI::Comm COMM_WORLD;

private:
    friend class AMPManager;
};


/*!
 * @brief Class AMPManager is a utility for managing startup and shutdown for AMP
 * applications and for changing the maximum number of patch data components supported
 * by AMP patches.  All applications should call AMPManager::startup() at the
 * beginning of the program.  Startup will initialize all packages including MPI.
 * AMPManager::shutdown() should be called at the end of the program, before calling exit(0).
 * Note that the shutdown, but it shuts down the packages, MPI, and deallocates memory.
 */
class AMPManager
{
public:
    /*!
     * Initialize the AMP package.  Depending on the architecture and
     * compile flags, this routine sets up MPI, initializes IEEE exception
     * handlers, and other architecture-specific details.
     */
    static void startup( int &argc,
                         char *argv[],
                         const AMPManagerProperties &properties = AMPManagerProperties() );

    /*!
     * Shutdown the AMP package.  Depending on the compile flags set at
     * compile-time, this routine shuts down MPI and calls registered shutdown
     * handlers.
     */
    static void shutdown();

    //! Restart as much of AMP as possible restoring it to a like new state
    static void restart();

    //! Function to check if AMP has been initialized
    static bool isInitialized();

    //! Function to check if AMP has been finalized
    static bool isFinalized();

    /*!
     * Return a reference to the original command line arguments that were used to initialize AMP.
     */
    static std::tuple<int, const char *const *> get_args();

    //! Function to return the AMPManagerProperties that was used to initialize AMP
    static AMPManagerProperties getAMPManagerProperties();

    //! Static function to terminate AMP
    static void terminate_AMP( std::string message );

    //! Set the default signal/terminate handlers (called on startup)
    static void setHandlers();

    //! Clearthe default signal/terminate handlers (called on shutdown)
    static void clearHandlers();

    //! Initialize the mpi error handler
    static void setMPIErrorHandler();

    //! Destroy the mpi error handler
    static void clearMPIErrorHandler();

    /*!
     * @brief  AMP version number
     * @details  This function returns the current version of AMP
     *   Note that a value of {0,0,xxx} will be returned for the development version.
     *   The build number is the revision number based on the commits in the development version.
     *   Note: when comparing versions, only the last number needs to be compared.
     * @return          Returns array containing the {major,minor,build} version
     */
    static std::array<int, 3> revision();

    //! Return detailed revision information
    static std::string info();

    //! Increment a resource counter
    static void incrementResource( const std::string &resource );

    //! Decrement a resource counter
    static void decrementResource( const std::string &resource );

    //! Get the global comm
    static const AMP::AMP_MPI &getCommWorld();

    //! Set the global comm
    static void setCommWorld( const AMP::AMP_MPI & );


private:
    // Private constructor (we do not actually want to create an object)
    AMPManager() = delete;

    // Static variables
    static int d_initialized;
    static int d_argc;
    static const char *const *d_argv;
    static AMPManagerProperties d_properties;

    // Function to control exit behavior
    static void exitFun();

    // Functions to start/shutdown the various packages
    static double start_SAMRAI();
    static double start_PETSc();
    static double start_HYPRE();
    static double start_CudaOrHip();
    static double stop_SAMRAI();
    static double stop_HYPRE();
    static double stop_PETSc();

    // Functions to set error handlers for specific packages
    static void set_PETSc_error_handler();
    static void clear_PETSc_error_handler();
};

} // namespace AMP

#endif
