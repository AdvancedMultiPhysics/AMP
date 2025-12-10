#include "AMP/utils/AMPManager.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/KokkosManager.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"
#include "StackTrace/ErrorHandlers.h"
#include "StackTrace/StackTrace.h"
#include "StackTrace/Utilities.h"

// Include external packages for startup/shutdown
// clang-format off
#ifdef AMP_USE_OPENMP
    #include <omp.h>
#endif
#ifdef AMP_USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime_api.h>
    #include "AMP/utils/cuda/Helper_Cuda.h"
#endif
#ifdef AMP_USE_HIP
    #include <hip/hip_runtime_api.h>
    #include "AMP/utils/hip/Helper_Hip.h"
#endif
#ifdef AMP_USE_TIMER
    #include "MemoryApp.h"
#endif
// clang-format on

#include <algorithm>
#include <array>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string.h>
#include <vector>


// Forward declares
namespace AMP {
class KeyData;
}

namespace AMP {


// Get the elapsed duration
static double getDuration( const std::chrono::time_point<std::chrono::steady_clock> &start )
{
    auto stop  = std::chrono::steady_clock::now();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( stop - start ).count();
    return 1e-9 * ns;
}


// Initialize static member variables
int AMPManager::d_initialized                 = 0;
int AMPManager::d_argc                        = 0;
const char *const *AMPManager::d_argv         = nullptr;
AMPManagerProperties AMPManager::d_properties = AMPManagerProperties();
std::vector<std::function<void()>> AMPManager::d_atShutdown;


/****************************************************************************
 *  Get the global communicator                                              *
 ****************************************************************************/
static AMP_MPI comm_world = AMP::AMP_MPI( AMP_COMM_NULL );
const AMP_MPI &AMPManager::getCommWorld() { return comm_world; }
void AMPManager::setCommWorld( const AMP::AMP_MPI &comm ) { comm_world = comm; }


/****************************************************************************
 * Functions to count resources                                              *
 ****************************************************************************/
static std::map<std::string, std::pair<int, int>> resourceMap;
void AMPManager::incrementResource( const std::string &resource )
{
    auto it = resourceMap.find( resource );
    if ( it == resourceMap.end() )
        resourceMap.insert( std::make_pair( resource, std::pair<int, int>( 1, 0 ) ) );
    else
        it->second.first++;
}
void AMPManager::decrementResource( const std::string &resource )
{
    auto it = resourceMap.find( resource );
    if ( it == resourceMap.end() )
        resourceMap.insert( std::make_pair( resource, std::pair<int, int>( 0, 1 ) ) );
    else
        it->second.second++;
}


/****************************************************************************
 * Initialize the AMP package.  This routine performs the following tasks:   *
 ****************************************************************************/
static size_t getStartupMemoryAllocations()
{
#ifdef AMP_USE_TIMER
    auto memory = MemoryApp::getMemoryStats();
    if ( memory.N_new > memory.N_delete )
        return memory.N_new - memory.N_delete;
#endif
    return 0;
}
static char **copy_args( int argc, char **argv )
{
    auto args = new char *[argc];
    for ( int i = 0; i < argc; i++ ) {
        int N   = strlen( argv[i] );
        args[i] = new char[N + 2];
        memset( args[i], 0, N + 2 );
        memcpy( args[i], argv[i], N );
    }
    return args;
}
static size_t N_memory_startup = getStartupMemoryAllocations();
void AMPManager::startup( int &argc, char *argv[], const AMPManagerProperties &properties )
{
    // Check if AMP was previously initialized
    if ( d_initialized == 1 )
        AMP_ERROR( "AMP was previously initialized.  It cannot be reinitialized" );
    if ( d_initialized == -1 )
        AMP_ERROR( "AMP was previously initialized and shutdown.  It cannot be reinitialized" );
    // Begin startup procedure
    auto start = std::chrono::steady_clock::now();
    // Copy full list of input arguments
    d_properties = properties;
    d_argc       = argc;
    d_argv       = copy_args( argc, argv );
    // Initialize the timers (default is disabled)
    PROFILE_DISABLE();
    // Initialize MPI
    auto MPI_start = std::chrono::steady_clock::now();
    AMP_MPI::start_MPI( argc, argv, d_properties.profile_MPI_level );
    auto MPI_time = getDuration( MPI_start );
    // Initialize AMP's MPI
    AMP_MPI::Comm new_comm = d_properties.COMM_WORLD;
#if defined( AMP_USE_MPI )
    AMP_INSIST( new_comm != AMP_COMM_NULL, "AMP comm world cannot be null" );
    AMP_INSIST( new_comm != MPI_COMM_NULL, "AMP comm world cannot be null" );
    if ( new_comm == AMP_COMM_WORLD )
        new_comm = MPI_COMM_WORLD;
    if ( new_comm == AMP_COMM_SELF )
        new_comm = MPI_COMM_SELF;
#endif
    comm_world = AMP_MPI( new_comm, true );
    // Initialize cuda/hip
    start_CudaOrHip();
    // Initialize OpenMP
    start_OpenMP();
    // Initialize Kokkos
    AMP::Utilities::initializeKokkos( argc, argv, d_properties );
    // Initialize Hypre
    double hypre_time = start_HYPRE();
    // Initialize PETSc
    double petsc_time = start_PETSc();
    // Initialize SAMRAI
    double SAMRAI_time = start_SAMRAI();
    // Initialize error handlers
    setHandlers();
    // Initialization finished
    d_initialized = 1;
    double time   = getDuration( start );
    if ( d_properties.print_startup ) {
        AMP::pout << "Version info:\n" << info() << std::endl;
        AMP::pout.flush();
    }
    if ( d_properties.print_times && comm_world.getRank() == 0 ) {
        printf( "startup time = %0.3f s\n", time );
        if ( MPI_time != 0 )
            printf( "  MPI startup time = %0.3f s\n", MPI_time );
        if ( hypre_time != 0 )
            printf( "  Hypre startup time = %0.3f s\n", hypre_time );
        if ( petsc_time != 0 )
            printf( "  PETSc startup time = %0.3f s\n", petsc_time );
        if ( SAMRAI_time != 0 )
            printf( "  SAMRAI startup time = %0.3f s\n", SAMRAI_time );
        printf( "\n" );
    }
}


/****************************************************************************
 * Shutdown the AMP package                                                  *
 ****************************************************************************/
void AMPManager::shutdown()
{
    auto start_time = std::chrono::steady_clock::now();
    int rank        = comm_world.getRank();
    if ( d_initialized == 0 )
        AMP_ERROR( "AMP is not initialized, did you forget to call startup" );
    if ( d_initialized == -1 )
        AMP_ERROR( "Calling shutdown more than once is invalid" );
    d_initialized = -1;
    // Clear error handlers
    clearHandlers();
    // Disable MPI_Abort
    d_properties.use_MPI_Abort = false;
    StackTrace::Utilities::setAbortBehavior( true, 2 );
    // Shutdown the parallel IO
    stopLogging();
    // Shutdown the profiler
    PROFILE_DISABLE();
    // Syncronize all ranks
    comm_world.barrier();
    // Shutdown SAMRAI
    double SAMRAI_time = stop_SAMRAI();
    // Shudown PETSc
    double petsc_time = stop_PETSc();
    // Shudown Hypre
    double hypre_time = stop_HYPRE();
    // shutdown Kokkos
    AMP::Utilities::finalizeKokkos();
    // Shutdown MPI
    auto MPI_start = std::chrono::steady_clock::now();
    comm_world     = AMP_COMM_NULL;
    AMP_MPI::stop_MPI();
    auto MPI_time = getDuration( MPI_start );
    // Print any AMP_MPI leaks
    if ( AMP_MPI::MPI_Comm_created() != AMP_MPI::MPI_Comm_destroyed() ) {
        printf( "Rank %i detected AMP_MPI comm leak: %i %i\n",
                rank,
                static_cast<int>( AMP_MPI::MPI_Comm_created() ),
                static_cast<int>( AMP_MPI::MPI_Comm_destroyed() ) );
    }
    // Clear input arguments
    for ( int i = 0; i < d_argc; i++ )
        delete[] d_argv[i];
    delete[] d_argv;
    d_argc = 0;
    d_argv = nullptr;
    // Call the shutdown routines in reverse order
    for ( int i = d_atShutdown.size() - 1; i >= 0; i-- )
        d_atShutdown[i]();
    // List shutdown times
    double shutdown_time = getDuration( start_time );
    if ( d_properties.print_times && rank == 0 ) {
        printf( "shutdown time = %0.3f s\n", shutdown_time );
        if ( SAMRAI_time != 0 )
            printf( "  SAMRAI shutdown time = %0.3f s\n", SAMRAI_time );
        if ( petsc_time != 0 )
            printf( "  PETSc shutdown time = %0.3f s\n", petsc_time );
        if ( hypre_time != 0 )
            printf( "  Hypre shutdown time = %0.3f s\n", hypre_time );
        if ( MPI_time != 0 )
            printf( "  MPI shutdown time = %0.3f s\n", MPI_time );
        printf( "\n" );
    }
    // Print resource counts
    for ( const auto &x : resourceMap ) {
        if ( x.second.first != x.second.second ) {
            printf( "%s:\n", x.first.data() );
            printf( "    %i created\n", x.second.first );
            printf( "    %i destroyed\n", x.second.second );
        }
    }
    resourceMap.clear();
    // Shutdown timer and print memory leaks on rank 0
    PROFILE_DISABLE();
#ifdef AMP_USE_TIMER
    if ( d_properties.print_memory != 0 ) {
        auto memory = MemoryApp::getMemoryStats();
        bool leaks  = memory.N_new > ( memory.N_delete + N_memory_startup );
        if ( ( rank == 0 && leaks ) || d_properties.print_memory == 1 ) {
            std::cout << std::endl << "N_memory_startup: " << N_memory_startup << std::endl;
            MemoryApp::print( std::cout );
        }
    }
#endif
    // Wait 50 milli-seconds for all processors to finish
    Utilities::sleep_ms( 50 );
}


/****************************************************************************
 * Restart AMP                                                               *
 ****************************************************************************/
void AMPManager::restart()
{
    if ( d_initialized != 1 )
        AMP_ERROR( "AMP is not initialized or has been shutdown" );

    restart_SAMRAI();
}


/****************************************************************************
 * Register shutdown routine                                                 *
 ****************************************************************************/
void AMPManager::registerShutdown( std::function<void()> fun ) { d_atShutdown.push_back( fun ); }


/****************************************************************************
 * Function to start/stop CUDA                                               *
 ****************************************************************************/
double AMPManager::start_CudaOrHip()
{
    if ( !d_properties.initialize_device )
        return 0;
    auto start = std::chrono::steady_clock::now();
#ifdef AMP_USE_DEVICE
    if ( d_properties.bind_process_to_accelerator ) {
        AMP::Utilities::setenv( "RDMAV_FORK_SAFE", "1" );
        auto nodeComm = comm_world.splitByNode();
        auto nodeRank = nodeComm.getRank();
        int deviceCount;

    #ifdef AMP_USE_CUDA
        checkCudaErrors( cudaGetDeviceCount( &deviceCount ) ); // How many GPUs?
        int device_id = nodeRank % deviceCount;
        checkCudaErrors( cudaSetDevice( device_id ) ); // Map MPI-process to a GPU
    #else
        checkHipErrors( hipGetDeviceCount( &deviceCount ) ); // How many GPUs?
        int device_id = nodeRank % deviceCount;
        checkHipErrors( hipSetDevice( device_id ) ); // Map MPI-process to a GPU
    #endif
    }

    void *tmp;
    #ifdef AMP_USE_CUDA
    checkCudaErrors( cudaMallocManaged( &tmp, 10, cudaMemAttachGlobal ) );
    checkCudaErrors( cudaFree( tmp ) );
    #else
    checkHipErrors( hipMallocManaged( &tmp, 10, hipMemAttachGlobal ) );
    checkHipErrors( hipFree( tmp ) );
    #endif
#endif
    return getDuration( start );
}


/****************************************************************************
 * Initialize OpenMP                                                         *
 ****************************************************************************/
double AMPManager::start_OpenMP()
{
#ifdef AMP_USE_OPENMP
    if ( AMP::Utilities::KokkosInitializedOpenMP() )
        return 0;
    auto start = std::chrono::steady_clock::now();
    // Set the number of threads
    int N_threads = 1;
    if ( d_properties.default_OpenMP_threads != 0 )
        N_threads = d_properties.default_OpenMP_threads;
    if ( d_properties.default_Kokkos_threads != 0 && AMP::Utilities::KokkosEnabled() )
        N_threads = d_properties.default_Kokkos_threads;
    auto OMP_NUM_THREADS = AMP::Utilities::getenv( "OMP_NUM_THREADS" );
    if ( !OMP_NUM_THREADS.empty() )
        N_threads = atoi( OMP_NUM_THREADS.data() );
    AMP::Utilities::setenv( "OMP_NUM_THREADS", std::to_string( N_threads ).data() );
    omp_set_num_threads( N_threads );
    // Verify the count
    int count = 0;
    #pragma omp parallel
    {
    #pragma omp atomic
        ++count;
    }
    if ( count != N_threads )
        printf( "OpenMP has %i threads but we expected %i\n", count, N_threads );
    return getDuration( start );
#else
    return 0;
#endif
}


/****************************************************************************
 * Empty constructor to setup default AMPManagerProperties                   *
 ****************************************************************************/
AMPManagerProperties::AMPManagerProperties() : COMM_WORLD( AMP_COMM_WORLD )
{
    auto defaultSignals = StackTrace::defaultSignalsToCatch();
    catch_signals       = std::set( defaultSignals.begin(), defaultSignals.end() );
}


/****************************************************************************
 *  Some simple functions                                                    *
 ****************************************************************************/
bool AMPManager::isInitialized() { return d_initialized != 0; }
bool AMPManager::isFinalized() { return d_initialized == -1; }
std::tuple<int, const char *const *> AMPManager::get_args()
{
    return std::tuple<int, const char *const *>( d_argc, d_argv );
}

AMPManagerProperties AMPManager::getAMPManagerProperties()
{
    AMP_INSIST( d_initialized, "AMP has not been initialized" );
    return d_properties;
}


} // namespace AMP

/***********************************************************************
 * C interfaces                                                         *
 ***********************************************************************/
extern "C" {
void amp_startup_basic_f()
{
    int argc     = 0;
    char *argv[] = { nullptr };
    AMP::AMPManagerProperties properties;
    properties.catch_signals.clear();
    properties.stack_trace_type  = 2;
    properties.initialize_device = false;
    properties.catch_exit        = false;
    properties.catch_MPI         = false;
    AMP::AMPManager::startup( argc, argv, properties );
}

void amp_startup_f( int argc, char **argv )
{
    AMP::AMPManagerProperties properties;
    properties.catch_signals.clear();
    properties.stack_trace_type  = 2;
    properties.initialize_device = false;
    properties.catch_exit        = false;
    properties.catch_MPI         = false;
    AMP::AMPManager::startup( argc, argv, properties );
}

void amp_shutdown_f( void ) { AMP::AMPManager::shutdown(); }
bool amp_initialized_f( void ) { return AMP::AMPManager::isInitialized(); }
bool amp_finalized_f( void ) { return AMP::AMPManager::isFinalized(); }
}
