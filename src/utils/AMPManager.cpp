#include "AMP/utils/AMPManager.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/FactoryStrategy.hpp"
#include "AMP/utils/KokkosManager.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"
#include "StackTrace/ErrorHandlers.h"
#include "StackTrace/StackTrace.h"
#include "StackTrace/Utilities.h"


// Include external packages for startup/shutdown
// clang-format off
#undef NULL_USE
#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime_api.h>
    #include "AMP/utils/cuda/helper_cuda.h"
#endif
#ifdef AMP_USE_PETSC
    #include "petsc.h"
    #include "petscerror.h"
#endif
#ifdef AMP_USE_TIMER
    #include "MemoryApp.h"
#endif
#ifdef AMP_USE_SAMRAI
    #include "SAMRAI/tbox/Logger.h"
    #include "SAMRAI/tbox/SAMRAIManager.h"
    #include "SAMRAI/tbox/Schedule.h"
    #include "SAMRAI/tbox/StartupShutdownManager.h"
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
namespace AMP::Materials {
class Material;
}


namespace AMP {


// Initialize static member variables
int AMPManager::d_initialized                 = 0;
int AMPManager::d_argc                        = 0;
const char *const *AMPManager::d_argv         = nullptr;
AMPManagerProperties AMPManager::d_properties = AMPManagerProperties();


/****************************************************************************
 *  Get the global communicator                                              *
 ****************************************************************************/
static AMP_MPI comm_world = AMP::AMP_MPI( MPI_COMM_NULL );
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
    double start = Utilities::time();
    // Copy full list of input arguments
    d_properties = properties;
    d_argc       = argc;
    d_argv       = copy_args( argc, argv );
    // Initialize the timers (default is disabled)
    PROFILE_DISABLE();
    // Set the abort method
    StackTrace::Utilities::setAbortBehavior( !d_properties.use_MPI_Abort, 3 );
    // Initialize MPI
    double MPI_start = Utilities::time();
    AMP_MPI::start_MPI( argc, argv, d_properties.profile_MPI_level );
    double MPI_time = Utilities::time() - MPI_start;
    // Initialize AMP's MPI
    comm_world = AMP_MPI( AMP_COMM_WORLD );
    if ( d_properties.COMM_WORLD != AMP_COMM_WORLD )
        comm_world = AMP_MPI( d_properties.COMM_WORLD );
    // Initialize cuda
    start_CUDA();
    // Initialize Kokkos
    AMP::Utilities::initializeKokkos( argc, argv );
    // Initialize PETSc
    double petsc_time = start_PETSc();
    // Initialize SAMRAI
    double SAMRAI_time = start_SAMRAI();
    // Initialize call stack
    if ( comm_world.getSize() == 1 )
        d_properties.stack_trace_type = std::min( d_properties.stack_trace_type, 2 );
    if ( d_properties.stack_trace_type == 3 )
        StackTrace::globalCallStackInitialize( comm_world.getCommunicator() );
    StackTrace::setDefaultStackType(
        static_cast<StackTrace::printStackType>( d_properties.stack_trace_type ) );
    // Set the signal/terminate handlers
    StackTrace::Utilities::setErrorHandlers();
    setHandlers();
    // Initialization finished
    d_initialized = 1;
    double time   = Utilities::time() - start;
    if ( d_properties.print_startup ) {
        AMP::pout << "Version info:\n" << info() << std::endl;
        AMP::pout.flush();
    }
    if ( d_properties.print_times && comm_world.getRank() == 0 ) {
        printf( "startup time = %0.3f s\n", time );
        if ( MPI_time != 0 )
            printf( "  MPI startup time = %0.3f s\n", MPI_time );
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
    double start_time = Utilities::time();
    int rank          = comm_world.getRank();
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
    // shutdown Kokkos
    AMP::Utilities::finalizeKokkos();
    // Shutdown MPI
    double MPI_start = Utilities::time();
    AMP_MPI::stop_MPI();
    double MPI_time = Utilities::time() - MPI_start;
    // Print any AMP_MPI leaks
    if ( AMP_MPI::MPI_Comm_created() != AMP_MPI::MPI_Comm_destroyed() ) {
        printf( "Rank %i detected AMP_MPI comm leak: %i %i\n",
                rank,
                static_cast<int>( AMP_MPI::MPI_Comm_created() ),
                static_cast<int>( AMP_MPI::MPI_Comm_destroyed() ) );
    }
    // List shutdown times
    double shutdown_time = Utilities::time() - start_time;
    if ( d_properties.print_times && rank == 0 ) {
        printf( "shutdown time = %0.3f s\n", shutdown_time );
        if ( SAMRAI_time != 0 )
            printf( "  SAMRAI shutdown time = %0.3f s\n", SAMRAI_time );
        if ( petsc_time != 0 )
            printf( "  PETSc shutdown time = %0.3f s\n", petsc_time );
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
    // Clear input arguments
    for ( int i = 0; i < d_argc; i++ )
        delete[] d_argv[i];
    delete[] d_argv;
    d_argc = 0;
    d_argv = nullptr;
    // Clear the factories
    AMP::FactoryStrategy<AMP::KeyData>::clear();
    AMP::FactoryStrategy<AMP::Materials::Material>::clear();
    AMP::Operator::OperatorFactory::clear();
    AMP::Solver::SolverFactory::clear();
    AMP::TimeIntegrator::TimeIntegratorFactory::clear();
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
#ifdef AMP_USE_SAMRAI
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::startup();
#endif
}


/****************************************************************************
 * Function to start/stop SAMRAI                                             *
 ****************************************************************************/
#ifdef AMP_USE_SAMRAI
template<typename T>
class hasClearTimers
{
private:
    template<typename C>
    static char &test( decltype( &C::clearTimers ) );
    template<typename C>
    static int &test( ... );

public:
    static constexpr bool value = sizeof( test<T>( 0 ) ) == sizeof( char );
};
template<typename T>
typename std::enable_if_t<hasClearTimers<T>::value, void> clearTimers( const T &obj )
{
    obj.clearTimers();
}
template<typename T>
typename std::enable_if_t<!hasClearTimers<T>::value, void> clearTimers( const T & )
{
}
double AMPManager::start_SAMRAI()
{
    double start = Utilities::time();
    #ifdef AMP_USE_MPI
    SAMRAI::tbox::SAMRAI_MPI::init( AMP_MPI( AMP_COMM_WORLD ).getCommunicator() );
    #else
    SAMRAI::tbox::SAMRAI_MPI::initMPIDisabled();
    #endif
    SAMRAI::tbox::SAMRAIManager::initialize();
    SAMRAI::tbox::SAMRAIManager::startup();
    SAMRAI::tbox::SAMRAIManager::setMaxNumberPatchDataEntries( 2048 );
    return Utilities::time() - start;
}
double AMPManager::stop_SAMRAI()
{
    double start = Utilities::time();
    SAMRAI::tbox::PIO::finalize();
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();
    clearTimers( SAMRAI::tbox::Schedule() );
    return Utilities::time() - start;
}
#else
double AMPManager::start_SAMRAI() { return 0; }
double AMPManager::stop_SAMRAI() { return 0; }
#endif


/****************************************************************************
 * Function to start/stop PETSc                                              *
 ****************************************************************************/
#ifdef AMP_USE_PETSC
static bool called_PetscInitialize = false;
double AMPManager::start_PETSc()
{
    double start = Utilities::time();
    if ( PetscInitializeCalled ) {
        called_PetscInitialize = false;
    } else {
        int nargsPetsc       = 1;
        const char *noMalloc = "-malloc no";
        char **petscArgs     = const_cast<char **>( &noMalloc );
        PetscInitialize( &nargsPetsc, &petscArgs, nullptr, nullptr );
        called_PetscInitialize = true;
    }
    #ifndef AMP_USE_MPI
    // Fix minor bug in petsc where first call to dup returns MPI_COMM_WORLD instead of a new comm
    AMP::AMP_MPI( MPI_COMM_WORLD ).dup();
    #endif
    return Utilities::time() - start;
}
double AMPManager::stop_PETSc()
{
    double time = 0;
    if ( called_PetscInitialize ) {
        double start = Utilities::time();
        PetscPopSignalHandler();
        PetscPopErrorHandler();
        PetscFinalize();
        time = Utilities::time() - start;
    }
    return time;
}
#else
double AMPManager::start_PETSc() { return 0; }
double AMPManager::stop_PETSc() { return 0; }
#endif


/****************************************************************************
 * Function to start/stop CUDA                                               *
 ****************************************************************************/
double AMPManager::start_CUDA()
{
    if ( !d_properties.initialize_CUDA )
        return 0;
    double start = Utilities::time();
#ifdef USE_CUDA
    if ( d_properties.bind_process_to_accelerator ) {
        auto nodeComm = comm_world.splitByNode();
        auto nodeRank = nodeComm.getRank();
        int deviceCount;
        cudaGetDeviceCount( &deviceCount ); // How many GPUs?
        int device_id = nodeRank % deviceCount;
        cudaSetDevice( device_id ); // Map MPI-process to a GPU
    }

    void *tmp;
    checkCudaErrors( cudaMallocManaged( &tmp, 10, cudaMemAttachGlobal ) );
    cudaFree( tmp );
#endif
    return Utilities::time() - start;
}


/****************************************************************************
 * Empty constructor to setup default AMPManagerProperties                   *
 ****************************************************************************/
AMPManagerProperties::AMPManagerProperties() : COMM_WORLD( AMP_COMM_WORLD ) {}


/****************************************************************************
 *  Some simple functions                                                    *
 ****************************************************************************/
bool AMPManager::isInitialized() { return d_initialized != 0; }
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
