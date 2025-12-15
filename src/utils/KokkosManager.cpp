#include "AMP/utils/KokkosManager.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Utilities.h"

#include <cstdio>
#include <string>

#ifdef AMP_USE_KOKKOS
    #include <Kokkos_Core.hpp>
#endif


namespace AMP::Utilities {


#ifdef AMP_USE_KOKKOS
static bool AMP_CalledKokkosInit = false;
void initializeKokkos( int &argc_in, char *argv_in[], const AMPManagerProperties &properties )
{
    if ( !Kokkos::is_initialized() ) {
        // Copy the input arguments, swap the kokkos arguments to the end,
        //    and remove them from the input argument list
        int argc = argc_in;
        for ( int i = argc_in - 1; i >= 0; i-- ) {
            if ( strncmp( argv_in[i], "--kokkos-", 9 ) == 0 )
                std::swap( argv_in[i], argv_in[--argc_in] );
        }
        char *argv[1024] = { nullptr };
        for ( int i = 0; i < argc; i++ )
            argv[i] = argv_in[i];
        // Set some basic environmental variables
        if ( getenv( "OMP_PROC_BIND" ).empty() )
            setenv( "OMP_PROC_BIND", "false" );
        // Check if we need to set the number of threads
        bool setThreads = !getenv( "OMP_NUM_THREADS" ).empty();
        for ( int i = 0; i < argc; i++ ) {
            setThreads = setThreads || strncmp( argv[i], "--threads", 9 ) == 0;
            setThreads = setThreads || strncmp( argv[i], "--kokkos-threads", 16 ) == 0;
            setThreads = setThreads || strncmp( argv[i], "--kokkos-num-threads", 20 ) == 0;
        }
        char threadArg[1024];
        if ( !setThreads ) {
            int N_threads = 1;
            if ( properties.default_OpenMP_threads != 0 )
                N_threads = properties.default_OpenMP_threads;
            if ( properties.default_Kokkos_threads != 0 )
                N_threads = properties.default_Kokkos_threads;
            snprintf( threadArg, sizeof( threadArg ), "--kokkos-num-threads=%i\n", N_threads );
            argv[argc++] = threadArg;
        }
        // Initialize kokkos
        Kokkos::initialize( argc, argv );
        AMP_CalledKokkosInit = true;
    } else {
        AMP_CalledKokkosInit = false;
    }
}
void finalizeKokkos()
{
    if ( AMP_CalledKokkosInit && !Kokkos::is_finalized() )
        Kokkos::finalize();
}
bool isKokkosInitialized() { return Kokkos::is_initialized() || AMP_CalledKokkosInit; }
bool KokkosInitializedOpenMP()
{
    #ifdef KOKKOS_ENABLE_OPENMP
    return isKokkosInitialized() && Kokkos::is_execution_space<Kokkos::OpenMP>::value;
    #else
    return false;
    #endif
}
bool KokkosEnabled() { return true; }
#else
void initializeKokkos( int &, char *[], const AMPManagerProperties & ) {}
void finalizeKokkos() {}
bool KokkosEnabled() { return false; }
bool isKokkosInitialized() { return false; }
bool KokkosInitializedOpenMP() { return false; }

#endif

} // namespace AMP::Utilities
