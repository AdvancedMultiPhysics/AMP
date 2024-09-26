#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"

#ifdef AMP_USE_HYPRE
    #include "HYPRE_utilities.h"
#endif

#include <chrono>


// Get the elapsed duration
[[maybe_unused]] static double
getDuration( const std::chrono::time_point<std::chrono::steady_clock> &start )
{
    auto stop  = std::chrono::steady_clock::now();
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>( stop - start ).count();
    return 1e-9 * ns;
}


namespace AMP {


/****************************************************************************
 * Function to start/stop HYPRE                                              *
 ****************************************************************************/
#ifdef AMP_USE_HYPRE
double AMPManager::start_HYPRE()
{
    auto start = std::chrono::steady_clock::now();
    if ( !HYPRE_Initialized() ) {
        HYPRE_Initialize();
    }
    return getDuration( start );
}
double AMPManager::stop_HYPRE()
{
    double time = 0;
    if ( !HYPRE_Finalized() ) {
        auto start = std::chrono::steady_clock::now();
        HYPRE_Finalize();
        time = getDuration( start );
    }
    return time;
}
#else
double AMPManager::start_HYPRE() { return 0; }
double AMPManager::stop_HYPRE() { return 0; }
#endif

} // namespace AMP
