#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"

#ifdef AMP_USE_SAMRAI
    #include "SAMRAI/tbox/Logger.h"
    #include "SAMRAI/tbox/SAMRAIManager.h"
    #include "SAMRAI/tbox/Schedule.h"
    #include "SAMRAI/tbox/StartupShutdownManager.h"
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
    auto start = std::chrono::steady_clock::now();
    #ifdef AMP_USE_MPI
    SAMRAI::tbox::SAMRAI_MPI::init( AMP_MPI( AMP_COMM_WORLD ).getCommunicator() );
    #else
    SAMRAI::tbox::SAMRAI_MPI::initMPIDisabled();
    #endif
    SAMRAI::tbox::SAMRAIManager::initialize();
    SAMRAI::tbox::SAMRAIManager::startup();
    SAMRAI::tbox::SAMRAIManager::setMaxNumberPatchDataEntries( 2048 );
    return getDuration( start );
}
double AMPManager::stop_SAMRAI()
{
    auto start = std::chrono::steady_clock::now();
    SAMRAI::tbox::PIO::finalize();
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::finalize();
    SAMRAI::tbox::SAMRAI_MPI::finalize();
    clearTimers( SAMRAI::tbox::Schedule() );
    return getDuration( start );
}

void AMPManager::restart_SAMRAI()
{
    SAMRAI::tbox::SAMRAIManager::shutdown();
    SAMRAI::tbox::SAMRAIManager::startup();
}
#else
double AMPManager::start_SAMRAI() { return 0; }
double AMPManager::stop_SAMRAI() { return 0; }
void AMPManager::restart_SAMRAI() {}
#endif
} // namespace AMP
