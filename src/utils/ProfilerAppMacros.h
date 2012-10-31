#ifndef included_ProfilerAppMacros
#define included_ProfilerAppMacros


//! Define some helper macros
#define GET_LEVEL(_0,N,...) N 
#define PROFILE_START_LEVEL(NAME,FILE,LINE,LEVEL)           \
    do {                                                    \
      if ( LEVEL <= global_profiler.get_level() )           \
        global_profiler.start( NAME, FILE, LINE, LEVEL );   \
    } while(0)
#define PROFILE_STOP_LEVEL(NAME,FILE,LINE,LEVEL)            \
    do {                                                    \
      if ( LEVEL <= global_profiler.get_level() )           \
        global_profiler.stop( NAME, FILE, LINE, LEVEL );    \
    } while(0)



/*! \addtogroup Macros
 *  @{
 */


/*! \def PROFILE_START(NAME,..)
 *  \brief Start the profiler
 *  \details This is the primary call to start a timer.  Only one call within a file 
 *      may call the timer.  Any other calls must use PROFILE_START2(X).
 *      This call will automatically add the file and line number to the timer.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 *  \param NAME  Name of the timer
 */
#define PROFILE_START(NAME,...) \
     PROFILE_START_LEVEL( NAME, __FILE__, __LINE__, GET_LEVEL(_0,##__VA_ARGS__,0) )


/*! \def PROFILE_STOP(NAME,..)
 *  \brief Stop the profiler
 *  \details This is the primary call to stop a timer.  Only one call within a file 
 *      may call the timer.  Any other calls must use PROFILE_STOP2(X).
 *      This call will automatically add the file and line number to the timer.
 *      An optional argument specifying the level to enable may be included.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 *  \param NAME  Name of the timer
 */
#define PROFILE_STOP(NAME,...) \
    PROFILE_STOP_LEVEL( NAME, __FILE__, __LINE__, GET_LEVEL(_0,##__VA_ARGS__,0) )


/*! \def PROFILE_START2(NAME,..)
 *  \brief Start the profiler
 *  \details This is a call to start a timer without the line number.
 *      An optional argument specifying the level to enable may be included.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 *  \param NAME  Name of the timer
 */
#define PROFILE_START2(NAME,...) \
     PROFILE_START_LEVEL( NAME, __FILE__, -1, GET_LEVEL(_0,##__VA_ARGS__,0) )


/*! \def PROFILE_STOP2(NAME,..)
 *  \brief Start the profiler
 *  \details This is a call to start a timer without the line number.
 *      An optional argument specifying the level to enable may be included.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 *  \param NAME  Name of the timer
 */
#define PROFILE_STOP2(NAME,...) \
    PROFILE_STOP_LEVEL( NAME, __FILE__, -1, GET_LEVEL(_0,##__VA_ARGS__,0) )


/*! \def PROFILE_SAVE(FILE)
 *  \brief Save the profile results
 *  \details This will save the results of the timers the file provided
 *      An optional argument specifying the level to enable may be included.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 *  \param FILE  Name of the file to save
 */
#define PROFILE_SAVE(FILE) \
    global_profiler.save( FILE )


/*! \def PROFILE_STORE_TRACE(X)
 *  \brief Enable/Disable the trace data
 *  \details This will enable or disable trace timers.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 *  \param X  Flag to indicate if we want to enable/disable the trace timers
 */
#define PROFILE_STORE_TRACE(X) \
    global_profiler.set_store_trace( X )


/*! \def PROFILE_ENABLE(...)
 *  \brief Enable the timers
 *  \details This will enable the timers.
 *      An optional argument specifying the level to enable may be included.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 */
#define PROFILE_ENABLE(...) \
    global_profiler.enable(__VA_ARGS__)


/*! \def PROFILE_DISABLE()
 *  \brief Disable the timers
 *  \details This will disable the timers.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 */
#define PROFILE_DISABLE() \
    global_profiler.disable()


/*! \def PROFILE_ENABLE_TRACE()
 *  \brief Enable the trace level timers
 *  \details This will enable the trace capabilites within the timers.
 *      It does not affect the which timers are enabled or disabled.
 *      By default trace cabailities are disabled and may affect the
 *      performance if enabled.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 */
#define PROFILE_ENABLE_TRACE() \
    global_profiler.set_store_trace(true)


/*! \def PROFILE_DISABLE_TRACE()
 *  \brief Disable the trace level timers
 *  \details This will disable the trace capabilites within the timers.
 *      It does not affect the which timers are enabled or disabled.
 *      By default trace cabailities are disabled and may affect the
 *      performance if enabled.
 *      See  \ref AMP::ProfilerApp "ProfilerApp" for more info.
 */
#define PROFILE_DISABLE_TRACE() \
    global_profiler.set_store_trace(false)


/*! @} */

#endif

