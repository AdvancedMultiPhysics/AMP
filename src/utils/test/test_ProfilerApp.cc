#include "utils/AMPManager.h"
#include "utils/ProfilerApp.h"

#include <string>
#include <vector>

using namespace AMP;

#ifdef USE_WINDOWS
    #define TIME_TYPE LARGE_INTEGER
    #define get_time(x) QueryPerformanceCounter(x)
    #define get_diff(start,end,f) (((double)(end.QuadPart-start.QuadPart))/((double)f.QuadPart))
    #define get_frequency(f) QueryPerformanceFrequency(f)
#elif defined(USE_LINUX) || defined(USE_MAC)
    #define TIME_TYPE timeval
    #define get_time(x) gettimeofday(x,NULL);
    #define get_diff(start,end,f) (((double)end.tv_sec-start.tv_sec)+1e-6*((double)end.tv_usec-start.tv_usec))
    #define get_frequency(f) (*f=timeval())
#else
    #error Unknown OS
#endif


bool call_recursive_scope( int N, int i=0 ) 
{
    char name[10];
    sprintf(name,"scoped-%i",i+1);
    bool pass = !global_profiler.active(name,__FILE__);
    PROFILE_SCOPED(timer,"scoped");
    pass = pass && global_profiler.active(name,__FILE__);
    if ( N > 0 )
        pass = pass && call_recursive_scope( --N, ++i );
    sprintf(name,"scoped-%i",i+2);
    pass = pass && !global_profiler.active(name,__FILE__);
    return pass;
}


int run_tests( bool enable_trace, std::string save_name ) 
{
    PROFILE_ENABLE();
    PROFILE_SYNCRONIZE();
    if ( enable_trace ) {
        PROFILE_ENABLE_TRACE();
        PROFILE_ENABLE_MEMORY();
    }
    PROFILE_START("MAIN");

    const int N_it = 100;
    const int N_timers = 1000;
    int N_errors = 0;
    int rank = AMP::AMP_MPI(AMP_COMM_WORLD).getRank();

    // Check that "MAIN" is active and "NULL" is not
    bool test1 = global_profiler.active("MAIN",__FILE__);
    bool test2 = global_profiler.active("NULL",__FILE__);
    if ( !test1 || test2 ) {
        std::cout << "Correct timers are not active\n";
        N_errors++;
    }

    // Test the scoped timer
    bool pass = call_recursive_scope( 5 );
    if ( !pass ) {
        std::cout << "Scoped timer fails\n";
        N_errors++;
    }

    // Get a list of timer names
    std::vector<std::string> names(N_timers);
    std::vector<size_t> ids(N_timers);
    for (int i=0; i<N_timers; i++) {
        char tmp[16];
        sprintf(tmp,"%04i",i);
        names[i] = std::string(tmp);
        ids[i] = ProfilerApp::get_timer_id(names[i].c_str(),__FILE__);
    }

    // Check that the start/stop command fail when they should
    try {   // Check basic start/stop
        PROFILE_START("dummy1"); 
        PROFILE_STOP("dummy1"); 
    } catch (... ) {
        N_errors++;
    }
    try {   // Check stop call before start
        PROFILE_STOP("dummy2"); 
        N_errors++;
    } catch (... ) {
    }
    try {   // Check multiple calls to start
        PROFILE_START("dummy3"); 
        PROFILE_START2("dummy3"); 
        N_errors++;
    } catch (... ) {
        PROFILE_STOP("dummy3"); 
    }
    try {   // Check multiple calls to start with different line numbers
        PROFILE_START("dummy1");
        N_errors++;
    } catch (... ) {
    }

    // Check the performance
    for (int i=0; i<N_it; i++) {
        // Test how long it takes to get the time
        PROFILE_START("gettime");
        TIME_TYPE time1;
        for (int j=0; j<N_timers; j++)
            get_time(&time1);
        PROFILE_STOP("gettime");
        // Test how long it takes to start/stop the timers
        PROFILE_START("level 0");
        for (int j=0; j<N_timers; j++) {
            global_profiler.start( names[j], __FILE__, __LINE__, 0, ids[j] );
            global_profiler.stop(  names[j], __FILE__, __LINE__, 0, ids[j] );
        }
        PROFILE_STOP("level 0");
        PROFILE_START("level 1");
        for (int j=0; j<N_timers; j++) {
            global_profiler.start( names[j], __FILE__, __LINE__, 1, ids[j] );
            global_profiler.stop(  names[j], __FILE__, __LINE__, 1, ids[j] );
        }
        PROFILE_STOP("level 1");
        // Test the memory around allocations
        PROFILE_START("allocate1");
        PROFILE_START("allocate2");
        double *tmp = new double[5000000];
        NULL_USE(tmp);
        PROFILE_STOP("allocate2");
        delete [] tmp;
        PROFILE_START("allocate3");
        tmp = new double[100000];
        NULL_USE(tmp);
        PROFILE_STOP("allocate3");
        delete [] tmp;
        PROFILE_STOP("allocate1");
    }

    // Profile the save
    PROFILE_START("SAVE");
    PROFILE_SAVE(save_name);
    PROFILE_STOP("SAVE");

    // Stop main
    PROFILE_STOP("MAIN");

    // Re-save the results
    PROFILE_SAVE(save_name);

    // Get the timers (sorting based on the timer ids)
    std::vector<TimerResults> data1 = global_profiler.getTimerResults();
    MemoryResults memory1 = global_profiler.getMemoryResults();
    size_t bytes1[2]={0,0};
    std::vector<id_struct> id1(data1.size());
    for (size_t i=0; i<data1.size(); i++) {
        bytes1[0] += data1[i].size(false);
        bytes1[1] += data1[i].size(true);
        id1[i] = data1[i].id;
    }
    Utilities::quicksort(id1,data1);

    // Load the data from the file (sorting based on the timer ids)
    PROFILE_START("LOAD");
    TimerMemoryResults load_results = ProfilerApp::load(save_name,rank);
    std::vector<TimerResults>& data2 = load_results.timers;
    MemoryResults memory2;
    if ( !load_results.memory.empty() )
        memory2 = load_results.memory[0];
    PROFILE_STOP("LOAD");
    size_t bytes2[2]={0,0};
    std::vector<id_struct> id2(data1.size());
    for (size_t i=0; i<data2.size(); i++) {
        bytes2[0] += data2[i].size(false);
        bytes2[1] += data2[i].size(true);
        id2[i] = data2[i].id;
    }
    Utilities::quicksort(id2,data2);

    // Compare the sets of timers
    bool error = false;
    if ( data1.size()!=data2.size() || bytes1[0]==0 || bytes1[0]!=bytes2[0] || bytes1[1]!=bytes2[1] ) {
        error = true;
    } else {
        for (size_t i=0; i<data1[i].trace.size(); i++) {
            if ( data1[i].id!=data2[i].id || data1[i].trace.size()!=data2[i].trace.size() ) {
                error = true;
            } else {
                for (size_t j=0; j<data1[i].trace.size(); j++) {
                    if ( data1[i].trace[j].id!=data2[i].trace[j].id )
                        error = true;
                }
            }
        }
    }
    if ( error ) {
        std::cout << "Timers do not match " << data1.size() << " " << data2.size() << 
            " " << bytes1[0] << " " << bytes2[0] << " " << bytes1[1] << " " << bytes2[1] << " " << std::endl;
        N_errors++;
    }

    // Compare the memory results
    error = false;
    if ( memory1.time.size()!=memory2.time.size() ) {
        error = true;
    } else {
        for (size_t i=0; i<memory1.time.size(); i++) {
            if ( memory1.time[i]!=memory2.time[i] || memory1.bytes[i]!=memory2.bytes[i] )
                error = true;
        }
    }
    if ( error ) {
        std::cout << "Memory trace does not match\n";
        N_errors++;
    }

    PROFILE_SAVE(save_name);
    PROFILE_SAVE(save_name,true);
    return N_errors;
}


int main(int argc, char* argv[])
{
    // Initialize AMP
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup(argc,argv,startup_properties);
    
    // Run the tests
    int N_errors=0;
    N_errors += run_tests( false, "test_ProfilerApp" );
    PROFILE_DISABLE();
    N_errors += run_tests( true, "test_ProfilerApp-trace" );
    
    // Finalize AMP
    if ( N_errors==0 ) 
        std::cout << "All tests passed" << std::endl;
    else
        std::cout << "Some tests failed" << std::endl;
    AMP::AMPManager::shutdown();
    return(N_errors);
}



