#include "utils/ProfilerApp.h"
#include "utils/Utilities.h"
#include "utils/AMP_MPI.h"

#include <stdio.h>
#include <iostream>
#include <sstream>

#define ERROR_MSG AMP_ERROR

#define MONITOR_PROFILER_PERFORMANCE 0


AMP::ProfilerApp global_profiler = AMP::ProfilerApp();

extern "C" {
    #include "assert.h"
}

#ifdef USE_WINDOWS
    #define get_time(x) QueryPerformanceCounter(x)
    #define get_diff(start,end,f) (((double)(end.QuadPart-start.QuadPart))/((double)f.QuadPart))
    #define get_frequency(f) QueryPerformanceFrequency(f)
#elif defined(USE_LINUX)
    #define get_time(x) gettimeofday(x,NULL);
    #define get_diff(start,end,f) (((double)end.tv_sec-start.tv_sec)+1e-6*((double)end.tv_usec-start.tv_usec))
    #define get_frequency(f) (*f=timeval())
#else
    #error Unknown OS
#endif

namespace AMP {

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

template <class type_a, class type_b>
static inline void quicksort2(int n, type_a *arr, type_b *brr);

#if MONITOR_PROFILER_PERFORMANCE==1
    double total_start_time = 0;
    double total_stop_time = 0;
    double total_block_time = 0;
    double total_thread_time = 0;
    double total_trace_id_time = 0;
#endif


// Inline function to get the current time/date string (without the newline character)
static inline std::string getDateString() {
    time_t rawtime;
    time ( &rawtime );
    std::string tmp(ctime(&rawtime));
    return tmp.substr(0,tmp.length()-1);
}


/******************************************************************
* Some inline functions to acquire/release a mutex                *
******************************************************************/
#ifdef USE_WINDOWS
    static inline bool GET_LOCK(HANDLE *lock) {
        int retval = WaitForSingleObject(*lock,INFINITE);
        if ( retval != WAIT_OBJECT_0 ) {
            printf("Error locking mutex\n");
            return true;
        }
	    return false;
    }
    static inline bool RELEASE_LOCK(HANDLE *lock) {
        int retval = ReleaseMutex(*lock);
        if ( retval == 0 ) {
            printf("Error unlocking mutex\n");
            return true;
        }
    	return false;
    }
#else
    static inline bool GET_LOCK(pthread_mutex_t *lock) {
        int retval = pthread_mutex_lock(lock);
        if ( retval == -1 ) {
            printf("Error locking mutex\n");
            return true;
        }
	    return false;
    }
    static inline bool RELEASE_LOCK(pthread_mutex_t *lock) {
        int retval = pthread_mutex_unlock(lock);
        if ( retval == -1 ) {
            printf("Error unlocking mutex\n");
            return true;
        }
	    return false;
    }
#endif


/***********************************************************************
* Inline functions to set or unset the ith bit of the bit array trace  *
***********************************************************************/
static inline void set_trace_bit( unsigned int i, unsigned int N, BIT_WORD *trace ) {
    unsigned int N_bits = 8*sizeof(BIT_WORD);
    unsigned int j = i/N_bits;
    unsigned int k = i%N_bits;
    BIT_WORD mask = ((BIT_WORD)0x1)<<k;
    if ( i < N*N_bits )
        trace[j] |= mask;
}
static inline void unset_trace_bit( unsigned int i, unsigned int N, BIT_WORD *trace ) {
    unsigned int N_bits = 8*sizeof(BIT_WORD);
    unsigned int j = i/N_bits;
    unsigned int k = i%N_bits;
    BIT_WORD mask = ((BIT_WORD)0x1)<<k;
    if ( i < N*N_bits )
        trace[j] &= ~mask;
}


/***********************************************************************
* Inline function to convert the timer id to a string                  *
***********************************************************************/
#define N_BITS_ID 24    // The probability of a collision is ~N^2/2^N_bits (N is the number of timers)
static inline void convert_timer_id( size_t key, char* str ) {
    int N_bits = MIN(N_BITS_ID,8*sizeof(unsigned int));
    // Get a new key that is representable by N bits
    size_t id = key;
    if ( N_BITS_ID < 8*sizeof(size_t) ) {
        if ( sizeof(size_t)==4 )
            id = (key*0x9E3779B9) >> (32-N_BITS_ID);
        else if ( sizeof(size_t)==8 )
            id = (key*0x9E3779B97F4A7C15) >> (64-N_BITS_ID);
        else
            ERROR_MSG("Unhandled case");
    }
    // Convert the new key to a string
    if ( N_bits <= 9 ) {
        // The id is < 512, store it as a 3-digit number        
        sprintf(str,"%03u",static_cast<unsigned int>(id));
    } else if ( N_bits <= 16 ) {
        // The id is < 2^16, store it as a 4-digit hex
        sprintf(str,"%04x",static_cast<unsigned int>(id));
    } else {
        // We will store the id use the 64 character set { 0-9 a-z A-Z & $ }
        int N = MAX(4,(N_bits+5)/6);    // The number of digits we need to use
        size_t tmp1 = id;
        for (int i=N-1; i>=0; i--) {
            unsigned char tmp2 = tmp1%64;
            tmp1 /= 64;
            if ( tmp2 < 10 )
                str[i] = tmp2+48;
            else if ( tmp2 < 36 )
                str[i] = tmp2+(97-10);
            else if ( tmp2 < 62 )
                str[i] = tmp2+(65-36);
            else if ( tmp2 < 63 )
                str[i] = '&';
            else if ( tmp2 < 64 )
                str[i] = '$';
            else
                str[i] = 0;   // We should never use this character
        }
        str[N] = 0;            
    }
}

/***********************************************************************
* Consructor                                                           *
***********************************************************************/
ProfilerApp::ProfilerApp() {
    if ( sizeof(BIT_WORD)%sizeof(size_t) )
        ERROR_MSG("sizeof(BIT_WORD) must be a product of sizeof(size_t)\n");
    get_frequency( &frequency );
    #ifdef USE_WINDOWS
        lock = CreateMutex (NULL, FALSE, NULL);
    #elif defined(USE_LINUX)
        pthread_mutex_init (&lock,NULL);
    #endif
    for (int i=0; i<THREAD_HASH_SIZE; i++)
        thread_head[i] = NULL;
    for (int i=0; i<TIMER_HASH_SIZE; i++)
        timer_table[i] = NULL;
    get_time(&construct_time);
    N_threads = 0;
    N_timers = 0;
    d_level = 0;
    store_trace_data = false;
}
void ProfilerApp::set_store_trace( bool profile ) { 
    if ( N_timers==0 ) 
        store_trace_data=profile;
    else
        ERROR_MSG("Cannot change trace status after a timer is started\n");
}


/***********************************************************************
* Deconsructor                                                         *
***********************************************************************/
ProfilerApp::~ProfilerApp() {
    // Delete the thread structures
    for (int i=0; i<THREAD_HASH_SIZE; i++) {
        volatile thread_info *thread = thread_head[i];
        while ( thread != NULL ) {
            // Delete the timers in the thread
            for (int j=0; j<TIMER_HASH_SIZE; j++) {
                store_timer *timer = thread->head[j];
                while ( timer != NULL ) {
                    // Delete the trace logs
                    store_trace *trace = timer->trace_head;
                    while ( trace != NULL ) {
                        store_trace *trace_tmp = trace;
                        trace = trace->next;
                        delete trace_tmp;
                    }
                    store_timer *tmp = timer;
                    timer = timer->next;
                    delete tmp;
                }
            }
            volatile thread_info *thread_next = thread->next;
            delete thread;
            thread = thread_next;
        }
        thread_head[i] = NULL;
    }
    // Delete the global timer info
    for (int i=0; i<TIMER_HASH_SIZE; i++) {
        volatile store_timer_data_info *timer = timer_table[i];
        while ( timer != NULL ) {
            volatile store_timer_data_info *timer_next = timer->next;
            delete timer;
            timer = timer_next;
        }
        timer_table[i] = NULL;
    }
}


/***********************************************************************
* Function to start profiling a block of code                          *
***********************************************************************/
void ProfilerApp::start( const std::string& message, const char* filename, const int line, const int level ) {
    if ( level<0 || level>=128 )
        ERROR_MSG("level must be in the range 0-127");
    if ( this->d_level<level )
        return;
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE start_time_local;
        get_time(&start_time_local);
    #endif
    // Get the thread data
    thread_info* thread_data = get_thread_data();
    // Get the appropriate timer
    store_timer* timer = get_block(thread_data,message.c_str(),filename,line,-1);
    if ( timer == NULL )
        ERROR_MSG("Failed to get the appropriate timer");
    if ( timer->is_active ) {
        std::stringstream msg;
        msg << "Timer is already active, did you forget to call stop? (" << message << " in " << filename << " at line " << line << ")\n";
        ERROR_MSG(msg.str());
    }
    // Start the timer 
    memcpy(timer->trace,thread_data->active,TRACE_SIZE*sizeof(BIT_WORD));
    timer->is_active = true;
    timer->N_calls++;
    set_trace_bit(timer->trace_index,TRACE_SIZE,thread_data->active);
    get_time(&timer->start_time);
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE stop_time_local;
        get_time(&stop_time_local);
        total_start_time += get_diff(start_time_local,stop_time_local,frequency);
    #endif
}


/***********************************************************************
* Function to stop profiling a block of code                           *
***********************************************************************/
void ProfilerApp::stop( const std::string& message, const char* filename, const int line, const int level ) {
    if ( level<0 || level>=128 )
        ERROR_MSG("level must be in the range 0-127");
    if ( this->d_level<level )
        return;
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE start_time_local;
        get_time(&start_time_local);
    #endif
    // Use the current time (minimize the effects of the overhead of the timer)
    TIME_TYPE end_time;
    get_time(&end_time);
    // Get the thread data
    thread_info* thread_data = get_thread_data();
    // Get the appropriate timer
    store_timer* timer = get_block(thread_data,message.c_str(),filename,-1,line);
    if ( timer == NULL )
        ERROR_MSG("Failed to get the appropriate timer");
    if ( !timer->is_active ) {
        std::stringstream msg;
        msg << "Timer is not active, did you forget to call start? (" << message << " in " << filename << " at line " << line << ")\n";
        ERROR_MSG(msg.str());
    }
    timer->is_active = false;
    // Update the active trace log
    unset_trace_bit(timer->trace_index,TRACE_SIZE,thread_data->active );
    // The timer is only a calling timer if it was active before and after the current timer
    BIT_WORD active[TRACE_SIZE];
    for (size_t i=TRACE_SIZE; i-- >0;)
        active[i] = thread_data->active[i] & timer->trace[i];
    size_t trace_id = get_trace_id( TRACE_SIZE, active );
    // Find the trace to save
    store_trace *trace = timer->trace_head;
    while ( trace != NULL) {
        if ( trace_id==trace->id )
            break;
        trace = trace->next;
    }
    if ( trace == NULL ) {
        trace = new store_trace;
        memcpy(trace->trace,active,TRACE_SIZE*sizeof(BIT_WORD));
        trace->id = trace_id;
        if ( timer->trace_head == NULL ) {
            timer->trace_head = trace;
        } else {
            store_trace *trace_list = timer->trace_head;
            while ( trace_list->next != NULL)
                trace_list = trace_list->next;
            trace_list->next = trace;
        }
    }
    // Calculate the time elapsed since start was called
    double time = get_diff(timer->start_time,end_time,frequency);
    // Save the starting and ending time if we are storing the detailed traces
    if ( store_trace_data && trace->N_calls<MAX_TRACE_TRACE) {
        // Check if we need to allocate more memory to store the times
        size_t size_old, size_new;
        size_t N = trace->N_calls;
        if ( trace->start_time==NULL ) {
            // We haven't allocated any memory yet
            size_old = 0;
            size_new = 1;
        } else {
            // We want to allocate memory in powers of 2
            // The current allocated size is the smallest power of 2 that is >= N
            size_old = 1;
            while ( size_old < N )
                size_old *= 2;
            // Double the storage space (if needed)
            if ( N == size_old )
                size_new = 2*size_old;
            else
                size_new = size_old;
            // Stop allocating memory if we reached the limit
            if ( size_new > MAX_TRACE_TRACE ) 
                size_new = MAX_TRACE_TRACE;
            if ( size_old > MAX_TRACE_TRACE ) 
                size_old = MAX_TRACE_TRACE;
        }
        if ( size_old != size_new ) {
            // Expand the trace list
            double *tmp_s = new double[size_new];
            double *tmp_e = new double[size_new];
            for (size_t i=0; i<size_old; i++) {
                tmp_s[i] = trace->start_time[i];
                tmp_e[i] = trace->end_time[i];
            }
            if ( trace->start_time!=NULL ) {
                delete [] trace->start_time;
                delete [] trace->end_time;
            }
            trace->start_time = tmp_s;
            trace->end_time = tmp_e;
        }
        // Calculate the time elapsed since the profiler was created
        trace->start_time[N] = get_diff(construct_time,timer->start_time,frequency);
        trace->end_time[N] = get_diff(construct_time,end_time,frequency);
    }
    // Save the minimum, maximum, and total times
    if ( timer->N_calls == 1 ) {
        timer->min_time = time;
        timer->max_time = time;
    } else {
        timer->max_time = MAX(timer->max_time,time);
        timer->min_time = MIN(timer->min_time,time);
    }
    timer->total_time += time;
    // Save the new time info to the trace
    if ( trace->N_calls == 0 ) {
        trace->min_time = time;
        trace->max_time = time;
    } else {
        trace->max_time = MAX(trace->max_time,time);
        trace->min_time = MIN(trace->min_time,time);
    }
    trace->total_time += time;
    trace->N_calls++;
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE stop_time_local;
        get_time(&stop_time_local);
        total_stop_time += get_diff(start_time_local,stop_time_local,frequency);
    #endif
}


/***********************************************************************
* Function to enable/disable the timers                                *
***********************************************************************/
void ProfilerApp::enable( int level )
{
    // This is a blocking function so it cannot be called at the same time as disable
    if ( level<0 || level>=128 )
        ERROR_MSG("level must be in the range 0-127");
    GET_LOCK(&lock);
    d_level = level;
    RELEASE_LOCK(&lock);
}
void ProfilerApp::disable( )
{
    // First, change the status flag
    GET_LOCK(&lock);
    d_level = -1;
    // Stop ALL timers
    TIME_TYPE end_time;
    get_time(&end_time);
    // Loop through the threads
    for (int i=0; i<THREAD_HASH_SIZE; i++) {
        volatile thread_info *thread = thread_head[i];
        while ( thread != NULL ) {
            // Disable clear the trace log
            for (int j=0; j<TRACE_SIZE; j++)
                thread->active[j] = 0;
            // Delete the active timers
            for (int j=0; j<TIMER_HASH_SIZE; j++) {
                store_timer *timer = thread->head[j];
                while ( timer != NULL ) {
                    store_timer *timer2 = timer->next;
                    delete timer;
                    timer = timer2;
                }
                thread->head[j] = NULL;
            }
            volatile thread_info *thread_next = thread->next;
            thread = thread_next;
        }
    }
    RELEASE_LOCK(&lock);
}


/***********************************************************************
* Function to save the profiling info                                  *
***********************************************************************/
void ProfilerApp::save( const std::string& filename ) {
    if ( this->d_level<0 ) {
        printf("Warning: Timers are not enabled, no data will be saved\n");
        return;
    }
    AMP::AMP_MPI global_comm(AMP_COMM_WORLD);
    int N_procs = global_comm.getSize();
    int rank = global_comm.getRank();
    // Get the current time in case we need to "stop" and timers
    TIME_TYPE end_time;
    get_time(&end_time);
    // Get the mutex for thread safety (we don't want the list changing while we are saving the data)
    // Note: Because we don't block for most operations in the timer, this is not full proof, but should help
    bool error = GET_LOCK(&lock);
    if ( error )
        return;
    // Get the thread specific data for each thread
    int N_threads2 = N_threads;     // Cache the number of threads since we are holing the lock
    thread_info **thread_data = new thread_info*[N_threads2];
    for (int i=0; i<N_threads2; i++)
        thread_data[i] = NULL;
    for (int i=0; i<THREAD_HASH_SIZE; i++) {
        thread_info *ptr = const_cast<thread_info*>(thread_head[i]);  // It is safe to case to a non-volatile object since we hold the lock
        while ( ptr != NULL ) {
            if ( ptr->thread_num >= N_threads2 )
                ERROR_MSG("Internal error (1)");
            if ( thread_data[ptr->thread_num] != NULL )
                ERROR_MSG("Internal error (2)");
            thread_data[ptr->thread_num] = ptr;
            ptr = const_cast<thread_info*>(ptr->next);    // It is safe to case to a non-volatile object since we hold the lock
        }
    }
    for (int i=0; i<N_threads2; i++) {
        if ( thread_data[i] == NULL ) {
            delete [] thread_data;
            RELEASE_LOCK(&lock);
            ERROR_MSG("Internal error (3)");
        }
    }
    // Get the timer ids and sort the ids by the total time (maximum value for each thread) to create a global order to save the results
    size_t *id_order = new size_t[N_timers];
    double *total_time = new double[N_timers];
    for (int i=0; i<N_timers; i++)
        total_time[i] = 0.0;
    int k = 0;
    for (int i=0; i<TIMER_HASH_SIZE; i++) {
        store_timer_data_info *timer_global = const_cast<store_timer_data_info*>(timer_table[i]);
        while ( timer_global!=NULL ) {
            id_order[k] = timer_global->id;
            store_timer* timer = NULL;
            for (int thread_id=0; thread_id<N_threads2; thread_id++) {
                thread_info *head = thread_data[thread_id];
                // Search for a timer that matches the current id, and save it
                unsigned int key = get_timer_hash( id_order[k] );
                timer = head->head[key];
                while ( timer != NULL ) {
                    if ( timer->id == id_order[k] )
                        break;
                    timer = timer->next;
                }
                if ( timer!=NULL ) {
                    // Get the total running time of the timer
                    total_time[k] = MAX(total_time[k],timer->total_time);
                    // If the timer is still running, add the current processing to the totals
                    if ( timer->is_active ) {
                        double time = get_diff(timer->start_time,end_time,frequency);
                        total_time[k] += time;
                    }
                }
            }
            k++;
            timer_global = const_cast<store_timer_data_info*>(timer_global->next);
        }
    }
    if ( k!=N_timers )
        ERROR_MSG("Not all timers were found");
    quicksort2(N_timers,total_time,id_order);
    delete [] total_time;
    // Open the file(s) for writing
    char filename_timer[1000], filename_trace[1000];
    sprintf(filename_timer,"%s.%i.timer",filename.c_str(),rank+1);
    sprintf(filename_trace,"%s.%i.trace",filename.c_str(),rank+1);
    FILE *timerFile = fopen(filename_timer,"wb");
    if ( timerFile == NULL ) {
        printf("Error opening file for writing (timer)");
        delete [] thread_data;
        delete [] id_order;
        RELEASE_LOCK(&lock);
        return;
    }
    FILE *traceFile = NULL;
    if ( store_trace_data ) {
        traceFile = fopen(filename_trace,"wb");
        if ( traceFile == NULL ) {
            printf("Error opening file for writing (trace)");
            delete [] thread_data;
            delete [] id_order;
            fclose(timerFile);
            RELEASE_LOCK(&lock);
            return;
        }
    }
    // Create the file header
    fprintf(timerFile,"                  Message                    Filename        Thread  Start Line  Stop Line  N_calls  Min Time  Max Time  Total Time\n");
    fprintf(timerFile,"-----------------------------------------------------------------------------------------------------------------------------------\n");
    // Loop through the list of timers, storing the most expensive first
    for (int i=N_timers-1; i>=0; i--) {
        size_t id = id_order[i];                    // Get the timer id
        unsigned int key = get_timer_hash( id );    // Get the timer hash key
        // Search for the global timer info
        store_timer_data_info *timer_global = const_cast<store_timer_data_info*>(timer_table[key]);
        while ( timer_global!=NULL ) {
            if ( timer_global->id == id ) 
                break;
            timer_global = const_cast<store_timer_data_info*>(timer_global->next);
        }
        if ( timer_global==NULL ) {
            delete [] thread_data;
            delete [] id_order;
            fclose(timerFile);
            if ( traceFile!=NULL)
                fclose(traceFile);
            RELEASE_LOCK(&lock);
            ERROR_MSG("Internal error");
        }
        const char* filename2 = timer_global->filename.c_str();
        const char* message = timer_global->message.c_str();
        int start_line = timer_global->start_line;
        int stop_line = timer_global->stop_line;
        // Loop through the thread entries
        for (int thread_id=0; thread_id<N_threads2; thread_id++) {
            thread_info *head = thread_data[thread_id];
            // Search for a timer that matches the current id
            store_timer* timer = head->head[key];
            while ( timer != NULL ) {
                if ( timer->id == id )
                    break;
                timer = timer->next;
            }
            if ( timer==NULL ) {
                // The current thread does not have a copy of this timer, move on
                continue;
            }
            // Get the running times of the timer
            double min_time = timer->min_time;
            double max_time = timer->max_time;
            double tot_time = timer->total_time;
            // If the timer is still running, add the current processing to the totals
            if ( timer->is_active ) {
                double time = get_diff(timer->start_time,end_time,frequency);
                if ( tot_time == 0.0 ) { 
                    min_time = time;
                    max_time = time;
                    tot_time = time;
                } else {
                    min_time = MIN(min_time,time);
                    max_time = MAX(max_time,time);
                    tot_time += time;
                }
            }
            // Save the timer to the file
            fprintf(timerFile,"%30s  %26s   %4i   %7i    %7i  %8i     %8.3f  %8.3f  %10.3f\n",
                message,filename2,thread_id,start_line,stop_line,timer->N_calls,min_time,max_time,tot_time);
            timer = timer->next;
        }
    }
    // Loop through all of the entries, saving the detailed data and the trace logs
    fprintf(timerFile,"\n\n");
    fprintf(timerFile,"<N_procs=%i,id=%i",N_procs,rank);
    if ( store_trace_data )
        fprintf(timerFile,",trace_file=%s",filename_trace);
    fprintf(timerFile,",date='%s'>\n",getDateString().c_str());
    get_time(&end_time);
    char id_str[16];
    // Loop through the list of timers, storing the most expensive first
    for (int i=N_timers-1; i>=0; i--) {
        size_t id = id_order[i];                    // Get the timer id
        unsigned int key = get_timer_hash( id );    // Get the timer hash key
        // Search for the global timer info
        store_timer_data_info *timer_global = const_cast<store_timer_data_info*>(timer_table[key]);
        while ( timer_global!=NULL ) {
            if ( timer_global->id == id ) 
                break;
            timer_global = const_cast<store_timer_data_info*>(timer_global->next);
        }
        if ( timer_global==NULL ) {
            delete [] thread_data;
            delete [] id_order;
            fclose(timerFile);
            if ( traceFile!=NULL)
                fclose(traceFile);
            RELEASE_LOCK(&lock);
            ERROR_MSG("Internal error");
        }
        const char* filename2 = timer_global->filename.c_str();
        const char* message = timer_global->message.c_str();
        int start_line = timer_global->start_line;
        int stop_line = timer_global->stop_line;
        // Loop through the thread entries
        for (int thread_id=0; thread_id<N_threads2; thread_id++) {
            thread_info *head = thread_data[thread_id];
            // Search for a timer that matches the current id
            store_timer* timer = head->head[key];
            while ( timer != NULL ) {
                if ( timer->id == id )
                    break;
                timer = timer->next;
            }
            if ( timer==NULL ) {
                // The current thread does not have a copy of this timer, move on
                continue;
            }
            // Get the running times of the timer
            double min_time = timer->min_time;
            double max_time = timer->max_time;
            double tot_time = timer->total_time;
            // If the timer is still running, add the current processing time to the totals
            bool add_trace = false;
            double time = 0.0;
            size_t trace_id = 0;
            BIT_WORD active[TRACE_SIZE];
            if ( timer->is_active ) {
                add_trace = true;
                time = get_diff(timer->start_time,end_time,frequency);
                min_time = MIN(min_time,time);
                max_time = MAX(min_time,time);
                tot_time += time;
                // The timer is only a calling timer if it was active before and after the current timer
                for (size_t i=0; i<TRACE_SIZE; i++)
                    active[i] = head->active[i] & timer->trace[i];
                unset_trace_bit(timer->trace_index,TRACE_SIZE,active);
                trace_id = get_trace_id( TRACE_SIZE, active );
            }
            // Save the timer info
            convert_timer_id(id,id_str);
            fprintf(timerFile,"<timer:id=%s,message=%s,file=%s,thread=%i,start=%i,stop=%i,N=%i,min=%e,max=%e,tot=%e>\n",
                id_str,message,filename2,thread_id,start_line,stop_line,timer->N_calls,min_time,max_time,tot_time);
            // Store each trace
            store_trace *trace = timer->trace_head;
            while ( trace != NULL ) {
                // Get the running times of the trace
                double trace_min_time = trace->min_time;
                double trace_max_time = trace->max_time;
                double trace_tot_time = trace->total_time;
                // Determine if we need to add the running trace
                if ( add_trace ) {
                    if ( trace_id == trace->id ) {
                        trace_min_time = MIN(trace_min_time,time);
                        trace_max_time = MAX(trace_min_time,time);
                        trace_tot_time += time;
                        add_trace = false;
                    }
                }
                // Save the trace results
                convert_timer_id(id,id_str);
                std::string active_list = get_active_list( trace->trace, timer->trace_index, head );
                fprintf(timerFile,"<trace:id=%s,thread=%i,N=%i,min=%e,max=%e,tot=%e,active=%s>\n",
                    id_str,thread_id,trace->N_calls,trace_min_time,trace_max_time,trace_tot_time,active_list.c_str());
                // Save the detailed trace results (this is a binary file)
                if ( store_trace_data ) { 
                    convert_timer_id(id,id_str);
                    fprintf(traceFile,"id=%s,thread=%i,active=%s,N=%i:",id_str,thread_id,active_list.c_str(),trace->N_calls);
                    fwrite(trace->start_time,sizeof(double),trace->N_calls,traceFile);
                    fwrite(trace->end_time,sizeof(double),trace->N_calls,traceFile);
                    fprintf(traceFile,"\n");
                }
                // Advance to the next trace
                trace = trace->next;
            }
            // Create a new trace if necessary
            if ( add_trace ) { 
                convert_timer_id(id,id_str);
                std::string active_list = get_active_list( active, timer->trace_index, head );
                fprintf(timerFile,"<trace:id=%s,thread=%i,N=%i,min=%e,max=%e,tot=%e,active=%s>\n",
                    id_str,thread_id,1,time,time,time,active_list.c_str());
                // Save the detailed trace results (this is a binary file)
                if ( store_trace_data ) { 
                    double start_time_trace = time = get_diff(construct_time,timer->start_time,frequency);
                    double end_time_trace = time = get_diff(construct_time,end_time,frequency);
                    convert_timer_id(id,id_str);
                    fprintf(traceFile,"id=%s,thread=%i,active=%s,N=%i:",id_str,thread_id,active_list.c_str(),1);
                    fwrite(&start_time_trace,sizeof(double),1,traceFile);
                    fwrite(&end_time_trace,sizeof(double),1,traceFile);
                    fprintf(traceFile,"\n");
                }
            }
        }
    }
    // Close the file(s)
    fclose(timerFile);
    if ( traceFile!=NULL)
        fclose(traceFile);
    // Free temporary memory
    delete [] thread_data;
    delete [] id_order;
    // Release the mutex
    RELEASE_LOCK(&lock);
    #if MONITOR_PROFILER_PERFORMANCE==1
        printf("start = %e, stop = %e, block = %e, thread = %e, trace_id = %e\n",
            total_start_time,total_stop_time,total_block_time,total_thread_time,total_trace_id_time);
    #endif
}


/***********************************************************************
* Function to get the list of active timers                            *
***********************************************************************/
std::string ProfilerApp::get_active_list( BIT_WORD *active, unsigned int myIndex, thread_info *head )
{
    char id_str[16];
    std::string active_list = "[";
    unsigned int BIT_WORD_size = 8*sizeof(BIT_WORD);
    for (unsigned int i=0; i<TRACE_SIZE; i++) {
        for (unsigned int j=0; j<BIT_WORD_size; j++) {
            unsigned int k = i*BIT_WORD_size + j;
            if ( k == myIndex )
                continue;
            BIT_WORD mask = ((BIT_WORD)0x1)<<j;
            if ( (mask&active[i])!=0 ) {
                // The kth timer is active, find the index and write it to the file
                store_timer* timer_tmp = NULL;
                for (int m=0; m<TIMER_HASH_SIZE; m++) {
                    timer_tmp = head->head[m];
                    while ( timer_tmp!=NULL ) {
                        if ( timer_tmp->trace_index==k )
                            break;
                        timer_tmp = timer_tmp->next;
                    }
                    if ( timer_tmp!=NULL )
                        break;
                }
                if ( timer_tmp==NULL )
                    ERROR_MSG("Internal Error");
                convert_timer_id(timer_tmp->id,id_str);
                active_list += " " + std::string(id_str);
            }
        }
    }
    active_list += " ]";
    return active_list;
}


/************************************************************************
* Function to get the data for the current thread                       *
* Note:  If a thread has called this function at some time in the past  *
* then it will be able to return without blocking. When a thread enters *
*  this function for the first time then it will block as necessary.    *
***********************************************************************/
ProfilerApp::thread_info* ProfilerApp::get_thread_data( ) 
{
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE start_time_local;
        get_time(&start_time_local);
    #endif
    // Get the thread id (as an integer)
    #ifdef USE_WINDOWS
        DWORD tmp_thread_id = GetCurrentThreadId();
        size_t thread_id = (size_t) tmp_thread_id;
    #elif defined(USE_LINUX)
        pthread_t tmp_thread_id = pthread_self();
        size_t thread_id = (size_t) tmp_thread_id;
    #endif
    // Get the hash key for the thread
    unsigned int key = get_thread_hash( thread_id );
    // Find the first entry with the given key (creating one if necessary)
    if ( thread_head[key]==NULL ) {
        // The entry in the hash table is empty
        // Acquire the lock
        bool error = GET_LOCK(&lock);
        if ( error )
            return NULL;
        // Check if the entry is still NULL
        if ( thread_head[key]==NULL ) {
            // Create a new entry
            thread_head[key] = new thread_info;
            thread_head[key]->id = thread_id;
            thread_head[key]->N_timers = 0;
            thread_head[key]->next = NULL;
            thread_head[key]->thread_num = N_threads;
            N_threads++;
        }
        // Release the lock
        RELEASE_LOCK(&lock);
    }
    volatile thread_info* head = thread_head[key];
    // Find the entry by looking through the list (creating the entry if necessary)
    while ( head->id != thread_id ) {
        // Check if there is another entry to check (and create one if necessary)
        if ( head->next==NULL ) {
            // Acquire the lock
            bool error = GET_LOCK(&lock);
            if ( error )
                return NULL;
            // Check if another thread created an entry while we were waiting for the lock
            if ( head->next==NULL ) {
                // Create a new entry
                thread_info* new_data = new thread_info;
                new_data = new thread_info;
                new_data->id = thread_id;
                new_data->N_timers = 0;
                new_data->next = NULL;
                new_data->thread_num = N_threads;
                N_threads++;
                head->next = new_data;
            }
            // Release the lock
            RELEASE_LOCK(&lock);
        } 
        // Advance to the next entry
        head = head->next;
    }
    // Return the pointer (Note: we no longer need volatile since we are accessing it from the creating thread)
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE stop_time_local;
        get_time(&stop_time_local);
        total_thread_time += get_diff(start_time_local,stop_time_local,frequency);
    #endif
    return const_cast<thread_info*>(head);
}


/***********************************************************************
* Function to get the timmer for a particular block of code            *
* Note: This function performs some blocking as necessary.             *
***********************************************************************/
inline ProfilerApp::store_timer* ProfilerApp::get_block( thread_info *thread_data, 
    const char* message, const char* filename1, const int start, const int stop ) 
{
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE start_time_local;
        get_time(&start_time_local);
    #endif
    // Get the name of the file without the path
    const char *s = filename1;
    int length = 1;
    while(*(++s)) { ++length; }
    const char* filename = filename1;
    for (int i=length-1; i>=0; --i) {
        if ( filename[i]==47 || filename[i]==92 ) {
            filename = &filename[i+1];
            break;
        }
    }
    // Get the id for the timer
    size_t id = get_timer_id(message,filename);
    unsigned int key = get_timer_hash( id );    // Get the hash index
    // Search for the thread-specific timer and create it if necessary (does not need blocking)
    if ( thread_data->head[key]==NULL ) {
        // The timer does not exist, create it
        store_timer *new_timer = new store_timer;
        new_timer->id = id;
        new_timer->is_active = false;
        new_timer->trace_index = thread_data->N_timers;
        thread_data->N_timers++;
        thread_data->head[key] = new_timer;
    }
    store_timer *timer = thread_data->head[key];
    while ( timer->id != id ) {
        // Check if there is another entry to check (and create one if necessary)
        if ( timer->next==NULL ) {
            store_timer *new_timer = new store_timer;
            new_timer->id = id;
            new_timer->is_active = false;
            new_timer->trace_index = thread_data->N_timers;
            thread_data->N_timers++;
            timer->next = new_timer;
        } 
        // Advance to the next entry
        timer = timer->next;
    }
    // Get the global timer info and create if necessary
    store_timer_data_info* global_info = timer->timer_data;
    if ( global_info == NULL ) {
        global_info = get_timer_data( id );
        timer->timer_data = global_info;
        if ( global_info->start_line==-2 ) {
            global_info->start_line = start;
            global_info->stop_line = stop;
            global_info->message = std::string(message);
            global_info->filename = std::string(filename);
        }
    }
    // Check the status of the timer
    if ( start==-1 ) {
        // We either are dealing with a stop statement, or the special case for multiple start lines
    } else if ( global_info->start_line==-1 ) {
        // The timer without a start line, assign it now 
        // Note:  Technically this should be a blocking call, however it is possible to update the start line directly.  
        global_info->start_line = start;
    } else if ( global_info->start_line != start ) {
        // Multiple start lines were detected indicating duplicate timers
        std::stringstream msg;
        msg << "Multiple start calls with the same message are not allowed ("
            << message << " in " << filename << " at lines " << start << ", " << global_info->start_line << ")\n";
        ERROR_MSG(msg.str());
    }
    if ( stop==-1 ) {
        // We either are dealing with a start statement, or the special case for multiple stop lines
    } else if ( global_info->stop_line==-1 ) {
        // The timer without a start line, assign it now (this requires blocking)
        // Note:  Technically this should be a blocking call, however it is possible to update the stop line directly.  
        global_info->stop_line = stop;
    } else if ( global_info->stop_line != stop ) {
        // Multiple start lines were detected indicating duplicate timers
        std::stringstream msg;
        msg << "Multiple start calls with the same message are not allowed ("
            << message << " in " << filename << " at lines " << stop << ", " << global_info->stop_line << ")\n";
        ERROR_MSG(msg.str());
    }
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE stop_time_local;
        get_time(&stop_time_local);
        total_block_time += get_diff(start_time_local,stop_time_local,frequency);
    #endif
    return timer;
}


/***********************************************************************
* Function to return a pointer to the global timer info and create it  *
* if necessary.                                                        *
***********************************************************************/
ProfilerApp::store_timer_data_info* ProfilerApp::get_timer_data( size_t id )
{
    unsigned int key = get_timer_hash( id );    // Get the hash index
    if ( timer_table[key]==NULL ) {
        // The global timer does not exist, create it (requires blocking)
        // Acquire the lock
        bool error = GET_LOCK(&lock);
        if ( error )
            return NULL;
        // Check if the entry is still NULL
        if ( timer_table[key]==NULL ) {
            // Create a new entry
            store_timer_data_info *info_tmp = new store_timer_data_info;
            info_tmp->id = id;
            info_tmp->start_line = -2;
            info_tmp->stop_line = -1;
            info_tmp->next = NULL;
            timer_table[key] = info_tmp;
            N_timers++;
        }
        // Release the lock
        RELEASE_LOCK(&lock);
    }
    volatile store_timer_data_info *info = timer_table[key];
    while ( info->id != id ) {
        // Check if there is another entry to check (and create one if necessary)
        if ( info->next==NULL ) {
            // Acquire the lock
            bool error = GET_LOCK(&lock);
            if ( error )
                return NULL;
            // Check if another thread created an entry while we were waiting for the lock
            if ( info->next==NULL ) {
                // Create a new entry
                store_timer_data_info *info_tmp = new store_timer_data_info;
                info_tmp->id = id;
                info_tmp->start_line = -2;
                info_tmp->stop_line = -1;
                info_tmp->next = NULL;
                info->next = info_tmp;
                N_timers++;
            }
            // Release the lock
            RELEASE_LOCK(&lock);
        } 
        // Advance to the next entry
        info = info->next;
    }
    return const_cast<store_timer_data_info*>(info);
}


/***********************************************************************
* Function to return a unique id based on the message and filename.    *
* Note:  We want to return a unique (but deterministic) id for each    *
* filename/message pair.  We want each process or thread to return the *
* same id independent of the other calls.                              *
***********************************************************************/
inline size_t ProfilerApp::get_timer_id( const char* message, const char* filename )
{
    unsigned int c;
    // Hash the filename using DJB2
    const char *s = filename;
    unsigned int hash1 = 5381;
    while((c = *s++)) {
        // hash = hash * 33 ^ c
        hash1 = ((hash1 << 5) + hash1) ^ c;
    }
    // Hash the message using DJB2
    s = message;
    unsigned int hash2 = 5381;
    while((c = *s++)) {
        // hash = hash * 33 ^ c
        hash2 = ((hash2 << 5) + hash2) ^ c;
    }
    // Combine the two hashes
    size_t key = 0;
    if ( sizeof(unsigned int)==sizeof(size_t) )
        key = hash1^hash2;
    else if ( sizeof(unsigned int)==4 && sizeof(size_t)==8 )
        key = (static_cast<size_t>(hash1)<<16) + static_cast<size_t>(hash2);
    else 
        ERROR_MSG("Unhandled case");
    return key;
}


/***********************************************************************
* Function to return a unique id based on the active timer bit array.  *
* This function works by performing a DJB2 hash on the bit array       *
***********************************************************************/
inline size_t ProfilerApp::get_trace_id( size_t N, const BIT_WORD *trace ) 
{
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE start_time_local;
        get_time(&start_time_local);
    #endif
    size_t hash = 5381;
    const size_t* s = reinterpret_cast<const size_t*>(trace);
    size_t N_words = N*sizeof(BIT_WORD)/sizeof(size_t);
    size_t c;
    if ( sizeof(size_t)==4 ) {
        for (size_t i=0; i<N_words; ++i) {
            // hash = hash * 33 ^ s[i]
            c = *s++;
            hash = ((hash << 5) + hash) ^ c;
        }
    } else if ( sizeof(size_t)==8 ) {
        for (size_t i=0; i<N_words; ++i) {
            // hash = hash * 65537 ^ s[i]
            c = *s++;
            hash = ((hash << 16) + hash) ^ c;
        }
    } else {
        ERROR_MSG("Unhandled case");
    }
    #if MONITOR_PROFILER_PERFORMANCE==1
        TIME_TYPE stop_time_local;
        get_time(&stop_time_local);
        total_trace_id_time += get_diff(start_time_local,stop_time_local,frequency);
    #endif
    return hash;
}


/***********************************************************************
* Function to return the hash index for a given timer id               *
***********************************************************************/
unsigned int ProfilerApp::get_timer_hash( size_t id )
{
    size_t key=0;
    if ( sizeof(size_t)==4 )
        key = (id*0x9E3779B9) >> 16;          // 2^32*0.5*(sqrt(5)-1) >> 0-65536
    else if ( sizeof(size_t)==8 )
        key = (id*0x9E3779B97F4A7C15) >> 48;  // 2^64*0.5*(sqrt(5)-1) >> 0-65536
    else
        ERROR_MSG("Unhandled case");
    return static_cast<unsigned int>(key%TIMER_HASH_SIZE);  // Convert the key to 0-TIMER_HASH_SIZE
}


/***********************************************************************
* Function to return the hash index for a given timer id               *
***********************************************************************/
unsigned int ProfilerApp::get_thread_hash( size_t id )
{
    size_t key=0;
    if ( sizeof(size_t)==4 )
        key = (id*0x9E3779B9) >> 16;          // 2^32*0.5*(sqrt(5)-1) >> 0-65536
    else if ( sizeof(size_t)==8 )
        key = (id*0x9E3779B97F4A7C15) >> 48;  // 2^64*0.5*(sqrt(5)-1) >> 0-65536
    else
        ERROR_MSG("Unhandled case");
    return static_cast<unsigned int>(key%THREAD_HASH_SIZE);  // Convert the key to 0-THREAD_HASH_SIZE
}


/***********************************************************************
* Subroutine to perform a quicksort                                    *
***********************************************************************/
template <class type_a, class type_b>
static inline void quicksort2(int n, type_a *arr, type_b *brr)
{
    bool test;
    int i, ir, j, jstack, k, l, istack[100];
    type_a a, tmp_a;
    type_b b, tmp_b;
    jstack = 0;
    l = 0;
    ir = n-1;
    while (1) {
        if ( ir-l < 7 ) {             // Insertion sort when subarray small enough.
            for ( j=l+1; j<=ir; j++ ) {
                a = arr[j];
                b = brr[j];
                test = true;
                for (i=j-1; i>=0; i--) {
                    if ( arr[i] < a ) {
                        arr[i+1] = a;
                        brr[i+1] = b;
                        test = false;
                        break;
                    }
                    arr[i+1] = arr[i];
                    brr[i+1] = brr[i];
                }
                if ( test ) {
                    i = l-1;
                    arr[i+1] = a;
                    brr[i+1] = b;
                }
            }
            if ( jstack==0 )
                return;
            ir = istack[jstack];    // Pop stack and begin a new round of partitioning.
            l = istack[jstack-1];
            jstack -= 2;
        } else {
            k = (l+ir)/2;           // Choose median of left, center and right elements as partitioning
                                    // element a. Also rearrange so that a(l) ? a(l+1) ? a(ir).
            tmp_a = arr[k];
            arr[k] = arr[l+1];
            arr[l+1] = tmp_a;
            tmp_b = brr[k];
            brr[k] = brr[l+1];
            brr[l+1] = tmp_b;
            if ( arr[l]>arr[ir] ) {
                tmp_a = arr[l];
                arr[l] = arr[ir];
                arr[ir] = tmp_a;
                tmp_b = brr[l];
                brr[l] = brr[ir];
                brr[ir] = tmp_b;
            }
            if ( arr[l+1] > arr[ir] ) {
                tmp_a = arr[l+1];
                arr[l+1] = arr[ir];
                arr[ir] = tmp_a;
                tmp_b = brr[l+1];
                brr[l+1] = brr[ir];
                brr[ir] = tmp_b;
            }
            if ( arr[l] > arr[l+1] ) {
                tmp_a = arr[l];
                arr[l] = arr[l+1];
                arr[l+1] = tmp_a;
                tmp_b = brr[l];
                brr[l] = brr[l+1];
                brr[l+1] = tmp_b;
            }
            // Scan up to find element > a
            j = ir;
            a = arr[l+1];           // Partitioning element.
            b = brr[l+1];
            for (i=l+2; i<=ir; i++) { 
                if ( arr[i]<a ) 
                    continue;
                while ( arr[j]>a )  // Scan down to find element < a.
                    j--;
                if ( j < i )
                    break;          // Pointers crossed. Exit with partitioning complete.
                tmp_a = arr[i];     // Exchange elements of both arrays.
                arr[i] = arr[j];
                arr[j] = tmp_a;
                tmp_b = brr[i];
                brr[i] = brr[j];
                brr[j] = tmp_b;
            }
            arr[l+1] = arr[j];      // Insert partitioning element in both arrays.
            arr[j] = a;
            brr[l+1] = brr[j];
            brr[j] = b;
            jstack += 2;
            // Push pointers to larger subarray on stack, process smaller subarray immediately.
            if ( ir-i+1 >= j-l ) {
                istack[jstack] = ir;
                istack[jstack-1] = i;
                ir = j-1;
            } else {
                istack[jstack] = j-1;
                istack[jstack-1] = l;
                l = i;
            }
        }
    }
}


}

