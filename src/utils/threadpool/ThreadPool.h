// Copyright 2004 Mark Berrill. All Rights Reserved. This work is distributed with permission,
// but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.
#ifndef included_AMP_ThreadPool
#define included_AMP_ThreadPool

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

#include "AMP/utils/threadpool/ThreadPoolId.h"
#include "AMP/utils/threadpool/ThreadPoolQueue.h"
#include "AMP/utils/threadpool/ThreadPoolWorkItem.h"


namespace AMP {
// clang-format off


/** \class ThreadPool
 *
 * \brief This is a concrete class that provides for a basic thread pool.
 * \details This class implements a basic thread pool that can be used for a wide variety of
 * applications.
 * An example call usage is provided below.  The ability to return a value is provided.  Note that
 * there
 * is a small overhead to using this functionality. <BR>
 * <pre>Example: <BR>
 *    Existing function call:
 *       double x = myfun_1(a,b);
 *       double y = myfun_2(c,d); <BR>
 *    Threaded call (processing in parallel):
 *       ThreadPoolID ids[2];
 *       ids[0] = TPOOL_ADD_WORK( tpool, myfun_1, (a,b) );
 *       ids[1] = TPOOL_ADD_WORK( tpool, myfun_2, (c,d) );
 *       wait_all(2,ids);
 *       double x = getFunctionRet(ids[0]);
 *       double y = getFunctionRet(ids[1]); <BR>
 *   </pre>
 */
class alignas(16) ThreadPool final
{
public:
    ///// Set some global properties
    constexpr static uint16_t MAX_THREADS = 128; // The maximum number of threads (must be a multiple of 64)

public:
    ///// Member classes


    //! Base class for the work item (users should derive from WorkItemRet)
    typedef ThreadPoolWorkItem WorkItem;


    /*!
     * \brief   Class to define a work item returning a variable
     * \details This is the class that defines a work item to be processed.  Users may derive their
     * own
     * class and add work using the add_work routine, or can use the TPOOL_ADD_WORK macro.
     * Note: this class is templated on the return argument type and may be a void type.
     */
    template <typename return_type>
    class WorkItemRet : public ThreadPool::WorkItem
    {
    public:
        //! Run the work item
        virtual void run() override = 0;
        //! Will the routine return a result
        virtual bool has_result() const override final { return !std::is_same<return_type,void>::value; }
        //! Return the results
        return_type get_results() const { return d_result; }
        //! Virtual destructor
        virtual ~WorkItemRet() {}
    protected:
        return_type d_result;
    protected:
        inline WorkItemRet() : d_result( return_type() ) { }
    private:
        WorkItemRet( const WorkItemRet & );            // Private copy constructor
        WorkItemRet &operator=( const WorkItemRet & ); // Private assignment operator
    };


public:
    ///// Member functions

    /*!
     *  Constructor that initialize the thread pool with N threads
     * @param N                 The desired number of worker threads
     * @param affinity          The affinity scheduler to use:
     *                          none - Let the OS handle the affinities (default)
     *                          independent - Give each thread an independent set of processors
     * @param procs             The processors to use (defaults to the process affinitiy list)
     * @param queueSize         The maximum number of items in the queue before forcing a wait
     */
    ThreadPool( const int N = 0, const std::string &affinity = "none",
        const std::vector<int> &procs = std::vector<int>(), int queueSize = 4096 );


    //! Copy constructor
    ThreadPool( const ThreadPool & ) = delete;


    //! Assignment operator
    ThreadPool &operator=( const ThreadPool & ) = delete;


    //! Destructor
    ~ThreadPool();


    //! Function to return the number of processors available
    static int getNumberOfProcessors();


    //! Function to return the processor number that the current thread is running on
    static int getCurrentProcessor();


    //! Function to return the affinity of the current process
    static std::vector<int> getProcessAffinity();


    //! Function to set the affinity of the current process
    static void setProcessAffinity( const std::vector<int>& procs );


    //! Function to return the affinity of the current thread
    static std::vector<int> getThreadAffinity();


    /*!
     *  Function to return the affinity of the given thread
     *  @param thread   The index of the thread
     */
    std::vector<int> getThreadAffinity( int thread ) const;


    /*!
     *  Function to set the affinity of the current thread
     *  @param procs    The processors to use
     */
    static void setThreadAffinity( const std::vector<int>& procs );


    /*!
     *  Set the given thread to have the specified affinity
     *  @param thread   The index of the thread
     *  @param procs    The processors to use
     */
    void setThreadAffinity( int thread, const std::vector<int>& procs ) const;


    //! Function to return the number of threads in the thread pool
    inline int getNumThreads() const { return d_N_threads; }


    /*!
     * \brief   Function to set the number of threads in the thread pool
     * \details  This function will change the number of worker threads in the ThreadPool
     *   to the number specified.  This function will immediately change the number of threads
     *   in the ThreadPool without checking the existing work unless the desired number of
     *   threads is 0.  In this case, the function will wait for all work items to finish
     *   before deleting the existing work threads.
     *   Member threads may not call this function.
     * @param N                 The desired number of worker threads
     * @param affinity          The affinity scheduler to use:
     *                          none - Let the OS handle the affinities (default)
     *                          independent - Give each thread an independent set of processors
     * @param procs             The processors to use (defaults to the process affinitiy list)
     */
    void setNumThreads( const int N, const std::string &affinity = "none",
        const std::vector<int> &procs = std::vector<int>() );


    /*!
     * \brief   Function to set the maximum wait time
     * \details  This function sets the maximum time the thread pool will
     *    wait before warning about a possible hung thread.
     *    Default is to wait 10 minutes.
     * @param time              The number of seconds to wait (seconds)
     */
    inline void setMaxWaitTimeDebug( const int time ) { d_max_wait_time = time; }


    /*!
     * \brief   Function to return the current thread number
     * \details  This function will return the thread number of current active thread.
     *    If the thread is not a member of the thread pool, this function will return 0.
     */
    int getThreadNumber() const;


    //! Function to check if the work item is valid
    /*!
     * This function checks if the work item has a valid id.
     *   Note: this function does not require blocking and will return immediately.
     * @param id                The id of the work item
     */
    inline bool isValid( const ThreadPoolID &id ) const;


    /*!
     * \brief    Function to check if the work item has finished processing
     * \details  This function checks if the work item has finished processing.
     * @param id                The id of the work item
     */
    inline bool isFinished( const ThreadPoolID& id ) const { return id.finished(); }


    /*!
     * \brief   Function to get the returned function value
     * \details This is the function returns the value that was returned from the working function.
     *   If the work item has not finished or was not found it will return 0.
     * @param id                The id of the work item
     */
    template <class return_type>
    static inline return_type getFunctionRet( const ThreadPoolID &id );


    /*!
     * \brief   Function to create a work item
     * \details This function creates a work item that can be added to the queue
     * @param routine           Function to call from the thread pool
     * @param args              Function arguments to pass
     */
    template<class Ret, class... Args1, class... Args2>
    static inline WorkItem* createWork( std::function<Ret(Args1...)> routine, std::tuple<Args2...> &&args );


    /*!
     * \brief   Function to create a work item
     * \details This function creates a work item that can be added to the queue
     * @param routine           Function to call from the thread pool
     * @param args              Function arguments to pass
     */
    template<class Ret, class... Args1, class... Args2>
    static inline WorkItem* createWork( Ret( *routine )( Args1... ), std::tuple<Args2...> &&args );


    /*!
     * \brief   Function to create a work item
     * \details This function creates a work item that can be added to the queue
     * @param routine           Function to call from the thread pool
     * @param args              Function arguments to pass
     */
    template<class Ret, class... Args1, class... Args2>
    static inline WorkItem* createWork( std::function<Ret(Args1...)> routine, Args2... args );


    /*!
     * \brief   Function to create a work item
     * \details This function creates a work item that can be added to the queue
     * @param routine           Function to call from the thread pool
     * @param args              Function arguments to pass
     */
    template <class Ret, class... Args1, class... Args2>
    static inline WorkItem* createWork( Ret( *routine )( Args1... ), Args2... args );


    /*!
     * \brief   Function to add a work item
     * \details This function adds a work item to the queue
     *   Note: any thread may call this routine.
     * @param work              Pointer to the work item to add
     *                          Note that the threadpool will automatically destroy the item when
     * finished
     * @param priority          A value indicating the priority of the work item (0-default)
     */
    inline ThreadPoolID add_work( ThreadPool::WorkItem *work, int priority = 0 );


    /*!
     * \brief   Function to add multiple work items
     * \details This function adds multiple work item to the queue
     *   Note: any thread may call this routine.
     * @param work              Vector of pointers to the work items to add
     *                          Note that the threadpool will automatically destroy the item when
     * finished
     * @param priority          Vector of values indicating the priority of the work items
     */
    inline std::vector<ThreadPoolID> add_work( const std::vector<ThreadPool::WorkItem *> &work,
        const std::vector<int> &priority = std::vector<int>() );


    /*!
     * \brief   Function to wait until a specific work item has finished
     * \details This is the function waits for a specific work item to finished.
     *   Note: any thread may call this routine, but they will block until finished.
     *   For worker threads this may eventually lead to a deadlock.
     * @param id                The work item to wait for
     */
    inline void wait( const ThreadPoolID &id ) const;


    /*!
     * \brief   Function to wait until any of the given work items have finished their work
     * \details This is the function waits for any of the given work items to finish.
     *   If successful it returns the index of a finished work item (the index in the array ids).
     *   Note: any thread may call this routine, but they will block until finished.
     *   For worker threads this may eventually lead to a deadlock.
     * @param ids               Vector of work items to wait for
     */
    inline size_t wait_any( const std::vector<ThreadPoolID> &ids ) const;


    /*!
     * \brief   Function to wait until all of the given work items have finished their work
     * \details This is the function waits for all given of the work items to finish.
     *   Note: any thread may call this routine, but they will block until finished.
     *   For worker threads this may eventually lead to a deadlock.
     * @param ids               Vector of work items to wait for
     */
    inline void wait_all( const std::vector<ThreadPoolID> &ids ) const;


    /*!
     * \brief   Function to wait until some of the given work items have finished their work
     * \details This is the function waits for some of the given work items to finish.
     *   If successful it returns the indicies of the finished work items (the index in the array ids).
     *   Note: any thread may call this routine, but they will block until finished.
     *   For worker threads this may eventually lead to a deadlock.
     * @param N_wait            Number of work items to wait for
     * @param ids               Vector of work items to wait for
     * @param max_wait          Maximum time to wait (seconds)
     * @return                  Returns the indicies items that have finished (in sorted order)
     */
    inline std::vector<int> wait_some( int N_wait, const std::vector<ThreadPoolID> &ids, int max_wait = 10000000 ) const;


    /*!
     * \brief   Function to wait until all work items in the thread pool have finished their work
     * \details This function will wait until all work has finished.
     *   Note: member threads may not call this function.
     *   Only one non-member thread should call this routine at a time.
     */
    void wait_pool_finished() const;


    /*!
     * \brief   Function to check if the thread pool is valid
     * \details Sometimes it is necessary to work with raw pointers for the thread pool.
     *    If the thread pool is invalid and used, the program will likely fail catastrophically.
     *    This function checks if the thread pool is valid is a relatively safe manner.
     *    If the thread pool is pointing to an invalid memory address, because it has been
     *    freed, never allocated, or otherwise corrupted, this function will return false.
     * @param tpool         Pointer to the ThreadPool to check
     */
    static bool is_valid( const ThreadPool *tpool );


    /*!
     * \brief   Function to enable/disable OS warning messages
     * \details Some of the functions such as setting/getting the thread affinities
     *      are not supported on all platforms.  This function controls the behavior
     *      of these functions on systems where they are not supported.  The default

     *      behavior is to print a warning message.  Other options include ignoring
     *      the messages (the functions will return empty sets), or throwing an exception.
     *      Note: this is a global property and will affect all thread pools in an application.
     * @param behavior      The behavior of OS specific messages/errors
     *                      0: Print a warning message
     *                      1: Ignore the messages
     *                      2: Throw an error
     */
    static void set_OS_warnings( int behavior = 0 );


    //! Return the number of items queued
    int N_queued( ) const { return d_queue.size(); }


    //! Set the error handler for threads
    void setErrorHandler( std::function<void(const std::string&)> fun );


public: // Static interface

    /*!
     * \brief   Function to return the number of work threads
     * \details This function returns the number of threads in the thread pool,
     *    or 0 if the thread pool is empty or does not exist
     * @param tpool         Threadpool to add work to (may be null)
     */
    static inline int numThreads( const ThreadPool* tpool ) { return tpool ? tpool->getNumThreads() : 0; }

    /*!
     * \brief   Function to add a work item
     * \details This function adds a work item to the queue
     *   Note: any thread may call this routine.
     * @param tpool         Threadpool to add work to (may be null)
     * @param work          Pointer to the work item to add
     *                      Note that the threadpool will automatically destroy the item when finished
     * @param priority      A value indicating the priority of the work item (0-default)
     */
    static inline ThreadPoolID add_work( ThreadPool* tpool, ThreadPool::WorkItem *work, int priority = 0 );


    /*!
     * \brief   Function to add multiple work items
     * \details This function adds multiple work item to the queue
     *   Note: any thread may call this routine.
     * @param tpool         Threadpool to add work to (may be null)
     * @param work          Vector of pointers to the work items to add
     *                      Note that the threadpool will automatically destroy the item when finished
     * @param priority      Vector of values indicating the priority of the work items
     */
    static inline std::vector<ThreadPoolID> add_work( ThreadPool* tpool, const std::vector<ThreadPool::WorkItem *> &work,
        const std::vector<int> &priority = std::vector<int>() );


    /*!
     * \brief   Function to wait until all of the given work items have finished their work
     * \details This is the function waits for all given of the work items to finish.
     *   Note: any thread may call this routine, but they will block until finished.
     *   For worker threads this may eventually lead to a deadlock.
     * @param tpool         Threadpool containing work (must match call to add_work)
     * @param ids           Vector of work items to wait for
     */
    static inline void wait_all( const ThreadPool* tpool, const std::vector<ThreadPoolID> &ids );


    /*!
     * \brief   Function to wait until all work items in the thread pool have finished their work
     * \details This function will wait until all work has finished.
     *   Note: member threads may not call this function.
     *   Only one non-member thread should call this routine at a time.
     * @param tpool         Threadpool containing work (must match call to add_work)
     */
    static inline void wait_pool_finished( const ThreadPool* tpool ) { if ( tpool ) { tpool->wait_pool_finished(); } }


private:
    ///// Member data structures    

    // Implimentation of condition_variable which does not require a lock
    class condition_variable final
    {
      public:
        condition_variable() { }
        ~condition_variable() { }
        inline void wait() const { std::unique_lock<std::mutex> lock(d_mutex); d_cv.wait(lock); }
        inline void wait_for( double seconds ) const
        {
            std::unique_lock<std::mutex> lock(d_mutex);
            if ( seconds < 4e-6 )
                d_cv.wait_for(lock,std::chrono::nanoseconds(static_cast<int>(1e9*seconds)));
            else if ( seconds < 4e-3 )
                d_cv.wait_for(lock,std::chrono::microseconds(static_cast<int>(1e6*seconds)));
            else if ( seconds < 4 )
                d_cv.wait_for(lock,std::chrono::milliseconds(static_cast<int>(1e3*seconds)));
            else
                d_cv.wait_for(lock,std::chrono::seconds(static_cast<int>(seconds)));
        }
        inline void notify_one() const { d_cv.notify_one(); }
        inline void notify_all() const { d_cv.notify_all(); }
      private:
        mutable std::condition_variable d_cv;
        mutable std::mutex d_mutex;
    };


private:
    ///// Member functions

    // Function to check the startup
    void check_startup( );

    // Function to add an array of work items
    void add_work(
        size_t N, ThreadPool::WorkItem *work[], const int *priority, ThreadPoolID *id );

    // Function to get a work item that has finished
    static inline WorkItem *getFinishedWorkItem( const ThreadPoolID& id )
    {
        return id.finished() ? id.getWork():nullptr;
    }

    // This function provides a wrapper (needed for the threads)
    static inline void create_new_thread( ThreadPool *tpool, int id )
    {
        tpool->tpool_thread( id );
    }

    /* This is the function that controls the individual thread and allows it to do work.
     * Note: this version uses a last in - first out work scheduling.
     * param thread_init - Structure address containing the startup information for the thread */
    void tpool_thread( int id );

    // Function to check if the current thread is a member of the thread pool
    inline bool isMemberThread() const { return getThreadNumber()>=0; }

    // Function to wait for some work items to finish
    std::vector<bool> wait_some( size_t N_work, const ThreadPoolID *ids, size_t N_wait, int max_wait ) const;
    
    // Function to wait for N work items to finish (may return early)
    void wait_N( int N, double time ) const;
    
    // Check if we are waiting too long and pring debug info
    void print_wait_warning( ) const;


private:
    ///// Member data

    // Typedefs
    typedef volatile std::atomic_uint32_t vint32_t; // volatile atomic int
    typedef volatile std::atomic_uint64_t vint64_t; // volatile atomic int64
    typedef condition_variable cond_t;              // condition variable

    // Internal data
    uint32_t d_NULL_HEAD;                 // Null data buffer to check memory bounds
    volatile mutable bool d_signal_empty; // Do we want to send a signal when the queue is empty
    uint16_t d_N_threads;                 // Number of threads
    int d_max_wait_time;                  // The maximum time waiting before printing a warning message
    mutable vint32_t d_signal_count;      // Signal count
    vint32_t d_num_active;                // Number of threads that are currently active
    vint64_t d_id_assign;                 // An internal variable used to store the current id to assign
    vint64_t d_active[MAX_THREADS/64];    // Which threads are currently active
    vint64_t d_cancel[MAX_THREADS/64];    // Which threads should be deleted
    vint64_t d_N_added;                   // Number of items added to the work queue
    vint64_t d_N_started;                 // Number of items started
    vint64_t d_N_finished;                // Number of items finished
    mutable cond_t d_wait_finished;       // Condition variable to signal when work is finished
    mutable cond_t d_wait_work;           // Condition variable to signal when there is new work
    ThreadPoolListQueue d_queue;          // The work queue
    std::function<void(const std::string&)> d_errorHandler; // Error handler
    std::thread *d_thread;                // Handles to the threads
    uint32_t d_NULL_TAIL;                 // Null data buffer to check memory bounds
};


} // AMP namespace


#include "AMP/utils/threadpool/ThreadPool.hpp"


// clang-format on
#endif
