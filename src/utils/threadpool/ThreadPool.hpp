// This file contains the template functions for the thread pool
#ifndef included_AMP_ThreadPoolTmpl
#define included_AMP_ThreadPoolTmpl

#include "AMP/utils/threadpool/ThreadPool.h"

#include "ProfilerApp.h"

#include <functional>
#include <stdexcept>
#include <tuple>


namespace AMP {


/*! \addtogroup Macros
 *  @{
 */


/*! \def id = TPOOL_ADD_WORK(tpool,function,args,priority)
 *  \brief Add an item to the thread pool
 *  \details This a macro to automatically create and add a work item to the thread pool.
 *  \param tpool        Pointer to the thread pool to use
 *  \param function     Pointer to the function to use
 *  \param args         The arguments to pass to the function in the form (arg1,arg2,...)
 *  \param priority     Optional argument specifying the priority of the work item
 */
#define TPOOL_ADD_WORK2( TPOOL, FUNCTION, ARGS, PRIORITY, ... ) \
    ThreadPool_add_work( TPOOL, PRIORITY, FUNCTION, std::make_tuple ARGS )
#define TPOOL_ADD_WORK( TPOOL, FUNCTION, ... ) TPOOL_ADD_WORK2( TPOOL, FUNCTION, __VA_ARGS__, 0, 0 )


/*! @} */

// \cond HIDDEN_SYMBOLS


// Specialization for no return argument
template<>
class ThreadPool::WorkItemRet<void> : public ThreadPool::WorkItem
{
public:
    virtual void run() override = 0;
    void get_results() {}
    virtual ~WorkItemRet() {}
    virtual bool has_result() const override final { return false; }
};


// Final class for the work item
template<class Ret, class FUN, class... Args>
class WorkItemFull;
template<class FUN, class... Args>
class WorkItemFull<void, FUN, Args...> final : public ThreadPool::WorkItemRet<void>
{
private:
    FUN routine;
    std::tuple<Args...> args;

public:
    WorkItemFull() = delete;
    WorkItemFull( FUN &&routine2, Args... ts )
        : ThreadPool::WorkItemRet<void>(), routine( std::move( routine2 ) ), args( ts... )
    {
    }
    WorkItemFull( FUN &&routine2, std::tuple<Args...> &&ts )
        : ThreadPool::WorkItemRet<void>(), routine( std::move( routine2 ) ), args( ts )
    {
    }
    virtual void run() override { std::apply( routine, args ); }
    virtual ~WorkItemFull() {}
};
template<class Ret, class FUN, class... Args>
class WorkItemFull final : public ThreadPool::WorkItemRet<Ret>
{
private:
    FUN routine;
    std::tuple<Args...> args;

public:
    WorkItemFull() = delete;
    WorkItemFull( FUN &&routine2, Args... ts )
        : ThreadPool::WorkItemRet<Ret>(), routine( std::move( routine2 ) ), args( ts... )
    {
    }
    WorkItemFull( FUN &&routine2, std::tuple<Args...> &&ts )
        : ThreadPool::WorkItemRet<Ret>(), routine( std::move( routine2 ) ), args( ts )
    {
    }
    virtual void run() override { this->d_result = std::apply( routine, args ); }
    virtual ~WorkItemFull() {}
};


// Functions to add work to the thread pool
// clang-format off
template<class Ret, class... Args1, class... Args2>
inline ThreadPoolID ThreadPool_add_work(
    ThreadPool *tpool, int priority, std::function<Ret( Args1... )> fun, std::tuple<Args2...> &&args )
{
    auto work = new WorkItemFull<Ret, decltype(fun), Args2...>( std::move( fun ), std::move( args ) );
    return ThreadPool::add_work( tpool, work, priority );
}
template<class Ret, class... Args1, class... Args2>
inline ThreadPoolID ThreadPool_add_work(
    ThreadPool *tpool, int priority, Ret ( *routine )( Args1... ), std::tuple<Args2...> &&args )
{
    std::function<Ret( Args1... )> fun = routine;
    auto work = new WorkItemFull<Ret, decltype(fun), Args2...>( std::move( fun ), std::move( args ) );
    return ThreadPool::add_work( tpool, work, priority );
}
template<class Ret>
inline ThreadPoolID ThreadPool_add_work(
    ThreadPool *tpool, int priority, Ret ( *routine )(), std::tuple<std::nullptr_t>&& )
{
    std::function<Ret()> fun = routine;
    auto work = new WorkItemFull<Ret, decltype(fun)>( std::move( fun ) );
    return ThreadPool::add_work( tpool, work, priority );
}
template<class Ret, class... Args1, class... Args2>
inline ThreadPoolID ThreadPool_add_work(
ThreadPool *tpool, int priority, std::function<Ret( Args1... )> fun, Args2... args )
{
    auto work = new WorkItemFull<Ret, decltype(fun), Args2...>( std::move( fun ), std::forward_as_tuple( args... ) );
    return ThreadPool::add_work( tpool, work, priority );
}
template<class Ret, class... Args1, class... Args2>
inline ThreadPoolID ThreadPool_add_work(
    ThreadPool *tpool, int priority, Ret ( *routine )( Args1... ), Args2... args )
{
    std::function<Ret( Args1... )> fun = routine;
    auto work = new WorkItemFull<Ret, decltype(fun), Args2...>( std::move( fun ), std::forward_as_tuple( args... ) );
    return ThreadPool::add_work( tpool, work, priority );
}
template<class Ret>
inline ThreadPoolID ThreadPool_add_work(
    ThreadPool *tpool, int priority, Ret ( *routine )(), void * )
{
    std::function<Ret(void)> fun = routine;
    auto work = new WorkItemFull<Ret,decltype(fun)>( std::move( fun ) );
    return ThreadPool::add_work( tpool, work, priority );
}
template<class Ret, class... Args1, class... Args2>
inline ThreadPool::WorkItem *ThreadPool::createWork(
    std::function<Ret( Args1... )> fun, Args2... args )
{
    return new WorkItemFull<Ret, decltype(fun), Args2...>( std::move( fun ), std::forward_as_tuple( args... ) );
}
template<class Ret, class... Args1, class... Args2>
inline ThreadPool::WorkItem *ThreadPool::createWork( Ret ( *routine )( Args1... ), Args2... args )
{
    std::function<Ret( Args1... )> fun = routine;
    return new WorkItemFull<Ret, decltype(fun), Args2...>( std::move( fun ), std::forward_as_tuple( args... ) );
}
template<class Ret, class... Args1, class... Args2>
inline ThreadPool::WorkItem *ThreadPool::createWork(
    std::function<Ret( Args1... )> fun, std::tuple<Args2...> &&args )
{
    return new WorkItemFull<Ret, decltype(fun), Args2...>( std::move( fun ), std::move( args ) );
}
template<class Ret, class... Args1, class... Args2>
inline ThreadPool::WorkItem *ThreadPool::createWork(
    Ret ( *routine )( Args1... ), std::tuple<Args2...> &&args )
{
    std::function<Ret( Args1... )> fun = routine;
    return new WorkItemFull<Ret, decltype(fun), Args2...>( std::move( fun ), std::move( args ) );
}
// clang-format on


/******************************************************************
 * Function to get the returned function value                     *
 ******************************************************************/
template<class TYPE>
inline TYPE ThreadPool::getFunctionRet( const ThreadPoolID &id )
{
    // Get the work and return result if finished
    auto work = dynamic_cast<WorkItemRet<TYPE> *>( getFinishedWorkItem( id ) );
    if ( work )
        return work->get_results();
    // Get default value
    static_assert( std::is_arithmetic_v<bool> );
    static_assert( std::is_arithmetic_v<char> );
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        return 0;
    } else {
        return TYPE();
    }
}


/******************************************************************
 * Inline functions to wait for the work items to finish           *
 ******************************************************************/
inline void ThreadPool::wait( const ThreadPoolID &id ) const
{
    PROFILE( "ThreadPool::wait" );
    auto finished = wait_some( 1, &id, 1, 10000000 );
    if ( !finished[0] )
        throw std::logic_error( "Failed to wait for id" );
}
inline size_t ThreadPool::wait_any( const std::vector<ThreadPoolID> &ids ) const
{
    if ( ids.empty() )
        return 0;
    PROFILE( "ThreadPool::wait_any" );
    auto finished = wait_some( ids.size(), &ids[0], 1, 10000000 );
    for ( size_t i = 0; i < ids.size(); i++ ) {
        if ( finished[i] )
            return i;
    }
    throw std::logic_error( "wait_any failed" );
}
inline void ThreadPool::wait_all( const std::vector<ThreadPoolID> &ids ) const
{
    if ( ids.empty() )
        return;
    PROFILE( "ThreadPool::wait_all" );
    auto finished = wait_some( ids.size(), ids.data(), ids.size(), 10000000 );
    size_t N      = 0;
    for ( bool test : finished ) {
        if ( test )
            N++;
    }
    if ( N != ids.size() )
        throw std::logic_error( "Failed to wait for all ids" );
}
inline void ThreadPool::wait_all( const ThreadPool *tpool, const std::vector<ThreadPoolID> &ids )
{
    if ( tpool )
        return tpool->wait_all( ids );
}
inline std::vector<int>
ThreadPool::wait_some( int N_wait, const std::vector<ThreadPoolID> &ids, int max_wait ) const
{
    PROFILE( "ThreadPool::wait_some" );
    auto finished = wait_some( ids.size(), ids.data(), N_wait, max_wait );
    std::vector<int> index;
    for ( size_t i = 0; i < ids.size(); i++ ) {
        if ( finished[i] )
            index.push_back( i );
    }
    return index;
}


/******************************************************************
 * Functions to add work items.                                    *
 ******************************************************************/
inline ThreadPoolID ThreadPool::add_work( WorkItem *work, int priority )
{
    ThreadPoolID id;
    add_work( 1, &work, &priority, &id );
    return id;
}
inline std::vector<ThreadPoolID>
ThreadPool::add_work( const std::vector<ThreadPool::WorkItem *> &work,
                      const std::vector<int> &priority )
{
    size_t N = work.size();
    if ( N == 0 )
        return std::vector<ThreadPoolID>();
    if ( priority.size() != N && !priority.empty() )
        throw std::logic_error( "size of work and priority do not match" );
    const int *priority2 = nullptr;
    if ( priority.empty() ) {
        priority2 = new int[N];
        memset( const_cast<int *>( priority2 ), 0, N * sizeof( int ) );
    } else {
        priority2 = &priority[0];
    }
    std::vector<ThreadPoolID> ids( N );
    add_work( N, const_cast<ThreadPool::WorkItem **>( &work[0] ), priority2, &ids[0] );
    if ( priority.empty() )
        delete[] priority2;
    return ids;
}
inline ThreadPoolID
ThreadPool::add_work( ThreadPool *tpool, ThreadPool::WorkItem *work, int priority )
{
    ThreadPoolID id;
    if ( tpool ) {
        id = tpool->add_work( work, priority );
    } else {
        id.reset( priority, std::rand(), work );
        work->d_state = ThreadPoolID::Status::started;
        work->run();
        work->d_state = ThreadPoolID::Status::finished;
    }
    return id;
}
inline std::vector<ThreadPoolID>
ThreadPool::add_work( ThreadPool *tpool,
                      const std::vector<ThreadPool::WorkItem *> &work,
                      const std::vector<int> &priority )
{
    if ( tpool ) {
        return tpool->add_work( work, priority );
    } else {
        std::vector<ThreadPoolID> ids( work.size() );
        for ( size_t i = 0; i < work.size(); i++ )
            ids[i] = add_work( tpool, work[i], priority[i] );
        return ids;
    }
}


/******************************************************************
 * This function checks if the id is valid                         *
 ******************************************************************/
inline bool ThreadPool::isValid( const ThreadPoolID &id ) const
{
    uint64_t local_id = id.getLocalID();
    uint64_t next_id  = d_id_assign - 1;
    return local_id != 0 && id.initialized() && local_id <= ThreadPoolID::maxThreadID &&
           local_id > next_id;
}


/******************************************************************
 * Function to get the thread number                               *
 * (-1 if it is not a member thread)                               *
 ******************************************************************/
inline int ThreadPool::getThreadNumber() const
{
    std::thread::id id = std::this_thread::get_id();
    for ( int i = 0; i < d_N_threads; i++ ) {
        if ( id == d_thread[i].get_id() )
            return i;
    }
    return -1;
}


} // namespace AMP


// \endcond


#endif
