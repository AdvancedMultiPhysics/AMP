// This file contains the template functions for the thread pool queues
#ifndef included_AMP_ThreadPoolQueue
#define included_AMP_ThreadPoolQueue


#include "AMP/utils/threadpool/AtomicList.h"
#include "AMP/utils/threadpool/ThreadPoolId.h"

#include <algorithm>


namespace AMP {


//! Class to store the queue for the ThreadPool using an thread-safe list
class ThreadPoolListQueue
{
public:
    //! Empty constructor
    ThreadPoolListQueue() = delete;

    //! Default constructor
    explicit ThreadPoolListQueue( size_t N ) : d_queue_list( N ) {}

    //! Copy constructor
    ThreadPoolListQueue( const ThreadPoolListQueue & ) = delete;

    //! Asignment operator
    ThreadPoolListQueue &operator=( const ThreadPoolListQueue & ) = delete;

    //! Check if the queue is empty
    inline bool empty() const { return d_queue_list.empty(); }

    //! The number of items that can be in the queue
    inline size_t capacity() const { return d_queue_list.capacity(); }

    //! The number of items that are in the queue
    inline size_t size() const { return d_queue_list.size(); }

    //! Get the next item to process
    inline ThreadPoolID pop()
    {
        return d_queue_list.remove( []( const ThreadPoolID &id ) { return id.ready(); } );
    }

    //! Add the given items to the queue
    inline void insert( size_t N, const ThreadPoolID *ids )
    {
        // Add the work items in reverse order for efficiency, queue will maintain order)
        for ( int64_t i = N - 1; i >= 0; i-- )
            d_queue_list.insert( ids[i] );
    }

    //! Change the prioirties of items in the queue
    inline void changePriorities( const std::vector<std::pair<uint64_t, int8_t>> &list )
    {
        auto compare = []( const ThreadPoolID &a, const uint64_t &b ) {
            return a.getLocalID() == b;
        };
        for ( auto item : list ) {
            // Remove and add the id back with a higher priority
            auto tmp = d_queue_list.remove( compare, item.first );
            if ( tmp.isNull() )
                continue;
            tmp.setPriority( std::max( item.second, tmp.getPriority() ) );
            d_queue_list.insert( tmp );
        }
    }

private: ///// Member data
    typedef AtomicList<ThreadPoolID, std::greater<ThreadPoolID>> queue_type;
    queue_type d_queue_list; // The work queue
};


//! Class to store the queue for the ThreadPool using a binary heap
class ThreadPoolHeapQueue
{
public:
    //! Empty constructor
    ThreadPoolHeapQueue() = delete;

    //! Default constructor
    explicit ThreadPoolHeapQueue( size_t N )
        : d_lock( 0 ), d_Nc( N ), d_Nh( 0 ), d_Nb( 0 ), d_ids( new ThreadPoolID[N] )
    {
    }

    //! Copy constructor
    ThreadPoolHeapQueue( const ThreadPoolHeapQueue & ) = delete;

    //! Asignment operator
    ThreadPoolHeapQueue &operator=( const ThreadPoolHeapQueue & ) = delete;

    //! The number of items that can be in the queue
    inline size_t capacity() const { return d_Nc; }

    //! The number of items that are in the queue
    inline size_t size() const { return d_Nh + d_Nb; }

    //! Check if the queue is empty
    inline bool empty() const { return size() == 0; }

    //! Get the next item to process
    inline ThreadPoolID pop()
    {
        lock();
        checkBlocked();
        ThreadPoolID id;
        if ( d_Nh > 0 ) {
            std::pop_heap( const_cast<ThreadPoolID *>( d_ids ),
                           const_cast<ThreadPoolID *>( d_ids + d_Nh ) );
            std::swap( id, const_cast<ThreadPoolID &>( d_ids[--const_cast<size_t &>( d_Nh )] ) );
        }
        unlock();
        return id;
    }

    //! Add the given items to the queue
    inline void insert( size_t N, const ThreadPoolID *ids )
    {
        lock();
        auto Nh2 = d_Nh;
        for ( size_t i = 0; i < N; i++ ) {
            if ( ids[i].ready() ) {
                d_ids[Nh2++] = ids[i];
            } else {
                const_cast<size_t &>( d_Nb )++;
                d_ids[d_Nc - d_Nb] = ids[i];
                if ( d_ids[d_Nc - d_Nb] > d_ids[d_Nc - 1] )
                    std::swap( d_ids[d_Nc - d_Nb], d_ids[d_Nc - 1] );
            }
        }
        if ( Nh2 - d_Nh > 3 ) {
            std::make_heap( d_ids, d_ids + d_Nh );
        } else {
            for ( size_t i = d_Nh + 1; i <= Nh2; i++ )
                std::push_heap( d_ids, d_ids + i );
        }
        d_Nh = Nh2;
        unlock();
    }

    //! Change the prioirties of items in the queue
    inline void changePriorities( const std::vector<std::pair<uint64_t, int8_t>> &list )
    {
        if ( list.empty() )
            return;
        lock();
        // Search the items in the heap
        bool changed = false;
        for ( size_t i = 0; i < d_Nh; i++ ) {
            auto id  = const_cast<ThreadPoolID &>( d_ids[i] );
            auto id2 = id.getLocalID();
            for ( size_t j = 0; j < list.size(); j++ ) {
                if ( list[j].first == id2 ) {
                    id.setPriority( std::max( list[j].second, id.getPriority() ) );
                    changed = true;
                }
            }
        }
        if ( changed )
            std::make_heap( d_ids, d_ids + d_Nh );
        // Search the items in the blocked list
        for ( int64_t i = d_Nc - 1; i >= static_cast<int64_t>( d_Nc - d_Nb ); i-- ) {
            auto id  = const_cast<ThreadPoolID &>( d_ids[i] );
            auto id2 = id.getLocalID();
            for ( size_t j = 0; j < list.size(); j++ ) {
                if ( list[j].first == id2 ) {
                    id.setPriority( std::max( list[j].second, id.getPriority() ) );
                    if ( d_ids[i] > d_ids[d_Nc - 1] )
                        std::swap( d_ids[i], d_ids[d_Nc - 1] );
                }
            }
        }
        unlock();
    }

private:
    inline void lock()
    {
        int expected = 0;
        while ( !d_lock.compare_exchange_weak( expected, 1 ) ) {
            expected = 0;
        }
    }
    inline void unlock()
    {
        int expected = 1;
        d_lock.compare_exchange_weak( expected, 0 );
    }
    inline void checkBlocked()
    {
        if ( d_Nb == 0 )
            return;
        bool test = d_Nh == 0 || d_ids[d_Nc - 1] > d_ids[0];
        if ( test ) {
            for ( size_t i = d_Nc - d_Nb; i < d_Nc; i++ ) {
                if ( const_cast<ThreadPoolID &>( d_ids[i] ).ready() ) {
                    std::swap( d_ids[const_cast<size_t &>( d_Nh )++], d_ids[i] );
                    std::push_heap( d_ids, d_ids + d_Nh );
                    if ( i > d_Nc - d_Nb )
                        std::swap( d_ids[d_Nc - d_Nb], d_ids[i] );
                    const_cast<size_t &>( d_Nb )--;
                }
            }
        }
    }

private: ///// Member data
    volatile std::atomic_int32_t d_lock;
    const size_t d_Nc;
    volatile size_t d_Nh;
    volatile size_t d_Nb;
    volatile ThreadPoolID *d_ids;
};


} // namespace AMP

#endif
