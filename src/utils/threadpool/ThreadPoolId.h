// Copyright © 2004 Mark Berrill. All Rights Reserved. This work is distributed with permission,
// but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.
#ifndef included_AMP_ThreadPoolID
#define included_AMP_ThreadPoolID


#include <atomic>


namespace AMP {


class ThreadPoolWorkItem;


/** \class ThreadPoolID
 *
 * \brief This a class to hold the work item id
 * \details This class hold the id of the work item that is being processed by the thread pool.
 *      It is created when a work item is added to the thread pool and is used by various
 * routines within the thread pool.
 */
class ThreadPoolID
{
public:
    // Work status
    enum class Status : int8_t { none = 0, added = 1, started = 2, finished = 3 };

    // nullID definitins
    static constexpr uint64_t nullThreadID = 0x0FFFFFFFFFFFFFFF;
    static constexpr uint64_t maxThreadID  = 0x00FFFFFFFFFFFFFD;

    //! Empty constructor
    inline ThreadPoolID();

    //! Destructor
    inline ~ThreadPoolID();

    //! Copy constructors
    inline ThreadPoolID( const volatile ThreadPoolID &rhs );
    inline ThreadPoolID( volatile ThreadPoolID &&rhs );
    inline ThreadPoolID &operator=( const ThreadPoolID &rhs ) volatile;
    inline ThreadPoolID &operator=( volatile ThreadPoolID &&rhs ) volatile;
#if !defined( WIN32 ) && !defined( _WIN32 ) && !defined( WIN64 ) && !defined( _WIN64 )
    inline ThreadPoolID( const ThreadPoolID &rhs );
    inline ThreadPoolID &operator=( ThreadPoolID &&rhs );
    inline ThreadPoolID &operator=( const ThreadPoolID &rhs );
    inline ThreadPoolID &operator=( const volatile ThreadPoolID &rhs );
    inline ThreadPoolID &operator=( const volatile ThreadPoolID &rhs ) volatile;
#endif

    // Overload key operators
    inline bool operator==( const ThreadPoolID &rhs ) const
    {
        return !( ( d_id ^ rhs.d_id ) & nullThreadID );
    }
    inline bool operator!=( const ThreadPoolID &rhs ) const
    {
        return ( d_id ^ rhs.d_id ) & nullThreadID;
    }
    inline bool operator>=( const ThreadPoolID &rhs ) const { return d_id >= rhs.d_id; }
    inline bool operator<=( const ThreadPoolID &rhs ) const { return d_id <= rhs.d_id; }
    inline bool operator>( const ThreadPoolID &rhs ) const { return d_id > rhs.d_id; }
    inline bool operator<( const ThreadPoolID &rhs ) const { return d_id < rhs.d_id; }
    inline bool operator==( const volatile ThreadPoolID &rhs ) const volatile
    {
        return !( ( d_id ^ rhs.d_id ) & nullThreadID );
    }
    inline bool operator!=( const volatile ThreadPoolID &rhs ) const volatile
    {
        return ( d_id ^ rhs.d_id ) & nullThreadID;
    }
    inline bool operator>=( const volatile ThreadPoolID &rhs ) const volatile
    {
        return d_id >= rhs.d_id;
    }
    inline bool operator<=( const volatile ThreadPoolID &rhs ) const volatile
    {
        return d_id <= rhs.d_id;
    }
    inline bool operator>( const volatile ThreadPoolID &rhs ) const volatile
    {
        return d_id > rhs.d_id;
    }
    inline bool operator<( const volatile ThreadPoolID &rhs ) const volatile
    {
        return d_id < rhs.d_id;
    }

    //! Reset the id back to a NULL id
    inline void reset() volatile;
    inline void reset();

    //! Return the status of the work item
    inline Status status() const;

    //! Check if the work has started (will return true if it has started or finished)
    inline bool started() const;

    //! Check if the work has finished
    inline bool finished() const;

    //! swap with rhs
    inline void swap( ThreadPoolID &rhs )
    {
        std::swap( this->d_id, rhs.d_id );
        std::swap( this->d_count, rhs.d_count );
        std::swap( this->d_work, rhs.d_work );
    }

    //! Check if thread id is null
    inline bool isNull() const { return d_id == nullThreadID; }

    //! Check if thread id is null
    inline ThreadPoolWorkItem *getWork() const
    {
        return reinterpret_cast<ThreadPoolWorkItem *>( d_work );
    }

public:
    // Is the id ready to process
    inline bool ready() const;
    // Reset the internal data to the given values
    inline void reset( int8_t priority, uint64_t local_id, void *work );
    // Generate an id
    static inline uint64_t createId( int8_t priority, uint64_t local_id );
    // Get the local id
    inline uint64_t getLocalID() const;
    // Get the priority
    inline int8_t getPriority() const;
    // Increase the priority
    inline void setPriority( int8_t priority );
    // Check if the id is initialized
    inline bool initialized() const volatile { return d_id != 0x0FFFFFFFFFFFFFFF; }

private:
    // Data
    uint64_t d_id;                         // 64-bit data to store id
    volatile std::atomic_int32_t *d_count; // Reference count
    void *d_work;                          // Pointer to the work item
};


} // namespace AMP


#include "AMP/utils/threadpool/ThreadPoolId.hpp"


#endif
