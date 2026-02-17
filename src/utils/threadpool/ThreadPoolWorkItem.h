// Copyright 2004 Mark Berrill. All Rights Reserved. This work is distributed with permission,
// but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.
#ifndef included_AMP_ThreadPoolWorkItem
#define included_AMP_ThreadPoolWorkItem


#include "AMP/utils/threadpool/ThreadPoolId.h"

#include <vector>


namespace AMP {


//! Base class for the work item (users should derive from ThreadPool::WorkItemRet)
class ThreadPoolWorkItem
{
public:
    //! Function to run the routine
    virtual void run() = 0;
    //! Will the routine return a result
    virtual bool has_result() const = 0;
    //! Empty deconstructor
    virtual ~ThreadPoolWorkItem()
    {
        delete[] d_ids;
        d_ids   = nullptr;
        d_N_ids = 0;
        d_size  = 0;
    }
    //! Get the number of work ids that this work item depends on
    inline std::size_t get_N_dependencies() const { return d_N_ids; }
    //! Return the list of work ids that we depend on
    std::vector<ThreadPoolID> get_dependencies() const;
    /*!
     * \brief Add a work item to the list of dependencies
     * \param id    Id of the work item to add
     */
    void add_dependency( const ThreadPoolID &id ) { add_dependencies( 1, &id ); }
    /*!
     * \brief Add a list of work item to the list of dependencies
     * \param ids   Ids of the work item to add
     */
    inline void add_dependencies( const std::vector<ThreadPoolID> &ids )
    {
        if ( !ids.empty() )
            add_dependencies( ids.size(), &ids[0] );
    }
    /*!
     * \brief Add a list of work item to the list of dependencies
     *    Note: this function is thread-safe for the threadpool and does not need blocking.
     * \param N     Number of items to add
     * \param ids   Ids of the work item to add
     */
    void add_dependencies( std::size_t N, const ThreadPoolID *ids );

    //! Get the current status
    inline auto getStatus() const { return d_state; }

protected:
    friend class ThreadPool;
    inline ThreadPoolWorkItem()
        : d_state( ThreadPoolID::Status::none ),
          d_N_ids( 0 ),
          d_size( 0 ),
          d_count( 0 ),
          d_ids( nullptr )
    {
    }

private:
    ThreadPoolWorkItem( const ThreadPoolWorkItem & ) = delete;
    ThreadPoolWorkItem &operator=( const ThreadPoolWorkItem & ) = delete;
    volatile ThreadPoolID::Status d_state; // Current state
    uint16_t d_N_ids;                      // Number of dependencies
    uint16_t d_size;                       // Size of d_ids
    volatile std::atomic_int32_t d_count;  // Count used by a thread_id
    ThreadPoolID *d_ids;                   // Pointer to id list
    // Friends
    friend class ThreadPoolID;
};


} // namespace AMP


#endif
