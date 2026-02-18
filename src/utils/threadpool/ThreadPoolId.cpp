// Copyright 2004 Mark Berrill. All Rights Reserved. This work is distributed with permission,
// but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.
#ifndef included_AMP_ThreadPoolIDTmpl
#define included_AMP_ThreadPoolIDTmpl


#include "AMP/utils/threadpool/ThreadPoolId.h"
#include "AMP/utils/threadpool/ThreadPoolWorkItem.h"

#include <stdexcept>


namespace AMP {


/******************************************************************
 * Class functions to for the thread id                            *
 ******************************************************************/
ThreadPoolID::ThreadPoolID( volatile ThreadPoolID &&rhs )
    : d_id( std::move( rhs.d_id ) ),
      d_count( std::move( rhs.d_count ) ),
      d_work( std::move( rhs.d_work ) )
{
    rhs.d_count = nullptr;
    rhs.d_work  = nullptr;
    rhs.d_id    = nullThreadID;
}
ThreadPoolID &ThreadPoolID::operator=( const ThreadPoolID &rhs ) volatile
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return const_cast<ThreadPoolID &>( *this );
    this->reset();
    d_id    = rhs.d_id;
    d_count = rhs.d_count;
    d_work  = rhs.d_work;
    if ( d_count )
        ++( *d_count );
    return const_cast<ThreadPoolID &>( *this );
}
ThreadPoolID &ThreadPoolID::operator=( volatile ThreadPoolID &&rhs ) volatile
{
    std::swap( d_id, rhs.d_id );
    std::swap( d_work, rhs.d_work );
    std::swap( d_count, rhs.d_count );
    return const_cast<ThreadPoolID &>( *this );
}
ThreadPoolID::ThreadPoolID( const volatile ThreadPoolID &rhs )
    : d_id( rhs.d_id ), d_count( rhs.d_count ), d_work( rhs.d_work )
{
    if ( d_count )
        ++( *d_count );
}
#if !defined( WIN32 ) && !defined( _WIN32 ) && !defined( WIN64 ) && !defined( _WIN64 )
ThreadPoolID::ThreadPoolID( const ThreadPoolID &rhs )
    : d_id( rhs.d_id ), d_count( rhs.d_count ), d_work( rhs.d_work )
{
    if ( d_count )
        ++( *d_count );
}
ThreadPoolID &ThreadPoolID::operator=( ThreadPoolID &&rhs )
{
    std::swap( d_id, rhs.d_id );
    std::swap( d_work, rhs.d_work );
    std::swap( d_count, rhs.d_count );
    return const_cast<ThreadPoolID &>( *this );
}
ThreadPoolID &ThreadPoolID::operator=( const ThreadPoolID &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return const_cast<ThreadPoolID &>( *this );
    this->reset();
    d_id    = rhs.d_id;
    d_count = rhs.d_count;
    d_work  = rhs.d_work;
    if ( d_count )
        ++( *d_count );
    return const_cast<ThreadPoolID &>( *this );
}
ThreadPoolID &ThreadPoolID::operator=( const volatile ThreadPoolID &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return const_cast<ThreadPoolID &>( *this );
    this->reset();
    d_id    = rhs.d_id;
    d_count = rhs.d_count;
    d_work  = rhs.d_work;
    if ( d_count )
        ++( *d_count );
    return const_cast<ThreadPoolID &>( *this );
}
ThreadPoolID &ThreadPoolID::operator=( const volatile ThreadPoolID &rhs ) volatile
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return const_cast<ThreadPoolID &>( *this );
    this->reset();
    d_id    = rhs.d_id;
    d_count = rhs.d_count;
    d_work  = rhs.d_work;
    if ( d_count )
        ++( *d_count );
    return const_cast<ThreadPoolID &>( *this );
}
#endif
void ThreadPoolID::reset() volatile
{
    if ( d_count ) {
        int count = --( *d_count );
        if ( count == 0 ) {
            ThreadPoolWorkItem *tmp = reinterpret_cast<ThreadPoolWorkItem *>( d_work );
            delete tmp;
        }
    }
    d_id    = nullThreadID;
    d_count = nullptr;
    d_work  = nullptr;
}
void ThreadPoolID::reset()
{
    if ( d_count ) {
        int count = --( *d_count );
        if ( count == 0 ) {
            ThreadPoolWorkItem *tmp = reinterpret_cast<ThreadPoolWorkItem *>( d_work );
            delete tmp;
        }
    }
    d_id    = nullThreadID;
    d_count = nullptr;
    d_work  = nullptr;
}
uint64_t ThreadPoolID::createId( int8_t priority, uint64_t local_id )
{
    if ( local_id > maxThreadID )
        throw std::logic_error( "Invalid local id" );
    uint64_t id = static_cast<int>( priority ) + 128;
    id          = ( id << 56 ) + local_id;
    return id;
}
void ThreadPoolID::reset( int8_t priority, uint64_t local_id, void *work )
{
    if ( d_count ) {
        int count = --( *d_count );
        if ( count == 0 ) {
            ThreadPoolWorkItem *tmp = reinterpret_cast<ThreadPoolWorkItem *>( d_work );
            delete tmp;
        }
    }
    // Create the id
    d_id = createId( priority, local_id );
    // Create the work and counter
    d_count = nullptr;
    d_work  = work;
    if ( d_work ) {
        d_count  = &( reinterpret_cast<ThreadPoolWorkItem *>( work )->d_count );
        *d_count = 1;
    }
}
uint64_t ThreadPoolID::getLocalID() const
{
    constexpr uint64_t null = ~( (uint64_t) 0 );
    return ( d_id == nullThreadID ) ? null : ( d_id & 0x00FFFFFFFFFFFFFF );
}
int8_t ThreadPoolID::getPriority() const
{
    if ( d_id == nullThreadID )
        return -128;
    uint64_t tmp = d_id >> 56;
    return static_cast<int>( tmp ) - 128;
}
void ThreadPoolID::setPriority( int8_t priority )
{
    if ( d_id == nullThreadID )
        return;
    d_id = createId( priority, getLocalID() );
}
ThreadPoolID::Status ThreadPoolID::status() const
{
    return d_id == nullThreadID ? Status::none :
                                  reinterpret_cast<ThreadPoolWorkItem *>( d_work )->d_state;
}
bool ThreadPoolID::ready() const
{
    bool ready = true;
    if ( !isNull() ) {
        auto tmp = getWork();
        for ( std::size_t i = 0; i < tmp->d_N_ids; i++ )
            ready = ready && tmp->d_ids[i].finished();
    }
    return ready;
}


} // namespace AMP

#endif
