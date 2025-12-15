#ifndef included_AMP_AtomicList_hpp
#define included_AMP_AtomicList_hpp


#include <iostream>
#include <stdexcept>
#include <thread>


namespace AMP {


/******************************************************************
 * Constructor                                                     *
 ******************************************************************/
template<class TYPE, class COMPARE>
AtomicList<TYPE, COMPARE>::AtomicList( size_t capacity,
                                       const TYPE &default_value,
                                       const COMPARE &comp )
    : d_compare( comp ),
      d_capacity( capacity ),
      d_default( default_value ),
      d_objects( new TYPE[capacity] ),
      d_N( 0 ),
      d_next( new std::atomic_int32_t[capacity + 1] ),
      d_unused( 1 ),
      d_N_insert( 0 ),
      d_N_remove( 0 )
{
    d_next[0] = -1;
    for ( int i = 0; i < (int) d_capacity; i++ ) {
        d_next[i + 1] = -5 - i;
        d_objects[i]  = const_cast<TYPE &>( d_default );
    }
}
template<class TYPE, class COMPARE>
AtomicList<TYPE, COMPARE>::~AtomicList()
{
    delete[] d_objects;
    delete[] d_next;
}


/******************************************************************
 * Remove an item                                                  *
 ******************************************************************/
template<class TYPE, class COMPARE>
template<class Compare, class... Args>
inline TYPE AtomicList<TYPE, COMPARE>::remove( Compare compare, const Args &...args )
{
    // Acquiring temporary ownership
    int pos   = 0;
    auto next = lock( 0 );
    while ( true ) {
        if ( next == -1 ) {
            // We have no more entires to search
            unlock( pos, -1 );
            pos = -1;
            break;
        }
        if ( next < 0 )
            throw std::logic_error( "Internal error" );
        // Acquire ownership of the next item
        int next2 = lock( next );
        // Test to see if the object passes compare
        bool test = compare( const_cast<TYPE &>( d_objects[next - 1] ), args... );
        if ( test ) {
            // We want to return this object, update next to point to another entry and remove
            unlock( next, -3 );
            unlock( pos, next2 );
            pos = next;
            break;
        }
        // Release the ownership and move on
        unlock( pos, next );
        pos  = next;
        next = next2;
    }
    TYPE rtn( d_default );
    if ( pos != -1 ) {
        std::swap( rtn, const_cast<TYPE &>( d_objects[pos - 1] ) );
        put_unused( pos );
        --d_N;
        ++d_N_remove;
    }
    return rtn;
}
template<class TYPE, class COMPARE>
inline TYPE AtomicList<TYPE, COMPARE>::remove_first()
{
    TYPE rtn( d_default );
    auto next = lock( 0 );
    if ( next != -1 ) {
        int next2 = lock( next );
        unlock( next, -3 );
        unlock( 0, next2 );
        std::swap( rtn, const_cast<TYPE &>( d_objects[next - 1] ) );
        put_unused( next );
        --d_N;
        ++d_N_remove;
    } else {
        unlock( 0, next );
    }
    return rtn;
}


/******************************************************************
 * Insert an item                                                  *
 ******************************************************************/
template<class TYPE, class COMPARE>
inline void AtomicList<TYPE, COMPARE>::insert( const TYPE &x )
{
    size_t N_used = ++d_N;
    if ( N_used > d_capacity ) {
        --d_N;
        throw std::logic_error( "No room in list" );
    }
    // Get an index to store the entry
    auto index = get_unused();
    if ( index < 1 )
        throw std::logic_error( "Internal error" );
    // Store the object in d_objects
    ++d_N_insert;
    d_objects[index - 1] = x;
    d_next[index]        = -1;
    // Find the position to store and update the next entires
    int pos   = 0;
    auto next = lock( pos );
    while ( true ) {
        // Get the next item in the list (acquiring temporary ownership)
        if ( next == -1 ) {
            // We have no more entires to search, store here
            unlock( pos, index );
            break;
        }
        // Test to see if the object is < the value being compared
        bool test = d_compare.operator()( x, const_cast<TYPE &>( d_objects[next - 1] ) );
        if ( test ) {
            // We want to store this object before next
            d_next[index] = next;
            unlock( pos, index );
            break;
        }
        // Release the ownership and move on
        int last = pos;
        pos      = next;
        next     = lock( next );
        unlock( last, pos );
    }
}


/******************************************************************
 * Check the internal structures of the list                       *
 * This is mostly thread-safe, but blocks all threads              *
 ******************************************************************/
template<class TYPE, class COMPARE>
inline bool AtomicList<TYPE, COMPARE>::check()
{
    // Get the lock and check for any other threads modifying the list
    auto start = lock( 0 );
    std::this_thread::sleep_for( std::chrono::microseconds( 100 ) );
    // Perform the checks on the list
    bool pass    = true;
    int N1       = 0;
    int N2       = 0;
    int N_unused = 0;
    int N_tail   = 0;
    for ( size_t i = 0; i < d_capacity; i++ ) {
        if ( d_objects[i] != d_default )
            N1++;
    }
    for ( size_t i = 0; i <= d_capacity; i++ ) {
        int next = i == 0 ? start : d_next[i].load();
        if ( next > 0 ) {
            N2++;
        } else if ( next < -3 ) {
            N_unused++;
        } else if ( next == -1 ) {
            N_tail++;
        } else {
            pass = false;
        }
    }
    pass    = pass && N_tail == 1 && N1 == d_N && N2 == d_N && N_unused + d_N == (int) d_capacity;
    int it  = 0;
    int pos = 0;
    while ( true ) {
        int next = pos == 0 ? start : d_next[pos].load();
        if ( next == -1 )
            break;
        pos = next;
        it++;
    }
    pass = pass && it == d_N;
    // Unlock the list and return the results
    unlock( 0, start );
    return pass;
}


} // namespace AMP

#endif
