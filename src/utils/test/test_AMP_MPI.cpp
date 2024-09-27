#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include "StackTrace/StackTrace.h"

#include "ProfilerApp.h"


#undef MPI_CLASS
#define MPI_CLASS AMP::AMP_MPI
#define MPI_ASSERT AMP_ASSERT
#define COMM_WORLD AMP_COMM_WORLD

using AMP::UnitTest;
using AMP::Utilities::stringf;


// Set some global define variables
#ifdef USE_MPI
bool useMPI = true;
#else
bool useMPI   = false;
#endif
#ifdef __APPLE__
bool useApple = true;
#else
bool useApple = false;
#endif


// Return the time elapsed in seconds
static inline double time() { return MPI_CLASS::time(); }


// Custom user type
struct mytype {
    int a;
    double b;
    mytype()
    {
        a = -1;
        b = -1.0;
    }
    mytype( int i )
    {
        a = i;
        b = -1.0;
    }
    mytype( int i, double d )
    {
        a = i;
        b = d;
    }
    bool operator==( const mytype &other )
    {
        if ( a == other.a && b == other.b )
            return true;
        return false;
    }
    bool operator!=( const mytype &other )
    {
        if ( a != other.a || b != other.b )
            return true;
        return false;
    }
};


// Record pass/fail/expected message
void record( AMP::UnitTest &ut, bool pass, const std::string &msg, bool expected = false )
{
    if ( pass )
        ut.passes( msg );
    else if ( expected )
        ut.expected_failure( msg );
    else
        ut.failure( msg );
}


// Routines to test Reduce with known data types
template<class type>
int testReduce( MPI_CLASS comm, UnitTest &ut );
template<>
int testReduce<std::complex<double>>( MPI_CLASS comm, UnitTest &ut )
{
    PROFILE( "testReduce<complex double>" );
    std::string typeName      = typeid( std::complex<double> ).name();
    std::complex<double> rank = comm.getRank() + 1;
    std::complex<double> N    = ( ( comm.getSize() * ( comm.getSize() + 1 ) ) / 2 );
    // Test sumReduce
    record( ut, comm.sumReduce<std::complex<double>>( rank ) == N, "sumReduce: " + typeName );
    std::complex<double> y;
    comm.sumReduce<std::complex<double>>( &rank, &y, 1 );
    record( ut, y == N, "sumReduce (x,y): " + typeName );
    return 2; // Return the number of tests
}
template<class type>
int testReduce( MPI_CLASS comm, UnitTest &ut )
{
    PROFILE( "testReduce" );
    std::string typeName = typeid( std::complex<double> ).name();
    auto rank            = (type) comm.getRank();
    auto size            = (type) comm.getSize();
    if ( (int) ( size ) != comm.getSize() ) {
        auto msg = stringf( "Reduce (%s) cannot represent the number of processors",
                            typeid( type ).name() );
        ut.expected_failure( msg );
        return 0;
    }
    type x = 0, y = 0;
    int N = ( ( comm.getSize() * ( comm.getSize() + 1 ) ) / 2 );
    // Test sumReduce
    auto msg = stringf( "sumReduce (%s)", typeid( type ).name() );
    if ( ( (int) ( (type) N ) ) != N )
        ut.expected_failure( msg ); // type cannot represent N
    else
        record( ut, comm.sumReduce<type>( rank + 1 ) == (type) N, msg );
    msg = stringf( "sumReduce (%s) (x,y)", typeid( type ).name() );
    x   = rank + 1;
    comm.sumReduce<type>( &x, &y, 1 );
    if ( ( (int) ( (type) N ) ) != N )
        ut.expected_failure( msg );
    else
        record( ut, y == (type) N, msg );
    // Test minReduce
    msg = stringf( "minReduce (%s)", typeid( type ).name() );
    record( ut, comm.minReduce<type>( rank + 1 ) == 1, msg );
    msg = stringf( "minReduce (%s) (x,y)", typeid( type ).name() );
    comm.minReduce<type>( &x, &y, 1, nullptr );
    record( ut, y == 1, msg );
    // Test maxReduce
    msg = stringf( "maxReduce (%s)", typeid( type ).name() );
    record( ut, comm.maxReduce<type>( rank + 1 ) == size, msg );
    msg = stringf( "maxReduce (%s) (x,y)", typeid( type ).name() );
    comm.maxReduce<type>( &x, &y, 1, nullptr );
    record( ut, y == size, msg );
    // Test minReduce with rank
    int rank_of_min = -1;
    int rank_of_max = -1;
    type rank_min   = rank + 1;
    type rank_max   = rank + 1;
    msg             = stringf( "minReduce-rank (%s)", typeid( type ).name() );
    try {
        comm.minReduce<type>( &rank_min, 1, &rank_of_min );
        record( ut, rank_min == 1 && rank_of_min == 0, msg );
    } catch ( StackTrace::abort_error &err ) {
        ut.failure( std::string( msg ) + " - " + err.message );
    } catch ( ... ) {
        ut.failure( std::string( msg ) + " - caught unknown exception" );
    }
    msg = stringf( "minReduce-rank (%s) (x,y)", typeid( type ).name() );
    try {
        comm.minReduce<type>( &x, &rank_min, 1, &rank_of_min );
        record( ut, rank_min == 1 && rank_of_min == 0, msg );
    } catch ( StackTrace::abort_error &err ) {
        ut.failure( std::string( msg ) + " - " + err.message );
    } catch ( ... ) {
        ut.failure( std::string( msg ) + " - caught unknown exception" );
    }
    // Test maxReduce with rank
    msg = stringf( "maxReduce-rank (%s)", typeid( type ).name() );
    try {
        comm.maxReduce<type>( &rank_max, 1, &rank_of_max );
        record( ut, rank_max == size && rank_of_max == comm.getSize() - 1, msg );
    } catch ( StackTrace::abort_error &err ) {
        ut.failure( std::string( msg ) + " - " + err.message );
    } catch ( ... ) {
        ut.failure( std::string( msg ) + " - caught unknown exception" );
    }
    msg = stringf( "maxReduce-rank (%s) (x,y)", typeid( type ).name() );
    try {
        comm.maxReduce<type>( &x, &rank_max, 1, &rank_of_max );
        record( ut, rank_max == size && rank_of_max == comm.getSize() - 1, msg );
    } catch ( StackTrace::abort_error &err ) {
        ut.failure( std::string( msg ) + " - " + err.message );
    } catch ( ... ) {
        ut.failure( std::string( msg ) + " - caught unknown exception" );
    }
    return 10; // Return the number of tests
}


// Routine to test Scan with known data types
template<class type>
int testScan( MPI_CLASS comm, UnitTest &ut )
{
    PROFILE( "testScan" );
    auto x   = (type) ( comm.getRank() + 1 );
    type y   = 0;
    auto msg = stringf( "sumScan (%s)", typeid( type ).name() );
    comm.sumScan<type>( &x, &y, 1 );
    auto N = (type) ( ( ( comm.getRank() + 1 ) * ( comm.getRank() + 2 ) ) / 2 );
    record( ut, y == N, msg );
    if ( std::is_same_v<type, std::complex<double>> ) {
        return 1;
    }
    msg = stringf( "minScan (%s)", typeid( type ).name() );
    comm.minScan<type>( &x, &y, 1 );
    record( ut, y == (type) 1, msg );
    msg = stringf( "maxScan (%s)", typeid( type ).name() );
    comm.maxScan<type>( &x, &y, 1 );
    record( ut, y == x, msg );
    return 3; // Return the number of tests
}


// Routine to test bcast
template<class type>
int testBcast( MPI_CLASS comm, UnitTest &ut, type default_val, type new_val )
{
    PROFILE( "testBcast" );
    for ( int i = 0; i < comm.getSize(); i++ ) {
        type tmp1 = default_val;
        if ( comm.getRank() == i )
            tmp1 = new_val;
        auto msg = stringf( "bcast scalar (%s) from rank %i", typeid( type ).name(), i );
        record( ut, comm.bcast( tmp1, i ) == new_val, msg );
        type tmp2[2];
        tmp2[0] = default_val;
        tmp2[1] = default_val;
        if ( comm.getRank() == i ) {
            tmp2[0] = new_val;
            tmp2[1] = new_val;
        }
        msg = stringf( "bcast vector (%s) from rank %i", typeid( type ).name(), i );
        comm.bcast( tmp2, 2, i );
        record( ut, tmp2[0] == new_val && tmp2[1] == new_val, msg );
    }
    return 2 * comm.getSize(); // Return the number of tests
}


// Routine to test allGather
template<class type>
int testAllGather( MPI_CLASS comm, UnitTest &ut )
{
    PROFILE( "testAllGather" );
    // Test scalar allGather
    auto x1  = (type) comm.getRank();
    auto *x2 = new type[comm.getSize()];
    comm.allGather( x1, x2 );
    bool pass = true;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        type test = i;
        if ( x2[i] != test )
            pass = false;
    }
    auto msg = stringf( "allGather scalar (%s)", typeid( type ).name() );
    record( ut, pass, msg );
    // Test vector allGather
    int N     = ( comm.getSize() * ( comm.getSize() + 1 ) ) / 2;
    auto *x3  = new type[comm.getRank() + 1];
    auto *x4  = new type[N];
    auto *x5  = new type[N];
    int *size = new int[comm.getSize()];
    for ( int i = 0; i <= comm.getRank(); i++ )
        x3[i] = (type) comm.getRank();
    int tot1 = comm.allGather( x3, comm.getRank() + 1, x4 );
    int tot2 = comm.allGather( x3, comm.getRank() + 1, x5, size );
    pass     = true;
    if ( tot1 != N || tot2 != N )
        pass = false;
    int k = 0;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        if ( size[i] != i + 1 )
            pass = false;
        if ( !pass )
            break;
        for ( int j = 0; j <= i; j++ ) {
            type test = i;
            if ( x4[k] != test || x5[k] != test )
                pass = false;
            k++;
        }
    }
    msg = stringf( "allGather vector (%s)", typeid( type ).name() );
    record( ut, pass, msg );
    delete[] x2;
    delete[] x3;
    delete[] x4;
    delete[] x5;
    delete[] size;
    // Test vector allGather with know recive sizes and non-zero displacements
    auto *send      = new type[comm.getRank() + 1];
    auto *recv      = new type[comm.getSize() * comm.getSize() + 1];
    auto *recv_size = new int[comm.getSize()];
    auto *recv_disp = new int[comm.getSize()];
    for ( int i = 0; i <= comm.getRank(); i++ )
        send[i] = i;
    for ( int i = 0; i < comm.getSize(); i++ )
        recv_size[i] = i + 1;
    for ( int i = 0; i < comm.getSize(); i++ )
        recv_disp[i] = 1 + i * comm.getSize() + comm.getSize() - i - 1;
    for ( int i = 0; i <= comm.getSize() * comm.getSize(); i++ )
        recv[i] = (type) -1;
    int tot = comm.allGather( send, comm.getRank() + 1, recv, recv_size, recv_disp, true );
    pass    = true;
    if ( tot != N )
        pass = false;
    auto test = (type) -1;
    if ( recv[0] != test )
        pass = false;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        for ( int j = 0; j < comm.getSize(); j++ ) {
            int l = j + i * comm.getSize() + 1 - recv_disp[i];
            if ( l >= 0 )
                test = l;
            else
                test = (type) -1;
            if ( recv[j + i * comm.getSize() + 1] != test )
                pass = false;
        }
    }
    msg = stringf( "allGather vector with known recv and non-zero displacements (%s)",
                   typeid( type ).name() );
    record( ut, pass, msg );
    delete[] send;
    delete[] recv;
    delete[] recv_size;
    delete[] recv_disp;
    // Test vector allGather with no elements
    size = new int[comm.getSize()];
    msg  = stringf( "allGather scalar (%s)", typeid( type ).name() );
    try {
        comm.allGather( &x1, 0, (type *) nullptr, size );
        ut.passes( msg );
    } catch ( ... ) {
        ut.failure( msg );
    }
    delete[] size;
    return 4; // Return the number of tests
}


// Routine to test setGather
template<class type>
int testSetGather( MPI_CLASS comm, UnitTest &ut )
{
    PROFILE( "testSetGather" );
    auto x1 = (type) comm.getRank();
    std::set<type> set;
    set.insert( x1 );
    comm.setGather( set );
    bool pass = true;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        type x2 = i;
        if ( set.find( x2 ) == set.end() )
            pass = false;
    }
    auto msg = stringf( "setGather (%s)", typeid( type ).name() );
    record( ut, pass, msg );
    return 1; // Return the number of tests
}


// Routine to test mapGather
template<class type>
int testMapGather( MPI_CLASS comm, UnitTest &ut )
{
    PROFILE( "testMapGather" );
    auto x1 = (type) comm.getRank();
    std::map<int, type> map;
    map.insert( std::pair<int, type>( comm.getRank(), x1 ) );
    comm.mapGather( map );
    bool pass = true;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        type x2 = i;
        auto it = map.find( i );
        if ( it == map.end() )
            pass = false;
        else if ( it->second != x2 )
            pass = false;
    }
    auto msg = stringf( "mapGather (%s)", typeid( type ).name() );
    record( ut, pass, msg );
    return 1; // Return the number of tests
}


// Routine to test allToAll
template<class type>
int testAllToAll( MPI_CLASS comm, UnitTest &ut )
{
    PROFILE( "testAllToAll" );
    bool pass;
    int size = 0;
    type *send_data, *recv_data;
    auto *send_cnt  = new int[comm.getSize()];
    auto *recv_cnt  = new int[comm.getSize()];
    auto *send_disp = new int[comm.getSize()];
    auto *recv_disp = new int[comm.getSize()];
    // Test allToAll with a scalar value to each processor
    send_data = new type[comm.getSize()];
    recv_data = new type[comm.getSize()];
    for ( int i = 0; i < comm.getSize(); i++ )
        send_data[i] = comm.getSize();
    comm.allToAll( 1, send_data, recv_data );
    pass = true;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        type test = comm.getSize();
        if ( recv_data[i] != test )
            pass = false;
    }
    delete[] send_data;
    delete[] recv_data;
    auto msg = stringf( "allToAll with scalar (%s)", typeid( type ).name() );
    record( ut, pass, msg );
    // Test allToAll vector with a scalar value to each processor
    send_data = new type[comm.getSize()];
    recv_data = new type[comm.getSize()];
    for ( int i = 0; i < comm.getSize(); i++ ) {
        send_cnt[i]  = 1;
        recv_cnt[i]  = 1;
        send_disp[i] = i;
        recv_disp[i] = i;
        send_data[i] = comm.getSize();
        recv_data[i] = 0;
    }
    size = comm.allToAll( send_data, send_cnt, send_disp, recv_data, recv_cnt, recv_disp, true );
    pass = true;
    if ( size != comm.getSize() )
        pass = false;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        type test = comm.getSize();
        if ( recv_data[i] != test )
            pass = false;
    }
    delete[] send_data;
    delete[] recv_data;
    msg = stringf( "allToAll vector with scalar (%s)", typeid( type ).name() );
    record( ut, pass, msg );
    // Test allToAll with a variable number of values per processor and spacing
    send_data = new type[comm.getSize() * comm.getSize()];
    recv_data = new type[2 * comm.getRank() * comm.getSize()];
    for ( int i = 0; i < comm.getSize(); i++ ) {
        send_cnt[i]  = i;
        recv_cnt[i]  = comm.getRank();
        send_disp[i] = i * comm.getSize();
        recv_disp[i] = 2 * i * comm.getRank();
        for ( int j = 0; j < comm.getSize(); j++ ) {
            if ( j < i )
                send_data[j + send_disp[i]] = i;
            else
                send_data[j + send_disp[i]] = (type) -1;
        }
    }
    for ( int i = 0; i < 2 * comm.getRank() * comm.getSize(); i++ )
        recv_data[i] = (type) -2;
    size = comm.allToAll( send_data, send_cnt, send_disp, recv_data, recv_cnt, recv_disp, true );
    pass = true;
    if ( size != comm.getRank() * comm.getSize() )
        pass = false;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        for ( int j = 0; j < 2 * comm.getRank(); j++ ) {
            if ( j < comm.getRank() ) {
                type test = comm.getRank();
                if ( recv_data[j + recv_disp[i]] != test )
                    pass = false;
            } else {
                auto test = (type) -2;
                if ( recv_data[j + recv_disp[i]] != test )
                    pass = false;
            }
        }
    }
    delete[] send_data;
    delete[] recv_data;
    msg = stringf( "allToAll with vector of known size and displacements (%s)",
                   typeid( type ).name() );
    record( ut, pass, msg );
    // Test allToAll with a unknown receive length
    send_data        = new type[comm.getSize() * comm.getSize()];
    auto *recv_data1 = new type[comm.getSize() * comm.getSize()];
    auto *recv_data2 = new type[comm.getSize() * comm.getSize()];
    for ( int i = 0; i < comm.getSize(); i++ ) {
        send_cnt[i]  = i;
        recv_cnt[i]  = -1;
        send_disp[i] = i * comm.getSize();
        recv_disp[i] = -1;
        for ( int j = 0; j < comm.getSize(); j++ ) {
            if ( j < i )
                send_data[j + send_disp[i]] = i;
            else
                send_data[j + send_disp[i]] = (type) -1;
        }
    }
    for ( int i = 0; i < comm.getSize() * comm.getSize(); i++ ) {
        recv_data1[i] = (type) -2;
        recv_data2[i] = (type) -2;
    }
    int size1 =
        comm.allToAll( send_data, send_cnt, send_disp, recv_data1, recv_cnt, recv_disp, false );
    int size2  = comm.allToAll( send_data, send_cnt, send_disp, recv_data2 );
    bool pass1 = true;
    bool pass2 = true;
    if ( size1 != comm.getRank() * comm.getSize() )
        pass1 = false;
    if ( size2 != comm.getRank() * comm.getSize() )
        pass2 = false;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        if ( recv_cnt[i] != comm.getRank() || recv_disp[i] != i * comm.getRank() )
            pass1 = false;
    }
    for ( int i = 0; i < comm.getRank() * comm.getSize(); i++ ) {
        type test = comm.getRank();
        if ( recv_data1[i] != test )
            pass1 = false;
        if ( recv_data2[i] != test )
            pass2 = false;
    }
    delete[] send_data;
    delete[] recv_data1;
    delete[] recv_data2;
    msg = stringf( "allToAll with vector of unknown size (%s)", typeid( type ).name() );
    record( ut, pass1, msg );
    msg =
        stringf( "allToAll with vector of unknown size with NULL recv(%s)", typeid( type ).name() );
    record( ut, pass2, msg );
    // Free temporary variables
    delete[] send_cnt;
    delete[] recv_cnt;
    delete[] send_disp;
    delete[] recv_disp;
    return 5; // Return the number of tests
}


// Routine to test send/recv
template<class type>
int testSendRecv( MPI_CLASS comm, UnitTest &ut, type v1, type v2 )
{
    PROFILE( "testSendRecv" );
    // Test send-recv with a known length
    for ( int i = 0; i < comm.getSize(); i++ ) {
        for ( int j = 0; j < comm.getSize(); j++ ) {
            type x   = v1;
            int tag  = i + j * comm.getSize();
            auto msg = stringf( "send-recv %i-%i known length (%s)", i, j, typeid( type ).name() );
            if ( i == j ) {
                // We are not allowed to send/receive from the same processor
                continue;
            } else if ( i == comm.getRank() ) {
                // We are sending
                x = v2;
                comm.send( &x, 1, j, tag );
            } else if ( j == comm.getRank() ) {
                // We are recieving
                int size = 1;
                comm.recv( &x, size, i, false, tag );
                record( ut, size == 1 && x == v2, msg );
            }
        }
    }
    // Test send-recv with an unknown length
    for ( int i = 0; i < comm.getSize(); i++ ) {
        for ( int j = 0; j < comm.getSize(); j++ ) {
            type x  = v1;
            int tag = i + j * comm.getSize();
            auto msg =
                stringf( "send-recv %i-%i unknown length (%s)", i, j, typeid( type ).name() );
            if ( i == j ) {
                // We are not allowed to send/receive from the same processor
                continue;
            } else if ( i == comm.getRank() ) {
                // We are sending
                x = v2;
                comm.send( &x, 1, j, tag );
            } else if ( j == comm.getRank() ) {
                // We are recieving
                int size = 1;
                comm.recv( &x, size, i, true, tag );
                record( ut, size == 1 && x == v2, msg );
            }
        }
    }
    // Test send-recv with an empty length
    for ( int i = 0; i < comm.getSize(); i++ ) {
        for ( int j = 0; j < comm.getSize(); j++ ) {
            type x   = v1;
            int tag  = i + j * comm.getSize();
            auto msg = stringf( "send-recv %i-%i empty length (%s)", i, j, typeid( type ).name() );
            if ( i == j ) {
                // We are not allowed to send/receive from the same processor
                continue;
            } else if ( i == comm.getRank() ) {
                // We are sending
                x = v2;
                comm.send( &x, 0, j, tag );
            } else if ( j == comm.getRank() ) {
                // We are recieving
                int size = comm.probe( i, tag );
                comm.recv( &x, size, i, false, tag );
                record( ut, size == 0, msg );
            }
        }
    }
    return 3 * comm.getSize() * comm.getSize(); // Return the number of tests
}


// Routine to test Isend/Irecv
template<class type>
int testIsendIrecv( MPI_CLASS comm, UnitTest &ut, type v1, type v2 )
{
    PROFILE( "testIsendIrecv" );
    std::vector<MPI_CLASS::Request> sendRequest;
    std::vector<MPI_CLASS::Request> recvRequest;
    // Send all msgs
    for ( int i = 0; i < comm.getSize(); i++ ) {
        // Check if the current rank is sending
        if ( i != comm.getRank() )
            continue;
        for ( int j = 0; j < comm.getSize(); j++ ) {
            // Start a non-blocking send
            int tag      = i + j * comm.getSize();
            auto request = comm.Isend( &v1, 1, j, tag );
            sendRequest.insert( sendRequest.begin(), request );
        }
    }
    // Recv all msgs
    auto *recv_buffer = new type[comm.getSize()];
    for ( int i = 0; i < comm.getSize(); i++ )
        recv_buffer[i] = v2;
    recv_buffer[comm.getRank()] = v1;
    for ( int j = 0; j < comm.getSize(); j++ ) {
        // Check if the current rank is recieving
        if ( j != comm.getRank() )
            continue;
        for ( int i = 0; i < comm.getSize(); i++ ) {
            // Start a non-blocking recv
            int tag      = i + j * comm.getSize();
            auto request = comm.Irecv( &recv_buffer[i], 1, i, tag );
            recvRequest.insert( recvRequest.begin(), request );
        }
    }
    // Wait for all communications to finish
    MPI_CLASS::wait( sendRequest[0] );
    sendRequest.erase( sendRequest.begin() + 0 );
    while ( !sendRequest.empty() ) {
        int index = comm.waitAny( sendRequest.size(), &( sendRequest[0] ) );
        sendRequest.erase( sendRequest.begin() + index );
    }
    auto finished = MPI_CLASS::waitSome( recvRequest.size(), recvRequest.data() );
    if ( !recvRequest.empty() ) {
        MPI_ASSERT( !finished.empty() );
        for ( auto it = finished.rbegin(); it != finished.rend(); ++it )
            recvRequest.erase( recvRequest.begin() + ( *it ) );
    }
    if ( !recvRequest.empty() )
        MPI_CLASS::waitAll( recvRequest.size(), &( recvRequest[0] ) );
    AMP::Utilities::unique( finished );
    // Check the received values
    bool pass = true;
    for ( int i = 0; i < comm.getSize(); i++ ) {
        if ( recv_buffer[i] != v1 )
            pass = false;
    }
    auto msg = stringf( "Isend-Irecv (%s)", typeid( type ).name() );
    record( ut, pass, msg );
    delete[] recv_buffer;
    return comm.getSize() * comm.getSize(); // Return the number of tests
}


// Routine to test CommRanks
int testCommRanks( MPI_CLASS comm, UnitTest &ut )
{
    std::vector<int> neighbors;
    for ( int i = 0; i < comm.getSize(); i++ )
        if ( ( i % 2 ) == 0 )
            neighbors.push_back( i );
    std::vector<int> neighbors2 = comm.commRanks( neighbors );
    bool pass                   = true;
    if ( comm.getRank() % 2 == 0 ) {
        pass = static_cast<int>( neighbors2.size() ) == comm.getSize();
        if ( pass ) {
            for ( int i = 0; i < comm.getSize(); i++ )
                pass = pass && neighbors2[i] == i;
        }
    } else {
        pass = neighbors2.empty();
    }
    auto ranks = comm.globalRanks();
    pass       = pass && (int) ranks.size() == comm.getSize();
    for ( int rank : ranks )
        pass = pass && rank >= 0;
    auto ranks2 = ranks;
    AMP::Utilities::unique( ranks2 );
    pass = pass && ranks.size() == ranks2.size();
    comm.barrier();
    record( ut, pass, "commRanks" );
    return 1; // Return the number of tests
}


// Structure to contain timer results
struct testCommTimerResults {
    int N_reduce        = 0;
    int N_scan          = 0;
    int N_bcast         = 0;
    int N_allGather     = 0;
    int N_setGather     = 0;
    int N_mapGather     = 0;
    int N_allToAll      = 0;
    int N_sendRecv      = 0;
    int N_IsendIrecv    = 0;
    double t_reduce     = 0;
    double t_scan       = 0;
    double t_bcast      = 0;
    double t_allGather  = 0;
    double t_setGather  = 0;
    double t_mapGather  = 0;
    double t_allToAll   = 0;
    double t_sendRecv   = 0;
    double t_IsendIrecv = 0;
    // Print the results
    void print()
    {
        auto printLine = []( const char *msg, int N, double time ) {
            printf( "   %-12s %4i  %3.3e %5.1f us\n", msg, N, time, 1e6 * time / N );
        };
        printf( "%16s  N     t_tot     t_avg\n", "" );
        printLine( "Reduce", N_reduce, t_reduce );
        printLine( "Scan", N_scan, t_scan );
        printLine( "Bcast", N_bcast, t_bcast );
        printLine( "allGather", N_allGather, t_allGather );
        printLine( "allToAll", N_allToAll, t_allToAll );
        printLine( "send-recv", N_sendRecv, t_sendRecv );
        printLine( "Isend-Irecv", N_IsendIrecv, t_IsendIrecv );
        printLine( "setGather", N_setGather, t_setGather );
        printLine( "mapGather", N_mapGather, t_mapGather );
    }
};


// This routine will test a single MPI communicator
testCommTimerResults testComm( MPI_CLASS comm, UnitTest &ut )
{
    PROFILE( "testComm" );
    testCommTimerResults timer;
    double start_time;
    // Test the tag
    int tag0        = comm.newTag();
    MPI_CLASS comm2 = comm;
    MPI_CLASS comm3( comm );
    bool pass = tag0 > 0 && tag0 < comm.maxTag();
    for ( int i = 1; i < 64; i++ ) {
        if ( comm.newTag() != tag0 + i )
            pass = false;
    }
    for ( int i = 1; i <= 64; i++ ) {
        if ( comm2.newTag() != tag0 + 63 + i )
            pass = false;
    }
    for ( int i = 1; i <= 128; i++ ) {
        if ( comm3.newTag() != tag0 + 127 + i )
            pass = false;
    }
    record( ut, pass, "newTag" );
    // Test min, max, and sum reduce
    start_time = time();
    timer.N_reduce += testReduce<unsigned char>( comm, ut );
    timer.N_reduce += testReduce<char>( comm, ut );
    timer.N_reduce += testReduce<unsigned int>( comm, ut );
    timer.N_reduce += testReduce<int>( comm, ut );
    timer.N_reduce += testReduce<unsigned long int>( comm, ut );
    timer.N_reduce += testReduce<long int>( comm, ut );
    timer.N_reduce += testReduce<size_t>( comm, ut );
    timer.N_reduce += testReduce<float>( comm, ut );
    timer.N_reduce += testReduce<double>( comm, ut );
    timer.N_reduce += testReduce<std::complex<double>>( comm, ut );
    mytype tmp1( 1, -1.0 );
    mytype tmp2;
    if ( comm.getSize() > 1 ) {
        // We can't perform a reduce on an unknown data type (this should throw an error)
        try {
            // This should fail
            tmp2 = comm.sumReduce<mytype>( tmp1 );
            ut.failure( "sumReduce should give an error with an unknown type" );
        } catch ( ... ) {
            ut.passes( "sumReduce should give an error with an unknown type" );
        }
        try {
            // This should fail
            tmp2 = comm.minReduce<mytype>( tmp1 );
            ut.failure( "minReduce should give an error with an unknown type" );
        } catch ( ... ) {
            ut.passes( "minReduce should give an error with an unknown type" );
        }
        try {
            // This should fail
            tmp2 = comm.maxReduce<mytype>( tmp1 );
            ut.failure( "maxReduce should give an error with an unknown type" );
        } catch ( ... ) {
            ut.passes( "maxReduce should give an error with an unknown type" );
        }
        timer.N_reduce += 3;
    }
    timer.t_reduce = time() - start_time;
    // Test min, max, and sum scan
    start_time = time();
    timer.N_scan += testScan<unsigned char>( comm, ut );
    timer.N_scan += testScan<char>( comm, ut );
    timer.N_scan += testScan<unsigned int>( comm, ut );
    timer.N_scan += testScan<int>( comm, ut );
    timer.N_scan += testScan<unsigned long int>( comm, ut );
    timer.N_scan += testScan<long int>( comm, ut );
    timer.N_scan += testScan<size_t>( comm, ut );
    timer.N_scan += testScan<float>( comm, ut );
    timer.N_scan += testScan<double>( comm, ut );
    timer.N_scan += testScan<std::complex<double>>( comm, ut );
    if ( comm.getSize() > 1 ) {
        // We can't perform a reduce on an unknown data type (this should throw an error)
        try {
            // This should fail
            comm.sumScan<mytype>( &tmp1, &tmp2, 1 );
            ut.failure( "sumReduce should give an error with an unknown type" );
        } catch ( ... ) {
            ut.passes( "sumReduce should give an error with an unknown type" );
        }
        try {
            // This should fail
            comm.minScan<mytype>( &tmp1, &tmp2, 1 );
            ut.failure( "minReduce should give an error with an unknown type" );
        } catch ( ... ) {
            ut.passes( "minReduce should give an error with an unknown type" );
        }
        try {
            // This should fail
            comm.maxScan<mytype>( &tmp1, &tmp2, 1 );
            ut.failure( "maxReduce should give an error with an unknown type" );
        } catch ( ... ) {
            ut.passes( "maxReduce should give an error with an unknown type" );
        }
        timer.N_scan += 3;
    }
    timer.t_scan = time() - start_time;
    // Test bcast
    start_time = time();
    timer.N_bcast += testBcast<unsigned char>( comm, ut, 0, 1 );
    timer.N_bcast += testBcast<char>( comm, ut, -1, 1 );
    timer.N_bcast += testBcast<unsigned int>( comm, ut, 0, 1 );
    timer.N_bcast += testBcast<int>( comm, ut, -1, 1 );
    timer.N_bcast += testBcast<unsigned long int>( comm, ut, 0, 1 );
    timer.N_bcast += testBcast<long int>( comm, ut, -1, 1 );
    timer.N_bcast += testBcast<size_t>( comm, ut, 0, 1 );
    timer.N_bcast += testBcast<float>( comm, ut, -1.0, 1.0 );
    timer.N_bcast += testBcast<double>( comm, ut, -1.0, 1.0 );
    mytype tmp3( -1, -1.0 );
    mytype tmp4( 1, 1.0 );
    timer.N_bcast += testBcast<mytype>( comm, ut, tmp3, tmp4 );
    timer.t_bcast = time() - start_time;
    // Test barrier
    comm.barrier();
    // Test gather
    start_time = time();
    timer.N_allGather += testAllGather<unsigned char>( comm, ut );
    timer.N_allGather += testAllGather<char>( comm, ut );
    timer.N_allGather += testAllGather<unsigned int>( comm, ut );
    timer.N_allGather += testAllGather<int>( comm, ut );
    timer.N_allGather += testAllGather<unsigned long int>( comm, ut );
    timer.N_allGather += testAllGather<long int>( comm, ut );
    timer.N_allGather += testAllGather<size_t>( comm, ut );
    timer.N_allGather += testAllGather<float>( comm, ut );
    timer.N_allGather += testAllGather<double>( comm, ut );
    timer.N_allGather += testAllGather<std::complex<double>>( comm, ut );
    timer.N_allGather += testAllGather<mytype>( comm, ut );
    timer.t_allGather = time() - start_time;
    // Test std::set gather
    start_time = time();
    timer.N_setGather += testSetGather<unsigned char>( comm, ut );
    timer.N_setGather += testSetGather<char>( comm, ut );
    timer.N_setGather += testSetGather<unsigned int>( comm, ut );
    timer.N_setGather += testSetGather<int>( comm, ut );
    timer.N_setGather += testSetGather<unsigned long int>( comm, ut );
    timer.N_setGather += testSetGather<long int>( comm, ut );
    timer.N_setGather += testSetGather<size_t>( comm, ut );
    timer.N_setGather += testSetGather<float>( comm, ut );
    timer.N_setGather += testSetGather<double>( comm, ut );
    timer.t_setGather = time() - start_time;
    // Test std::map gather
    start_time = time();
    timer.N_mapGather += testMapGather<unsigned char>( comm, ut );
    timer.N_mapGather += testMapGather<char>( comm, ut );
    timer.N_mapGather += testMapGather<unsigned int>( comm, ut );
    timer.N_mapGather += testMapGather<int>( comm, ut );
    timer.N_mapGather += testMapGather<unsigned long int>( comm, ut );
    timer.N_mapGather += testMapGather<long int>( comm, ut );
    timer.N_mapGather += testMapGather<size_t>( comm, ut );
    timer.N_mapGather += testMapGather<float>( comm, ut );
    timer.N_mapGather += testMapGather<double>( comm, ut );
    timer.t_mapGather = time() - start_time;
    // Test allToAlll
    start_time = time();
    timer.N_allToAll += testAllToAll<unsigned char>( comm, ut );
    timer.N_allToAll += testAllToAll<char>( comm, ut );
    timer.N_allToAll += testAllToAll<unsigned int>( comm, ut );
    timer.N_allToAll += testAllToAll<int>( comm, ut );
    timer.N_allToAll += testAllToAll<unsigned long int>( comm, ut );
    timer.N_allToAll += testAllToAll<long int>( comm, ut );
    timer.N_allToAll += testAllToAll<size_t>( comm, ut );
    timer.N_allToAll += testAllToAll<float>( comm, ut );
    timer.N_allToAll += testAllToAll<double>( comm, ut );
    timer.N_allToAll += testAllToAll<std::complex<double>>( comm, ut );
    timer.N_allToAll += testAllToAll<mytype>( comm, ut );
    timer.t_allToAll = time() - start_time;
    // Test send/recv
    start_time = time();
    timer.N_sendRecv += testSendRecv<unsigned char>( comm, ut, 0, 1 );
    timer.N_sendRecv += testSendRecv<char>( comm, ut, -1, 1 );
    timer.N_sendRecv += testSendRecv<unsigned int>( comm, ut, 0, 1 );
    timer.N_sendRecv += testSendRecv<int>( comm, ut, -1, 1 );
    timer.N_sendRecv += testSendRecv<unsigned long int>( comm, ut, 0, 1 );
    timer.N_sendRecv += testSendRecv<long int>( comm, ut, -1, 1 );
    timer.N_sendRecv += testSendRecv<size_t>( comm, ut, 0, 1 );
    timer.N_sendRecv += testSendRecv<float>( comm, ut, -1.0, 1.0 );
    timer.N_sendRecv += testSendRecv<double>( comm, ut, -1.0, 1.0 );
    timer.N_sendRecv += testSendRecv<mytype>( comm, ut, tmp3, tmp4 );
    timer.t_sendRecv = time() - start_time;
    // Test Isend/Irecv
    start_time = time();
    timer.N_IsendIrecv += testIsendIrecv<unsigned char>( comm, ut, 0, 1 );
    timer.N_IsendIrecv += testIsendIrecv<char>( comm, ut, -1, 1 );
    timer.N_IsendIrecv += testIsendIrecv<unsigned int>( comm, ut, 0, 1 );
    timer.N_IsendIrecv += testIsendIrecv<int>( comm, ut, -1, 1 );
    timer.N_IsendIrecv += testIsendIrecv<unsigned long int>( comm, ut, 0, 1 );
    timer.N_IsendIrecv += testIsendIrecv<long int>( comm, ut, -1, 1 );
    timer.N_IsendIrecv += testIsendIrecv<size_t>( comm, ut, 0, 1 );
    timer.N_IsendIrecv += testIsendIrecv<float>( comm, ut, -1.0, 1.0 );
    timer.N_IsendIrecv += testIsendIrecv<double>( comm, ut, -1.0, 1.0 );
    timer.N_IsendIrecv += testIsendIrecv<mytype>( comm, ut, tmp3, tmp4 );
    timer.t_IsendIrecv = time() - start_time;
    // Test all/any reduce
    int size = comm.getSize();
    int rank = comm.getRank();
    std::vector<bool> src( size + 2, false );
    src[rank] = true;
    src[size] = true;
    auto all  = src;
    auto any  = src;
    comm.anyReduce( any );
    comm.allReduce( all );
    pass = comm.allReduce( true ) && comm.anyReduce( rank == 0 );
    pass = pass && any[size] && all[size] && !any[size + 1] && !all[size + 1];
    for ( int i = 0; i < rank; i++ )
        pass = pass && any[i] && !all[i];
    record( ut, pass, "anyReduce/allReduce" );
    // Test serializeStart()
    if ( size < 64 ) {
        double start = MPI_CLASS::time();
        comm.serializeStart();
        std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
        comm.serializeStop();
        double stop = MPI_CLASS::time();
        double avg  = 1e3 * ( stop - start ) / size;
        if ( avg > 98 && avg < 130 )
            ut.passes( "serialize" );
        else
            ut.failure( "serialize: " + std::to_string( avg ) );
    }
    // Test commRanks
    testCommRanks( comm, ut );
    return timer;
}


// Test comm dup and the number of communicators that can be created
void testCommDup( UnitTest &ut )
{
#if defined( __clang__ ) && defined( __APPLE__ )
    // The MPI error handler crashes so this test fails
    // This seems to be a MAC? + Clang + MPICH? issue only
    ut.expected_failure( "testCommDup skipped for this architecture/compiler" );
#else
    MPI_CLASS globalComm( COMM_WORLD );
    MPI_CLASS dupComm = globalComm.dup();
    if ( globalComm.getCommunicator() != dupComm.getCommunicator() &&
         dupComm.getSize() == globalComm.getSize() && dupComm.getRank() == globalComm.getRank() ) {
        ut.passes( "dup comm" );
    } else {
        ut.failure( "dup comm" );
        return;
    }
    #if defined( USE_PETSC ) && !defined( USE_MPI )
    ut.expected_failure( "Skipping dup tests, PETSc (no-mpi) has a limit of 128 unique comms" );
    return;
    #endif
    int N_comm_try = 2000; // Maximum number of comms to try and create
    std::vector<MPI_CLASS> comms;
    comms.reserve( N_comm_try );
    try {
        for ( int i = 0; i < N_comm_try; i++ ) {
            MPI_CLASS tmp_comm = globalComm.dup();
            comms.push_back( tmp_comm );
            MPI_ASSERT( globalComm.getCommunicator() != comms[i].getCommunicator() );
            MPI_ASSERT( comms.back().sumReduce<int>( 1 ) ==
                        globalComm.getSize() ); // We need to communicate as part of the test
        }
        ut.passes( AMP::Utilities::stringf( "Created %i comms", N_comm_try ) );
    } catch ( ... ) {
        if ( comms.size() < 252 ) {
            ut.failure( "Could not create 252 different communicators" );
        } else {
            int N = comms.size();
            ut.expected_failure( AMP::Utilities::stringf(
                "Failed to create an unlimited number of comms (%i)", N ) );
        }
        AMP::pout << "Maximum number of concurrent communicators: " << comms.size() << std::endl;
    }
    comms.clear();
    size_t N_dup = 0;
    globalComm.barrier();
    try {
        double start = MPI_CLASS::time();
        for ( int i = 0; i < N_comm_try; i++ ) {
            MPI_CLASS tmp_comm1 = globalComm.dup();
            MPI_CLASS tmp_comm2 = globalComm.dup();
            MPI_ASSERT( globalComm.getCommunicator() != tmp_comm1.getCommunicator() );
            MPI_ASSERT( globalComm.getCommunicator() != tmp_comm2.getCommunicator() );
            MPI_ASSERT( tmp_comm1.getCommunicator() != tmp_comm2.getCommunicator() );
            MPI_ASSERT( tmp_comm1.sumReduce<int>( 1 ) ==
                        globalComm.getSize() ); // We need to communicate as part of the test
            MPI_ASSERT( tmp_comm2.sumReduce<int>( 1 ) ==
                        globalComm.getSize() ); // We need to communicate as part of the test
            N_dup += 2;
        }
        double stop = MPI_CLASS::time();
        ut.passes( "Created/Destroyed an unlimited number of comms" );
        auto msg = stringf( "Time to create/destroy comm using MPI_CLASS::dup() is: %0.1f us",
                            1e6 * ( stop - start ) / N_dup );
        AMP::pout << msg << std::endl;
    } catch ( ... ) {
        ut.failure( "Failed to create/destroy an unlimited number of comms" );
        AMP::pout << "Maximum number of communicators created with destruction: " << N_dup
                  << std::endl;
    }
#endif
}


// Test rand
bool addRand( const AMP::AMP_MPI &comm, std::vector<size_t> &vals )
{
    auto v1 = comm.rand();
    auto v2 = comm.rand();
    vals.push_back( v1 );
    vals.push_back( v2 );
    return v1 == comm.bcast( v1, 0 ) && v2 == comm.bcast( v2, 0 );
}
void testRand( UnitTest &ut )
{
    bool pass = true;
    // Create communicators and call rand on them
    std::vector<size_t> vals;
    pass = pass && addRand( AMP::AMP_MPI(), vals );
    pass = pass && addRand( AMP_COMM_NULL, vals );
    pass = pass && addRand( AMP_COMM_NULL, vals );
    pass = pass && addRand( AMP_COMM_SELF, vals );
    pass = pass && addRand( AMP_COMM_SELF, vals );
    pass = pass && addRand( AMP_COMM_WORLD, vals );
#ifdef AMP_USE_MPI
    pass = pass && addRand( MPI_COMM_WORLD, vals );
#endif
    auto comm  = AMP::AMP_MPI( AMP_COMM_WORLD );
    pass       = pass && addRand( comm, vals );
    auto comm1 = comm.dup();
    auto comm2 = comm.dup();
    pass       = pass && addRand( comm, vals );
    pass       = pass && addRand( comm1, vals );
    pass       = pass && addRand( comm2, vals );
#ifdef AMP_USE_MPI
    MPI_Comm tmp;
    MPI_Comm_dup( MPI_COMM_WORLD, &tmp );
    auto comm3 = AMP::AMP_MPI( tmp, true );
    pass       = pass && addRand( comm2, vals );
#endif
    if ( !pass )
        ut.failure( "rand() is not consistent across all ranks" );
    // Verify that all values are unique
    auto vals2 = vals;
    AMP::Utilities::unique( vals2 );
    if ( vals2.size() != vals.size() ) {
        pass = false;
        ut.failure( "Failed to generate unique random numbers" );
    }
    if ( pass )
        ut.passes( "rand()" );
}


//  This test will test the AMP::AMP_MPI class
int main( int argc, char *argv[] )
{
    // Start MPI
    AMP::AMPManager::startup( argc, argv );

    // Create the unit test
    UnitTest ut;
    PROFILE_ENABLE( 0 );
    PROFILE( "Main" );


    // Limit the scope so objects are destroyed
    {

        // Get the start time for the tests
        double start_time = time();

        // Print the global size (if we are using MPI)
        int global_size = 0;
#ifdef AMP_USE_MPI
        MPI_Comm_size( MPI_COMM_WORLD, &global_size );
#else
        global_size = 1;
#endif

        // Test the global communicator (AMP_COMM_WORLD)
        MPI_CLASS globalComm = MPI_CLASS( AMP_COMM_WORLD );
        record( ut, !globalComm.isNull(), "Global communicator created" );
        record( ut, globalComm.getSize() == global_size, "Global communicator size" );
        if ( globalComm.getRank() == 0 ) {
            std::cout << "MPI_COMM_WORLD = " << global_size << " processors" << std::endl;
            std::cout << "   Largest tag value = " << globalComm.maxTag() << std::endl << std::endl;
        }
#ifdef AMP_USE_MPI
        record(
            ut, globalComm.getCommunicator() == MPI_COMM_WORLD, "Communicator == MPI_COMM_WORLD" );
#endif
        auto commTimer = testComm( globalComm, ut );
        if ( globalComm.getRank() == 0 ) {
            std::cout << "Results for global timer (rank 0)" << std::endl;
            commTimer.print();
            std::cout << std::endl;
        }

        // Test a call to sleepBarrier
        globalComm.sleepBarrier();

        // Test bcast with std::string
        std::string rank_string;
        if ( globalComm.getRank() == 0 )
            rank_string = "Rank 0";
        rank_string = globalComm.bcast( rank_string, 0 );
        record( ut, rank_string == "Rank 0", "Bcast std::string" );

        // Test AMP_COMM_SELF
        MPI_CLASS selfComm = MPI_CLASS( AMP_COMM_SELF );
        record( ut, !selfComm.isNull(), "Self communicator created" );
#ifdef AMP_USE_MPI
        record( ut,
                selfComm.getCommunicator() == MPI_COMM_SELF,
                "Communicator == MPI_COMM_SELF",
                !useMPI );
#endif
        testComm( selfComm, ut );

        // Test == and !=
        record( ut, globalComm == globalComm && !( selfComm == globalComm ), "==", !useMPI );
        record( ut, selfComm != globalComm && !( globalComm != globalComm ), "!=", !useMPI );

        // Test AMP_COMM_NULL
        MPI_CLASS nullComm = MPI_CLASS( AMP_COMM_NULL );
        record( ut, nullComm.isNull(), "Null communicator created" );
        record( ut, nullComm.getSize() == 0, "Null communicator has zero size" );
#ifdef AMP_USE_MPI
        record( ut,
                nullComm.getCommunicator() == MPI_COMM_NULL,
                "Communicator == MPI_COMM_NULL",
                !useMPI );
#endif

        // Test dup
#if !defined( AMP_USE_MPI ) && defined( AMP_USE_PETSC )
        MPI_CLASS dupComm2 = globalComm.dup();
#endif
        MPI_CLASS dupComm = globalComm.dup();
        record( ut, nullComm.dup().isNull(), "Null communicator duplicates a Null communicator" );
        testCommDup( ut );

        // Test rand()
        testRand( ut );

        // Test compare
        record( ut, globalComm.compare( globalComm ) == 1, "compare comm global==global" );
        record( ut, globalComm.compare( dupComm ) == 3, "compare comm global~=dup" );
        if ( global_size == 1 ) {
            record( ut,
                    globalComm.compare( selfComm ) == 3,
                    "compare comm global~=self (global size=1)",
                    !useMPI );
        } else {
            record( ut, globalComm.compare( selfComm ) == 0, "compare comm global!=self" );
        }

        // Split the global comm and test
        int color;
        if ( globalComm.getRank() == 0 )
            color = 0;
        else if ( globalComm.getRank() < 3 )
            color = 1;
        else
            color = 2 + ( globalComm.getRank() - 2 ) / 4;
        std::vector<MPI_CLASS> splitComms( 4 );
        splitComms[0] = globalComm.split( color );
        splitComms[1] = globalComm.split( color, globalComm.getRank() );
        record( ut,
                splitComms[0].getCommunicator() != globalComm.getCommunicator() &&
                    splitComms[1].getCommunicator() != globalComm.getCommunicator() &&
                    splitComms[0].getCommunicator() != splitComms[1].getCommunicator(),
                "split comm has different communicator" );
        if ( globalComm.getSize() > 1 ) {
            record( ut, splitComms[0].getSize() < globalComm.getSize(), "split comm is smaller" );
        }
        record( ut, splitComms[0].getRank() == splitComms[1].getRank(), "split sort by rank" );
        testComm( splitComms[0], ut );
        splitComms[2] = globalComm.split( -1 );
        record( ut, splitComms[2].isNull(), "split with color=-1 returns NULL communicator" );
        splitComms[3] = splitComms[0]; // Make a copy to ensure there are no memory leaks
        splitComms[3] = splitComms[2]; // Perform assignement to check memory leaks
        MPI_ASSERT( splitComms[3] == splitComms[2] );

        // Test  <  <=  >  >=
        if ( globalComm.getSize() > 1 ) {
            bool test1 = splitComms[0] < globalComm && splitComms[1] < globalComm &&
                         !( globalComm < globalComm ) && !( globalComm < splitComms[0] );
            bool test2 = splitComms[0] <= globalComm && splitComms[1] <= globalComm &&
                         globalComm <= globalComm && !( globalComm <= splitComms[0] );
            bool test3 = globalComm > splitComms[0] && globalComm > splitComms[1] &&
                         !( globalComm > globalComm ) && !( splitComms[0] > globalComm );
            bool test4 = globalComm >= splitComms[0] && globalComm >= splitComms[1] &&
                         globalComm >= globalComm && !( splitComms[0] >= globalComm );
            record( ut, test1, "< comm" );
            record( ut, test2, "<= comm" );
            record( ut, test3, "> comm" );
            record( ut, test4, ">= comm" );
        }

        // Test intersection
        // Test globalComm with selfComm
        if ( globalComm.getSize() > 1 ) {
            MPI_CLASS comm1 = MPI_CLASS::intersect( globalComm, selfComm );
            MPI_CLASS comm2 = MPI_CLASS::intersect( selfComm, globalComm );
            MPI_CLASS comm3 = MPI_CLASS::intersect( globalComm, globalComm );
            record( ut,
                    comm1.compare( globalComm ) == 0 && comm1.compare( selfComm ) != 0 &&
                        comm2.compare( globalComm ) == 0 && comm2.compare( selfComm ) != 0 &&
                        comm3.compare( globalComm ) != 0 && comm3.compare( selfComm ) == 0,
                    "intersection of globalComm and selfComm" );
        }

        // Test case where we have disjoint sets (this can only happen of one of the comms is null)
        {
            MPI_CLASS intersection = MPI_CLASS::intersect( globalComm, nullComm );
            record( ut, intersection.isNull(), "intersection of non-overlapping comms" );
        }

        // Test case where the comms partially overlap
        if ( globalComm.getSize() > 2 ) {
            int n = globalComm.getSize() - 1;
            // Intersect 2 comms (all other ranks will be null)
            MPI_CLASS split1       = globalComm.split( globalComm.getRank() == 0 ? -1 : 0 );
            MPI_CLASS split2       = globalComm.split( globalComm.getRank() == n ? -1 : 0 );
            MPI_CLASS intersection = MPI_CLASS::intersect( split1, split2 );
            bool pass              = true;
            if ( globalComm.getRank() == 0 || globalComm.getRank() == n ) {
                if ( !intersection.isNull() )
                    pass = false;
            } else {
                if ( intersection.compare( split1 ) != 0 || intersection.compare( split2 ) != 0 ||
                     intersection.getSize() != globalComm.getSize() - 2 )
                    pass = false;
            }
            // Intersect 2 sets for ranks (3 groups should result)
            // split1 = globalComm.split(globalComm.getRank()==0?1:2);
            // split2 = globalComm.split(globalComm.getRank()==n?1:2);
            // intersection = MPI_CLASS::intersect( split1, split2 );
            // bool pass = true;
            // if ( globalComm.getRank()==0 || globalComm.getRank()==n ) {
            //    if ( intersection.compare(selfComm)==0 )
            //        pass = false;
            //} else {
            //    if ( intersection.compare(split1)!=0 || intersection.compare(split2)!=0 ||
            //         intersection.getSize()!=globalComm.getSize()-2 )
            //        pass = false;
            //}
            record( ut, pass, "intersection of partially overlapping comms" );
        }

        // Test splitByNode
        MPI_CLASS nodeComm    = globalComm.splitByNode();
        std::string localName = MPI_CLASS::getNodeName();
        std::vector<std::string> globalStrings( globalComm.getSize() );
        std::vector<std::string> nodeStrings( nodeComm.getSize() );
        globalComm.allGather<std::string>( localName, &globalStrings[0] );
        nodeComm.allGather<std::string>( localName, &nodeStrings[0] );
        int N_local = 0;
        for ( auto &nodeString : nodeStrings ) {
            if ( nodeString == localName )
                N_local++;
        }
        int N_global = 0;
        for ( auto &globalString : globalStrings ) {
            if ( globalString == localName )
                N_global++;
        }
        record( ut,
                !nodeComm.isNull() && N_local == nodeComm.getSize() && N_local == N_global,
                "splitByNode" );

        // Test the call to load balance the processes
        MPI_CLASS::balanceProcesses( globalComm, 1 );
        std::vector<int> cpus = MPI_CLASS::getProcessAffinity();
        size_t maxProcNode    = nodeComm.maxReduce( cpus.size() );
        bool pass_balance     = cpus.size() == maxProcNode && !cpus.empty();
        MPI_CLASS::balanceProcesses( globalComm, 2 );
        cpus = MPI_CLASS::getProcessAffinity();
        if ( cpus.size() < 1 || cpus.size() > maxProcNode / nodeComm.getSize() )
            pass_balance = false;
        record( ut, pass_balance, "balanceProcesses", useApple );

        // Test the performance of yield (used internally by AMP_MPI wait routines)
        globalComm.barrier();
        double start_yield = time();
        for ( int i = 0; i < 10000; i++ )
            std::this_thread::yield();
        double time_yield = ( time() - start_yield ) / 10000;
        if ( globalComm.getRank() == 0 )
            std::cout << "Time to yield: " << time_yield * 1e6 << " us" << std::endl;

        // Test time and tick
        double end_time = MPI_CLASS::time();
        double time_res = MPI_CLASS::tick();
        if ( globalComm.getRank() == 0 ) {
            std::cout << "Time to run tests: " << end_time - start_time << std::endl;
            std::cout << "Timer resolution: " << time_res << std::endl;
            if ( time_res > 0 && time_res < 1 && ( end_time - start_time ) >= time_res )
                ut.passes( "time and tick" );
            else
                ut.failure( "time and tick" );
            std::cout << std::endl;
        }

    } // Limit the scope so objects are destroyed

    // Finished testing, report the results
    ut.report();
    int num_failed = ut.NumFailGlobal();

    // Shutdown
    PROFILE_SAVE( "test_AMP_MPI" );
    ut.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
