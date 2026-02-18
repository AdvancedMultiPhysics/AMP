#define NOMINMAX
#include "AMP/utils/Utilities.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"

#include "StackTrace/StackTrace.h"
#include "StackTrace_Version.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

#ifdef AMP_USE_TIMER
    #include "MemoryApp.h"
#endif

#ifdef AMP_USE_CUDA
    #include "AMP/utils/cuda/Helper_Cuda.h"
#endif
#ifdef AMP_USE_HIP
    #include "AMP/utils/hip/Helper_Hip.h"
#endif

// Include system dependent headers
// clang-format off
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 ) || defined( _MSC_VER )
    #include <process.h>
    #include <stdio.h>
    #include <tchar.h>
    #include <windows.h>
    #include <psapi.h>  // Must be after windows.h
#else
    #include <dlfcn.h>
    #include <execinfo.h>
    #include <sched.h>
    #include <sys/time.h>
    #include <unistd.h>
#endif
#if defined( __APPLE__ )
    #include <mach/mach.h>
    #include <sys/sysctl.h>
    #include <sys/types.h>
#endif
// clang-format on


namespace AMP::Utilities {


// Mutex for Utility functions
static std::mutex Utilities_mutex;


/****************************************************************************
 * Check if valgrind is running                                              *
 * We will eveuntually remove this function once everyone has had time to    *
 * update to a more recent version of StackTrace                             *
 ****************************************************************************/
#ifndef StackTrace_BUILD_VERSION
    #define StackTrace_BUILD_VERSION 0
#endif
bool running_valgrind()
{
#if StackTrace_BUILD_VERSION >= 125
    return StackTrace::Utilities::running_valgrind();
#else
    auto x = getenv( "LD_PRELOAD" );
    return std::min( x.find( "/valgrind/" ), x.find( "/vgpreload" ) ) != std::string::npos;
#endif
}


/****************************************************************************
 *  Check if a number is nan or infinity                                     *
 *  Note this routine needs to be robust in case -ffast_math is set          *
 ****************************************************************************/
/*template<class TYPE>
static std::string getHex( const TYPE& x )
{
    auto data = reinterpret_cast<const uint8_t*>( &x );
    std::string str;
    str.resize( 2 * ( sizeof(TYPE) + 1 ), '0' );
    str[0] = '0';
    str[1] = 'x';
    for ( size_t i = 0, j = 2*sizeof(TYPE); i < sizeof(TYPE); i++, j-=2) {
        char tmp[8];
        snprintf( tmp, 8, "%02x", data[i] );
        str[j+0] = tmp[0];
        str[j+1] = tmp[1];
    }
    return str;
}*/
template<class TYPE>
bool isInf( TYPE x )
{
    auto y = std::numeric_limits<TYPE>::infinity();
    if ( x == y )
        return true;
    if ( std::memcmp( &x, &y, sizeof( x ) ) == 0 )
        return true;
    return false;
}
template<class TYPE>
bool isNaN( TYPE x )
{
    if ( x != x )
        return true;
    auto y1 = std::numeric_limits<TYPE>::quiet_NaN();
    auto y2 = -std::numeric_limits<TYPE>::quiet_NaN();
    if ( std::memcmp( &x, &y1, sizeof( x ) ) == 0 || std::memcmp( &x, &y2, sizeof( x ) ) == 0 )
        return true;
    return false;
}
template bool isInf<float>( float );
template bool isNaN<float>( float );
template bool isInf<double>( double );
template bool isNaN<double>( double );
template bool isInf<long double>( long double );
template bool isNaN<long double>( long double );


/****************************************************************************
 *  Basic string functions                                                   *
 ****************************************************************************/
std::string intToString( int num, int min_width )
{
    int tmp_width = ( min_width > 0 ? min_width : 1 );
    std::ostringstream os;
    if ( num < 0 ) {
        os << '-' << std::setw( tmp_width - 1 ) << std::setfill( '0' ) << -num;
    } else {
        os << std::setw( tmp_width ) << std::setfill( '0' ) << num;
    }
    os << std::flush;
    return os.str();
}
std::string nodeToString( int num ) { return intToString( num, 5 ); }
std::string processorToString( int num ) { return intToString( num, 5 ); }
std::string patchToString( int num ) { return intToString( num, 4 ); }
std::string levelToString( int num ) { return intToString( num, 4 ); }
std::string blockToString( int num ) { return intToString( num, 4 ); }
std::string strrep( const std::string &in, const std::string &s, const std::string &r )
{
    auto out = in;
    size_t i = 0;
    while ( i < out.length() ) {
        i = out.find( s, i );
        if ( i == std::string::npos ) {
            break;
        }
        out.replace( i, s.length(), r );
        i += r.length();
    }
    return out;
}


/****************************************************************************
 *  Basic checks                                                             *
 ****************************************************************************/
static_assert( getOS() != OS::Unknown );


/****************************************************************************
 *  Function to set an environemental variable                               *
 ****************************************************************************/
void setenv( const char *name, const char *value )
{
    Utilities_mutex.lock();
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 ) || \
    defined( _MSC_VER )
    bool pass = SetEnvironmentVariable( name, value ) != 0;
#else
    bool pass = false;
    if ( value == nullptr )
        pass = ::unsetenv( name ) == 0;
    else
        pass = ::setenv( name, value, 1 ) == 0;
#endif
    Utilities_mutex.unlock();
    if ( !pass ) {
        std::string msg;
        if ( value != nullptr )
            msg = stringf( "Error setting enviornmental variable: %s=%s\n", name, value );
        else
            msg = stringf( "Error clearing enviornmental variable: %s\n", name );
        AMP_ERROR( msg );
    }
}
std::string getenv( const char *name )
{
    std::string var;
    Utilities_mutex.lock();
    auto tmp = std::getenv( name );
    if ( tmp )
        var = std::string( tmp );
    Utilities_mutex.unlock();
    return var;
}


/****************************************************************************
 *  Print AMP Banner                                                         *
 ****************************************************************************/
// clang-format off
void printBanner()
{
    constexpr char banner[] =
        R"(            _____                    _____                    _____)" "\n"
        R"(           /\    \                  /\    \                  /\    \)" "\n"
        R"(          /::\    \                /::\____\                /::\    \)" "\n"
        R"(         /::::\    \              /::::|   |               /::::\    \)" "\n"
        R"(        /::::::\    \            /:::::|   |              /::::::\    \)" "\n"
        R"(       /:::/\:::\    \          /::::::|   |             /:::/\:::\    \)" "\n"
        R"(      /:::/__\:::\    \        /:::/|::|   |            /:::/__\:::\    \)" "\n"
        R"(     /::::\   \:::\    \      /:::/ |::|   |           /::::\   \:::\    \)" "\n"
        R"(    /::::::\   \:::\    \    /:::/  |::|___|______    /::::::\   \:::\    \)" "\n"
        R"(   /:::/\:::\   \:::\    \  /:::/   |::::::::\    \  /:::/\:::\   \:::\____\)" "\n"
        R"(  /:::/  \:::\   \:::\____\/:::/    |:::::::::\____\/:::/  \:::\   \:::|    |)" "\n"
        R"(  \::/    \:::\  /:::/    /\::/    / ~~~~~/:::/    /\::/    \:::\  /:::|____|)" "\n"
        R"(   \/____/ \:::\/:::/    /  \/____/      /:::/    /  \/_____/\:::\/:::/    /)" "\n"
        R"(            \::::::/    /               /:::/    /            \::::::/    /)" "\n"
        R"(             \::::/    /               /:::/    /              \::::/    /)" "\n"
        R"(             /:::/    /               /:::/    /                \::/____/)" "\n"
        R"(            /:::/    /               /:::/    /)" "\n"
        R"(           /:::/    /               /:::/    /)" "\n"
        R"(          /:::/    /               /:::/    /)" "\n"
        R"(          \::/    /                \::/    /)" "\n"
        R"(           \/____/                  \/____/)" "\n";
    AMP::pout << "\n" << banner << "\n" << std::endl;
}
// clang-format on


/****************************************************************************
 *  Prime number functions                                                   *
 ****************************************************************************/
std::vector<int> factor( uint64_t n )
{
    // Handle trival case
    if ( n <= 3 )
        return { static_cast<int>( n ) };
    // Initialize factors
    size_t N        = 0;
    int factors[64] = { 0 };
    // Remove all factors of 2
    while ( ( n & 0x01 ) == 0 ) {
        factors[N++] = 2;
        n >>= 1;
    }
    // Remove all factors of 3
    while ( n % 3 == 0 ) {
        factors[N++] = 3;
        n /= 3;
    }
    // Use brute force to find remaining factors
    uint64_t f = 5;
    while ( n > 1 ) {
        // Determine the largest number we need to check
        auto f_max = static_cast<uint64_t>( floor( 1.000000000000001 * std::sqrt( n ) ) );
        // Search all remaining numbers (note: we skip every 3rd odd number)
        bool found = false;
        for ( ; f <= f_max && !found; f += 6 ) {
            while ( n % f == 0 ) {
                factors[N++] = f;
                n /= f;
                found = true;
            }
            while ( n % ( f + 2 ) == 0 ) {
                factors[N++] = f + 2;
                n /= f + 2;
                found = true;
            }
        }
        if ( !found ) {
            factors[N++] = n;
            break;
        }
    }
    return std::vector<int>( factors, factors + N );
}
bool isPrime( uint64_t n )
{
    if ( n <= 3 )
        return true;
    if ( ( n & 0x01 ) == 0 || n % 3 == 0 )
        return false;
    // Determine the largest number we need to check
    auto f_max = static_cast<uint64_t>( floor( 1.000000000000001 * std::sqrt( n ) ) );
    // Check if the number is prime
    for ( uint64_t f = 5; f <= f_max; f += 6 ) {
        if ( ( n % f == 0 ) || ( n % ( f + 2 ) == 0 ) )
            return false;
    }
    return true;
}
std::vector<uint64_t> primes( uint64_t n )
{
    // Handle special cases
    if ( n < 2 )
        return { 1u };
    if ( n == 2 )
        return { 2u };
    // Create our bit array
    uint64_t n2 = ( n + 1 ) / 2;
    uint64_t ub = static_cast<uint64_t>( std::sqrt( n + 0.1 ) + 1 );
    auto N      = ( n2 + 63 ) / 64;
    auto p      = new uint64_t[N];
    memset( p, 0xFF, sizeof( uint64_t ) * N );
    // Helper functions to get/set the bits
    auto get = [p]( uint64_t i ) {
        uint64_t i1 = i >> 6;
        uint64_t i2 = i & 0x3F;
        return ( p[i1] & ( ( (uint64_t) 1 ) << i2 ) ) != 0;
    };
    auto unset = [p]( uint64_t i ) {
        uint64_t i1 = i >> 6;
        uint64_t i2 = i & 0x3F;
        p[i1] &= ~( ( (uint64_t) 1 ) << i2 );
    };
    // Set all non-prime values to false
    for ( uint64_t k = 1; k <= ub; k++ ) {
        if ( get( k ) ) {
            uint64_t k2 = 2 * k + 1;
            for ( uint64_t j = 2 * k * ( k + 1 ); j < n2; j += k2 )
                unset( j );
        }
    }
    // Store the prime numbers (note: this takes longer than computing them)
    auto M = static_cast<uint64_t>( n / log2( n ) );
    M      = ( (uint64_t) 1 ) << static_cast<int>( round( log2( M ) ) );
    std::vector<uint64_t> p2;
    p2.reserve( M );
    p2.push_back( 2 );
    for ( uint64_t i = 1; i < n2; i++ ) {
        if ( get( i ) )
            p2.push_back( 2 * i + 1 );
    }
    delete[] p;
    return p2;
}


/****************************************************************************
 *  Interpolation routines                                                   *
 ****************************************************************************/
double linear( const std::vector<double> &x, const std::vector<double> &f, double xi )
{
    size_t Nx = x.size();
    AMP_ASSERT( Nx > 1 );
    AMP_ASSERT( f.size() == Nx );
    size_t i = findfirst( x, xi );
    if ( i == 0 ) {
        i = 1;
    }
    if ( i == x.size() ) {
        i = x.size() - 1;
    }
    double dx = ( xi - x[i - 1] ) / ( x[i] - x[i - 1] );
    return dx * f[i] + ( 1.0 - dx ) * f[i - 1];
}
double bilinear( const std::vector<double> &x,
                 const std::vector<double> &y,
                 const std::vector<double> &f,
                 double xi,
                 double yi )
{
    size_t Nx = x.size();
    size_t Ny = y.size();
    AMP_ASSERT( Nx > 1 && Ny > 1 );
    AMP_ASSERT( f.size() == Nx * Ny );
    size_t i = findfirst( x, xi );
    size_t j = findfirst( y, yi );
    if ( i == 0 ) {
        i = 1;
    }
    if ( j == 0 ) {
        j = 1;
    }
    if ( i == x.size() ) {
        i = x.size() - 1;
    }
    if ( j == y.size() ) {
        j = y.size() - 1;
    }
    double dx  = ( xi - x[i - 1] ) / ( x[i] - x[i - 1] );
    double dy  = ( yi - y[j - 1] ) / ( y[j] - y[j - 1] );
    double f1  = f[i - 1 + ( j - 1 ) * Nx];
    double f2  = f[i + ( j - 1 ) * Nx];
    double f3  = f[i - 1 + j * Nx];
    double f4  = f[i + j * Nx];
    double dx2 = 1.0 - dx;
    double dy2 = 1.0 - dy;
    return ( dx * f2 + dx2 * f1 ) * dy2 + ( dx * f4 + dx2 * f3 ) * dy;
}
double trilinear( const std::vector<double> &x,
                  const std::vector<double> &y,
                  const std::vector<double> &z,
                  const std::vector<double> &f,
                  double xi,
                  double yi,
                  double zi )
{
    size_t Nx = x.size();
    size_t Ny = y.size();
    size_t Nz = z.size();
    AMP_ASSERT( Nx > 1 && Ny > 1 && Nz > 1 );
    AMP_ASSERT( f.size() == Nx * Ny * Nz );
    size_t i = findfirst( x, xi );
    size_t j = findfirst( y, yi );
    size_t k = findfirst( z, zi );
    if ( i == 0 ) {
        i = 1;
    }
    if ( j == 0 ) {
        j = 1;
    }
    if ( k == 0 ) {
        k = 1;
    }
    if ( i == x.size() ) {
        i = x.size() - 1;
    }
    if ( j == y.size() ) {
        j = y.size() - 1;
    }
    if ( k == z.size() ) {
        k = z.size() - 1;
    }
    double dx  = ( xi - x[i - 1] ) / ( x[i] - x[i - 1] );
    double dy  = ( yi - y[j - 1] ) / ( y[j] - y[j - 1] );
    double dz  = ( zi - z[k - 1] ) / ( z[k] - z[k - 1] );
    double f1  = f[i - 1 + ( j - 1 ) * Nx + ( k - 1 ) * Nx * Ny];
    double f2  = f[i + ( j - 1 ) * Nx + ( k - 1 ) * Nx * Ny];
    double f3  = f[i - 1 + j * Nx + ( k - 1 ) * Nx * Ny];
    double f4  = f[i + j * Nx + ( k - 1 ) * Nx * Ny];
    double f5  = f[i - 1 + ( j - 1 ) * Nx + k * Nx * Ny];
    double f6  = f[i + ( j - 1 ) * Nx + k * Nx * Ny];
    double f7  = f[i - 1 + j * Nx + k * Nx * Ny];
    double f8  = f[i + j * Nx + k * Nx * Ny];
    double dx2 = 1.0 - dx;
    double dy2 = 1.0 - dy;
    double dz2 = 1.0 - dz;
    double h0  = ( dx * f2 + dx2 * f1 ) * dy2 + ( dx * f4 + dx2 * f3 ) * dy;
    double h1  = ( dx * f6 + dx2 * f5 ) * dy2 + ( dx * f8 + dx2 * f7 ) * dy;
    return h0 * dz2 + h1 * dz;
}


/****************************************************************************
 *  Dummy function to prevent compiler from optimizing away variable         *
 ****************************************************************************/
void nullUse( const void * ) {}


/****************************************************************************
 *  Busy wait functions                                                      *
 ****************************************************************************/
void busy_ms( int N )
{
    auto t0 = std::chrono::high_resolution_clock::now();
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dis;
    auto duration = [t0]() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>( t1 - t0 ).count();
    };
    while ( duration() < N ) {
        [[maybe_unused]] double x = dis( gen );
        for ( int i = 0; i < 500; i++ )
            x = sqrt( x );
    }
}


/****************************************************************************
 *   Function to return a unique alpha-numeric string on the comm            *
 ****************************************************************************/
std::string randomString( const AMP::AMP_MPI &comm )
{
    // Get a random number that is consistent across the communicator
    //    key = rand() * 2^64*0.5*(sqrt(5)-1)
    uint64_t key = ( ( (uint64_t) comm.rand() ) * 0x9E3779B97F4A7C15 );
    // Convert the hash key to an alpha-numeric string
    char cname[64] = { 0 };
    size_t i       = 0;
    while ( key != 0 ) {
        size_t tmp = key % 62;
        key /= 62;
        if ( tmp < 10 )
            cname[i] = static_cast<char>( tmp + 48 );
        else if ( tmp < 36 )
            cname[i] = static_cast<char>( tmp - 10 + 97 );
        else if ( tmp < 62 )
            cname[i] = static_cast<char>( tmp - 36 + 65 );
        else
            AMP_ERROR( "Internal error" );
        i++;
    }
    return std::string( cname, i );
}


/****************************************************************************
 *  Fill with random values in [0,1]                                         *
 ****************************************************************************/
void fillRandom( std::vector<double> &x )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( 0, 1 );
    for ( size_t i = 0; i < x.size(); i++ )
        x[i] = dis( gen );
}
template<size_t NDIM>
void fillRandom( std::vector<std::array<double, NDIM>> &x )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( 0, 1 );
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( size_t d = 0; d < NDIM; d++ )
            x[i][d] = dis( gen );
    }
}
template void fillRandom<1>( std::vector<std::array<double, 1>> & );
template void fillRandom<2>( std::vector<std::array<double, 2>> & );
template void fillRandom<3>( std::vector<std::array<double, 3>> & );
template void fillRandom<4>( std::vector<std::array<double, 4>> & );


/****************************************************************************
 *  Function to demangle a string (e.g. from typeid)                         *
 ****************************************************************************/
#ifdef __GNUC__
    #define USE_ABI
    #include <cxxabi.h>
#endif
std::string demangle( const std::string &name )
{
    std::string out;
#ifdef __GNUC__
    int status;
    char *demangled = abi::__cxa_demangle( name.data(), nullptr, nullptr, &status );
    if ( status == 0 && demangled != nullptr )
        out = demangled;
    free( demangled );
#endif
    if ( out.empty() )
        out = name;
    return out;
}


/****************************************************************************
 *  Get the string for the last errno                                        *
 ****************************************************************************/
std::string getLastErrnoString()
{
    int err = errno;
    Utilities_mutex.lock();
    std::string msg( std::strerror( err ) );
    Utilities_mutex.unlock();
    return msg;
}


/****************************************************************************
 *  Modify database entries to ensure consistent memory spaces               *
 ****************************************************************************/
void setNestedOperatorMemoryLocations( std::shared_ptr<AMP::Database> input_db,
                                       std::string outerOperatorName,
                                       std::vector<std::string> nestedOperatorNames )
{
    auto outer_db = input_db->getDatabase( outerOperatorName );
    AMP_INSIST( outer_db, "OperatorBuilder: outer DB is null" );

    if ( outer_db->keyExists( "MemoryLocation" ) ) {
        // if outer operator requests a memory location it takes precedent
        auto memLoc = outer_db->getScalar<std::string>( "MemoryLocation" );
        for ( auto &innerName : nestedOperatorNames ) {
            auto inner_db = input_db->getDatabase( innerName );
            AMP_INSIST( inner_db, "OperatorBuilder: inner DB is null" );
            inner_db->putScalar(
                "MemoryLocation", memLoc, Units(), Database::Check::WarnOverwrite );
        }
    } else {
        // outer db does not specify a memory location, check if any internal one does
        // if multiple are specified use most restrictive one
        std::string memLoc{ "device" };
        auto memRestrict = []( const std::string &m1, const std::string &m2 ) -> std::string {
            int c1 = 3, c2 = 3;
            if ( m1 == "device" || m1 == "Device" ) {
                c1 = 2;
            } else if ( m1 == "managed" || m1 == "Managed" ) {
                c1 = 1;
            } else if ( m1 == "host" || m1 == "host" ) {
                c1 = 0;
            }
            if ( m2 == "device" || m2 == "Device" ) {
                c2 = 2;
            } else if ( m2 == "managed" || m2 == "Managed" ) {
                c2 = 1;
            } else if ( m2 == "host" || m2 == "host" ) {
                c2 = 0;
            }
            if ( c1 == 3 && c2 == 3 ) {
                // both spaces unrecognized
                AMP_WARNING( "Unrecognized memory space, returning host" );
                return "host";
            } else if ( c1 < c2 ) {
                return m1;
            }
            return m2;
        };
        bool found = false;
        for ( auto &innerName : nestedOperatorNames ) {
            auto inner_db = input_db->getDatabase( innerName );
            AMP_INSIST( inner_db, "OperatorBuilder: inner DB is null (" + innerName + ")" );
            if ( inner_db->keyExists( "MemoryLocation" ) ) {
                found        = true;
                auto memLocI = inner_db->getScalar<std::string>( "MemoryLocation" );
                memLoc       = memRestrict( memLoc, memLocI );
            }
        }
        if ( found ) {
            outer_db->putScalar(
                "MemoryLocation", memLoc, Units(), Database::Check::WarnOverwrite );
            for ( auto &innerName : nestedOperatorNames ) {
                auto inner_db = input_db->getDatabase( innerName );
                AMP_INSIST( inner_db, "OperatorBuilder: inner DB is null" );
                inner_db->putScalar(
                    "MemoryLocation", memLoc, Units(), Database::Check::WarnOverwrite );
            }
        }
    }
}


} // namespace AMP::Utilities


/************************************************************************
 * Explicit instantiations                                               *
 ************************************************************************/
using Point1D = std::array<double, 1>;
using Point2D = std::array<double, 2>;
using Point3D = std::array<double, 3>;
AMP_INSTANTIATE_SORT( int8_t );
AMP_INSTANTIATE_SORT( int16_t );
AMP_INSTANTIATE_SORT( int32_t );
AMP_INSTANTIATE_SORT( int64_t );
AMP_INSTANTIATE_SORT( uint8_t );
AMP_INSTANTIATE_SORT( uint16_t );
AMP_INSTANTIATE_SORT( uint32_t );
AMP_INSTANTIATE_SORT( uint64_t );
AMP_INSTANTIATE_SORT( float );
AMP_INSTANTIATE_SORT( double );
AMP_INSTANTIATE_SORT( long double );
AMP_INSTANTIATE_SORT( Point1D );
AMP_INSTANTIATE_SORT( Point2D );
AMP_INSTANTIATE_SORT( Point3D );
template void AMP::Utilities::quicksort<double, size_t>( size_t, double *, size_t * );
template std::string AMP::Utilities::to_string<int8_t>( std::vector<int8_t> const & );
template std::string AMP::Utilities::to_string<int16_t>( std::vector<int16_t> const & );
template std::string AMP::Utilities::to_string<int32_t>( std::vector<int32_t> const & );
template std::string AMP::Utilities::to_string<int64_t>( std::vector<int64_t> const & );
template std::string AMP::Utilities::to_string<uint8_t>( std::vector<uint8_t> const & );
template std::string AMP::Utilities::to_string<uint16_t>( std::vector<uint16_t> const & );
template std::string AMP::Utilities::to_string<uint32_t>( std::vector<uint32_t> const & );
template std::string AMP::Utilities::to_string<uint64_t>( std::vector<uint64_t> const & );
template std::string AMP::Utilities::to_string<float>( std::vector<float> const & );
template std::string AMP::Utilities::to_string<double>( std::vector<double> const & );
template std::string AMP::Utilities::to_string<long double>( std::vector<long double> const & );
