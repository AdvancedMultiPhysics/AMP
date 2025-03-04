#include "AMP/IO/FileSystem.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>


/****************************************************************************
 *  Filesystem utilities                                                     *
 ****************************************************************************/
constexpr static char slash[] = { 0x2F, 0x5C, 0x0 };
bool AMP::IO::exists( const std::string &filename ) { return std::filesystem::exists( filename ); }
std::string AMP::IO::path( const std::string &filename )
{
    size_t pos = filename.find_last_of( slash );
    if ( pos == std::string::npos )
        return {};
    return filename.substr( 0, pos );
}
std::string AMP::IO::filename( const std::string &filename )
{
    size_t pos = filename.find_last_of( slash );
    if ( pos == std::string::npos )
        return filename;
    return filename.substr( pos + 1 );
}
void AMP::IO::rename( const std::string &old_filename, const std::string &new_filename )
{
    AMP_ASSERT( !old_filename.empty() );
    AMP_ASSERT( !new_filename.empty() );
    std::filesystem::rename( old_filename, new_filename );
}
void AMP::IO::deleteFile( const std::string &filename )
{
    std::filesystem::remove( filename );
    if ( std::filesystem::exists( filename ) )
        AMP_ERROR( "Error deleting file" );
}
size_t AMP::IO::fileSize( const std::string &filename )
{
    if ( !std::filesystem::exists( filename ) )
        return 0;
    return std::filesystem::file_size( filename );
}
std::string AMP::IO::getSuffix( const std::string &filename )
{
    size_t pos = filename.rfind( '.' );
    if ( pos == std::string::npos )
        return std::string();
    std::string suffix = filename.substr( pos + 1 );
    for ( auto &c : suffix )
        c = std::tolower( c );
    return suffix;
}
static bool recursiveMkdir2( const std::string &path )
{
    if ( path.empty() || path == "." )
        return false;
    // Create the directories
    bool created = std::filesystem::create_directories( path );
    // Verify they exist
    bool exists = std::filesystem::exists( path );
    if ( !exists ) {
        std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
        exists = std::filesystem::exists( path );
    }
    if ( !exists )
        AMP_ERROR( "Cannot create directory " + path );
    return created;
}
void AMP::IO::recursiveMkdir( const std::string &path ) { recursiveMkdir2( path ); }
void AMP::IO::permissions( const std::string &filename, std::filesystem::perms mode )
{
    std::filesystem::permissions( filename, mode );
}


/****************************************************************************
 *  deprecated functions                                                     *
 ****************************************************************************/
bool AMP::IO::fileExists( const std::string &filename ) { return exists( filename ); }
void AMP::IO::renameFile( const std::string &old_filename, const std::string &new_filename )
{
    rename( old_filename, new_filename );
}
#if !( defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 ) || \
       defined( _MSC_VER ) )
void AMP::IO::recursiveMkdir( const std::string &path, mode_t mode )
{
    bool created = recursiveMkdir2( path );
    if ( created )
        std::filesystem::permissions( path, static_cast<std::filesystem::perms>( mode ) );
}
void AMP::IO::recursiveMkdir( const std::string &path, mode_t mode, bool )
{
    bool created = recursiveMkdir2( path );
    if ( created )
        std::filesystem::permissions( path, static_cast<std::filesystem::perms>( mode ) );
}
#endif
