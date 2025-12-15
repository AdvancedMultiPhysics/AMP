// This file contains helper functions and interfaces for filesystem access
#ifndef included_AMP_FileSystem
#define included_AMP_FileSystem


#include <filesystem>
#include <string>
#include <string_view>


namespace AMP::IO {


/*
 * Create the directory specified by the path string.
 */
void recursiveMkdir( const std::string &path );


//! Return the path to the file
std::string path( const std::string &filename );


//! Return the filename (strip the path)
std::string filename( const std::string &filename );


//! Set the permissions for the file or directory
void permissions( const std::string &filename, std::filesystem::perms mode );


//! Check if a file exists and return true if it does
bool exists( const std::string &filename );


//! Return the file size
size_t fileSize( const std::string &filename );


//! Rename a file from old file name to new file name
void rename( const std::string &old_filename, const std::string &new_filename );


//! Delete a file.  If the file does not exist, nothing will happen
void deleteFile( const std::string &filename );


//! Get the lower case suffix for a file
std::string getSuffix( const std::string &filename );


// Deprecated functions
#if !( defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 ) || \
       defined( _MSC_VER ) )
[[deprecated( "recursiveMkdir(path,mode,rank_create) is deprecated, no need to specify "
              "only_node_zero_creates, use recursiveMkdir(path)" )]] void
recursiveMkdir( const std::string &, mode_t, bool );
[[deprecated( "recursiveMkdir(path,mode) is deprecated, use recursiveMkdir(path) and "
              "permissions(mode)" )]] void
recursiveMkdir( const std::string &, mode_t );
#endif
[[deprecated( "fileExists() is deprecated, use exists()" )]] bool fileExists( const std::string & );
[[deprecated( "renameFile() is deprecated, use rename()" )]] void
renameFile( const std::string &old_filename, const std::string &new_filename );


} // namespace AMP::IO

#endif
