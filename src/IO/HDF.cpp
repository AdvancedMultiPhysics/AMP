#include "AMP/IO/HDF.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/IO/HDF.hpp"
#include "AMP/utils/Array.h"
#include "AMP/utils/Utilities.h"

#include <complex>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>


namespace AMP::IO {


#ifdef AMP_USE_HDF5 // USE HDF5


/******************************************************************
 * Open/close HDF5 files                                           *
 ******************************************************************/
bool HDF5enabled() { return true; }
hid_t openHDF5( const std::string &filename, const char *mode, Compression compress )
{
    // Set cache size to 3MBs and instruct the cache to discard the fully read chunk
    auto pid = H5P_DEFAULT;
    /*auto pid = H5Pcreate( H5P_FILE_ACCESS );
    int nelemts;
    size_t nslots, nbytes;
    double w0;
    H5Pget_cache(pid,& nelemts,& nslots,& nbytes,& w0);
    H5Pset_cache(pid, nelemts, 1999, 3*1024*1024, 1.0); */
    // Open the file
    hid_t fid = 0;
    if ( strcmp( mode, "r" ) == 0 ) {
        fid = H5Fopen( filename.data(), H5F_ACC_RDONLY, pid );
    } else if ( strcmp( mode, "w" ) == 0 ) {
        auto pwd = IO::path( filename );
        IO::recursiveMkdir( pwd );
        fid = H5Fcreate( filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
    } else if ( strcmp( mode, "rw" ) == 0 ) {
        fid = H5Fopen( filename.data(), H5F_ACC_RDWR, H5P_DEFAULT );
    } else {
        AMP_ERROR( "Invalid mode for opening HDF5 file" );
    }
    if ( strcmp( mode, "w" ) == 0 ) {
        int defaultCompression = 0;
        if ( compress == Compression::None ) {
            // No compression
        } else if ( compress == Compression::GZIP ) {
            // Use gzip (if available)
            if ( H5Zfilter_avail( H5Z_FILTER_DEFLATE ) )
                defaultCompression = 1;
            else
                AMP::perr << "HDF5 gzip filter is not available" << std::endl;
        } else if ( compress == Compression::SZIP ) {
            // Use szip (if available)
            if ( H5Zfilter_avail( H5Z_FILTER_SZIP ) )
                defaultCompression = 2;
            else
                AMP::perr << "HDF5 szip filter is not available" << std::endl;
        } else {
            AMP_ERROR( "Internal error" );
        }
        writeHDF5<int>( fid, "DefaultCompression", defaultCompression );
    }
    // H5Pclose( pid );
    return fid;
}
void closeHDF5( hid_t fid, bool print )
{
    // Get a list of the open objects
    auto [file, set, group, type, attr] = openObjects( fid );
    int N_open = file.size() + set.size() + group.size() + type.size() + attr.size();
    // Print the resource leaks
    if ( print && N_open > 1 ) {
        printf( "Open objects by HDF:\n" );
        printf( "   files: (%i)\n", static_cast<int>( file.size() ) );
        printf( "   datasets: (%i)\n", static_cast<int>( set.size() ) );
        printf( "   groups: (%i)\n", static_cast<int>( group.size() ) );
        printf( "   types: (%i)\n", static_cast<int>( type.size() ) );
        printf( "   attributes: (%i)\n", static_cast<int>( attr.size() ) );
    }
    if ( N_open >= 1000 )
        AMP_WARNING( "Lots of open objects detected in HDF5 while closing file" );
    // Try to close any remaining objects (needed to ensure we can reopen the data if desired)
    for ( auto id : file ) {
        if ( id != fid )
            H5Fclose( id );
    }
    for ( auto id : set )
        H5Dclose( id );
    for ( auto id : group )
        H5Gclose( id );
    for ( auto id : type )
        H5Tclose( id );
    for ( auto id : attr )
        H5Aclose( id );
    // Flush the data (needed to ensure we can reopen the data if desired)
    unsigned intent;
    H5Fget_intent( fid, &intent );
    if ( intent == H5F_ACC_RDWR || intent == H5F_ACC_TRUNC )
        H5Fflush( fid, H5F_SCOPE_GLOBAL );
    // Close the file
    H5Fclose( fid );
}


/************************************************************************
 * Get list of open objects                                              *
 ************************************************************************/
std::tuple<std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>>
openObjects( hid_t fid )
{
    std::vector<hid_t> file, set, group, type, attr;
    file.resize( H5Fget_obj_count( fid, H5F_OBJ_FILE ) );
    set.resize( H5Fget_obj_count( fid, H5F_OBJ_DATASET ) );
    group.resize( H5Fget_obj_count( fid, H5F_OBJ_GROUP ) );
    type.resize( H5Fget_obj_count( fid, H5F_OBJ_DATATYPE ) );
    attr.resize( H5Fget_obj_count( fid, H5F_OBJ_ATTR ) );
    if ( !file.empty() ) {
        size_t N_file = H5Fget_obj_ids( fid, H5F_OBJ_FILE, file.size(), file.data() );
        AMP_ASSERT( N_file == file.size() );
    }
    if ( !set.empty() ) {
        size_t N_set = H5Fget_obj_ids( fid, H5F_OBJ_DATASET, set.size(), set.data() );
        AMP_ASSERT( N_set == set.size() );
    }
    if ( !group.empty() ) {
        size_t N_group = H5Fget_obj_ids( fid, H5F_OBJ_GROUP, group.size(), group.data() );
        AMP_ASSERT( N_group == group.size() );
    }
    if ( !type.empty() ) {
        size_t N_type = H5Fget_obj_ids( fid, H5F_OBJ_DATATYPE, type.size(), type.data() );
        AMP_ASSERT( N_type == type.size() );
    }
    if ( !attr.empty() ) {
        size_t N_attr = H5Fget_obj_ids( fid, H5F_OBJ_ATTR, attr.size(), attr.data() );
        AMP_ASSERT( N_attr == attr.size() );
    }
    return std::tie( file, set, group, type, attr );
}


/************************************************************************
 * Check if we support compression                                       *
 ************************************************************************/
Compression defaultCompression( hid_t fid )
{
    hid_t root = H5Gopen2( fid, "/", H5P_DEFAULT );
    if ( !H5Dexists( root, "DefaultCompression" ) )
        return Compression::None;
    int tmp;
    readHDF5( root, "DefaultCompression", tmp );
    Compression compress = Compression::None;
    if ( tmp == 0 ) {
        compress = Compression::None;
    } else if ( tmp == 1 ) {
        compress = Compression::GZIP;
    } else if ( tmp == 2 ) {
        compress = Compression::SZIP;
    } else {
        AMP_ERROR( "Internal error" );
    }
    H5Gclose( root );
    return compress;
}


/************************************************************************
 * Create a default chunk size                                           *
 ************************************************************************/
static size_t ceil2( size_t x )
{
    size_t y = 2;
    while ( y < x )
        y <<= 1;
    return y;
}
static size_t floor2( size_t x ) { return ceil2( x ) >> 1; }
hid_t createChunk( AMP::ArraySize dims, Compression compress, size_t objSize )
{
    if ( compress == Compression::None || dims.length() < 512 )
        return H5P_DEFAULT;
    if ( objSize == 0 )
        objSize = 512;
    // Limit chunk size to < 2 GB
    int d = dims.ndim() - 1;
    while ( dims.length() * objSize >= 0x80000000 ) {
        if ( dims[d] == 1 ) {
            d--;
        } else {
            dims.resize( d, floor2( dims[d] ) );
        }
    }
    hid_t plist = H5Pcreate( H5P_DATASET_CREATE );
    auto status = H5Pset_chunk( plist, dims.ndim(), arraySize( dims ).data() );
    AMP_ASSERT( status == 0 );
    if ( compress == Compression::GZIP ) {
        status = H5Pset_deflate( plist, 7 );
        AMP_ASSERT( status == 0 );
    } else if ( compress == Compression::SZIP ) {
        status = H5Pset_szip( plist, H5_SZIP_NN_OPTION_MASK, 16 );
        AMP_ASSERT( status == 0 );
    }
    return plist;
}


/************************************************************************
 * HDF5 helper routines                                                  *
 ************************************************************************/
bool H5Gexists( hid_t fid, const std::string &name )
{
    H5E_auto2_t func;
    void *client;
    H5Eget_auto2( H5E_DEFAULT, &func, &client );
    H5Eset_auto2( H5E_DEFAULT, nullptr, nullptr );
    int status = H5Gget_objinfo( fid, name.data(), 0, nullptr );
    H5Eset_auto2( H5E_DEFAULT, func, client );
    return status == 0;
}
bool H5Dexists( hid_t fid, const std::string &name )
{
    H5E_auto2_t func;
    void *client;
    H5Eget_auto2( H5E_DEFAULT, &func, &client );
    H5Eset_auto2( H5E_DEFAULT, nullptr, nullptr );
    hid_t dataset = H5Dopen2( fid, name.data(), H5P_DEFAULT );
    H5Eset_auto2( H5E_DEFAULT, func, client );
    bool exists = dataset > 0;
    if ( exists )
        H5Dclose( dataset );
    return exists;
}
static char nullName[] = "---null---";
hid_t createGroup( hid_t fid, const std::string &name )
{
    if ( name.empty() )
        return H5Gcreate2( fid, nullName, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    return H5Gcreate2( fid, name.data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
}
hid_t openGroup( hid_t fid, const std::string &name )
{
    if ( name.empty() )
        return openGroup( fid, nullName );
    if ( !H5Gexists( fid, name ) )
        AMP_ERROR( "Group " + std::string( name ) + " does not exist" );
    return H5Gopen2( fid, name.data(), H5P_DEFAULT );
}
void closeGroup( hid_t gid ) { H5Gclose( gid ); }
void closeDataset( hid_t fid ) { H5Dclose( fid ); }
void closeDatatype( hid_t fid ) { H5Tclose( fid ); }
void closeDataspace( hid_t fid ) { H5Sclose( fid ); }
hid_t getHDF5datatype( hid_t fid, const std::string &name )
{
    hid_t dataset  = H5Dopen2( fid, name.data(), H5P_DEFAULT );
    hid_t datatype = H5Dget_type( dataset );
    H5Dclose( dataset );
    return datatype;
}


/************************************************************************
 * Read/write bytes to HDF5                                              *
 ************************************************************************/
void writeHDF5( hid_t fid, const std::string &name, size_t N_bytes, const void *data )
{
    hsize_t dim[1] = { N_bytes };
    hid_t plist    = H5P_DEFAULT;
    if ( N_bytes < 0x7500 ) {
        // Use compact storage (limited to < 30K)
        plist       = H5Pcreate( H5P_DATASET_CREATE );
        auto status = H5Pset_layout( plist, H5D_COMPACT );
        AMP_ASSERT( status == 0 );
    } else {
        // Use compression if available
        plist = createChunk( N_bytes, defaultCompression( fid ), 1 );
    }
    hid_t dataspace = H5Screate_simple( 1, dim, NULL );
    hid_t dataset   = H5Dcreate2(
        fid, name.data(), H5T_NATIVE_UCHAR, dataspace, H5P_DEFAULT, plist, H5P_DEFAULT );
    H5Dwrite( dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data );
    H5Dclose( dataset );
    H5Sclose( dataspace );
    if ( plist != H5P_DEFAULT )
        H5Pclose( plist );
}
void readHDF5( hid_t fid, const std::string &name, size_t N_bytes, void *data )
{
    if ( !H5Dexists( fid, name ) ) {
        AMP_ERROR( "Variable does not exist in file: " + std::string( name ) );
        return;
    }
    hid_t dataset   = H5Dopen2( fid, name.data(), H5P_DEFAULT );
    hid_t dataspace = H5Dget_space( dataset );
    hsize_t dims0[10];
    int ndim = H5Sget_simple_extent_dims( dataspace, dims0, NULL );
    AMP_ASSERT( ndim == 1 && dims0[0] == N_bytes );
    H5Dread( dataset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, data );
    H5Dclose( dataset );
    H5Sclose( dataspace );
}


#else // No HDF5
// Dummy implementations for no HDF5
bool HDF5enabled() { return false; }
hid_t openHDF5( const std::string &, const char *, AMP::IO::Compression )
{
    AMP_ERROR( "Attempting to open HDF5 file without enabling HDF5" );
    return 0;
}
void closeHDF5( hid_t, bool ) {}
bool H5Gexists( hid_t, const std::string & ) { return false; }
bool H5Dexists( hid_t, const std::string & ) { return false; }
hid_t createGroup( hid_t, const std::string & ) { return 0; }
hid_t openGroup( hid_t, const std::string & ) { return 0; }
void closeGroup( hid_t ) {}
void writeHDF5( hid_t, const std::string &, size_t, const std::byte * );
void readHDF5( hid_t, const std::string &, size_t, std::byte * );
std::tuple<std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>,
           std::vector<hid_t>>
openObjects( hid_t )
{
    std::vector<hid_t> file, set, group, type, attr;
    return std::tie( file, set, group, type, attr );
}
#endif

} // namespace AMP::IO
