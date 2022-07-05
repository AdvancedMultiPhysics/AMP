#include "AMP/IO/HDF5_IO.h"
#include "AMP/IO/HDF5_IO.hpp"
#include "AMP/utils/Array.h"
#include "AMP/utils/Utilities.h"

#include <complex>
#include <sstream>
#include <string>
#include <vector>


namespace AMP {


#ifdef AMP_USE_HDF5 // USE HDF5


// Function to round up to the nearest power of 2
static size_t ceil2( size_t x )
{
    size_t y = 2;
    while ( y < x )
        y <<= 1;
    return y;
}
static size_t floor2( size_t x ) { return ceil2( x ) >> 1; }


/************************************************************************
 * HDF5 helper routines                                                  *
 ************************************************************************/
inline const void *H5Ptr( const void *x ) { return x == nullptr ? ( (void *) 1 ) : x; }
bool H5Gexists( hid_t fid, const std::string_view &name )
{
    H5E_auto2_t func;
    void *client;
    H5Eget_auto2( H5E_DEFAULT, &func, &client );
    H5Eset_auto2( H5E_DEFAULT, nullptr, nullptr );
    int status = H5Gget_objinfo( fid, name.data(), 0, nullptr );
    H5Eset_auto2( H5E_DEFAULT, func, client );
    return status == 0;
}
bool H5Dexists( hid_t fid, const std::string_view &name )
{
    H5E_auto2_t func;
    void *client;
    H5Eget_auto2( H5E_DEFAULT, &func, &client );
    H5Eset_auto2( H5E_DEFAULT, nullptr, nullptr );
    hid_t dataset = H5Dopen2( fid, name.data(), H5P_DEFAULT );
    H5Eset_auto2( H5E_DEFAULT, func, client );
    bool exists = dataset > 0;
    // if ( exists )
    //    H5Dclose( dataset );
    return exists;
}
hid_t createGroup( hid_t fid, const std::string_view &name )
{
    return H5Gcreate2( fid, name.data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
}
hid_t openGroup( hid_t fid, const std::string_view &name )
{
    if ( !H5Gexists( fid, name ) )
        AMP_ERROR( "Group " + std::string( name ) + " does not exist" );
    return H5Gopen2( fid, name.data(), H5P_DEFAULT );
}
void closeGroup( hid_t gid ) { H5Gclose( gid ); }


/************************************************************************
 * Complex struct that is compatible with HDF5                           *
 ************************************************************************/
template<class TYPE>
struct complex_t {
    TYPE re;
    TYPE im;
};
template<class TYPE>
inline void convert( size_t N, const std::complex<TYPE> *x, complex_t<TYPE> *y )
{
    for ( size_t i = 0; i < N; i++ ) {
        y[i].re = x[i].real();
        y[i].im = x[i].imag();
    }
}
template<class TYPE>
inline void convert( size_t N, const complex_t<TYPE> *x, std::complex<TYPE> *y )
{
    for ( size_t i = 0; i < N; i++ ) {
        y[i] = std::complex<TYPE>( x[i].re, x[i].im );
    }
}


/************************************************************************
 * Get the HDF5 data type                                                *
 ************************************************************************/
template<>
hid_t getHDF5datatype<bool>()
{
    return H5Tcopy( H5T_NATIVE_UCHAR );
}
template<>
hid_t getHDF5datatype<char>()
{
    return H5Tcopy( H5T_NATIVE_CHAR );
}
template<>
hid_t getHDF5datatype<uint8_t>()
{
    return H5Tcopy( H5T_NATIVE_UINT8 );
}
template<>
hid_t getHDF5datatype<int8_t>()
{
    return H5Tcopy( H5T_NATIVE_INT8 );
}
template<>
hid_t getHDF5datatype<uint16_t>()
{
    return H5Tcopy( H5T_NATIVE_UINT16 );
}
template<>
hid_t getHDF5datatype<int16_t>()
{
    return H5Tcopy( H5T_NATIVE_INT16 );
}
template<>
hid_t getHDF5datatype<int>()
{
    return H5Tcopy( H5T_NATIVE_INT );
}
template<>
hid_t getHDF5datatype<unsigned int>()
{
    return H5Tcopy( H5T_NATIVE_UINT );
}
template<>
hid_t getHDF5datatype<long int>()
{
    return H5Tcopy( H5T_NATIVE_LONG );
}
template<>
hid_t getHDF5datatype<unsigned long int>()
{
    return H5Tcopy( H5T_NATIVE_ULONG );
}
template<>
hid_t getHDF5datatype<float>()
{
    return H5Tcopy( H5T_NATIVE_FLOAT );
}
template<>
hid_t getHDF5datatype<double>()
{
    return H5Tcopy( H5T_NATIVE_DOUBLE );
}
template<>
hid_t getHDF5datatype<std::complex<float>>()
{
    hid_t datatype = H5Tcreate( H5T_COMPOUND, sizeof( complex_t<float> ) );
    H5Tinsert( datatype, "real", HOFFSET( complex_t<float>, re ), H5T_NATIVE_FLOAT );
    H5Tinsert( datatype, "imag", HOFFSET( complex_t<float>, im ), H5T_NATIVE_FLOAT );
    return datatype;
}
template<>
hid_t getHDF5datatype<std::complex<double>>()
{
    hid_t datatype = H5Tcreate( H5T_COMPOUND, sizeof( complex_t<double> ) );
    H5Tinsert( datatype, "real", HOFFSET( complex_t<double>, re ), H5T_NATIVE_DOUBLE );
    H5Tinsert( datatype, "imag", HOFFSET( complex_t<double>, im ), H5T_NATIVE_DOUBLE );
    return datatype;
}
template<>
hid_t getHDF5datatype<char *>()
{
    hid_t datatype = H5Tcopy( H5T_C_S1 );
    H5Tset_size( datatype, H5T_VARIABLE );
    return datatype;
}


/************************************************************************
 * Read/write Array types                                                *
 ************************************************************************/
template<>
void readHDF5<AMP::Array<std::string>>( hid_t fid,
                                        const std::string_view &name,
                                        AMP::Array<std::string> &data )
{
    if ( !H5Dexists( fid, name ) ) {
        // Dataset does not exist
        data.resize( 0 );
        return;
    }
    hid_t dataset   = H5Dopen2( fid, name.data(), H5P_DEFAULT );
    hid_t datatype  = H5Dget_type( dataset );
    hid_t dataspace = H5Dget_space( dataset );
    hsize_t dims0[10];
    int ndim  = H5Sget_simple_extent_dims( dataspace, dims0, nullptr );
    auto dims = convertSize( ndim, dims0 );
    data.resize( dims );
    hid_t datatype2 = getHDF5datatype<char *>();
    if ( data.empty() ) {
        // The data is empty
    } else if ( H5Tequal( datatype, datatype2 ) ) {
        // The type of Array and the data in HDF5 match
        auto **tmp = new char *[data.length() * sizeof( char * )];
        memset( tmp, 0, data.length() * sizeof( char * ) );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp );
        for ( size_t i = 0; i < data.length(); i++ )
            data( i ) = std::string( tmp[i] );
        H5Dvlen_reclaim( datatype, dataspace, H5P_DEFAULT, tmp );
        delete[] tmp;
    } else {
        AMP_ERROR( "Unknown format for std::string" );
    }
    H5Dclose( dataset );
    H5Tclose( datatype );
    H5Tclose( datatype2 );
    H5Sclose( dataspace );
}
template<>
void readHDF5<AMP::Array<std::string_view>>( hid_t,
                                             const std::string_view &,
                                             AMP::Array<std::string_view> & )
{
    AMP_ERROR( "Reading data to a string view is not supported" );
}
template<class TYPE>
static void
readHDF5complex( hid_t fid, const std::string_view &name, AMP::Array<std::complex<TYPE>> &data )
{
    if ( !H5Dexists( fid, name ) ) {
        // Dataset does not exist
        data.resize( 0 );
        return;
    }
    hid_t dataset   = H5Dopen2( fid, name.data(), H5P_DEFAULT );
    hid_t datatype  = H5Dget_type( dataset );
    hid_t dataspace = H5Dget_space( dataset );
    hsize_t dims0[10];
    int ndim  = H5Sget_simple_extent_dims( dataspace, dims0, nullptr );
    auto dims = convertSize( ndim, dims0 );
    data.resize( dims );
    hid_t datatype2 = getHDF5datatype<std::complex<TYPE>>();
    if ( data.empty() ) {
        // The data is empty
    } else if ( H5Tequal( datatype, datatype2 ) ) {
        // The type of Array and the data in HDF5 match
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data() );
    } else {
        AMP_ERROR( "We need to convert data formats" );
    }
    H5Dclose( dataset );
    H5Tclose( datatype );
    H5Tclose( datatype2 );
    H5Sclose( dataspace );
}
template<>
void readHDF5<AMP::Array<std::complex<float>>>( hid_t fid,
                                                const std::string_view &name,
                                                AMP::Array<std::complex<float>> &data )
{
    readHDF5complex( fid, name, data );
}
template<>
void readHDF5<AMP::Array<std::complex<double>>>( hid_t fid,
                                                 const std::string_view &name,
                                                 AMP::Array<std::complex<double>> &data )
{
    readHDF5complex( fid, name, data );
}
// clang-format off
#define readWriteHDF5Array( TYPE )                                                          \
    template<>                                                                              \
    void writeHDF5<AMP::Array<TYPE>>( hid_t fid, const std::string_view &name, const AMP::Array<TYPE> &data ) \
    {                                                                                       \
        writeHDF5ArrayDefault<TYPE>( fid, name, data );                                     \
    }                                                                                       \
    template<>                                                                              \
    void readHDF5<AMP::Array<TYPE>>( hid_t fid, const std::string_view &name, AMP::Array<TYPE> &data ) \
    {                                                                                       \
        readHDF5ArrayDefault<TYPE>( fid, name, data );                                      \
    }
readWriteHDF5Array( bool )
readWriteHDF5Array( char )
readWriteHDF5Array( int8_t )
readWriteHDF5Array( int16_t )
readWriteHDF5Array( int32_t )
readWriteHDF5Array( int64_t )
readWriteHDF5Array( uint8_t )
readWriteHDF5Array( uint16_t )
readWriteHDF5Array( uint32_t )
readWriteHDF5Array( uint64_t )
readWriteHDF5Array( float )
readWriteHDF5Array( double )
    // clang-format on


    /************************************************************************
     * Read/write scalar types                                               *
     ************************************************************************/
    template<>
    void readHDF5<std::string>( hid_t fid, const std::string_view &name, std::string &data )
{
    hid_t dataset   = H5Dopen2( fid, name.data(), H5P_DEFAULT );
    hid_t datatype  = H5Dget_type( dataset );
    hid_t datatype0 = getHDF5datatype<char *>();
    if ( H5Tequal( datatype, datatype0 ) ) {
        hid_t dataspace = H5Dget_space( dataset );
        char *tmp[1]    = { nullptr };
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp );
        data = std::string( tmp[0] );
        H5Dvlen_reclaim( datatype, dataspace, H5P_DEFAULT, tmp );
        H5Sclose( dataspace );
    } else {
        AMP::Array<char> tmp;
        readHDF5( fid, name, tmp );
        data = std::string( tmp.data(), tmp.length() );
    }
    H5Dclose( dataset );
    H5Tclose( datatype );
    H5Tclose( datatype0 );
}
template<>
void writeHDF5<std::string>( hid_t fid, const std::string_view &name, const std::string &data )
{
    AMP::Array<char> tmp;
    tmp.viewRaw( { data.length() }, (char *) data.data() );
    writeHDF5( fid, name, tmp );
}
// clang-format off
#define readWriteHDF5Scalar( TYPE )                                                         \
    template<>                                                                              \
    void writeHDF5<TYPE>( hid_t fid, const std::string_view &name, const TYPE &data )       \
    {                                                                                       \
        hid_t dataspace = H5Screate( H5S_SCALAR );                                          \
        hid_t datatype  = getHDF5datatype<TYPE>();                                          \
        hid_t dataset   = H5Dcreate2(                                                       \
            fid, name.data(), datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT ); \
        H5Dwrite( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, H5Ptr( &data ) );       \
        H5Dclose( dataset );                                                                \
        H5Tclose( datatype );                                                               \
        H5Sclose( dataspace );                                                              \
    }                                                                                       \
    template<>                                                                              \
    void readHDF5<TYPE>( hid_t fid, const std::string_view &name, TYPE &data )              \
    {                                                                                       \
        AMP::Array<TYPE> tmp;                                                                    \
        readHDF5( fid, name, tmp );                                                         \
        AMP_INSIST( tmp.ndim() == 1 && tmp.length() == 1, "Error loading " + std::string( name ) ); \
        data = tmp( 0 );                                                                    \
    }
readWriteHDF5Scalar( bool )
readWriteHDF5Scalar( char )
readWriteHDF5Scalar( int8_t )
readWriteHDF5Scalar( int16_t )
readWriteHDF5Scalar( int32_t )
readWriteHDF5Scalar( int64_t )
readWriteHDF5Scalar( uint8_t )
readWriteHDF5Scalar( uint16_t )
readWriteHDF5Scalar( uint32_t )
readWriteHDF5Scalar( uint64_t )
readWriteHDF5Scalar( float )
readWriteHDF5Scalar( double )
readWriteHDF5Scalar( std::complex<float> )
readWriteHDF5Scalar( std::complex<double> )
    // clang-format on


    /******************************************************************
     * Create custom error handler                                     *
     ******************************************************************/
    herr_t hdf5_error_handler( hid_t err_stack, void * )
{
    FILE *fid = tmpfile();
    H5Eprint2( err_stack, fid );
    H5Eclear2( err_stack );
    rewind( fid );
    char msg[1024];
    size_t N = fread( msg, 1, sizeof( msg ) - 1, fid );
    fclose( fid );
    msg[N]           = 0;
    std::string msg2 = "Error calling HDF5 routine:\n";
    AMP_ERROR( msg2 + msg );
    return 0;
}
bool set_hdf5_error_handler()
{
    hid_t error_stack = 0;
    H5E_auto2_t fun   = hdf5_error_handler;
    H5Eset_auto2( error_stack, fun, nullptr );
    return true;
}
bool global_is_hdf5_error_handler_set = set_hdf5_error_handler();


/******************************************************************
 * Open/close HDF5 files                                           *
 ******************************************************************/
hid_t openHDF5( const std::string_view &filename, const char *mode, Compression compress )
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
void closeHDF5( hid_t fid )
{
    // Try to close any remaining objects (needed to ensure we can reopen the data if desired)
    hid_t file[1000], set[1000], group[1000], type[1000], attr[1000];
    size_t N_file  = H5Fget_obj_ids( fid, H5F_OBJ_FILE, 1000, file );
    size_t N_set   = H5Fget_obj_ids( fid, H5F_OBJ_DATASET, 1000, set );
    size_t N_group = H5Fget_obj_ids( fid, H5F_OBJ_GROUP, 1000, group );
    size_t N_type  = H5Fget_obj_ids( fid, H5F_OBJ_DATATYPE, 1000, type );
    size_t N_attr  = H5Fget_obj_ids( fid, H5F_OBJ_ATTR, 1000, attr );
    for ( size_t i = 0; i < N_file; i++ ) {
        if ( file[i] != fid )
            H5Fclose( file[i] );
    }
    for ( size_t i = 0; i < N_set; i++ )
        H5Dclose( set[i] );
    for ( size_t i = 0; i < N_group; i++ )
        H5Gclose( group[i] );
    for ( size_t i = 0; i < N_type; i++ )
        H5Tclose( type[i] );
    for ( size_t i = 0; i < N_attr; i++ )
        H5Aclose( attr[i] );
    // Flush the data (needed to ensure we can reopen the data if desired)
    unsigned intent;
    H5Fget_intent( fid, &intent );
    if ( intent == H5F_ACC_RDWR || intent == H5F_ACC_TRUNC )
        H5Fflush( fid, H5F_SCOPE_GLOBAL );
    // Close the file
    H5Fclose( fid );
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
    return compress;
}


/************************************************************************
 * Create a default chunk size                                           *
 ************************************************************************/
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
 * Write Array                                                           *
 ************************************************************************/
template<class TYPE>
static void writeHDF5complex( hid_t fid,
                              const std::string_view &name,
                              const AMP::Array<std::complex<TYPE>> &data )
{
    hid_t datatype = getHDF5datatype<std::complex<TYPE>>();
    // Create the storage properties
    size_t objSize = sizeof( std::complex<TYPE> );
    hid_t plist    = createChunk( data.size(), defaultCompression( fid ), objSize );
    // Copy the data
    size_t N = data.length();
    auto *y  = new complex_t<TYPE>[N];
    convert( N, data.data(), y );
    // Save the array
    auto dim        = arraySize( data );
    hid_t dataspace = H5Screate_simple( dim.size(), dim.data(), nullptr );
    hid_t dataset =
        H5Dcreate2( fid, name.data(), datatype, dataspace, H5P_DEFAULT, plist, H5P_DEFAULT );
    H5Dwrite( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, H5Ptr( y ) );
    H5Dclose( dataset );
    H5Sclose( dataspace );
    H5Pclose( plist );
    H5Tclose( datatype );
    delete[] y;
}
template<>
void writeHDF5<AMP::Array<std::complex<float>>>( hid_t fid,
                                                 const std::string_view &name,
                                                 const AMP::Array<std::complex<float>> &data )
{
    writeHDF5complex( fid, name, data );
}
template<>
void writeHDF5<AMP::Array<std::complex<double>>>( hid_t fid,
                                                  const std::string_view &name,
                                                  const AMP::Array<std::complex<double>> &data )
{
    writeHDF5complex( fid, name, data );
}
template<>
void writeHDF5<AMP::Array<std::string>>( hid_t fid,
                                         const std::string_view &name,
                                         const AMP::Array<std::string> &data )
{
    auto dim        = arraySize( data );
    hid_t dataspace = H5Screate_simple( dim.size(), dim.data(), nullptr );
    auto **tmp      = new char *[data.length() + 1];
    memset( tmp, 0, ( data.length() + 1 ) * sizeof( char * ) );
    for ( size_t i = 0; i < data.length(); i++ ) {
        tmp[i] = const_cast<char *>( data( i ).data() );
    }
    hid_t datatype = getHDF5datatype<char *>();
    hid_t props    = H5Pcreate( H5P_DATASET_CREATE );
    hid_t dataset  = H5Dcreate1( fid, name.data(), datatype, dataspace, props );
    H5Dwrite( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp );
    H5Pclose( props );
    H5Dclose( dataset );
    H5Tclose( datatype );
    H5Sclose( dataspace );
    delete[] tmp;
}
template<>
void writeHDF5<AMP::Array<std::string_view>>( hid_t,
                                              const std::string_view &,
                                              const AMP::Array<std::string_view> & )
{
    AMP_ERROR( "Writing data from a string_view if not yet supported" );
}


/************************************************************************
 * Specializations for std::vector                                       *
 ************************************************************************/
template<>
void readHDF5<std::vector<bool>>( hid_t fid, const std::string_view &name, std::vector<bool> &data )
{
    AMP::Array<bool> tmp;
    readHDF5( fid, name, tmp );
    data.resize( tmp.length() );
    for ( size_t i = 0; i < data.size(); i++ )
        data[i] = tmp( i );
}
template<>
void writeHDF5<std::vector<bool>>( hid_t fid,
                                   const std::string_view &name,
                                   const std::vector<bool> &x )
{
    AMP::Array<bool> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y( i ) = x[i];
    writeHDF5( fid, name, y );
}


/************************************************************************
 * Explicit instantiations                                              *
 ***********************************************************************/
template void readHDF5<bool>( hid_t, const std::string_view &, bool & );
template void writeHDF5<bool>( hid_t, const std::string_view &, const bool & );
// clang-format off
#define instantiate( FUN )       \
    FUN( char );                 \
    FUN( int8_t );               \
    FUN( int16_t );              \
    FUN( int32_t );              \
    FUN( int64_t );              \
    FUN( uint8_t );              \
    FUN( uint16_t );             \
    FUN( uint32_t );             \
    FUN( uint64_t );             \
    FUN( float );                \
    FUN( double );               \
    FUN( long double );
#define instantiateScalerData( TYPE )                             \
    template void readHDF5<TYPE>( hid_t, const std::string_view &, TYPE & ); \
    template void writeHDF5<TYPE>( hid_t, const std::string_view &, const TYPE& )
#define instantiateVectorData( TYPE )                             \
    template void readHDF5<std::vector<TYPE>>( hid_t, const std::string_view &, std::vector<TYPE> & ); \
    template void writeHDF5<std::vector<TYPE>>( hid_t, const std::string_view &, const std::vector<TYPE>& )
instantiate( instantiateScalerData )        // Scalar data
instantiate( instantiateVectorData )        // Vector data
// clang-format on


#else // No HDF5
// Dummy implementations for no HDF5
hid_t openHDF5( const std::string_view &, const char *, Compression ) { return 0; }
void closeHDF5( hid_t ) {}
bool H5Gexists( hid_t, const std::string_view & ) { return false; }
bool H5Dexists( hid_t, const std::string_view & ) { return false; }
hid_t createGroup( hid_t, const std::string_view & ) { return 0; }
hid_t openGroup( hid_t, const std::string_view & ) { return 0; }
void closeGroup( hid_t ) {}
#endif

} // namespace AMP
