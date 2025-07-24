#include "AMP/AMP_TPLs.h"
#include "AMP/IO/HDF.h"
#include "AMP/IO/HDF.hpp"
#include "AMP/utils/Array.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/Utilities.h"

#include <array>
#include <complex>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>


#ifdef AMP_USE_HDF5


namespace AMP::IO {


static inline const void *H5Ptr( const void *x ) { return x == nullptr ? ( (void *) 1 ) : x; }


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
template<class TYPE>
hid_t getHDF5datatype()
{
    if constexpr ( std::is_same_v<TYPE, bool> ) {
        return H5Tcopy( H5T_NATIVE_UCHAR );
    } else if constexpr ( std::is_same_v<TYPE, char> ) {
        return H5Tcopy( H5T_NATIVE_CHAR );
    } else if constexpr ( std::is_same_v<TYPE, uint8_t> ) {
        return H5Tcopy( H5T_NATIVE_UINT8 );
    } else if constexpr ( std::is_same_v<TYPE, int8_t> ) {
        return H5Tcopy( H5T_NATIVE_INT8 );
    } else if constexpr ( std::is_same_v<TYPE, uint16_t> ) {
        return H5Tcopy( H5T_NATIVE_UINT16 );
    } else if constexpr ( std::is_same_v<TYPE, int16_t> ) {
        return H5Tcopy( H5T_NATIVE_INT16 );
    } else if constexpr ( std::is_same_v<TYPE, int> ) {
        return H5Tcopy( H5T_NATIVE_INT );
    } else if constexpr ( std::is_same_v<TYPE, unsigned int> ) {
        return H5Tcopy( H5T_NATIVE_UINT );
    } else if constexpr ( std::is_same_v<TYPE, long int> ) {
        return H5Tcopy( H5T_NATIVE_LONG );
    } else if constexpr ( std::is_same_v<TYPE, unsigned long int> ) {
        return H5Tcopy( H5T_NATIVE_ULONG );
    } else if constexpr ( std::is_same_v<TYPE, float> ) {
        return H5Tcopy( H5T_NATIVE_FLOAT );
    } else if constexpr ( std::is_same_v<TYPE, double> ) {
        return H5Tcopy( H5T_NATIVE_DOUBLE );
    } else if constexpr ( std::is_same_v<TYPE, std::complex<float>> ) {
        hid_t datatype = H5Tcreate( H5T_COMPOUND, sizeof( complex_t<float> ) );
        H5Tinsert( datatype, "real", HOFFSET( complex_t<float>, re ), H5T_NATIVE_FLOAT );
        H5Tinsert( datatype, "imag", HOFFSET( complex_t<float>, im ), H5T_NATIVE_FLOAT );
        return datatype;
    } else if constexpr ( std::is_same_v<TYPE, std::complex<double>> ) {
        hid_t datatype = H5Tcreate( H5T_COMPOUND, sizeof( complex_t<double> ) );
        H5Tinsert( datatype, "real", HOFFSET( complex_t<double>, re ), H5T_NATIVE_DOUBLE );
        H5Tinsert( datatype, "imag", HOFFSET( complex_t<double>, im ), H5T_NATIVE_DOUBLE );
        return datatype;
    } else if constexpr ( std::is_same_v<TYPE, char *> ) {
        hid_t datatype = H5Tcopy( H5T_C_S1 );
        H5Tset_size( datatype, H5T_VARIABLE );
        return datatype;
    } else {
        throw std::logic_error( "Unknown type" );
    }
}


// clang-format off


/************************************************************************
 * std::complex                                                          *
 ************************************************************************/
template<class TYPE>
static void
readHDF5complex( hid_t fid, const std::string &name, AMP::Array<std::complex<TYPE>> &data )
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
template<class TYPE>
static void writeHDF5complex( hid_t fid,
                              const std::string &name,
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
void readHDF5Array<std::complex<float>>( hid_t fid,
                                                const std::string &name,
                                                AMP::Array<std::complex<float>> &data )
{
    readHDF5complex( fid, name, data );
}
template<>
void readHDF5Array<std::complex<double>>( hid_t fid,
                                                 const std::string &name,
                                                 AMP::Array<std::complex<double>> &data )
{
    readHDF5complex( fid, name, data );
}
template<>
void writeHDF5Array<std::complex<float>>( hid_t fid,
                                          const std::string &name,
                                          const AMP::Array<std::complex<float>> &data )
{
    writeHDF5complex( fid, name, data );
}
template<>
void writeHDF5Array<std::complex<double>>( hid_t fid,
                                           const std::string &name,
                                          const AMP::Array<std::complex<double>> &data )
{
    writeHDF5complex( fid, name, data );
}
template<>
void readHDF5Scalar<std::complex<float>>( hid_t fid,
                                                const std::string &name,
                                                std::complex<float> &data )
{
    AMP::Array<std::complex<float>> data2;
    readHDF5complex( fid, name, data2 );
    AMP_ASSERT( data2.length() == 1 );
    data = data2( 0 );
}
template<>
void readHDF5Scalar<std::complex<double>>( hid_t fid,
                                                 const std::string &name,
                                                 std::complex<double> &data )
{
    AMP::Array<std::complex<double>> data2;
    readHDF5complex( fid, name, data2 );
    AMP_ASSERT( data2.length() == 1 );
    data = data2( 0 );
}
template<>
void writeHDF5Scalar<std::complex<float>>( hid_t fid,
                                          const std::string &name,
                                          const std::complex<float> &data )
{
    AMP::Array<std::complex<float>> data2 = { data };
    writeHDF5complex( fid, name, data2 );
}
template<>
void writeHDF5Scalar<std::complex<double>>( hid_t fid,
                                           const std::string &name,
                                           const std::complex<double> &data )
{
    AMP::Array<std::complex<double>> data2 = { data };
    writeHDF5complex( fid, name, data2 );
}

template hid_t getHDF5datatype<std::complex<float>>();
template hid_t getHDF5datatype<std::complex<double>>();


/************************************************************************
 * std::string                                                          *
 ************************************************************************/
template<>
void readHDF5Scalar<std::string>( hid_t fid, const std::string &name, std::string &data )
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
void writeHDF5Scalar<std::string>( hid_t fid, const std::string &name, const std::string &data )
{
    AMP::Array<char> tmp;
    tmp.viewRaw( { data.length() }, (char *) data.data() );
    writeHDF5Array( fid, name, tmp );
}
template<>
void readHDF5Array<std::string>( hid_t fid,
                                        const std::string &name,
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
void writeHDF5Array<std::string>( hid_t fid,
                                         const std::string &name,
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
void writeHDF5Array<std::string_view>( hid_t fid,
                                       const std::string &name,
                                       const AMP::Array<std::string_view> &x )
{
    AMP::Array<std::string> y( x.size() );
    for ( size_t i=0; i<x.length(); i++)
        y(i) = x(i);
    writeHDF5Array( fid, name, y );
}
template<class T>
void writeHDF5Array( hid_t fid, const std::string &name, const AMP::Array<T> &data )
{
    writeHDF5ArrayDefault<T>( fid, name, data );
}


/************************************************************************
 * MeshPoint                                                            *
 ***********************************************************************/
template<>
void readHDF5Scalar<AMP::Mesh::Point>( hid_t fid, const std::string &name, AMP::Mesh::Point &x )
{
    std::array<double,4> y;
    readHDF5( fid, name, y );
    x = AMP::Mesh::Point( y[0], { y[1], y[2], y[3] } );
}
template<>
void writeHDF5Scalar<AMP::Mesh::Point>( hid_t fid, const std::string &name, const AMP::Mesh::Point &x )
{
    std::array<double,4> y = { (double) x.ndim(), x[0], x[1], x[2] };
    writeHDF5( fid, name, y );
}
template<>
void readHDF5Array<AMP::Mesh::Point>( hid_t fid, const std::string &name, AMP::Array<AMP::Mesh::Point> &x )
{
    AMP::Array<double> y;
    readHDF5Array( fid, name, y );
    x.resize( pop( y.size() ) );
    for ( size_t i = 0; i < x.length(); i++ )
        x( i ) = AMP::Mesh::Point( y(0,i), { y(1,i), y(2,i), y(3,i) } );
}
template<>
void writeHDF5Array<AMP::Mesh::Point>( hid_t fid, const std::string &name, const AMP::Array<AMP::Mesh::Point> &x )
{
    AMP::Array<double> y( AMP::cat( AMP::ArraySize( 4 ), x.size() ) );
    for ( size_t i = 0; i < x.length(); i++ ) {
        y( 0, i ) = x( i ).ndim();
        y( 1, i ) = x( i )[0];
        y( 2, i ) = x( i )[1];
        y( 3, i ) = x( i )[2];
    }
    writeHDF5Array( fid, name, y );
}


/************************************************************************
 * Read/write scalar types                                               *
 ************************************************************************/
template<class TYPE>
void writeHDF5Scalar( hid_t fid, const std::string &name, const TYPE &data )
{
    hid_t dataspace = H5Screate( H5S_SCALAR );
    hid_t datatype  = getHDF5datatype<TYPE>();
    hid_t dataset   = H5Dcreate2( fid, name.data(), datatype,
        dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    H5Dwrite( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, H5Ptr( &data ) );
    H5Dclose( dataset );
    H5Tclose( datatype );
    H5Sclose( dataspace );
}
template<class TYPE>
void readHDF5Scalar( hid_t fid, const std::string &name, TYPE &data )
{
    AMP::Array<TYPE> tmp;
    readHDF5( fid, name, tmp );
    if ( tmp.ndim() != 1 || tmp.length() != 1 ) {
        auto msg = AMP::Utilities::stringf( "Error loading %s: (%i,%i)", name.data(), tmp.ndim(), (int) tmp.length() );
        AMP_ERROR( msg );
    }
    data = tmp( 0 );
}
template<class T>
void readHDF5Array( hid_t fid, const std::string &name, AMP::Array<T> &data )
{
    readHDF5ArrayDefault<T>( fid, name, data );
}


/************************************************************************
 * TypeID                                                               *
 ***********************************************************************/
template<>
hid_t getHDF5datatype<AMP::typeID>()
{
  hid_t datatype = H5Tcreate( H5T_COMPOUND, sizeof( typeID ) );
  H5Tinsert( datatype, "bytes", HOFFSET( typeID, bytes ), H5T_NATIVE_UINT32 );
  H5Tinsert( datatype, "hash", HOFFSET( typeID, hash ), H5T_NATIVE_UINT32 );
  const hsize_t rank = 120;
  hid_t array_id = H5Tarray_create(H5T_NATIVE_CHAR, 1, &rank);
  H5Tinsert( datatype, "name", HOFFSET( typeID, name ), array_id );
  return datatype;
}
template<>
void readHDF5Scalar<AMP::typeID>( hid_t fid, const std::string &name, AMP::typeID &x )
{
  auto datatype = getHDF5datatype<AMP::typeID>();
  hid_t dataset   = H5Dopen2( fid, name.data(), H5P_DEFAULT );
  H5Dread(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &x);
  H5Dclose( dataset );
  H5Tclose( datatype );
}
template<>
void writeHDF5Scalar<AMP::typeID>( hid_t fid, const std::string &name, const AMP::typeID &data )
{
  auto datatype = getHDF5datatype<AMP::typeID>();
  hid_t dataset   = H5Dopen2( fid, name.data(), H5P_DEFAULT );
  H5Dwrite( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, H5Ptr( &data ) );
  H5Dclose( dataset );
  H5Tclose( datatype );
}
template<>
void readHDF5Array<AMP::typeID>( hid_t fid, const std::string &name, AMP::Array<AMP::typeID> &data )
{
    readHDF5ArrayDefault<AMP::typeID>( fid, name, data );
}
template<>
void writeHDF5Array<AMP::typeID>( hid_t fid, const std::string &name, const AMP::Array<AMP::typeID> &data )
{
    writeHDF5ArrayDefault<AMP::typeID>( fid, name, data );
}


} // namespace AMP::IO


#include "AMP/utils/Array.hpp"
instantiateArrayConstructors( AMP::Mesh::Point );


INSTANTIATE_HDF5( AMP::typeID );
INSTANTIATE_HDF5( std::string );
INSTANTIATE_HDF5( std::string_view );
INSTANTIATE_HDF5( AMP::Mesh::Point );
INSTANTIATE_HDF5( std::complex<float> );
INSTANTIATE_HDF5( std::complex<double> );
INSTANTIATE_SCALAR_HDF5( std::string_view );
INSTANTIATE_AMPARRAY_HDF5( AMP::typeID );
INSTANTIATE_AMPARRAY_HDF5( std::string );
INSTANTIATE_AMPARRAY_HDF5( AMP::Mesh::Point );
INSTANTIATE_AMPARRAY_HDF5( std::complex<float> );
INSTANTIATE_AMPARRAY_HDF5( std::complex<double> );

#else
INSTANTIATE_HDF5( AMP::typeID );
INSTANTIATE_HDF5( std::string );
INSTANTIATE_HDF5( std::string_view );
INSTANTIATE_HDF5( AMP::Mesh::Point );
INSTANTIATE_HDF5( std::complex<float> );
INSTANTIATE_HDF5( std::complex<double> );
INSTANTIATE_AMPARRAY_HDF5( AMP::typeID );
INSTANTIATE_AMPARRAY_HDF5( std::string );
INSTANTIATE_AMPARRAY_HDF5( std::string_view );
INSTANTIATE_AMPARRAY_HDF5( AMP::Mesh::Point );
INSTANTIATE_AMPARRAY_HDF5( std::complex<float> );
INSTANTIATE_AMPARRAY_HDF5( std::complex<double> );
#endif




/************************************************************************
 * Explicit instantiations                                              *
 ***********************************************************************/
#define INSTANTIATE_DEFAULT( TYPE )   \
    INSTANTIATE_HDF5( TYPE );         \
    INSTANTIATE_SCALAR_HDF5( TYPE );  \
    INSTANTIATE_RWARRAY_HDF5( TYPE ); \
    INSTANTIATE_AMPARRAY_HDF5( TYPE )
INSTANTIATE_DEFAULT( bool );
INSTANTIATE_DEFAULT( char );
INSTANTIATE_DEFAULT( int8_t );
INSTANTIATE_DEFAULT( int16_t );
INSTANTIATE_DEFAULT( int32_t );
INSTANTIATE_DEFAULT( int64_t );
INSTANTIATE_DEFAULT( uint8_t );
INSTANTIATE_DEFAULT( uint16_t );
INSTANTIATE_DEFAULT( uint32_t );
INSTANTIATE_DEFAULT( uint64_t );
INSTANTIATE_DEFAULT( float );
INSTANTIATE_DEFAULT( double );
INSTANTIATE_DEFAULT( long double );
INSTANTIATE_DEFAULT( std::byte );
#if defined( WIN32 ) || defined( _WIN32 ) || defined( WIN64 ) || defined( _WIN64 )
INSTANTIATE_DEFAULT( long int );
#endif


/************************************************************************
 * std::array                                                           *
 ***********************************************************************/
#define INSTANTIATE_HDF5_ARRAY( TYPE, N ) \
    template void AMP::IO::readHDF5<std::array<TYPE,N>>( hid_t, const std::string &, std::array<TYPE,N> & ); \
    template void AMP::IO::writeHDF5<std::array<TYPE,N>>( hid_t, const std::string &, const std::array<TYPE,N> & )
INSTANTIATE_HDF5_ARRAY( bool, 1 );
INSTANTIATE_HDF5_ARRAY( bool, 2 );
INSTANTIATE_HDF5_ARRAY( bool, 3 );
INSTANTIATE_HDF5_ARRAY( int, 1 );
INSTANTIATE_HDF5_ARRAY( int, 2 );
INSTANTIATE_HDF5_ARRAY( int, 3 );
INSTANTIATE_HDF5_ARRAY( int, 6 );
INSTANTIATE_HDF5_ARRAY( double, 1 );
INSTANTIATE_HDF5_ARRAY( double, 2 );
INSTANTIATE_HDF5_ARRAY( double, 3 );
INSTANTIATE_HDF5_ARRAY( double, 4 );
INSTANTIATE_HDF5_ARRAY( double, 6 );
INSTANTIATE_HDF5_ARRAY( double, 9 );

