// This file contains helper functions and interfaces for reading/writing HDF5

#ifndef included_AMP_HDF5_hpp
#define included_AMP_HDF5_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/IO/HDF.h"
#include "AMP/IO/HDF5_Class.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/TypeTraits.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/typeid.h"

#include <array>
#include <complex>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>


namespace AMP::Geometry {
class Geometry;
}
namespace AMP::Mesh {
class Mesh;
}
namespace AMP::LinearAlgebra {
class Vector;
class Matrix;
} // namespace AMP::LinearAlgebra
namespace AMP::Operator {
class Operator;
}
namespace AMP::Solver {
class SolverStrategy;
}
namespace AMP::TimeIntegrator {
class TimeIntegrator;
}
namespace AMP::Utilities {
void nullUse( const void * );
}


namespace AMP::IO {


#ifdef AMP_USE_HDF5


/******************************************************************
 * Define some specializations                                     *
 ******************************************************************/
template<>
void writeHDF5<std::shared_ptr<AMP::Geometry::Geometry>>(
    hid_t, const std::string &, const std::shared_ptr<AMP::Geometry::Geometry> & );
template<>
void writeHDF5<std::shared_ptr<const AMP::Geometry::Geometry>>(
    hid_t, const std::string &, const std::shared_ptr<const AMP::Geometry::Geometry> & );
template<>
void readHDF5<std::shared_ptr<AMP::Geometry::Geometry>>(
    hid_t, const std::string &, std::shared_ptr<AMP::Geometry::Geometry> & );


/******************************************************************
 * Default implementation                                          *
 ******************************************************************/
template<class T>
void writeHDF5Array( hid_t, const std::string &, const AMP::Array<T> & );
template<class T>
void readHDF5Array( hid_t, const std::string &, AMP::Array<T> & );
template<class T>
void writeHDF5Scalar( hid_t, const std::string &, const T & );
template<class T>
void readHDF5Scalar( hid_t, const std::string &, T & );
template<class TYPE>
void writeHDF5( hid_t fid, const std::string &name, const TYPE &x )
{
    if constexpr ( AMP::is_shared_ptr_v<TYPE> ) {
        // We are dealing with a std::shared_ptr
        writeHDF5( fid, name, *x );
    } else if constexpr ( AMP::is_vector_v<TYPE> ) {
        // We are dealing with a std::vector
        typedef decltype( *x.begin() ) TYPE2;
        typedef typename AMP::remove_cvref_t<TYPE2> TYPE3;
        if constexpr ( std::is_same_v<TYPE3, bool> ) {
            AMP::Array<bool> y( x.size() );
            for ( size_t i = 0; i < x.size(); i++ )
                y( i ) = x[i];
            writeHDF5Array( fid, name, y );
        } else {
            AMP::Array<TYPE3> y;
            y.viewRaw( { x.size() }, const_cast<TYPE3 *>( x.data() ) );
            writeHDF5( fid, name, y );
        }
    } else if constexpr ( std::is_array_v<TYPE> ) {
        // We are dealing with an C array
        typedef decltype( *x ) TYPE2;
        typedef typename AMP::remove_cvref_t<TYPE2> TYPE3;
        AMP::Array<TYPE3> y;
        y.viewRaw( { std::size( x ) }, const_cast<TYPE3 *>( x ) );
        writeHDF5( fid, name, y );
    } else if constexpr ( AMP::is_Array_v<TYPE> ) {
        // We are dealing with an Array
        if constexpr ( AMP::is_shared_ptr_v<typename TYPE::value_type> ) {
            typedef typename TYPE::value_type TYPE2;
            typedef typename TYPE2::element_type TYPE3;
            AMP::Array<TYPE3> y( x.size() );
            for ( size_t i = 0; i < x.length(); i++ )
                y( i ) = *x( i );
            writeHDF5Array( fid, name, y );
        } else {
            writeHDF5Array( fid, name, x );
        }
    } else if constexpr ( std::is_same_v<TYPE, std::string> ) {
        // We are dealing with a std::string
        writeHDF5Scalar( fid, name, x );
    } else if constexpr ( std::is_same_v<TYPE, std::string> || std::is_same_v<TYPE, char *> ||
                          std::is_same_v<TYPE, const char *> ) {
        // We are dealing with a string or char array
        writeHDF5( fid, name, std::string( x ) );
    } else if constexpr ( AMP::is_container_v<TYPE> ) {
        // We are dealing with a container
        typedef decltype( *x.begin() ) TYPE2;
        typedef typename AMP::remove_cvref_t<TYPE2> TYPE3;
        std::vector<TYPE3> x2( x.begin(), x.end() );
        writeHDF5<std::vector<TYPE3>>( fid, name, x2 );
    } else {
        writeHDF5Scalar( fid, name, x );
    }
}
template<class TYPE>
void readHDF5( hid_t fid, const std::string &name, TYPE &x )
{
    if constexpr ( AMP::is_shared_ptr_v<TYPE> ) {
        // We are dealing with a std::shared_ptr
        readHDF5( fid, name, *x );
    } else if constexpr ( AMP::is_vector_v<TYPE> ) {
        // We are dealing with a std::vector
        typedef typename AMP::remove_cvref_t<decltype( *x.begin() )> TYPE2;
        if constexpr ( std::is_same_v<TYPE2, std::vector<bool>::reference> ) {
            AMP::Array<bool> y;
            readHDF5Array( fid, name, y );
            x.resize( y.length() );
            for ( size_t i = 0; i < x.size(); i++ )
                x[i] = y( i );
        } else {
            AMP::Array<TYPE2> y;
            readHDF5( fid, name, y );
            x.resize( y.length() );
            // Swap the elements in the arrays to use the move operator
            for ( size_t i = 0; i < x.size(); i++ )
                std::swap( x[i], y( i ) );
        }
    } else if constexpr ( std::is_array_v<TYPE> ) {
        // We are dealing with a C array
        typedef typename AMP::remove_cvref_t<decltype( *x )> TYPE2;
        AMP::Array<TYPE2> y;
        readHDF5( fid, name, y );
        AMP_ASSERT( y.length() == std::size( x ) );
        // Swap the elements in the arrays to use the move operator
        for ( size_t i = 0; i < std::size( x ); i++ )
            std::swap( x[i], y( i ) );
    } else if constexpr ( AMP::is_array_v<TYPE> ) {
        // We are dealing with a std::array
        typedef typename AMP::remove_cvref_t<decltype( *x.begin() )> TYPE2;
        AMP::Array<TYPE2> y;
        readHDF5( fid, name, y );
        AMP_ASSERT( y.length() == x.size() );
        // Swap the elements in the arrays to use the move operator
        for ( size_t i = 0; i < x.size(); i++ )
            std::swap( x[i], y( i ) );
    } else if constexpr ( AMP::is_Array_v<TYPE> ) {
        // We are dealing with an Array
        if constexpr ( AMP::is_shared_ptr_v<typename TYPE::value_type> ) {
            typedef typename TYPE::value_type TYPE2;
            typedef typename TYPE2::element_type TYPE3;
            AMP::Array<TYPE3> y( x.size() );
            readHDF5Array( fid, name, y );
            x.resize( y.size() );
            for ( size_t i = 0; i < x.length(); i++ )
                x( i ) = std::make_shared<TYPE3>( y( i ) );
        } else {
            readHDF5Array( fid, name, x );
        }
    } else if constexpr ( std::is_same_v<TYPE, std::string> ) {
        // We are dealing with a std::string
        readHDF5Scalar( fid, name, x );
    } else if constexpr ( std::is_same_v<TYPE, std::string_view> || std::is_same_v<TYPE, char *> ||
                          std::is_same_v<TYPE, const char *> ) {
        // We are dealing with a string or char array
        AMP::Utilities::nullUse( &fid );
        throw std::logic_error(
            "Reading data into a string_view, char*, const char* is not supported" );
    } else if constexpr ( AMP::is_container_v<TYPE> ) {
        // We are dealing with a container
        typedef typename AMP::remove_cvref_t<decltype( *x.begin() )> TYPE2;
        AMP::Array<TYPE2> y;
        readHDF5( fid, name, y );
        if ( x.size() == y.length() ) {
            auto it = x.begin();
            for ( size_t i = 0; i < y.length(); i++, ++it )
                *it = y( i );
        } else {
            throw std::logic_error( "Reading data into an arbitrary container is not finished" );
        }
    } else {
        readHDF5Scalar( fid, name, x );
    }
}


/************************************************************************
 * readAndConvertHDF5Data                                                *
 ************************************************************************/
template<class T>
typename std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>, void>
readAndConvertHDF5Data( hid_t dataset, hid_t datatype, AMP::Array<T> &data )
{
    if ( H5Tequal( datatype, H5T_NATIVE_CHAR ) ) {
        AMP::Array<char> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_UCHAR ) ) {
        AMP::Array<unsigned char> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_INT8 ) ) {
        AMP::Array<int8_t> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_UINT8 ) ) {
        AMP::Array<uint8_t> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_INT ) ) {
        AMP::Array<int> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_UINT ) ) {
        AMP::Array<unsigned int> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_LONG ) ) {
        AMP::Array<long int> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_ULONG ) ) {
        AMP::Array<unsigned long int> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_FLOAT ) ) {
        AMP::Array<float> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else if ( H5Tequal( datatype, H5T_NATIVE_DOUBLE ) ) {
        AMP::Array<double> data2( data.size() );
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data2.data() );
        data.copy( data2 );
    } else {
        AMP_ERROR( "We need to convert unknown data format" );
    }
}
template<class T>
typename std::enable_if_t<!std::is_integral_v<T> && !std::is_floating_point_v<T>, void>
readAndConvertHDF5Data( hid_t, hid_t, AMP::Array<T> & )
{
    AMP_ERROR( "Unable to convert data" );
}


/************************************************************************
 * Helper function to get the size of an Array                           *
 * Note that HDF5 uses C ordered arrays so we need to flip the dimensions*
 ************************************************************************/
inline std::vector<hsize_t> arraySize( const AMP::ArraySize &s1 )
{
    int N = s1.ndim();
    std::vector<hsize_t> s2( std::max( N, 1 ), 0 );
    for ( int i = 0; i < N; i++ )
        s2[N - i - 1] = static_cast<hsize_t>( s1[i] );
    return s2;
}
template<class T>
inline std::vector<hsize_t> arraySize( const AMP::Array<T> &x )
{
    return arraySize( x.size() );
}
inline std::vector<size_t> convertSize( int N, const hsize_t *dims )
{
    if ( N == 0 )
        return std::vector<size_t>( 1, 1 );
    std::vector<size_t> size( N, 0 );
    for ( int i = 0; i < N; i++ )
        size[N - i - 1] = static_cast<size_t>( dims[i] );
    return size;
}


/******************************************************************
 * Helper functions for reading/writing AMP::Array<TYPE>           *
 ******************************************************************/
template<class T>
void readHDF5ArrayDefault( hid_t fid, const std::string &name, AMP::Array<T> &data )
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
    int ndim  = H5Sget_simple_extent_dims( dataspace, dims0, NULL );
    auto dims = convertSize( ndim, dims0 );
    data.resize( dims );
    hid_t datatype2 = getHDF5datatype<T>();
    if ( data.empty() ) {
        // The data is empty
    } else if ( H5Tequal( datatype, datatype2 ) ) {
        // The type of Array and the data in HDF5 match
        H5Dread( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data() );
    } else {
        // Try to convert the data
        readAndConvertHDF5Data( dataset, datatype, data );
    }
    H5Dclose( dataset );
    H5Tclose( datatype );
    H5Tclose( datatype2 );
    H5Sclose( dataspace );
}
template<class T>
void writeHDF5ArrayDefault( hid_t fid, const std::string &name, const AMP::Array<T> &data )
{
    size_t N_bytes = data.length() * sizeof( T );
    auto dim       = arraySize( data );
    hid_t plist    = H5P_DEFAULT;
    if ( N_bytes < 0x7500 ) {
        // Use compact storage (limited to < 30K)
        plist       = H5Pcreate( H5P_DATASET_CREATE );
        auto status = H5Pset_layout( plist, H5D_COMPACT );
        AMP_ASSERT( status == 0 );
    } else {
        // Use compression if available
        plist = createChunk( data.size(), defaultCompression( fid ), sizeof( T ) );
    }
    hid_t dataspace = H5Screate_simple( dim.size(), dim.data(), NULL );
    hid_t datatype  = getHDF5datatype<T>();
    hid_t dataset =
        H5Dcreate2( fid, name.data(), datatype, dataspace, H5P_DEFAULT, plist, H5P_DEFAULT );
    const void *ptr = data.data() == NULL ? ( (void *) 1 ) : data.data();
    H5Dwrite( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptr );
    H5Dclose( dataset );
    H5Tclose( datatype );
    H5Sclose( dataspace );
    if ( plist != H5P_DEFAULT )
        H5Pclose( plist );
}


#else


// Default no-op implementations for use without HDF5
// clang-format off
template<class T> hid_t getHDF5datatype() { return 0; }
template<class T> void writeHDF5( hid_t, const std::string &, const T & ) {}
template<class T> void writeHDF5Array( hid_t, const std::string &, const AMP::Array<T> & ) {}
template<class T> void writeHDF5Scalar( hid_t, const std::string &, const T & ) {}
template<class T> void readHDF5( hid_t, const std::string &, T & ) {}
template<class T> void readHDF5Array( hid_t, const std::string &, AMP::Array<T> & ) {}
template<class T> void readHDF5Scalar( hid_t, const std::string &, T & ) {}
template<class T> std::unique_ptr<T> readHDF5( hid_t, const std::string &, AMP_MPI ) {}
    // clang-format on


#endif


} // namespace AMP::IO


#ifdef AMP_USE_HDF5 // USE HDF5
    #define INSTANTIATE_SCALAR_HDF5( TYPE )                                                \
        template hid_t AMP::IO::getHDF5datatype<TYPE>();                                   \
        template void AMP::IO::readHDF5Scalar<TYPE>( hid_t, const std::string &, TYPE & ); \
        template void AMP::IO::writeHDF5Scalar<TYPE>( hid_t, const std::string &, const TYPE & )

    #define INSTANTIATE_RWARRAY_HDF5( TYPE )                  \
        template void AMP::IO::readHDF5Array<TYPE>(           \
            hid_t, const std::string &, AMP::Array<TYPE> & ); \
        template void AMP::IO::writeHDF5Array<TYPE>(          \
            hid_t, const std::string &, const AMP::Array<TYPE> & )

    #define INSTANTIATE_HDF5( TYPE )                                                        \
        template void AMP::IO::readHDF5<TYPE>( hid_t, const std::string &, TYPE & );        \
        template void AMP::IO::readHDF5<std::vector<TYPE>>(                                 \
            hid_t, const std::string &, std::vector<TYPE> & );                              \
        template void AMP::IO::writeHDF5<TYPE>( hid_t, const std::string &, const TYPE & ); \
        template void AMP::IO::writeHDF5<std::vector<TYPE>>(                                \
            hid_t, const std::string &, const std::vector<TYPE> & )

    #define INSTANTIATE_AMPARRAY_HDF5( TYPE )                 \
        template void AMP::IO::readHDF5<AMP::Array<TYPE>>(    \
            hid_t, const std::string &, AMP::Array<TYPE> & ); \
        template void AMP::IO::writeHDF5<AMP::Array<TYPE>>(   \
            hid_t, const std::string &, const AMP::Array<TYPE> & )
#else
    #define INSTANTIATE_SCALAR_HDF5( TYPE )
    #define INSTANTIATE_RWARRAY_HDF5( TYPE )

    #define INSTANTIATE_HDF5( TYPE )                                                        \
        template void AMP::IO::readHDF5<TYPE>( hid_t, const std::string &, TYPE & );        \
        template void AMP::IO::readHDF5<std::vector<TYPE>>(                                 \
            hid_t, const std::string &, std::vector<TYPE> & );                              \
        template void AMP::IO::writeHDF5<TYPE>( hid_t, const std::string &, const TYPE & ); \
        template void AMP::IO::writeHDF5<std::vector<TYPE>>(                                \
            hid_t, const std::string &, const std::vector<TYPE> & )

    #define INSTANTIATE_AMPARRAY_HDF5( TYPE )                 \
        template void AMP::IO::readHDF5<AMP::Array<TYPE>>(    \
            hid_t, const std::string &, AMP::Array<TYPE> & ); \
        template void AMP::IO::writeHDF5<AMP::Array<TYPE>>(   \
            hid_t, const std::string &, const AMP::Array<TYPE> & )
#endif


#endif
