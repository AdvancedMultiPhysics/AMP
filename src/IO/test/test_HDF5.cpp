// This file tests the HDF5 interfaces
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "AMP/IO/HDF.h"
#include "AMP/IO/HDF.hpp"
#include "AMP/IO/HDF5_Class.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Array.hpp"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"

#include <random>


// Record a pass/fail
static inline void record( bool pass, const std::string &name, AMP::UnitTest &ut )
{
    if ( pass )
        ut.passes( name );
    else
        ut.failure( name );
}


// Structure to test HDF5 compounds
struct compoundStruct {
    compoundStruct()                                    = default;
    compoundStruct( const compoundStruct & )            = default;
    compoundStruct &operator=( const compoundStruct & ) = default;
    compoundStruct( int x, float y, double z ) : a( x ), b( y ), c( z ) {}
    bool operator==( const compoundStruct &x ) const { return x.a == a && x.b == b && x.c == c; }
    int a;
    float b;
    double c;
};
template<>
hid_t AMP::IO::getHDF5datatype<compoundStruct>()
{
    auto datatype = H5Tcreate( H5T_COMPOUND, sizeof( compoundStruct ) );
    H5Tinsert( datatype, "a", HOFFSET( compoundStruct, a ), H5T_NATIVE_INT );
    H5Tinsert( datatype, "b", HOFFSET( compoundStruct, b ), H5T_NATIVE_FLOAT );
    H5Tinsert( datatype, "c", HOFFSET( compoundStruct, c ), H5T_NATIVE_DOUBLE );
    return datatype;
}
template<>
void AMP::IO::writeHDF5Array<compoundStruct>( hid_t fid,
                                              const std::string &name,
                                              const AMP::Array<compoundStruct> &data )
{
    AMP::IO::writeHDF5ArrayDefault<compoundStruct>( fid, name, data );
}
template<>
void AMP::IO::writeHDF5Scalar<compoundStruct>( hid_t fid,
                                               const std::string &name,
                                               const compoundStruct &data )
{
    AMP::IO::writeHDF5ArrayDefault( fid, name, AMP::Array<compoundStruct>( { 1 }, &data ) );
}


// Function to write an opaque object
template<class TYPE>
void writeOpaque( hid_t fid, const std::string &name, const TYPE &data )
{
    hid_t datatype  = H5Tcreate( H5T_OPAQUE, sizeof( TYPE ) );
    hid_t dataspace = H5Screate( H5S_SCALAR );
    hid_t dataset =
        H5Dcreate2( fid, name.data(), datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT );
    H5Dwrite( dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data );
    H5Dclose( dataset );
    H5Tclose( datatype );
    H5Sclose( dataspace );
}


// Test a basic read
template<class TYPE>
void checkHDF5( hid_t fid, const std::string &name, const TYPE &x, AMP::UnitTest &ut )
{
    TYPE y;
    AMP::IO::readHDF5( fid, name, y );
    record( x == y, name, ut );
}


// Read the data from HDF5 using the class interface
template<class TYPE>
void checkScalar( const AMP::IO::HDF5data &ptr,
                  const std::string &name,
                  const TYPE &x,
                  AMP::UnitTest &ut )
{
    AMP::Array<TYPE> data;
    auto ptr2 = ptr.getData( 0, name );
    AMP_ASSERT( *ptr2 == *ptr2 );
    ptr2->getData( data );
    AMP_ASSERT( data.length() == 1 );
    auto test = data( 0 ) == x;
    if ( std::is_arithmetic_v<TYPE> ) {
        AMP::Array<double> data2;
        ptr2->getData( data2 );
        test = test && data( 0 ) == x;
    }
    record( test, "HDF5Class: " + name, ut );
}
template<class TYPE>
void checkVector( const AMP::IO::HDF5data &ptr,
                  const std::string &name,
                  const std::vector<TYPE> &x,
                  AMP::UnitTest &ut )
{
    AMP::Array<TYPE> data;
    auto ptr2 = ptr.getData( 0, name );
    ptr2->getData( data );
    auto y = std::vector<TYPE>( data.begin(), data.end() );
    record( x == y, "HDF5Class: " + name, ut );
}
template<class TYPE, std::size_t N>
void checkArray( const AMP::IO::HDF5data &ptr,
                 const std::string &name,
                 const std::array<TYPE, N> &x,
                 AMP::UnitTest &ut )
{
    AMP::Array<TYPE> data;
    auto ptr2 = ptr.getData( 0, name );
    ptr2->getData( data );
    AMP_ASSERT( data.length() == N );
    std::array<TYPE, N> y;
    for ( size_t i = 0; i < N; i++ )
        y[i] = data( i );
    record( x == y, "HDF5Class: " + name, ut );
}
template<class TYPE>
void checkArray( const AMP::IO::HDF5data &ptr,
                 const std::string &name,
                 const AMP::Array<TYPE> &x,
                 AMP::UnitTest &ut )
{
    AMP::Array<TYPE> y;
    auto ptr2 = ptr.getData( 0, name );
    ptr2->getData( y );
    record( x == y, "HDF5Class: " + name, ut );
}


// Class to generate random values
class uniform_complex
{
public:
    template<class Generator>
    std::complex<double> operator()( Generator &g )
    {
        static std::uniform_real_distribution<double> dist( 0, 1 );
        return { dist( g ), dist( g ) };
    }
};
class random_char
{
public:
    template<class Generator>
    char operator()( Generator &g )
    {
        static std::uniform_int_distribution<int> dist( 32, 126 );
        return dist( g );
    }
};
template<class TYPE>
class random_integral
{
public:
    template<class Generator>
    TYPE operator()( Generator &g )
    {
        if constexpr ( std::is_signed_v<TYPE> ) {
            static std::uniform_int_distribution<long long> dist(
                std::numeric_limits<TYPE>::min(), std::numeric_limits<TYPE>::max() );
            return dist( g );
        } else {
            static std::uniform_int_distribution<unsigned long long> dist(
                std::numeric_limits<TYPE>::min(), std::numeric_limits<TYPE>::max() );
            return dist( g );
        }
    }
};
class random_string
{
public:
    template<class Generator>
    std::string operator()( Generator &g )
    {
        static std::uniform_int_distribution<int> length( 3, 10 );
        static std::uniform_int_distribution<int> dist( 32, 126 );
        std::string str;
        str.resize( length( g ) );
        for ( size_t i = 0; i < str.size(); i++ )
            str[i] = dist( g );
        return str;
    }
};
class random_compound
{
public:
    template<class Generator>
    compoundStruct operator()( Generator &g )
    {
        static std::uniform_int_distribution<int> dist_i( 3, 10 );
        static std::uniform_real_distribution<double> dist_d( 0, 1 );
        return compoundStruct( dist_i( g ), dist_d( g ), dist_d( g ) );
    }
};


// Class to hold/test a specific type
template<class TYPE>
class testTYPE
{
public:
    testTYPE()
    {
        name = AMP::getTypeID<TYPE>().name;
        vec.resize( 10 );
        Array.resize( 4, 3 );
        std::random_device rd;
        std::mt19937 gen( rd() );
        if constexpr ( std::is_same_v<TYPE, char> ) {
            random_char dist;
            fill( dist, gen );
        } else if constexpr ( std::is_integral_v<TYPE> ) {
            random_integral<TYPE> dist;
            fill( dist, gen );
        } else if constexpr ( std::is_floating_point_v<TYPE> ) {
            std::uniform_real_distribution<TYPE> dist( 0, 1 );
            fill( dist, gen );
        } else if constexpr ( std::is_same_v<TYPE, std::complex<double>> ) {
            uniform_complex dist;
            fill( dist, gen );
        } else if constexpr ( std::is_same_v<TYPE, std::string> ) {
            random_string dist;
            fill( dist, gen );
        } else if constexpr ( std::is_same_v<TYPE, compoundStruct> ) {
            random_compound dist;
            fill( dist, gen );
        } else {
            static_assert( !std::is_same_v<TYPE, TYPE>, "Not finished" );
        }
    }
    testTYPE( const testTYPE & ) = delete;
    template<class DIST, class GEN>
    void fill( DIST &dist, GEN &gen )
    {
        scalar = dist( gen );
        for ( size_t i = 0; i < 10; i++ )
            vec[i] = dist( gen );
        for ( size_t i = 0; i < array.size(); i++ )
            array[i] = dist( gen );
        for ( size_t i = 0; i < Array.length(); i++ )
            Array( i ) = dist( gen );
    }
    void write( hid_t fid )
    {
        AMP::IO::writeHDF5( fid, name, scalar );
        AMP::IO::writeHDF5( fid, "std::vector<" + name + ">", vec );
        AMP::IO::writeHDF5( fid, "std::array<" + name + ">", array );
        AMP::IO::writeHDF5( fid, "AMP::Array<" + name + ">", Array );
    }
    void check( hid_t fid, const AMP::IO::HDF5data *ptr, AMP::UnitTest &ut )
    {
        if constexpr ( std::is_same_v<TYPE, compoundStruct> ) {
            // Special case (skip loading the data)
        } else {
            // Check a direct read
            checkHDF5( fid, name, scalar, ut );
            checkHDF5( fid, "std::vector<" + name + ">", vec, ut );
            checkHDF5( fid, "std::array<" + name + ">", array, ut );
            checkHDF5( fid, "AMP::Array<" + name + ">", Array, ut );
            // Check reading through HDF5 class interface
            if ( ptr ) {
                checkScalar( *ptr, name, scalar, ut );
                checkVector( *ptr, "std::vector<" + name + ">", vec, ut );
                checkArray( *ptr, "std::array<" + name + ">", array, ut );
                checkArray( *ptr, "AMP::Array<" + name + ">", Array, ut );
            }
        }
    }

private:
    std::string name;
    TYPE scalar;
    std::vector<TYPE> vec;
    std::array<TYPE, 3> array;
    AMP::Array<TYPE> Array;
};


// Structure to contain set of types to test
struct data_struct {
public: // Functions
    data_struct()                      = default;
    data_struct( const data_struct & ) = delete;
    void write( hid_t fid )
    {
        c.write( fid );
        u8.write( fid );
        u16.write( fid );
        u32.write( fid );
        u64.write( fid );
        i8.write( fid );
        i16.write( fid );
        i32.write( fid );
        i64.write( fid );
        f.write( fid );
        d.write( fid );
        cmplx.write( fid );
        str.write( fid );
        compound.write( fid );
        writeOpaque( fid, "opaque", 3.0 );
    }
    void check( hid_t fid, AMP::UnitTest &ut )
    {
        auto data = AMP::IO::readHDF5( fid, "/" );
        auto ptr  = data.get();
        c.check( fid, ptr, ut );
        u8.check( fid, ptr, ut );
        u16.check( fid, ptr, ut );
        u32.check( fid, ptr, ut );
        u64.check( fid, ptr, ut );
        i8.check( fid, ptr, ut );
        i16.check( fid, ptr, ut );
        i32.check( fid, ptr, ut );
        i64.check( fid, ptr, ut );
        f.check( fid, ptr, ut );
        d.check( fid, ptr, ut );
        cmplx.check( fid, ptr, ut );
        str.check( fid, ptr, ut );
        compound.check( fid, ptr, ut );
    }

public: // Data members
    testTYPE<char> c;
    testTYPE<uint8_t> u8;
    testTYPE<uint16_t> u16;
    testTYPE<uint32_t> u32;
    testTYPE<uint64_t> u64;
    testTYPE<int8_t> i8;
    testTYPE<int16_t> i16;
    testTYPE<int32_t> i32;
    testTYPE<int64_t> i64;
    testTYPE<float> f;
    testTYPE<double> d;
    testTYPE<std::complex<double>> cmplx;
    testTYPE<std::string> str;
    testTYPE<compoundStruct> compound;
};


// Test compression
void testCompression( [[maybe_unused]] AMP::UnitTest &ut )
{
    // Write data using different compression formats
    AMP::Array<int> zeros1( 500, 1000 );
    AMP::Array<double> zeros2( 500, 500 );
    AMP::Array<std::complex<double>> zeros3( 500, 200 );
    zeros1.fill( 0.0 );
    zeros2.fill( 0.0 );
    zeros3.fill( 0.0 );
    auto fid1 = AMP::IO::openHDF5( "test_HDF5.none.hdf5", "w", AMP::IO::Compression::None );
    auto fid2 = AMP::IO::openHDF5( "test_HDF5.gzip.hdf5", "w", AMP::IO::Compression::GZIP );
    auto fid3 = AMP::IO::openHDF5( "test_HDF5.szip.hdf5", "w", AMP::IO::Compression::SZIP );
    AMP::IO::writeHDF5( fid1, "zeros1", zeros1 );
    AMP::IO::writeHDF5( fid1, "zeros2", zeros2 );
    AMP::IO::writeHDF5( fid1, "zeros3", zeros3 );
    AMP::IO::writeHDF5( fid2, "zeros1", zeros1 );
    AMP::IO::writeHDF5( fid2, "zeros2", zeros2 );
    AMP::IO::writeHDF5( fid2, "zeros3", zeros3 );
    AMP::IO::writeHDF5( fid3, "zeros1", zeros1 );
    AMP::IO::writeHDF5( fid3, "zeros2", zeros2 );
    AMP::IO::writeHDF5( fid3, "zeros3", zeros3 );
    AMP::IO::closeHDF5( fid1 );
    AMP::IO::closeHDF5( fid2 );
    AMP::IO::closeHDF5( fid3 );
}


// Test writing a large array with compression
void testLarge( [[maybe_unused]] AMP::UnitTest &ut )
{
    printf( "Allocating large array\n" );
    AMP::Array<int> data( 40000, 40000 );
    data.fill( 0 );
    // std::default_random_engine gen;
    // std::uniform_int_distribution<int> dist( 0, 10000 );
    // for ( size_t i=0; i<data.length(); i++)
    //    data(i) = dist(gen);
    printf( "Writing large array\n" );
    auto fid = AMP::IO::openHDF5( "test_HDF5.large.hdf5", "w", AMP::IO::Compression::GZIP );
    AMP::IO::writeHDF5( fid, "data", data );
    AMP::IO::closeHDF5( fid, true );
}


// Main
int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    { // Limit scope

        // Test catching an hdf5 error
        try {
            AMP::IO::openHDF5( "", "w" );
            ut.failure( "Failed to catch HDF5 error" );
        } catch ( ... ) {
            ut.passes( "Caught HDF5 error" );
        }

        // Write variables to HDF5
        data_struct data;
        auto fid = AMP::IO::openHDF5( "test_HDF5.hdf5", "w" );
        data.write( fid );
        AMP::IO::closeHDF5( fid, true );

        // Read the variables from HDF5
        fid = AMP::IO::openHDF5( "test_HDF5.hdf5", "r" );
        data.check( fid, ut );
        AMP::IO::closeHDF5( fid, true );

        // Test compression
        testCompression( ut );
        // testLarge( ut );
    }

    int N_errors = ut.NumFailGlobal();
    ut.report();
    ut.reset();
    AMP::AMPManager::shutdown();
    return N_errors;
}
