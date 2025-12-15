#include "AMP/IO/HDF.hpp"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/utils/Array.h"


using MeshElementIndex = AMP::Mesh::BoxMesh::MeshElementIndex;


/********************************************************
 * Read/Write BoxMesh::MeshElementIndex                  *
 ********************************************************/
#ifdef AMP_USE_HDF5
static inline std::array<int, 4> convert( const MeshElementIndex &x )
{
    int y1 = ( ( (int) x.type() ) << 16 ) | ( ( (int) x.side() ) );
    return { y1, x.index()[0], x.index()[1], x.index()[2] };
}
static inline MeshElementIndex convert( const std::array<int, 4> &x )
{
    auto type    = static_cast<AMP::Mesh::GeomType>( x[0] >> 16 );
    uint8_t side = x[0] & 0xFFFF;
    return MeshElementIndex( type, side, x[1], x[2], x[3] );
}
static_assert( sizeof( MeshElementIndex ) == 16 );
template<>
hid_t AMP::IO::getHDF5datatype<MeshElementIndex>()
{
    AMP_ERROR( "Not finished" );
}
template<>
void AMP::IO::writeHDF5Array<MeshElementIndex>( hid_t fid,
                                                const std::string &name,
                                                const AMP::Array<MeshElementIndex> &data )
{
    AMP::Array<int> data2( AMP::cat( AMP::ArraySize( 4 ), data.size() ) );
    for ( size_t i = 0; i < data.length(); i++ ) {
        auto y        = convert( data( i ) );
        data2( 0, i ) = y[0];
        data2( 1, i ) = y[1];
        data2( 2, i ) = y[2];
        data2( 3, i ) = y[3];
    }
    writeHDF5Array( fid, name, data2 );
}
template<>
void AMP::IO::readHDF5Array<MeshElementIndex>( hid_t fid,
                                               const std::string &name,
                                               AMP::Array<MeshElementIndex> &data )
{
    AMP::Array<int> data2;
    readHDF5Array<int>( fid, name, data2 );
    data.resize( pop( data2.size() ) );
    for ( size_t i = 0; i < data.length(); i++ ) {
        data( i ) = convert( { data2( 0, i ), data2( 1, i ), data2( 2, i ), data2( 3, i ) } );
    }
}
template<>
void AMP::IO::writeHDF5Scalar<MeshElementIndex>( hid_t fid,
                                                 const std::string &name,
                                                 const MeshElementIndex &data )
{
    auto data2 = convert( data );
    IO::writeHDF5( fid, name, data2 );
}
template<>
void AMP::IO::readHDF5Scalar<MeshElementIndex>( hid_t fid,
                                                const std::string &name,
                                                MeshElementIndex &data )
{
    std::array<int, 4> data2 = { 0 };
    IO::readHDF5( fid, name, data2 );
    data = convert( data2 );
}
#endif
INSTANTIATE_HDF5( MeshElementIndex );


/********************************************************
 * HDF5 operators                                        *
 ********************************************************/
#include "AMP/utils/Array.hpp"
instantiateArrayConstructors( AMP::Mesh::BoxMesh::MeshElementIndex );
