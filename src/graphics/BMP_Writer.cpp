#include "AMP/graphics/BMP_Writer.h"
#include "AMP/utils/Utilities.h"

#include <cmath>
#include <cstdio>


namespace AMP::Graphics {


struct BITMAPFILEHEADER {
    const char bfType[2]      = { 'B', 'M' };
    char bfSize[4]            = { 0 };
    const char bfReserved1[2] = { 0, 0 };
    const char bfReserved2[2] = { 0, 0 };
    char bfOffBits[4]         = { 0 };
};
struct BITMAPINFOHEADER {
    char biSize[4];
    char biWidth[4];
    char biHeight[4];
    char biPlanes[2];
    char biBitCount[2];
    char biCompression[4];
    char biSizeImage[4];
    char biXPelsPerMeter[4];
    char biYPelsPerMeter[4];
    char biClrUsed[4];
    char biClrImportant[4];
};
static_assert( sizeof( BITMAPFILEHEADER ) == 14, "Unexpected size" );
static_assert( sizeof( BITMAPINFOHEADER ) == 40, "Unexpected size" );


// Colormap (jet-rgb)
const uint8_t jet_rgb[768] = {
    0,   0,   131, 0,   0,   135, 0,   0,   139, 0,   0,   143, 0,   0,   147, 0,   0,   151, 0,
    0,   155, 0,   0,   159, 0,   0,   163, 0,   0,   167, 0,   0,   171, 0,   0,   175, 0,   0,
    179, 0,   0,   183, 0,   0,   187, 0,   0,   191, 0,   0,   195, 0,   0,   199, 0,   0,   203,
    0,   0,   207, 0,   0,   211, 0,   0,   215, 0,   0,   219, 0,   0,   223, 0,   0,   227, 0,
    0,   231, 0,   0,   235, 0,   0,   239, 0,   0,   243, 0,   0,   247, 0,   0,   251, 0,   0,
    255, 0,   4,   255, 0,   8,   255, 0,   12,  255, 0,   16,  255, 0,   20,  255, 0,   24,  255,
    0,   28,  255, 0,   32,  255, 0,   36,  255, 0,   40,  255, 0,   44,  255, 0,   48,  255, 0,
    52,  255, 0,   56,  255, 0,   60,  255, 0,   64,  255, 0,   68,  255, 0,   72,  255, 0,   76,
    255, 0,   80,  255, 0,   84,  255, 0,   88,  255, 0,   92,  255, 0,   96,  255, 0,   100, 255,
    0,   104, 255, 0,   108, 255, 0,   112, 255, 0,   116, 255, 0,   120, 255, 0,   124, 255, 0,
    127, 255, 0,   131, 255, 0,   135, 255, 0,   139, 255, 0,   143, 255, 0,   147, 255, 0,   151,
    255, 0,   155, 255, 0,   159, 255, 0,   163, 255, 0,   167, 255, 0,   171, 255, 0,   175, 255,
    0,   179, 255, 0,   183, 255, 0,   187, 255, 0,   191, 255, 0,   195, 255, 0,   199, 255, 0,
    203, 255, 0,   207, 255, 0,   211, 255, 0,   215, 255, 0,   219, 255, 0,   223, 255, 0,   227,
    255, 0,   231, 255, 0,   235, 255, 0,   239, 255, 0,   243, 255, 0,   247, 255, 0,   251, 255,
    0,   255, 255, 0,   255, 247, 0,   255, 239, 0,   255, 231, 0,   255, 223, 0,   255, 215, 0,
    255, 207, 0,   255, 199, 0,   255, 191, 0,   255, 183, 0,   255, 175, 0,   255, 167, 0,   255,
    159, 0,   255, 151, 0,   255, 143, 0,   255, 135, 0,   255, 128, 0,   255, 120, 0,   255, 112,
    0,   255, 104, 0,   255, 96,  0,   255, 88,  0,   255, 80,  0,   255, 72,  0,   255, 64,  0,
    255, 56,  0,   255, 48,  0,   255, 40,  0,   255, 32,  0,   255, 24,  0,   255, 16,  0,   255,
    8,   0,   255, 0,   8,   255, 0,   16,  255, 0,   24,  255, 0,   32,  255, 0,   40,  255, 0,
    48,  255, 0,   56,  255, 0,   64,  255, 0,   72,  255, 0,   80,  255, 0,   88,  255, 0,   96,
    255, 0,   104, 255, 0,   112, 255, 0,   120, 255, 0,   128, 255, 0,   135, 255, 0,   143, 255,
    0,   151, 255, 0,   159, 255, 0,   167, 255, 0,   175, 255, 0,   183, 255, 0,   191, 255, 0,
    199, 255, 0,   207, 255, 0,   215, 255, 0,   223, 255, 0,   231, 255, 0,   239, 255, 0,   247,
    255, 0,   255, 255, 0,   255, 251, 0,   255, 247, 0,   255, 243, 0,   255, 239, 0,   255, 235,
    0,   255, 231, 0,   255, 227, 0,   255, 223, 0,   255, 219, 0,   255, 215, 0,   255, 211, 0,
    255, 207, 0,   255, 203, 0,   255, 199, 0,   255, 195, 0,   255, 191, 0,   255, 187, 0,   255,
    183, 0,   255, 179, 0,   255, 175, 0,   255, 171, 0,   255, 167, 0,   255, 163, 0,   255, 159,
    0,   255, 155, 0,   255, 151, 0,   255, 147, 0,   255, 143, 0,   255, 139, 0,   255, 135, 0,
    255, 131, 0,   255, 128, 0,   255, 124, 0,   255, 120, 0,   255, 116, 0,   255, 112, 0,   255,
    108, 0,   255, 104, 0,   255, 100, 0,   255, 96,  0,   255, 92,  0,   255, 88,  0,   255, 84,
    0,   255, 80,  0,   255, 76,  0,   255, 72,  0,   255, 68,  0,   255, 64,  0,   255, 60,  0,
    255, 56,  0,   255, 52,  0,   255, 48,  0,   255, 44,  0,   255, 40,  0,   255, 36,  0,   255,
    32,  0,   255, 28,  0,   255, 24,  0,   255, 20,  0,   255, 16,  0,   255, 12,  0,   255, 8,
    0,   255, 4,   0,   255, 0,   0,   251, 0,   0,   247, 0,   0,   243, 0,   0,   239, 0,   0,
    235, 0,   0,   231, 0,   0,   227, 0,   0,   223, 0,   0,   219, 0,   0,   215, 0,   0,   211,
    0,   0,   207, 0,   0,   203, 0,   0,   199, 0,   0,   195, 0,   0,   192, 0,   0,   188, 0,
    0,   184, 0,   0,   180, 0,   0,   176, 0,   0,   172, 0,   0,   168, 0,   0,   164, 0,   0,
    160, 0,   0,   156, 0,   0,   152, 0,   0,   148, 0,   0,   144, 0,   0,   140, 0,   0,   136,
    0,   0,   132, 0,   0,   128, 0,   0
};


static void set( char ( &data )[4], int value ) { memcpy( data, &value, 4 ); }
static void set( char ( &data )[2], short value ) { memcpy( data, &value, 2 ); }


void writeBMP( const std::string &file, const AMP::Array<uint8_t> &data )
{
    // Copy the data to a buffer, reordering and packing as needed
    std::vector<uint8_t> data2;
    data2.reserve( data.length() + data.size( 0 ) );
    for ( int i = data.size( 0 ) - 1; i >= 0; i-- ) {
        for ( size_t j = 0; j < data.size( 1 ); j++ )
            data2.push_back( data( i, j ) );
        while ( data2.size() % 4 != 0 )
            data2.push_back( 0 );
    }
    // Create the colormap
    uint8_t colormap[1024];
    for ( int i = 0; i < 256; i++ ) {
        colormap[4 * i + 0] = jet_rgb[3 * i + 2];
        colormap[4 * i + 1] = jet_rgb[3 * i + 1];
        colormap[4 * i + 2] = jet_rgb[3 * i + 0];
        colormap[4 * i + 3] = 0;
    }
    // Create the headers
    size_t offset = sizeof( BITMAPFILEHEADER ) + sizeof( BITMAPINFOHEADER ) + sizeof( colormap );
    BITMAPFILEHEADER header;
    BITMAPINFOHEADER info;
    // NULL_USE( header.bfType );
    // NULL_USE( header.bfReserved1 );
    // NULL_USE( header.bfReserved2 );
    set( header.bfSize, offset + data2.size() );
    set( header.bfOffBits, offset );
    set( info.biSize, sizeof( BITMAPINFOHEADER ) );
    set( info.biWidth, data.size( 1 ) );
    set( info.biHeight, data.size( 0 ) );
    set( info.biPlanes, 1 );
    set( info.biBitCount, 8 );
    set( info.biCompression, 0 );
    set( info.biSizeImage, 0 );
    set( info.biXPelsPerMeter, 0 );
    set( info.biYPelsPerMeter, 0 );
    set( info.biClrUsed, 0 );
    set( info.biClrImportant, 0 );

    // Write the file
    auto fid = fopen( file.c_str(), "wb" );
    AMP_ASSERT( fid != nullptr );
    fwrite( &header, sizeof( BITMAPFILEHEADER ), 1, fid );
    fwrite( &info, sizeof( BITMAPINFOHEADER ), 1, fid );
    fwrite( colormap, 1, sizeof( colormap ), fid );
    fwrite( data2.data(), 1, data2.size(), fid );
    fclose( fid );
}


void writeBMP( const std::string &file, const AMP::Array<double> &data )
{
    // Scale the data
    AMP::Array<uint8_t> data2( data.size() );
    double x1 = data.min();
    double x2 = data.max();
    data2.fill( 0 );
    if ( x2 > x1 ) {
        for ( size_t i = 0; i < data.length(); i++ ) {
            double v   = ( data( i ) - x1 ) / ( x2 - x1 );
            data2( i ) = round( 255 * v );
        }
    }
    // Write the data
    writeBMP( file, data2 );
}


} // namespace AMP::Graphics
