// This file tests the .bmp interfaces

#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/graphics/BMP_Writer.h"

#include <cmath>


// Main
int main( int, char *[] )
{
    // Create dummy data
    const int Nx = 100;
    const int Ny = 200;
    AMP::Array<double> data( Nx, Ny );
    data.fill( 0 );
    for ( int i = 0; i < Nx; i++ ) {
        double x = static_cast<double>( i ) / static_cast<double>( Nx - 1 );
        for ( int j = 0; j < Ny; j++ ) {
            double y     = static_cast<double>( j ) / static_cast<double>( Ny - 1 );
            data( i, j ) = 3.5 * exp( -x ) * exp( -5.0 * ( y - 0.5 ) * ( y - 0.5 ) );
        }
    }
    AMP::Graphics::writeBMP( "testData.bmp", data );
    data.clear();

    // Return
    return 0;
}
