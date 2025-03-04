// This file tests the HDF5 interfaces
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

#include "AMP/IO/FileSystem.h"
#include "AMP/IO/HDF5.h"
#include "AMP/IO/HDF5_Class.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"


// Main
int main( int argc, char *argv[] )
{
    AMP::AMPManagerProperties properties;
    properties.print_memory = 0;
    AMP::AMPManager::startup( argc, argv, properties );

    if ( argc == 1 ) {
        std::cerr << "view_HDF5 filename" << std::endl;
        return -1;
    }

    // Loop through the input files loading and printing the variables
    for ( int i = 1; i < argc; i++ ) {
        std::cout << argv[i] << ":\n";
        if ( !AMP::IO::exists( argv[i] ) ) {
            std::cerr << "File does not exist\n";
            return -1;
        }
        auto fid  = AMP::IO::openHDF5( argv[i], "r" );
        auto data = AMP::IO::readHDF5( fid, "/" );
        AMP::IO::closeHDF5( fid );
        data->print( 2, "  " );
        std::cout << std::endl;
    }

    AMP::AMPManager::shutdown();
    return 0;
}
