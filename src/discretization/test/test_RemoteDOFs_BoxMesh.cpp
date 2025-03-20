#include "AMP/discretization/testHelpers/test_RemoteDOFs.h"

int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::string> files;
    PROFILE_ENABLE();

    if ( argc > 1 ) {
        files.emplace_back( argv[1] );
    } else {
        files.emplace_back( "input_testRemoteDOFs-boxmesh-1" );
        files.emplace_back( "input_testRemoteDOFs-boxmesh-2" );
    }

    for ( auto &file : files ) {
        remoteDOFTest( &ut, true, file );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
