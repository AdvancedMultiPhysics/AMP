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
#ifdef AMP_USE_LIBMESH
        files.emplace_back( "input_testRemoteDOFs-libmesh-1" );
        files.emplace_back( "input_testRemoteDOFs-libmesh-2" );
#endif
    }

    for ( auto &file : files ) {
        // do not use verbose output, way too many failing DOFs
        remoteDOFTest( &ut, false, file );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
