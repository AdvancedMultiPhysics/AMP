#include "test_Matrix.h"

#include "AMP/AMP_TPLs.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/vectors/VectorBuilder.h"

#include <chrono>


#define to_ms( x ) std::chrono::duration_cast<std::chrono::milliseconds>( x ).count()


namespace AMP::unit_test {
class AMPCubeGenerator5 : public AMPCubeGenerator
{
public:
    AMPCubeGenerator5() : AMPCubeGenerator( 5 ) {}
};
#ifdef AMP_USE_LIBMESH
class ExodusReaderGenerator2 : public ExodusReaderGenerator
{
public:
    ExodusReaderGenerator2() : ExodusReaderGenerator( "pellet_1x.e" ) {}
};
#endif
} // namespace AMP::unit_test


using namespace AMP::LinearAlgebra;


int main( int argc, char **argv )
{

    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;
    PROFILE_ENABLE();


    // Get the types of matrices to test
    std::vector<std::string> types;
    //    types.emplace_back( "CSRMatrix" );
#ifdef AMP_USE_TRILINOS_TPETRA
    types.emplace_back( "ManagedTpetraMatrix" );
#endif
#if 0
    #ifdef AMP_USE_TRILINOS_EPETRA
    types.emplace_back( "ManagedEpetraMatrix" );
    #endif
    #ifdef AMP_USE_PETSC
    types.emplace_back( "NativePetscMatrix" );
    #endif
    if ( AMP::AMP_MPI( AMP_COMM_WORLD ).getSize() == 1 )
        types.emplace_back( "DenseSerialMatrix" );
#endif

    // Test some basic properties
    AMP::pout << "Running basic tests" << std::endl << std::endl;
    testBasics( ut, "auto" );
    for ( auto &type : types )
        testBasics( ut, type );


    // Run general tests for each matrix type
    for ( auto type : types ) {
        AMP::pout << "Running tests for " << type;
        auto t1    = std::chrono::high_resolution_clock::now();
        using DOF1 = DOFMatrixTestFactory<1, 1, AMPCubeGenerator5>;
        using DOF3 = DOFMatrixTestFactory<3, 3, AMPCubeGenerator5>;
        test_matrix_loop( ut, std::make_shared<DOF1>( type ) );
        test_matrix_loop( ut, std::make_shared<DOF3>( type ) );
#if defined( AMP_USE_LIBMESH ) && !defined( _GLIBCXX_DEBUG )
        if ( AMP::IO::exists( "pellet_1x.e" ) ) {
            using libmeshFactory = DOFMatrixTestFactory<3, 3, ExodusReaderGenerator2>;
            test_matrix_loop( ut, std::make_shared<libmeshFactory>( type ) );
        }
#endif
        auto t2 = std::chrono::high_resolution_clock::now();
        AMP::pout << " (" << 1e-3 * to_ms( t2 - t1 ) << " s)" << std::endl;
    }
    AMP::pout << std::endl;


    // Test using the copy factories between types
    for ( auto type1 : types ) {
        for ( auto type2 : types ) {
            if ( ( type1 == "DenseSerialMatrix" ) != ( type2 == "DenseSerialMatrix" ) )
                continue;
            AMP::pout << "Running copy tests for " << type1 << " --> " << type2;
            auto t1       = std::chrono::high_resolution_clock::now();
            using DOF     = DOFMatrixTestFactory<3, 3, AMPCubeGenerator5>;
            auto factory1 = std::make_shared<DOF>( type1 );
            auto factory2 = std::make_shared<DOF>( type2 );
            test_matrix_loop( ut, factory1, factory2 );
            auto t2 = std::chrono::high_resolution_clock::now();
            AMP::pout << " (" << 1e-3 * to_ms( t2 - t1 ) << " s)" << std::endl;
        }
    }


    ut.report();
    PROFILE_SAVE( "test_Matrix" );
    int num_failed = ut.NumFailGlobal();
    ut.reset();
    types = {};
    AMP::AMPManager::shutdown();
    return num_failed;
}
