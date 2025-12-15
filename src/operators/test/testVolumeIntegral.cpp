#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/libmesh/SourceNonlinearElement.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <exception>
#include <memory>
#include <string>


static void adjust( const AMP::LinearAlgebra::Vector::shared_ptr vec,
                    AMP::LinearAlgebra::Vector::shared_ptr work )
{
    work->setToScalar( 301. );
    AMP::LinearAlgebra::Vector &x = *vec;
    AMP::LinearAlgebra::Vector &y = *work;
    vec->add( x, y );
    vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

static void applyTest( AMP::UnitTest *ut,
                       const std::string &msgPrefix,
                       std::shared_ptr<AMP::Operator::Operator> testOperator,
                       AMP::LinearAlgebra::Vector::shared_ptr rhsVec,
                       AMP::LinearAlgebra::Vector::shared_ptr solVec,
                       AMP::LinearAlgebra::Vector::shared_ptr resVec,
                       AMP::LinearAlgebra::Vector::shared_ptr workVec )
{
    // first test for apply - random values in all three input vectors
    try {
        for ( int j = 0; j < 3; j++ ) {
            solVec->setRandomValues();
            rhsVec->setRandomValues();
            resVec->setRandomValues();
            adjust( solVec, workVec );
            testOperator->residual( rhsVec, solVec, resVec );
        } // end for j
        ut->passes( msgPrefix + " : apply with random f, u, r, a=1, b=-1.0" );
    } catch ( const std::exception & ) {
        ut->failure( msgPrefix + " : apply with random f, u, r, a=1, b=-1.0" );
    }

    // second test for apply - f NULL, u, r, random values
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr fVec;
            solVec->setRandomValues();
            resVec->setRandomValues();
            adjust( solVec, workVec );
            testOperator->residual( fVec, solVec, resVec );
        } // end for j
        ut->passes( msgPrefix + " : apply with f NULL, random u, r, a=1, b=-1.0" );
    } catch ( const std::exception & ) {
        ut->failure( msgPrefix + " : apply with f NULL, random u, r, a=1, b=-1.0" );
    }

    // R.S.: u is allowed to be NULL for some operators. For example, operators
    // with an in-place apply. However, this test is not meant to be used with those operators.
    // third test for apply - u NULL, f, r, random values
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr uVec;
            rhsVec->setRandomValues();
            resVec->setRandomValues();
            testOperator->residual( rhsVec, uVec, resVec );
        } // end for j
        ut->failure( msgPrefix +
                     " : apply with u NULL, random values in the vectors f,r, a=1, b=-1.0" );
    } catch ( const std::exception & ) {
        ut->passes( msgPrefix +
                    " : apply with u NULL, random values in the vectors f,r, a=1, b=-1.0" );
    }

    // fourth test for apply - r NULL, f, u, random values
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr rVec;
            solVec->setRandomValues();
            rhsVec->setRandomValues();
            adjust( solVec, workVec );
            testOperator->residual( rhsVec, solVec, rVec );
        } // end for j
        ut->failure( msgPrefix +
                     " : apply with r NULL, random values in the vectors f,u, a=1, b=-1.0" );
    } catch ( const std::exception & ) {
        ut->passes( msgPrefix +
                    " : apply with r NULL, random values in the vectors f,u, a=1, b=-1.0" );
    }

    // fifth test for apply - f NULL, u NULL, r, random values
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr fVec;
            AMP::LinearAlgebra::Vector::shared_ptr uVec;
            resVec->setRandomValues();
            testOperator->residual( fVec, uVec, resVec );
        } // end for j
        ut->failure( msgPrefix +
                     " : apply with f NULL, u NULL random values in the vector r, a=1, b=-1.0" );
    } catch ( const std::exception & ) {
        ut->passes( msgPrefix +
                    " : apply with f NULL, u NULL random values in the vector r, a=1, b=-1.0" );
    }

    // sixth test for apply - u NULL, r NULL, f, random values
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr uVec;
            AMP::LinearAlgebra::Vector::shared_ptr rVec;
            rhsVec->setRandomValues();
            testOperator->residual( rhsVec, uVec, rVec );
        } // end for j
        ut->failure( msgPrefix +
                     " : apply with u NULL, r NULL, random values in the vector f, a=1, b=-1.0" );
    } catch ( const std::exception & ) {
        ut->passes( msgPrefix +
                    " : apply with u NULL, r NULL, random values in the vector f, a=1, b=-1.0" );
    }

    // seventh test for apply - r NULL, f NULL, u random values
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr rVec;
            AMP::LinearAlgebra::Vector::shared_ptr fVec;
            solVec->setRandomValues();
            adjust( solVec, workVec );
            testOperator->residual( fVec, solVec, rVec );
        } // end for j
        ut->failure( msgPrefix +
                     " : apply with f, r NULL, random values in the vector u, a=1, b=-1.0" );
    } catch ( const std::exception & ) {
        ut->passes( msgPrefix +
                    " : apply with f, r NULL, random values in the vector u, a=1, b=-1.0" );
    }

    // eighth test for apply - r NULL, f NULL, u NULL
    try {
        for ( int j = 0; j < 3; j++ ) {
            AMP::LinearAlgebra::Vector::shared_ptr rVec;
            AMP::LinearAlgebra::Vector::shared_ptr fVec;
            AMP::LinearAlgebra::Vector::shared_ptr uVec;
            testOperator->residual( fVec, uVec, rVec );
        } // end for j
        ut->failure( msgPrefix + " : apply with f, u, r NULL, a=1, b=-1.0" );
    } catch ( const std::exception & ) {
        ut->passes( msgPrefix + " : apply with f, u, r NULL, a=1, b=-1.0" );
    }
}


static void sourceTest( AMP::UnitTest *ut, const std::string &exeName )
{
    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logAllNodes( log_file );

    std::cout << "testing with input file " << input_file << std::endl;
    std::cout.flush();


    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );

    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    //   CREATE THE VOLUME INTEGRAL OPERATOR -----------
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    auto sourceDatabase = input_db->getDatabase( "VolumeIntegralOperator" );
    AMP::pout << "before sourceOp" << std::endl;
    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "VolumeIntegralOperator", input_db ) );
    AMP::pout << "after sourceOp" << std::endl;
    auto inputVariable  = sourceOperator->getInputVariable();
    auto outputVariable = sourceOperator->getOutputVariable();

    // Create a DOF manager for a gauss point vector
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, 0, 8, true );
    auto nodalDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto solVec  = AMP::LinearAlgebra::createVector( gaussPointDofMap, inputVariable, true );
    auto rhsVec  = AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable, true );
    auto resVec  = AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable, true );
    auto workVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, inputVariable, true );

    ut->passes( exeName + " : create" );

    auto msgPrefix = exeName + ": VolumeIntegralOperator reset";
    // test reset
    try {
        auto volumeIntegralParameters =
            std::make_shared<AMP::Operator::VolumeIntegralOperatorParameters>( sourceDatabase );
        sourceOperator->reset( volumeIntegralParameters );
        ut->passes( msgPrefix + " : pass " );
    } catch ( const std::exception & ) {
        ut->failure( msgPrefix + " : fail " );
    }

    // test apply
    msgPrefix         = exeName + ": VolumeIntegralOperator apply";
    auto testOperator = std::dynamic_pointer_cast<AMP::Operator::Operator>( sourceOperator );
    applyTest( ut, msgPrefix, testOperator, rhsVec, solVec, resVec, workVec );

    ut->passes( msgPrefix );
}


// Input and output file names
int testVolumeIntegral( int argc, char *argv[] )
{

    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    PROFILE_ENABLE( 5 );
#ifdef AMP_USE_TIMER
    global_profiler.ignoreTimerErrors( true );
#endif
    PROFILE( "main" );

    AMP::UnitTest ut;

    const int NUMFILES          = 3;
    std::string files[NUMFILES] = { "testVolumeIntegral-1",
                                    "testVolumeIntegral-2",
                                    "testVolumeIntegral-3" };

    for ( auto &file : files )
        sourceTest( &ut, file );

    ut.report();
    PROFILE_SAVE( "testVolumeIntegral-1" );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
