#include "AMP/IO/PIO.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/subchannel/FlowFrapconOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


#define ITFAILS ut.failure( __LINE__ );
#define UNIT_TEST( a ) \
    if ( !( a ) )      \
        ut.failure( __LINE__ );


static void adjust( const AMP::LinearAlgebra::Vector::shared_ptr vec,
                    AMP::LinearAlgebra::Vector::shared_ptr work )
{
    work->setToScalar( 301. );
    AMP::LinearAlgebra::Vector &x = *vec;
    AMP::LinearAlgebra::Vector &y = *work;
    vec->add( x, y );
    vec->makeConsistent();
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

static void flowTest( AMP::UnitTest *ut )
{
    // Input and output file names
    std::string exeName( "testFlowOperator" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    AMP::logAllNodes( log_file );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto manager = AMP::Mesh::MeshFactory::create( params );
    auto mesh    = manager->Subset( "cube_3d" );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    // CREATE THE FLOW OPERATOR
    AMP_INSIST( input_db->keyExists( "FlowFrapconOperator" ),
                "Key ''FlowFrapconOperator'' is missing!" );
    auto flowOperator = std::dynamic_pointer_cast<AMP::Operator::FlowFrapconOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "FlowFrapconOperator", input_db ) );

    auto inputVariable  = flowOperator->getInputVariable();
    auto outputVariable = flowOperator->getOutputVariable();

    auto solVec  = AMP::LinearAlgebra::createSimpleVector<double>( 10, inputVariable );
    auto cladVec = AMP::LinearAlgebra::createSimpleVector<double>( 10, inputVariable );

    auto rhsVec  = AMP::LinearAlgebra::createSimpleVector<double>( 10, outputVariable );
    auto resVec  = AMP::LinearAlgebra::createSimpleVector<double>( 10, outputVariable );
    auto workVec = AMP::LinearAlgebra::createSimpleVector<double>( 10, inputVariable );

    cladVec->setToScalar( 300 );
    flowOperator->setVector( cladVec );

    ut->passes( exeName + ": create" );

    auto testOperator = std::dynamic_pointer_cast<AMP::Operator::Operator>( flowOperator );
    //  flowOperator->apply(nullVec, TemperatureInKelvinVec, flowOutputVec, 1., 0.);

    // test apply
    std::string msgPrefix = exeName + ": apply";
    applyTest( ut, msgPrefix, testOperator, rhsVec, solVec, resVec, workVec );

    ut->passes( msgPrefix );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    flowTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
