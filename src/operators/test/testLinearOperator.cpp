#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "applyTests.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>


static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testLinearOperator-1" );
    std::string outerInput_file = "input_" + exeName;
    std::string log_file        = "output_" + exeName;
    std::string msgPrefix;

    AMP::logOnlyNodeZero( log_file );

    auto outerInput_db = AMP::Database::parseInputFile( outerInput_file );
    outerInput_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    AMP_INSIST( outerInput_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto database = outerInput_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    AMP_INSIST( outerInput_db->keyExists( "number_of_tests" ), "key missing!" );
    int numTests = outerInput_db->getScalar<int>( "number_of_tests" );

    for ( int i = 0; i < numTests; i++ ) {
        char key[256];
        snprintf( key, sizeof key, "test_%d", i );

        AMP_INSIST( outerInput_db->keyExists( key ), "key missing!" );
        auto innerInput_file = outerInput_db->getString( key );

        auto innerInput_db = AMP::Database::parseInputFile( innerInput_file );
        innerInput_db->print( AMP::plog );

        AMP_INSIST( innerInput_db->keyExists( "testOperator" ), "key missing!" );

        auto testOperator =
            AMP::Operator::OperatorBuilder::createOperator( mesh, "testOperator", innerInput_db );

        msgPrefix = exeName + " : " + innerInput_file;

        if ( testOperator ) {
            ut->passes( msgPrefix + " : create" );
        } else {
            ut->failure( msgPrefix + " : create" );
        }

        auto myLinOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( testOperator );

        AMP_INSIST( myLinOp != nullptr, "Is not a linear operator!" );

        AMP_INSIST( innerInput_db->keyExists( "dofsPerNode" ), "key missing!" );
        int dofsPerNode = innerInput_db->getScalar<int>( "dofsPerNode" );

        auto myInpVar = myLinOp->getInputVariable();
        auto myOutVar = myLinOp->getOutputVariable();
        auto NodalDOF = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, dofsPerNode, true );

        {
            auto solVec = AMP::LinearAlgebra::createVector( NodalDOF, myInpVar, true );
            auto rhsVec = AMP::LinearAlgebra::createVector( NodalDOF, myOutVar, true );
            auto resVec = AMP::LinearAlgebra::createVector( NodalDOF, myOutVar, true );
            // test apply with single variable vectors
            applyTests( ut, msgPrefix, testOperator, rhsVec, solVec, resVec );
        }

        // now run apply tests with multi-vectors
        auto postfix   = std::to_string( i );
        auto auxInpVar = std::make_shared<AMP::LinearAlgebra::Variable>(
            "testLinearOperator-1-auxInpVar" + postfix );
        auto auxOutVar = std::make_shared<AMP::LinearAlgebra::Variable>(
            "testLinearOperator-1-auxOutVar" + postfix );

        auto myMultiInpVar =
            std::make_shared<AMP::LinearAlgebra::MultiVariable>( "MultiInputVariable" );
        myMultiInpVar->add( myInpVar );
        myMultiInpVar->add( auxInpVar );

        auto myMultiOutVar =
            std::make_shared<AMP::LinearAlgebra::MultiVariable>( "MultiOutputVariable" );
        myMultiOutVar->add( myOutVar );
        myMultiOutVar->add( auxOutVar );

        {
            auto solVec = AMP::LinearAlgebra::createVector( NodalDOF, myMultiInpVar, true );
            auto rhsVec = AMP::LinearAlgebra::createVector( NodalDOF, myMultiOutVar, true );
            auto resVec = AMP::LinearAlgebra::createVector( NodalDOF, myMultiOutVar, true );

            // test apply with single multivariable vectors
            applyTests( ut, msgPrefix, testOperator, rhsVec, solVec, resVec );
        }

        // test getJacobianParameters
        msgPrefix = exeName + " : " + innerInput_file;
        std::shared_ptr<AMP::LinearAlgebra::Vector> nullGuess;
        auto jacobianParameters = testOperator->getParameters( "Jacobian", nullGuess );

        if ( jacobianParameters.get() == nullptr ) {
            ut->passes( msgPrefix + "getJacobianParameters (should return NULL for now)" );
        } else {
            ut->failure( msgPrefix + "getJacobianParameters (should return NULL for now)" );
        }

    } // end for i

    ut->passes( "testLinearOperator-1" );
}


int testLinearOperator( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
