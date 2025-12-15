#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    AMP_INSIST( input_db->keyExists( "testNonlinearMechanicsOperator" ), "key missing!" );

    auto testNonlinOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "testNonlinearMechanicsOperator", input_db ) );
    auto elementPhysicsModel = testNonlinOperator->getMaterialModel();

    AMP_INSIST( input_db->keyExists( "testLinearMechanicsOperator" ), "key missing!" );

    auto testLinOperator = std::make_shared<AMP::Operator::MechanicsLinearFEOperator>(
        testNonlinOperator->getParameters( "Jacobian", nullptr ) );

    ut->passes( exeName + " : create" );

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    auto var    = testNonlinOperator->getOutputVariable();
    auto solVec = AMP::LinearAlgebra::createVector( dofMap, var, true );
    auto rhsVec = solVec->clone();
    auto resVec = solVec->clone();

    for ( int j = 0; j < 3; j++ ) {
        solVec->setRandomValues();
        rhsVec->setRandomValues();
        resVec->setRandomValues();
        testNonlinOperator->residual( rhsVec, solVec, resVec );
        resVec->scale( -1.0 );
    } // end for j

    ut->passes( exeName + " : apply" );

    auto resetParams = testNonlinOperator->getParameters( "Jacobian", solVec );

    ut->passes( exeName + " : getJac" );

    testLinOperator->reset( resetParams );

    ut->passes( exeName + " : Linear::reset" );
}

int testNonlinearMechanics( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testNonlinearMechanics-1-normal" );
    exeNames.emplace_back( "testNonlinearMechanics-1-reduced" );
    // exeNames.push_back("testNonlinearMechanics_Creep-1-normal");
    // exeNames.push_back("testNonlinearMechanics_Creep-1-reduced");
    // exeNames.push_back("testNonlinearMechanics_Creep_MatLib-1-normal");
    // exeNames.push_back("testNonlinearMechanics_Creep_MatLib-1-reduced");
    // exeNames.push_back("testNonlinearMechanics_NonlinearStrainHardening-1-normal");
    // exeNames.push_back("testNonlinearMechanics_NonlinearStrainHardening-1-reduced");
    // exeNames.push_back("testVonMises_IsotropicKinematicHardening-1-normal");
    // exeNames.push_back("testNonlinearMechanics_Matpro_Fuel_Creep-1-normal");
    // exeNames.push_back("testFrapconCladCreepMaterialModel-1");
    // exeNames.push_back("testPoroElasticMaterialModel-1");

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
