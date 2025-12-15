#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include "libmesh/libmesh.h"
#include "libmesh/mesh_communication.h"

#include <cstdlib>
#include <iostream>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &exeName, int callLinReset )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    std::string msgName    = exeName;
    if ( callLinReset ) {
        log_file = log_file + "-1";
        msgName  = msgName + "-1";
    } else {
        log_file = log_file + "-0";
        msgName  = msgName + "-0";
    }

    AMP::logOnlyNodeZero( log_file );


    AMP::AMP_MPI globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );
    auto input_db           = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    auto mesh_file = input_db->getString( "mesh_file" );
    auto mesh      = AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD );

    auto nonlinOperator = std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "NonlinearMechanicsOperator", input_db ) );
    auto elementPhysicsModel = nonlinOperator->getMaterialModel();

    auto linOperator = std::make_shared<AMP::Operator::MechanicsLinearFEOperator>(
        nonlinOperator->getParameters( "Jacobian", nullptr ) );

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    auto var = nonlinOperator->getOutputVariable();

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto solVec       = AMP::LinearAlgebra::createVector( dofMap, var, true );
    auto resVecNonlin = solVec->clone();
    auto resVecLin    = solVec->clone();
    auto resDiffVec   = solVec->clone();
    auto tmpNonlinVec = solVec->clone();
    auto tmpLinVec    = solVec->clone();

    solVec->setToScalar( 0.0 );
    double solNorm = static_cast<double>( solVec->L2Norm() );
    AMP::pout << "sol-Norm-2 = " << solNorm << std::endl;

    nonlinOperator->apply( solVec, resVecNonlin );
    linOperator->reset( nonlinOperator->getParameters( "Jacobian", solVec ) );
    linOperator->apply( solVec, resVecLin );
    resDiffVec->subtract( *resVecNonlin, *resVecLin );

    double epsilon =
        1.0e-13 * static_cast<double>( ( linOperator->getMatrix() )->extractDiagonal()->L1Norm() );
    AMP::pout << "epsilon = " << epsilon << std::endl;

    double nonLinNorm = static_cast<double>( resVecNonlin->L1Norm() );
    double linNorm    = static_cast<double>( resVecLin->L1Norm() );
    double diffNorm   = static_cast<double>( resDiffVec->L1Norm() );

    AMP::pout << "nonLin-Norm-1 = " << nonLinNorm << std::endl;
    AMP::pout << "lin-Norm-1 = " << linNorm << std::endl;
    AMP::pout << "diff-Norm-1 = " << diffNorm << std::endl;

    if ( nonLinNorm > epsilon ) {
        ut->failure( msgName );
    }

    if ( linNorm > epsilon ) {
        ut->failure( msgName );
    }

    if ( diffNorm > epsilon ) {
        ut->failure( msgName );
    }

    solVec->setRandomValues();
    solVec->scale( 100.0 );
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    solNorm = static_cast<double>( solVec->L2Norm() );
    AMP::pout << "sol-Norm-2 = " << solNorm << std::endl;

    nonlinOperator->apply( solVec, resVecNonlin );
    if ( callLinReset ) {
        linOperator->reset( nonlinOperator->getParameters( "Jacobian", solVec ) );
    }
    linOperator->apply( solVec, resVecLin );
    resDiffVec->subtract( *resVecNonlin, *resVecLin );

    nonLinNorm = static_cast<double>( resVecNonlin->L2Norm() );
    linNorm    = static_cast<double>( resVecLin->L2Norm() );
    diffNorm   = static_cast<double>( resDiffVec->L1Norm() );

    AMP::pout << "nonLin-Norm-2 = " << nonLinNorm << std::endl;
    AMP::pout << "lin-Norm-2 = " << linNorm << std::endl;
    AMP::pout << "diff-Norm-1 = " << diffNorm << std::endl;

    if ( diffNorm > epsilon ) {
        ut->failure( msgName );
    }

    tmpNonlinVec->copyVector( resVecNonlin );
    tmpNonlinVec->scale( 10.0 );

    tmpLinVec->copyVector( resVecLin );
    tmpLinVec->scale( 10.0 );

    solVec->scale( 10.0 );
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    solNorm = static_cast<double>( solVec->L2Norm() );
    AMP::pout << "sol-Norm-2 = " << solNorm << std::endl;

    nonlinOperator->apply( solVec, resVecNonlin );
    if ( callLinReset ) {
        linOperator->reset( nonlinOperator->getParameters( "Jacobian", solVec ) );
    }
    linOperator->apply( solVec, resVecLin );

    nonLinNorm = static_cast<double>( resVecNonlin->L2Norm() );
    linNorm    = static_cast<double>( resVecLin->L2Norm() );
    AMP::pout << "nonLin-Norm-2 = " << nonLinNorm << std::endl;
    AMP::pout << "lin-Norm-2 = " << linNorm << std::endl;

    resDiffVec->subtract( *resVecNonlin, *tmpNonlinVec );
    diffNorm = static_cast<double>( resDiffVec->L1Norm() );

    if ( diffNorm > ( 10.0 * epsilon ) ) {
        ut->failure( msgName );
    }

    resDiffVec->subtract( *resVecLin, *tmpLinVec );
    diffNorm = static_cast<double>( resDiffVec->L1Norm() );

    if ( diffNorm > ( 10.0 * epsilon ) ) {
        ut->failure( msgName );
    }

    ut->passes( msgName );
}

int testConsistentTangent( int argc, char *argv[] )
{

    AMP::AMPManager::startup( argc, argv );
    auto libmeshInit = std::make_shared<AMP::Mesh::initializeLibMesh>( AMP_COMM_WORLD );

    AMP::UnitTest ut;

    std::vector<std::string> exeNames;

    exeNames.emplace_back( "testConsistentTangent-1-mesh1-normal" );
    exeNames.emplace_back( "testConsistentTangent-1-mesh1-reduced" );
    exeNames.emplace_back( "testConsistentTangent-1-cookMesh1-normal" );
    exeNames.emplace_back( "testConsistentTangent-1-cookMesh1-reduced" );

    exeNames.emplace_back( "testConsistentTangent-2-mesh1-normal" );
    exeNames.emplace_back( "testConsistentTangent-2-mesh1-reduced" );
    exeNames.emplace_back( "testConsistentTangent-2-cookMesh1-normal" );
    exeNames.emplace_back( "testConsistentTangent-2-cookMesh1-reduced" );

    exeNames.emplace_back( "testConsistentTangent-3-mesh1-normal" );
    exeNames.emplace_back( "testConsistentTangent-3-mesh1-reduced" );
    exeNames.emplace_back( "testConsistentTangent-3-cookMesh1-normal" );
    exeNames.emplace_back( "testConsistentTangent-3-cookMesh1-reduced" );

    exeNames.emplace_back( "testConsistentTangent-4-mesh1-normal" );
    exeNames.emplace_back( "testConsistentTangent-4-mesh1-reduced" );
    exeNames.emplace_back( "testConsistentTangent-4-cookMesh1-normal" );
    exeNames.emplace_back( "testConsistentTangent-4-cookMesh1-reduced" );

    for ( int j = 0; j < 2; j++ ) {
        for ( auto &exeName : exeNames )
            myTest( &ut, exeName, j );
    }

    ut.report();
    int num_failed = ut.NumFailGlobal();

    libmeshInit.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
