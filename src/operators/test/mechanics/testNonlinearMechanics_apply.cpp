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

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    auto var = testNonlinOperator->getOutputVariable();

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto solVec = AMP::LinearAlgebra::createVector( dofMap, var, true );
    auto resVec = solVec->clone();

    solVec->setToScalar( 5.0 );

    AMP::pout << "Solution Norm: " << ( solVec->L2Norm() ) << std::endl;

    testNonlinOperator->apply( solVec, resVec );

    double resNorm1 = static_cast<double>( resVec->L2Norm() );

    AMP::pout << "resNorm1 = " << resNorm1 << std::endl;

    testNonlinOperator->apply( solVec, resVec );

    double resNorm2 = static_cast<double>( resVec->L2Norm() );

    AMP::pout << "resNorm2 = " << resNorm2 << std::endl;

    AMP_ASSERT( resNorm1 == resNorm2 );

    ut->passes( exeName );
}

int testNonlinearMechanics_apply( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testNonlinearMechanics-apply-1" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
