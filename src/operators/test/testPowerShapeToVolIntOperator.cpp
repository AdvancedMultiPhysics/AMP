#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/libmesh/PowerShape.h"
#include "AMP/operators/libmesh/SourceNonlinearElement.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void test_with_shape( AMP::UnitTest *ut, const std::string &exeName )
{
    //  Read Input File
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logAllNodes( log_file );
    auto input_db = AMP::Database::parseInputFile( input_file );

    //   Create the Mesh.
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    std::string interfaceVarName = "interVar";

    //  Construct PowerShape
    AMP_INSIST( input_db->keyExists( "PowerShape" ), "Key ''PowerShape'' is missing!" );
    auto shape_db        = input_db->getDatabase( "PowerShape" );
    auto shape_params    = std::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
    shape_params->d_Mesh = mesh;
    auto shape           = std::make_shared<AMP::Operator::PowerShape>( shape_params );

    // Create a DOF manager for a gauss point vector
    int DOFsPerElement    = 8;
    int DOFsPerNode       = 1;
    int ghostWidth        = 0;
    int nodalGhostWidth   = 1;
    bool split            = true;
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, ghostWidth, DOFsPerElement, split );
    auto nodalDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // Create a shared pointer to a Variable - Power - Output because it will be used in the
    // "residual" location of apply
    auto shapeVar = std::make_shared<AMP::LinearAlgebra::Variable>( interfaceVarName );

    // Create input and output vectors associated with the Variable.
    auto shapeInpVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, shapeVar, split );
    auto shapeOutVec = shapeInpVec->clone();

    shapeInpVec->setToScalar( 1. );

    // CREATE THE VOLUME INTEGRAL OPERATOR
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    auto volumeDatabase = input_db->getDatabase( "VolumeIntegralOperator" );
    auto inputVarDB     = volumeDatabase->getDatabase( "ActiveInputVariables" );
    inputVarDB->putScalar( "ActiveVariable_0", interfaceVarName );
    auto volumeOp = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "VolumeIntegralOperator", input_db ) );

    auto outputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "heatsource" );

    auto resVec = AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable, split );
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    try {
        shape->apply( shapeInpVec, shapeOutVec );
    } catch ( std::exception const &a ) {
        std::cout << a.what() << std::endl;
        ut->failure( "error" );
    }

    AMP::pout << "shapeOutVec->max/min"
              << " : " << shapeOutVec->min() << " : " << shapeOutVec->max() << std::endl;
    ut->passes( "PowerShape didn't crash the system" );

    try {
        volumeOp->apply( shapeOutVec, resVec );
    } catch ( std::exception const &a ) {
        std::cout << a.what() << std::endl;
        ut->failure( "error" );
    }

    ut->passes( "VolumeIntegralOperator didn't either" );
}


int testPowerShapeToVolIntOperator( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName( "testPowerShapeToVolIntOperator" );
    test_with_shape( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
