#include "utils/AMPManager.h"
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/PIO.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/Utilities.h"
#include "utils/shared_ptr.h"
#include "vectors/Variable.h"
#include <string>

#include "utils/Writer.h"
#include "vectors/Vector.h"

#include "operators/ElementOperationFactory.h"
#include "operators/ElementPhysicsModelFactory.h"
#include "operators/OperatorBuilder.h"
#include "operators/libmesh/SourceNonlinearElement.h"
#include "operators/libmesh/VolumeIntegralOperator.h"

#include "ampmesh/Mesh.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/VectorBuilder.h"

#include "operators/libmesh/PowerShape.h"

void test_with_shape( AMP::UnitTest *ut, std::string exeName )
{

    //--------------------------------------------------
    //  Read Input File.
    //--------------------------------------------------
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logAllNodes( log_file );

    AMP::shared_ptr<AMP::InputDatabase> input_db( new AMP::InputDatabase( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile( input_file, input_db );

    //--------------------------------------------------
    //   Create the Mesh.
    //--------------------------------------------------
    AMP::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase( "Mesh" );
    AMP::shared_ptr<AMP::Mesh::MeshParameters> mgrParams(
        new AMP::Mesh::MeshParameters( mesh_db ) );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    AMP::shared_ptr<AMP::Mesh::Mesh> meshAdapter = AMP::Mesh::Mesh::buildMesh( mgrParams );

    std::string interfaceVarName = "interVar";

    //--------------------------------------------------
    //  Construct PowerShape.
    //--------------------------------------------------
    AMP_INSIST( input_db->keyExists( "PowerShape" ), "Key ''PowerShape'' is missing!" );
    AMP::shared_ptr<AMP::Database> shape_db = input_db->getDatabase( "PowerShape" );
    AMP::shared_ptr<AMP::Operator::PowerShapeParameters> shape_params(
        new AMP::Operator::PowerShapeParameters( shape_db ) );
    shape_params->d_Mesh = meshAdapter;
    AMP::shared_ptr<AMP::Operator::PowerShape> shape(
        new AMP::Operator::PowerShape( shape_params ) );

    // Create a DOF manager for a gauss point vector
    int DOFsPerElement  = 8;
    int DOFsPerNode     = 1;
    int ghostWidth      = 0;
    int nodalGhostWidth = 1;
    bool split          = true;
    AMP::Discretization::DOFManager::shared_ptr gaussPointDofMap =
        AMP::Discretization::simpleDOFManager::create(
            meshAdapter, AMP::Mesh::Volume, ghostWidth, DOFsPerElement, split );
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap =
        AMP::Discretization::simpleDOFManager::create(
            meshAdapter, AMP::Mesh::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // Create a shared pointer to a Variable - Power - Output because it will be used in the
    // "residual" location of
    // apply.
    AMP::LinearAlgebra::Variable::shared_ptr shapeVar(
        new AMP::LinearAlgebra::Variable( interfaceVarName ) );

    // Create input and output vectors associated with the Variable.
    AMP::LinearAlgebra::Vector::shared_ptr shapeInpVec =
        AMP::LinearAlgebra::createVector( gaussPointDofMap, shapeVar, split );
    AMP::LinearAlgebra::Vector::shared_ptr shapeOutVec = shapeInpVec->cloneVector();

    shapeInpVec->setToScalar( 1. );

    //--------------------------------------------------
    //   CREATE THE VOLUME INTEGRAL OPERATOR -----------
    //--------------------------------------------------

    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );

    AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> transportModel;
    AMP::shared_ptr<AMP::Database> volumeDatabase =
        input_db->getDatabase( "VolumeIntegralOperator" );
    AMP::shared_ptr<AMP::Database> inputVarDB =
        volumeDatabase->getDatabase( "ActiveInputVariables" );
    inputVarDB->putString( "ActiveVariable_0", interfaceVarName );
    AMP::shared_ptr<AMP::Operator::VolumeIntegralOperator> volumeOp =
        AMP::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter, "VolumeIntegralOperator", input_db, transportModel ) );

    AMP::LinearAlgebra::Variable::shared_ptr outputVariable(
        new AMP::LinearAlgebra::Variable( "heatsource" ) );

    AMP::LinearAlgebra::Vector::shared_ptr resVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable, split );
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


int main( int argc, char *argv[] )
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
