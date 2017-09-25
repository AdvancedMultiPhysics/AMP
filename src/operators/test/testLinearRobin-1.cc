#include "materials/Material.h"
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

#include "operators/ElementOperationFactory.h"
#include "operators/ElementPhysicsModelFactory.h"
#include "operators/diffusion/DiffusionLinearElement.h"
#include "operators/diffusion/DiffusionLinearFEOperator.h"
#include "operators/diffusion/DiffusionTransportModel.h"
#include "utils/Writer.h"
#include "vectors/Vector.h"

#include "operators/LinearBVPOperator.h"
#include "operators/OperatorBuilder.h"

#include "operators/boundary/libmesh/RobinMatrixCorrection.h"

#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Variable.h"
#include "vectors/Vector.h"
#include "vectors/VectorBuilder.h"


void bcTests( AMP::UnitTest *ut,
              std::string msgPrefix,
              AMP::shared_ptr<AMP::Operator::Operator> &feOperator,
              AMP::shared_ptr<AMP::Operator::Operator> &bcOperator,
              AMP::shared_ptr<AMP::InputDatabase>
                  bcDatabase,
              AMP::LinearAlgebra::Vector::shared_ptr bcCorrectionVec )
//             AMP::shared_ptr<AMP::Operator::OperatorParameters> &bcParameters)
{

    bool passed = true;

    try {
        AMP::shared_ptr<AMP::InputDatabase> tmp_db( new AMP::InputDatabase( "Dummy" ) );
        tmp_db->putBool( "skip_params", false );
        tmp_db->putInteger( "print_info_level", 3 );
        tmp_db->putDouble( "alpha", 0.0 );

        AMP::shared_ptr<AMP::Operator::RobinMatrixCorrectionParameters> dummyParameters(
            new AMP::Operator::RobinMatrixCorrectionParameters( tmp_db ) );

        bcOperator->reset( dummyParameters );

        ut->failure( "Test" );
    } catch ( std::exception ) {

        ut->passes( msgPrefix + ": catches when prefactor alpha is set to zero " );
    }

    ut->passes( msgPrefix );

    passed = true;
    try {
        AMP::shared_ptr<AMP::Operator::RobinMatrixCorrectionParameters> bcParameters(
            new AMP::Operator::RobinMatrixCorrectionParameters( bcDatabase ) );
        bcParameters->d_inputMatrix =
            ( AMP::dynamic_pointer_cast<AMP::Operator::LinearFEOperator>( feOperator ) )
                ->getMatrix();
        bcParameters->d_variable = feOperator->getOutputVariable();
        bcOperator->reset( bcParameters );

        bcCorrectionVec->setToScalar( 0.0 );
        ( AMP::dynamic_pointer_cast<AMP::Operator::BoundaryOperator>( bcOperator ) )
            ->addRHScorrection( bcCorrectionVec );
        AMP_INSIST( ( ( bcCorrectionVec.get() ) != nullptr ), "NULL rhs correction vector" );

        ut->passes( msgPrefix + ": Robin returns a rhs correction vector " );

        // ut.failure(msgPrefix+": BoundaryOperators have changed and this needs to be updated.");
    } catch ( ... ) {

        ut->failure( "Exception" );
    }

    if ( passed ) {
        ut->passes( msgPrefix + ": Robin Matrix Correction for Linear Operator  " );
    } else {
        ut->failure( msgPrefix + ": Robin Matrix Correction for Linear Operator  " );
    }

    ut->passes( msgPrefix );
    std::cout.flush();
}

void linearRobinTest( AMP::UnitTest *ut, std::string exeName )
{
    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logAllNodes( log_file );

    std::cout << "testing with input file " << input_file << std::endl;
    std::cout.flush();

    AMP::shared_ptr<AMP::InputDatabase> input_db( new AMP::InputDatabase( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile( input_file, input_db );
    input_db->printClassData( AMP::plog );

    // Get the mesh name
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    std::string mesh_file = input_db->getString( "Mesh" );

    // Create the mesh parameter object
    AMP::shared_ptr<AMP::MemoryDatabase> database( new AMP::MemoryDatabase( "Mesh" ) );
    database->putInteger( "dim", 3 );
    database->putString( "MeshName", "mesh" );
    database->putString( "MeshType", "libMesh" );
    database->putString( "FileName", mesh_file );
    AMP::shared_ptr<AMP::Mesh::MeshParameters> params( new AMP::Mesh::MeshParameters( database ) );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the mesh
    AMP::Mesh::Mesh::shared_ptr meshAdapter = AMP::Mesh::Mesh::buildMesh( params );

    /////////////////////////////////////////////////
    //   CREATE THE LINEAR DIFFUSION BVP OPERATOR  //
    /////////////////////////////////////////////////
    AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> elementModel;
    AMP::shared_ptr<AMP::Operator::LinearBVPOperator> diffusionOperator =
        AMP::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter, "DiffusionBVPOperator", input_db, elementModel ) );

    AMP::shared_ptr<AMP::InputDatabase> bcDatabase = AMP::dynamic_pointer_cast<AMP::InputDatabase>(
        input_db->getDatabase( "RobinMatrixCorrection" ) );

    AMP::Operator::Operator::shared_ptr boundaryOp, volumeOp;
    boundaryOp = diffusionOperator->getBoundaryOperator();
    volumeOp   = diffusionOperator->getVolumeOperator();

    // AMP::LinearAlgebra::Vector::shared_ptr bndVec = meshAdapter->createVector(
    // volumeOp->getOutputVariable() );
    AMP::Discretization::DOFManager::shared_ptr NodalScalarDOF =
        AMP::Discretization::simpleDOFManager::create(
            meshAdapter, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    AMP::LinearAlgebra::Vector::shared_ptr bndVec =
        AMP::LinearAlgebra::createVector( NodalScalarDOF, volumeOp->getOutputVariable(), true );

    std::string msgPrefix = exeName + "- Boundary Conditions";
    // Test Robin Boundary Conditions
    {
        bcTests( ut, msgPrefix, volumeOp, boundaryOp, bcDatabase, bndVec );
    }

    std::cout.flush();
}
// Input and output file names
int main( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    const int NUMFILES          = 1;
    std::string files[NUMFILES] = { "LinearOp-Robin-1" };

    for ( auto &file : files )
        linearRobinTest( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
