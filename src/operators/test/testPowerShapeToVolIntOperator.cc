#include <string>
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "boost/shared_ptr.hpp"
#include "utils/InputDatabase.h"
#include "utils/Utilities.h"
#include "utils/InputManager.h"
#include "utils/PIO.h"
#include "utils/Database.h"
#include "vectors/Variable.h"

#include "ampmesh/SiloIO.h"
#include "vectors/Vector.h"

#include "operators/VolumeIntegralOperator.h"
#include "operators/SourceNonlinearElement.h"
#include "operators/ElementPhysicsModelFactory.h"
#include "operators/ElementOperationFactory.h"
#include "operators/OperatorBuilder.h"

#include "ampmesh/Mesh.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/VectorBuilder.h"

#include "../PowerShape.h"

void test_with_shape(AMP::UnitTest *ut, std::string exeName )
{

//--------------------------------------------------
//  Read Input File.
//--------------------------------------------------
    std::string input_file = "input_" + exeName;
    std::string log_file = "output_" + exeName;

    AMP::PIO::logAllNodes(log_file);

    boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
    AMP::InputManager::getManager()->parseInputFile(input_file, input_db);

//--------------------------------------------------
//   Create the Mesh.
//--------------------------------------------------
    boost::shared_ptr<AMP::Database>  mesh_db = input_db->getDatabase("Mesh");
    boost::shared_ptr<AMP::Mesh::MeshParameters> mgrParams(new AMP::Mesh::MeshParameters(mesh_db));
    mgrParams->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));
    boost::shared_ptr<AMP::Mesh::Mesh> meshAdapter = AMP::Mesh::Mesh::buildMesh(mgrParams);
    
    std::string interfaceVarName = "interVar";

//--------------------------------------------------
//  Construct PowerShape.
//--------------------------------------------------
    AMP_INSIST(input_db->keyExists("PowerShape"), "Key ''PowerShape'' is missing!");
    boost::shared_ptr<AMP::Database>  shape_db = input_db->getDatabase("PowerShape");
    boost::shared_ptr<AMP::Operator::PowerShapeParameters> shape_params(new AMP::Operator::PowerShapeParameters( shape_db ));
    shape_params->d_Mesh = meshAdapter;
    boost::shared_ptr<AMP::Operator::PowerShape> shape(new AMP::Operator::PowerShape( shape_params ));

    // Create a DOF manager for a gauss point vector 
    int DOFsPerNode = 8;
    int ghostWidth = 0;
    bool split = true;
    AMP::Discretization::DOFManager::shared_ptr dof_map = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Volume, ghostWidth, DOFsPerNode, split);

    // Create a shared pointer to a Variable - Power - Output because it will be used in the "residual" location of apply. 
    AMP::LinearAlgebra::Variable::shared_ptr shapeVar(new AMP::LinearAlgebra::Variable( interfaceVarName ));
  
    // Create input and output vectors associated with the Variable.
    AMP::LinearAlgebra::Vector::shared_ptr  shapeInpVec = AMP::LinearAlgebra::createVector( dof_map, shapeVar, split );
    AMP::LinearAlgebra::Vector::shared_ptr  shapeoutVec = shapeInpVec->cloneVector( );
    
    shapeInpVec->setToScalar(1.);
    
//--------------------------------------------------
//   CREATE THE VOLUME INTEGRAL OPERATOR -----------
//--------------------------------------------------

  AMP_INSIST( input_db->keyExists("VolumeIntegralOperator"), "key missing!" );

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> transportModel;
  boost::shared_ptr<AMP::Database> volumeDatabase = input_db->getDatabase("VolumeIntegralOperator");
  boost::shared_ptr<AMP::Database> inputVarDB = volumeDatabase->getDatabase("ActiveInputVariables");
  inputVarDB->putString("ActiveVariable_0",interfaceVarName);
  boost::shared_ptr<AMP::Operator::VolumeIntegralOperator> volumeOp = boost::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
																							"VolumeIntegralOperator",
																							input_db,
																							transportModel));

  AMP::LinearAlgebra::Variable::shared_ptr outputVariable = volumeOp->getOutputVariable();

  AMP::LinearAlgebra::Vector::shared_ptr resVec = meshAdapter->createVector( outputVariable );
  AMP::LinearAlgebra::Vector::shared_ptr nullVec;
  
    try   { shape->apply(nullVec, shapeInpVec, shapeOutVec, 1., 0.); }
    catch ( std::exception const & a ) {  
        std::cout << a.what() << std::endl;  
        ut->failure("error");
    } 

    AMP::pout << "shapeOutVec->max/min" << " : " <<  shapeOutVec->min() << " : " <<  shapeOutVec->max() <<  std::endl; 
    ut->passes("PowerShape didn't crash the system");
    
    try   { volumeOp->apply(nullVec, shapeOutVec, resVec, 1., 0.); }
    catch ( std::exception const & a ) {  
        std::cout << a.what() << std::endl;  
        ut->failure("error");
    } 

    ut->passes("VolumeIntegralOperator didn't either");
}


int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    try {
        std::string exeName("testPowerShapeToVolIntOperator");
        test_with_shape(&ut, exeName);
    } catch (std::exception &err) {
        std::cout << "ERROR: While testing "<<argv[0] << err.what() << std::endl;
        ut.failure("ERROR: While testing");
    } catch( ... ) {
        std::cout << "ERROR: While testing "<<argv[0] << "An unknown exception was thrown." << std::endl;
        ut.failure("ERROR: While testing");
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}   



