#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "boost/shared_ptr.hpp"

#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/PIO.h"

#include "ampmesh/Mesh.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/VectorBuilder.h"

#include "operators/ParameterFactory.h"
#include "operators/boundary/DirichletMatrixCorrectionParameters.h"


void ParameterFactoryTest(AMP::UnitTest *ut)
{
  std::string exeName("testParameterFactory-1");
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName;

  AMP::PIO::logOnlyNodeZero(log_file);

  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);

  AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
  std::string mesh_file = input_db->getString("Mesh");

  // Create the mesh parameter object
  boost::shared_ptr<AMP::MemoryDatabase> database(new AMP::MemoryDatabase("Mesh"));
  database->putInteger("dim",3);
  database->putString("MeshName","mesh");
  database->putString("MeshType","libMesh");
  database->putString("FileName",mesh_file);
  boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
  params->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));

  // Create the mesh
  AMP::Mesh::Mesh::shared_ptr  meshAdapter = AMP::Mesh::Mesh::buildMesh(params);

  AMP_INSIST(input_db->keyExists("Parameter"), "Key ''Parameter'' is missing!");
  boost::shared_ptr<AMP::Database> elemOp_db = input_db->getDatabase("Parameter");
  boost::shared_ptr<AMP::Operator::OperatorParameters> operatorParameters = AMP::Operator::ParameterFactory::createParameter(elemOp_db, meshAdapter);

  if(elemOp_db->getString("name")=="DirichletMatrixCorrection")
    {
      
      boost::shared_ptr<AMP::Operator::DirichletMatrixCorrectionParameters> operatorParams = boost::dynamic_pointer_cast<AMP::Operator::DirichletMatrixCorrectionParameters>(operatorParameters);
      
      if(operatorParams.get()!=NULL)
    {
      ut->passes(exeName.c_str());
    }
      else
    {
      ut->failure(exeName.c_str());
    }
    }
  else
    {
      ut->failure(exeName.c_str());      
    }

}

int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;
 
    try {
        ParameterFactoryTest(&ut);
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


