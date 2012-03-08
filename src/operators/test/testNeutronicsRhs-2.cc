#include <string>
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "materials/Material.h"
#include "boost/shared_ptr.hpp"
#include "utils/InputDatabase.h"
#include "utils/Utilities.h"
#include "utils/InputManager.h"
#include "utils/PIO.h"
#include "utils/Database.h"
#include "vectors/Variable.h"

#include "ampmesh/SiloIO.h"
#include "vectors/Vector.h"
#include "operators/NeutronicsRhs.h"
#include "operators/SourceNonlinearElement.h"
#include "operators/ElementPhysicsModelFactory.h"
#include "operators/ElementOperationFactory.h"

#include "operators/LinearBVPOperator.h"
#include "operators/NonlinearBVPOperator.h"
#include "operators/OperatorBuilder.h"

#include <exception>


void sourceTest(AMP::UnitTest *ut , std::string exeName)
{
  // Initialization
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName;

  AMP::PIO::logAllNodes(log_file);

  std::cout << "testing with input file " << input_file << std::endl;
  

  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);

  AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
  boost::shared_ptr<AMP::Database>  mesh_db = input_db->getDatabase("Mesh");
  boost::shared_ptr<AMP::Mesh::MeshParameters> mgrParams(new AMP::Mesh::MeshParameters(mesh_db));
  mgrParams->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));
  boost::shared_ptr<AMP::Mesh::Mesh> meshAdapter = AMP::Mesh::Mesh::buildMesh(mgrParams);

//--------------------------------------------------
//   CREATE THE VOLUME INTEGRAL OPERATOR -----------
//--------------------------------------------------

  AMP_INSIST( input_db->keyExists("NeutronicsRhs"), "key missing!" );

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> unusedModel;

  // Construct with OperatorBuilder
  {
    boost::shared_ptr<AMP::Operator::NeutronicsRhs> ntxBld = boost::dynamic_pointer_cast<AMP::Operator::NeutronicsRhs>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
																				      "NeutronicsRhs",
																				      input_db,
																				      unusedModel));
    AMP_INSIST(ntxBld.get()!=NULL, "NULL rhs out of OperatorBuilder");
    ut->passes( "NeutronicsRhs was constructed by OperatorBuilder for: " + input_file);
    // ntxBld->setTimeStep(0); 
    // ut->passes( "NeutronicsRhs, constructed by OperatorBuilder, set the time for: " + input_file);
  }

}


  // Input and output file names
int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

  const int NUMFILES=1;
  std::string files[NUMFILES] = {
        "testNeutronicsRhs-db"
  };

    try {
        for (int i=0; i<NUMFILES; i++) {
            sourceTest(&ut, files[i]);
        }
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


