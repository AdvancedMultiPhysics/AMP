#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include <iostream>
#include <string>

#include "boost/shared_ptr.hpp"

#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/PIO.h"

#include "ampmesh/Mesh.h"
#include "vectors/VectorBuilder.h"
#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"

#include "operators/diffusion/DiffusionTransportModel.h"
#include "operators/diffusion/DiffusionConstants.h"
#include "operators/diffusion/DiffusionLinearElement.h"
#include "operators/diffusion/DiffusionLinearFEOperator.h"
#include "operators/diffusion/DiffusionLinearFEOperatorParameters.h"
#include "operators/diffusion/DiffusionNonlinearElement.h"
#include "operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "operators/diffusion/DiffusionNonlinearFEOperatorParameters.h"
#include "operators/ElementPhysicsModelParameters.h"
#include "operators/ElementPhysicsModelFactory.h"
#include "operators/OperatorBuilder.h"

#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/VectorBuilder.h"
#include "vectors/Variable.h"
#include "vectors/Vector.h"

#include "applyTests.h"

#include "materials/Material.h"


void nonlinearTest(AMP::UnitTest *ut, std::string exeName)
{
  // Initialization
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName;

  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

  std::cout << "testing with input file " << input_file << std::endl;
  std::cout.flush();

  // Test create
  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);

  // Get the Mesh database and create the mesh parameters
  boost::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
  boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
  params->setComm(globalComm);

  // Create the meshes from the input database
  boost::shared_ptr<AMP::Mesh::Mesh> meshAdapter = AMP::Mesh::Mesh::buildMesh(params);

  // nonlinear operator
  boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> diffOp;
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> elementModel;
  boost::shared_ptr<AMP::InputDatabase> diffFEOp_db =
    boost::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase("NonlinearDiffusionOp"));
  boost::shared_ptr<AMP::Operator::Operator> nonlinearOperator = AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
														"NonlinearDiffusionOp",
														input_db,
														elementModel);
  diffOp = boost::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(nonlinearOperator);

  // linear operator
  boost::shared_ptr<AMP::Operator::DiffusionLinearFEOperator> linOp;
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> linElementModel;
  boost::shared_ptr<AMP::Operator::Operator> linearOperator = AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
													     "LinearDiffusionOp",
													     input_db,
													     linElementModel);
  linOp = boost::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(linearOperator);

  ut->passes(exeName+": create");
  std::cout.flush();

  // set up defaults for materials arguments and create transport model
  boost::shared_ptr<AMP::Database> transportModel_db;
  if (input_db->keyExists("DiffusionTransportModel"))
	  transportModel_db = input_db->getDatabase("DiffusionTransportModel");
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel =
          AMP::Operator::ElementPhysicsModelFactory::createElementPhysicsModel(transportModel_db);
  boost::shared_ptr<AMP::Operator::DiffusionTransportModel> transportModel =
          boost::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>(elementPhysicsModel);

  double defTemp = transportModel_db->getDoubleWithDefault("Default_Temperature", 400.0);
  double defConc = transportModel_db->getDoubleWithDefault("Default_Concentration", .33);
  double defBurn = transportModel_db->getDoubleWithDefault("Default_Burnup", .5);

  std::string property=transportModel_db->getString("Property");

  // create parameters for reset test
  boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperatorParameters> diffOpParams(new
  AMP::Operator::DiffusionNonlinearFEOperatorParameters( diffFEOp_db ));

  // nullify vectors in parameters
  diffOpParams->d_FrozenTemperature.reset();
  diffOpParams->d_FrozenConcentration.reset();
  diffOpParams->d_FrozenBurnup.reset();

  // create vectors for parameters
  boost::shared_ptr<AMP::Database> active_db = diffFEOp_db->getDatabase("ActiveInputVariables");
  AMP::LinearAlgebra::Variable::shared_ptr tVar(new AMP::LinearAlgebra::Variable(
          active_db->getStringWithDefault("temperature","not_specified")));
  AMP::LinearAlgebra::Variable::shared_ptr cVar(new AMP::LinearAlgebra::Variable(
          active_db->getStringWithDefault("concentration","not_specified")));
  AMP::LinearAlgebra::Variable::shared_ptr bVar(new AMP::LinearAlgebra::Variable(
          active_db->getStringWithDefault("burnup","not_specified")));

  //----------------------------------------------------------------------------------------------------------------------------------------------//
  // Create a DOF manager for a nodal vector 
  int DOFsPerNode = 1;
  int nodalGhostWidth = 1;
  bool split = true;
  AMP::Discretization::DOFManager::shared_ptr nodalDofMap = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, nodalGhostWidth, DOFsPerNode, split);
  //----------------------------------------------------------------------------------------------------------------------------------------------//

  // create solution, rhs, and residual vectors
  AMP::LinearAlgebra::Vector::shared_ptr tVec = AMP::LinearAlgebra::createVector( nodalDofMap, tVar );
  AMP::LinearAlgebra::Vector::shared_ptr cVec = AMP::LinearAlgebra::createVector( nodalDofMap, cVar );
  AMP::LinearAlgebra::Vector::shared_ptr bVec = AMP::LinearAlgebra::createVector( nodalDofMap, bVar );
  tVec->setToScalar(defTemp);
  cVec->setToScalar(defConc);
  bVec->setToScalar(defBurn);

  // set principal variable vector and shift for applyTests
  double shift=0., scale=1.;
  std::vector<double> range(2);
  AMP::Materials::Material::shared_ptr mat = transportModel->getMaterial();
  if (diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::TEMPERATURE) {
      diffOpParams->d_FrozenTemperature = tVec;
      if( (mat->property(property))->is_argument("temperature") ) {
          range = (mat->property(property))->get_arg_range("temperature");  // Compile error
          scale = range[1]-range[0];
          shift = range[0]+0.001*scale;
          scale *= 0.999;
      }
  }
  if (diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::CONCENTRATION) {
      diffOpParams->d_FrozenConcentration = cVec;
      if( (mat->property(property))->is_argument("concentration") ) {
          range = (mat->property(property))->get_arg_range("concentration");  // Compile error
          scale = range[1]-range[0];
          shift = range[0]+0.001*scale;
          scale *= 0.999;
      }
  }
  if (diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::BURNUP) {
      AMP_INSIST(false, "do not know what to do");
  }

  // set frozen vectors in parameters
  if (diffFEOp_db->getBoolWithDefault("Freezetemperature",false))
    diffOpParams->d_FrozenTemperature = tVec;
  if (diffFEOp_db->getBoolWithDefault("Freezeconcentration",false))
    diffOpParams->d_FrozenConcentration = cVec;
  if (diffFEOp_db->getBoolWithDefault("Freezeburnup",false))
    diffOpParams->d_FrozenBurnup = bVec;

  // set transport model
  diffOpParams->d_transportModel = transportModel;

  // Test reset
  {
      diffOp->reset(diffOpParams);
      ut->passes(exeName+": reset");
      std::cout.flush();
  }

  // set up variables for apply tests
  //AMP::LinearAlgebra::Variable::shared_ptr diffSolVar = diffOp->getInputVariable(diffOp->getPrincipalVariableId());
  AMP::LinearAlgebra::Variable::shared_ptr diffSolVar = diffOp->getOutputVariable();

  AMP::LinearAlgebra::Variable::shared_ptr diffRhsVar = diffOp->getOutputVariable();
  AMP::LinearAlgebra::Variable::shared_ptr diffResVar = diffOp->getOutputVariable();
  AMP::LinearAlgebra::Variable::shared_ptr workVar(new AMP::LinearAlgebra::Variable("work"));
  std::vector<unsigned int> nonPrincIds = diffOp->getNonPrincipalVariableIds();
  unsigned int numNonPrincIds = nonPrincIds.size();
  std::vector<AMP::LinearAlgebra::Variable::shared_ptr> nonPrincVars(numNonPrincIds);
  AMP::LinearAlgebra::Variable::shared_ptr inputVar = diffOp->getInputVariable();
  for (size_t i=0; i<numNonPrincIds; i++) {
      //nonPrincVars[i] = diffOp->getInputVariable(nonPrincIds[i]);
      nonPrincVars[i] = boost::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(inputVar)->getVariable(i);
  }

  // Test apply
  {
      std::string msgPrefix=exeName+": apply";
      AMP::LinearAlgebra::Vector::shared_ptr diffSolVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffSolVar );
      AMP::LinearAlgebra::Vector::shared_ptr diffRhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffRhsVar );
      AMP::LinearAlgebra::Vector::shared_ptr diffResVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffResVar );
      std::vector<AMP::LinearAlgebra::Vector::shared_ptr> nonPrincVecs(numNonPrincIds);
      for (unsigned int i=0; i<numNonPrincIds; i++) {
          nonPrincVecs[i] = AMP::LinearAlgebra::createVector( nodalDofMap, nonPrincVars[i] );
          if (nonPrincIds[i] == AMP::Operator::Diffusion::TEMPERATURE) nonPrincVecs[i]->setToScalar(defTemp);
          if (nonPrincIds[i] == AMP::Operator::Diffusion::CONCENTRATION) nonPrincVecs[i]->setToScalar(defConc);
          if (nonPrincIds[i] == AMP::Operator::Diffusion::BURNUP) nonPrincVecs[i]->setToScalar(defBurn);
      }
      diffRhsVec->setToScalar(0.0);
      applyTests(ut, msgPrefix, nonlinearOperator, diffRhsVec, diffSolVec, diffResVec, shift, scale);
      std::cout.flush();

      // Test getJacobianParameters and linear operator creation
      {
          diffSolVec->setRandomValues();
          adjust(diffSolVec, shift, scale);
          boost::shared_ptr<AMP::Operator::OperatorParameters> jacParams =
                  diffOp->getJacobianParameters(diffSolVec);
          linOp->reset(boost::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperatorParameters>(jacParams));
          ut->passes(exeName+": getJacobianParameters");
          std::cout.flush();
      }
  }

  // now run apply tests with multi-vectors
  AMP::LinearAlgebra::Variable::shared_ptr auxInpVar = diffSolVar->cloneVariable("NonlinearDiffusionOperator-auxInpVar");
  AMP::LinearAlgebra::Variable::shared_ptr auxOutVar = diffResVar->cloneVariable("NonlinearDiffusionOperator-auxOutVar");
  AMP::LinearAlgebra::Variable::shared_ptr auxWorkVar = diffSolVar->cloneVariable("NonlinearDiffusionOperator-auxWorkVar");

  boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> myMultiInpVar( new AMP::LinearAlgebra::MultiVariable("MultiInputVariable"));
  myMultiInpVar->add(diffSolVar);
  myMultiInpVar->add(auxInpVar);

  boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> myMultiOutVar( new AMP::LinearAlgebra::MultiVariable("MultiOutputVariable"));
  myMultiOutVar->add(diffResVar);
  myMultiOutVar->add(auxOutVar);

  boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> myMultiWorkVar( new AMP::LinearAlgebra::MultiVariable("MultiWorkVariable"));
  myMultiWorkVar->add(workVar);
  myMultiWorkVar->add(auxWorkVar);

  {
    std::string msgPrefix=exeName+": apply MultiVector ";
    AMP::LinearAlgebra::Vector::shared_ptr solVec = AMP::LinearAlgebra::createVector( nodalDofMap, myMultiInpVar );
    AMP::LinearAlgebra::Vector::shared_ptr rhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, myMultiOutVar );
    AMP::LinearAlgebra::Vector::shared_ptr resVec = AMP::LinearAlgebra::createVector( nodalDofMap, myMultiOutVar );

    // test apply with single variable vectors
    applyTests(ut, msgPrefix, nonlinearOperator, rhsVec, solVec, resVec, shift, scale);
    std::cout.flush();
  }

  // Test isValidInput function
  {
      AMP::LinearAlgebra::Vector::shared_ptr testVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffSolVar );

      testVec->setToScalar(-1000.);
      if (not diffOp->isValidInput(testVec)) ut->passes(exeName+": validInput-1");
      else {
        if( (diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::TEMPERATURE) &&
            ( (mat->property(property))->is_argument("temperature") ) ) {
              ut->failure(exeName+": validInput-1");
        } else if( (diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::CONCENTRATION) &&
            ( (mat->property(property))->is_argument("concentration") ) ) {
              ut->failure(exeName+": validInput-1");
        }
      }
      testVec->setToScalar(1.e99);
      if (not diffOp->isValidInput(testVec)) ut->passes(exeName+": validInput-2");
      else {
        if( (diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::TEMPERATURE) &&
            ( (mat->property(property))->is_argument("temperature") ) ) {
              ut->failure(exeName+": validInput-2");
        } else if( (diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::CONCENTRATION) &&
            ( (mat->property(property))->is_argument("concentration") ) ) {
              ut->failure(exeName+": validInput-2");
        }
      }
      testVec->setToScalar(1.e99);
      std::cout.flush();
  }
}

int main(int argc, char *argv[])
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup(argc,argv,startup_properties);

    AMP::UnitTest ut;

    const int NUMFILES=14;
    std::string files[NUMFILES] = {
        "Diffusion-CylindricalFick-1", "Diffusion-TUI-Thermal-1",
        "Diffusion-TUI-Fick-1", "Diffusion-TUI-Soret-1",
        "Diffusion-UO2MSRZC09-Thermal-1", "Diffusion-UO2MSRZC09-Fick-1", "Diffusion-UO2MSRZC09-Soret-1",
        "Diffusion-TUI-Thermal-ActiveTemperatureAndConcentration-1",
        "Diffusion-TUI-Fick-ActiveTemperatureAndConcentration-1",
        "Diffusion-TUI-Soret-ActiveTemperatureAndConcentration-1",
        "Diffusion-UO2MSRZC09-Thermal-ActiveTemperatureAndConcentration-1",
        "Diffusion-UO2MSRZC09-Fick-ActiveTemperatureAndConcentration-1",
        "Diffusion-UO2MSRZC09-Soret-ActiveTemperatureAndConcentration-1",
        "Diffusion-TUI-TensorFick-1"
    };

    for (int i=0; i<NUMFILES; i++) {
        try {
            nonlinearTest(&ut, files[i]);    
        } catch (std::exception &err) {
            std::cout << "ERROR: While testing "<<argv[0] << err.what() << std::endl;
            ut.failure("ERROR: While testing: "+files[i]);
        } catch( ... ) {
            std::cout << "ERROR: While testing "<<argv[0] << "An unknown exception was thrown." << std::endl;
            ut.failure("ERROR: While testing: "+files[i]);
        }
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}   



