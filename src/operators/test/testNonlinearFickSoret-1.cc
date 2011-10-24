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

#include "ampmesh/MeshManager.h"
#include "ampmesh/MeshVariable.h"

#include "libmesh.h"

#include "operators/diffusion/DiffusionTransportModel.h"
#include "operators/diffusion/DiffusionConstants.h"
#include "operators/diffusion/DiffusionLinearElement.h"
#include "operators/diffusion/DiffusionLinearFEOperator.h"
#include "operators/diffusion/DiffusionLinearFEOperatorParameters.h"
#include "operators/diffusion/DiffusionNonlinearElement.h"
#include "operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "operators/diffusion/DiffusionNonlinearFEOperatorParameters.h"
#include "operators/diffusion/FickSoretNonlinearFEOperatorParameters.h"
#include "operators/diffusion/FickSoretNonlinearFEOperator.h"
#include "../ElementPhysicsModelParameters.h"
#include "../ElementPhysicsModelFactory.h"
#include "../OperatorBuilder.h"

#include "applyTests.h"

#include "materials/Material.h"


void nonlinearTest(AMP::UnitTest *ut, std::string exeName)
{
    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero(log_file);

    std::cout << "testing with input file " << input_file << std::endl;
    std::cout.flush();

    // Test create
    boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase(
            "input_db"));
    AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
    input_db->printClassData(AMP::plog);

    AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
    std::string mesh_file = input_db->getString("Mesh");

    AMP::Mesh::MeshManager::Adapter::shared_ptr meshAdapter =
            AMP::Mesh::MeshManager::Adapter::shared_ptr(
                    new AMP::Mesh::MeshManager::Adapter());
    meshAdapter->readExodusIIFile(mesh_file.c_str());

    boost::shared_ptr<AMP::Operator::FickSoretNonlinearFEOperator> fsOp;
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> elementModel;
    boost::shared_ptr<AMP::InputDatabase> fsOp_db =
            boost::dynamic_pointer_cast<AMP::InputDatabase>(
                    input_db->getDatabase("NonlinearFickSoretOp"));
    boost::shared_ptr<AMP::Operator::Operator> nonlinearOperator =
      AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
						     "NonlinearFickSoretOp",
						     input_db,
						     elementModel);
    fsOp = boost::dynamic_pointer_cast<AMP::Operator::FickSoretNonlinearFEOperator>(
            nonlinearOperator);
    boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> fickOp =
            fsOp->getFickOperator();
    boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> soretOp =
            fsOp->getSoretOperator();

    ut->passes(exeName + ": create");
    std::cout.flush();

    // set up defaults for materials arguments and create transport model
    boost::shared_ptr<AMP::Operator::DiffusionTransportModel> fickModel =
            fickOp->getTransportModel();
    boost::shared_ptr<AMP::Operator::DiffusionTransportModel> soretModel =
            soretOp->getTransportModel();

    // create parameters for reset test
    boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperatorParameters>
            fickOpParams(new AMP::Operator::DiffusionNonlinearFEOperatorParameters(fsOp_db));
    boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperatorParameters>
            soretOpParams(new AMP::Operator::DiffusionNonlinearFEOperatorParameters(fsOp_db));
    fickOpParams->d_transportModel = fickModel;
    soretOpParams->d_transportModel = soretModel;
    boost::shared_ptr<AMP::Database> fsOpBase_db(boost::dynamic_pointer_cast<AMP::Database>(fsOp_db));
    boost::shared_ptr<AMP::Operator::FickSoretNonlinearFEOperatorParameters>
        fsOpParams(new AMP::Operator::FickSoretNonlinearFEOperatorParameters(fsOpBase_db));
    fsOpParams->d_FickParameters = fickOpParams;
    fsOpParams->d_SoretParameters = soretOpParams;

    // create vectors for parameters
    AMP::LinearAlgebra::Variable::shared_ptr tVar(soretOp->getInputVariable(AMP::Operator::Diffusion::TEMPERATURE));
    AMP::LinearAlgebra::Variable::shared_ptr cVar(fickOp->getInputVariable(AMP::Operator::Diffusion::CONCENTRATION));
    AMP::LinearAlgebra::Variable::shared_ptr bVar(new AMP::Mesh::NodalScalarVariable("burnup"));
    AMP::LinearAlgebra::Vector::shared_ptr tVec = meshAdapter->createVector(tVar);
    AMP::LinearAlgebra::Vector::shared_ptr cVec = meshAdapter->createVector(cVar);
    AMP::LinearAlgebra::Vector::shared_ptr bVec = meshAdapter->createVector(bVar);

    // set vectors in parameters
    fickOpParams->d_temperature = tVec;
    fickOpParams->d_concentration = cVec;
    fickOpParams->d_burnup = bVec;
    soretOpParams->d_temperature = tVec;
    soretOpParams->d_concentration = cVec;
    soretOpParams->d_burnup = bVec;

    // Test reset
    {
        fsOp->reset(fsOpParams);
        ut->passes(exeName + ": reset");
        std::cout.flush();
    }

    // set shift and scale for applyTests
  double shift[2];
  double scale[2];
  shift[0] = 0.;
  shift[1] = 0.;
  scale[0] = 1.;
  scale[1] = 1.;
  std::vector<double> range(2);
  std::vector<double> defaults;
  AMP::Materials::Material::shared_ptr matFick = fickModel->getMaterial();  // compile error
  AMP::Materials::Material::shared_ptr matSoret = soretModel->getMaterial();  // compile error
  // the Soret has a principal variable of temperature
  if (soretOp->getPrincipalVariableId() == AMP::Operator::Diffusion::TEMPERATURE) {
      std::string property="ThermalDiffusionCoefficient";
      if( (matSoret->property(property))->is_argument("temperature") ) {
          range = (matSoret->property(property))->get_arg_range("temperature");  // Compile error
          scale[0] = range[1]-range[0];
          shift[0] = range[0]+0.001*scale[0];
          scale[0] *= 0.999;
          defaults = (matSoret->property(property))->get_defaults();  // compile error
      }
  }
  // the fick has a principal variable of oxygen
  if (fickOp->getPrincipalVariableId() == AMP::Operator::Diffusion::CONCENTRATION) {
      std::string property="FickCoefficient";
      if( (matFick->property(property))->is_argument("concentration") ) {
          range = (matFick->property(property))->get_arg_range("concentration");  // Compile error
          scale[1] = range[1]-range[0];
          shift[1] = range[0]+0.001*scale[1];
          scale[1] *= 0.999;
          defaults = (matFick->property(property))->get_defaults();  // compile error
      }
  }
    if(defaults.size() > 0) tVec->setToScalar(defaults[0]);  // compile error
    if(defaults.size() > 1) cVec->setToScalar(defaults[1]);  // compile error
    if(defaults.size() > 2) bVec->setToScalar(defaults[2]);  // compile error
    // set up input multivariable and output variable
    boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> fsInpVar(new AMP::LinearAlgebra::MultiVariable("fsInput"));
    fsInpVar->add(tVar);
    fsInpVar->add(cVar);
    fsInpVar->add(bVar);
    boost::shared_ptr<AMP::LinearAlgebra::Variable> fsOutVar(fickOp->getOutputVariable());

    std::string msgPrefix = exeName + ": apply ";
    AMP::LinearAlgebra::Vector::shared_ptr solVec = meshAdapter->createVector(fsInpVar);
    AMP::LinearAlgebra::Vector::shared_ptr rhsVec = meshAdapter->createVector(fsOutVar);
    AMP::LinearAlgebra::Vector::shared_ptr resVec = meshAdapter->createVector(fsOutVar);

    // set default values of input variables
    AMP::LinearAlgebra::Vector::shared_ptr inTempVec = solVec->subsetVectorForVariable(tVar);
    AMP::LinearAlgebra::Vector::shared_ptr inConcVec = solVec->subsetVectorForVariable(cVar);
    AMP::LinearAlgebra::Vector::shared_ptr inBurnVec = solVec->subsetVectorForVariable(bVar);
    if(defaults.size() > 0) inTempVec->setToScalar(defaults[0]);  // compile error
    if(defaults.size() > 1) inConcVec->setToScalar(defaults[1]);  // compile error
    if(defaults.size() > 2) inBurnVec->setToScalar(defaults[2]);  // compile error

    AMP_INSIST(fsOp->isValidInput(solVec), "input variable not set up correctly");

    // test apply
    applyTests(ut, msgPrefix, nonlinearOperator, rhsVec, solVec, resVec, shift, scale, 2);
    std::cout.flush();
}

int main(int argc, char *argv[])
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup(argc,argv,startup_properties);

    AMP::UnitTest ut;
    const int NUMFILES = 2;
    std::string files[NUMFILES] = {"FickSoret-TUI-1", "FickSoret-UO2MSRZC09-1"};

    try {
        for (int i = 0; i < NUMFILES; i++)
            nonlinearTest(&ut, files[i]);
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

