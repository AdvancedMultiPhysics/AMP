
#include <iostream>
#include <string>
#include <cassert>
#include <fstream>

/* libMesh files */
#include "libmesh/mesh.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/mesh_communication.h"
#include "libmesh/elem.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/boundary_info.h"

/* AMP files */
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/PIO.h"
#include "utils/WriteSolutionToFile.h"
#include "utils/ReadTestMesh.h"

#include "ampmesh/Mesh.h"
#include "ampmesh/libmesh/libMesh.h"

#include "discretization/simpleDOF_Manager.h"
#include "vectors/VectorBuilder.h"

#include "operators/LinearBVPOperator.h"
#include "operators/OperatorBuilder.h"
#include "operators/boundary/DirichletVectorCorrection.h"

#include "vectors/Vector.h"
#include "utils/Writer.h"

#include "solvers/trilinos/TrilinosMLSolver.h"

void linearElasticTest(AMP::UnitTest *ut, int reduced, std::string mesh_file)
{
  std::string exeName;
  std::string input_file;

  if(reduced) {
    exeName = "testLinearElasticity-reduced-" + mesh_file;
    input_file = "input_testLinearElasticity-reduced-mesh2elem";
  } else {
    exeName = "testLinearElasticity-normal-" + mesh_file;
    input_file = "input_testLinearElasticity-normal-mesh2elem";
  }

  std::string log_file = "output_" + exeName;

  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm = AMP::AMP_MPI(AMP_COMM_WORLD);

  if( globalComm.getSize() == 1 ) {
    AMP::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
    AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
    input_db->printClassData(AMP::plog);

    AMP::shared_ptr<AMP::InputDatabase> mesh_file_db(new AMP::InputDatabase("mesh_file_db"));
    AMP::InputManager::getManager()->parseInputFile(mesh_file, mesh_file_db);

    const unsigned int mesh_dim = 3;
    AMP::shared_ptr< ::Mesh > mesh(new ::Mesh(mesh_dim));

    AMP::readTestMesh(mesh_file_db, mesh);
    mesh->prepare_for_use(false);

    AMP::Mesh::Mesh::shared_ptr meshAdapter = AMP::Mesh::Mesh::shared_ptr (
        new AMP::Mesh::libMesh (mesh, "TestMesh") );

    AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel;
    AMP::shared_ptr<AMP::Operator::LinearBVPOperator> bvpOperator =
      AMP::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
          AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
            "MechanicsBVPOperator", input_db, elementPhysicsModel));

    AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> dummyModel;
    AMP::shared_ptr<AMP::Operator::DirichletVectorCorrection> dirichletVecOp =
      AMP::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
          AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
            "Load_Boundary", input_db, dummyModel));

    AMP::LinearAlgebra::Variable::shared_ptr var = bvpOperator->getOutputVariable();

    //This has an in-place apply. So, it has an empty input variable and
    //the output variable is the same as what it is operating on. 
    dirichletVecOp->setVariable(var);

    AMP::Discretization::DOFManager::shared_ptr dofMap = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::Vertex, 1, 3, true); 

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    AMP::LinearAlgebra::Vector::shared_ptr mechSolVec = AMP::LinearAlgebra::createVector(dofMap, var, true);
    AMP::LinearAlgebra::Vector::shared_ptr mechRhsVec = mechSolVec->cloneVector();
    AMP::LinearAlgebra::Vector::shared_ptr mechResVec = mechSolVec->cloneVector();

    mechSolVec->setToScalar(0.5);
    mechRhsVec->setToScalar(0.0);
    mechResVec->setToScalar(0.0);

    dirichletVecOp->apply(nullVec, nullVec, mechRhsVec, 1.0, 0.0);

    double rhsNorm = mechRhsVec->L2Norm();

    AMP::pout<<"RHS Norm: "<<rhsNorm<<std::endl;

    double initSolNorm = mechSolVec->L2Norm();

    AMP::pout<<"Initial Solution Norm: "<<initSolNorm<<std::endl;

    bvpOperator->apply(mechRhsVec, mechSolVec, mechResVec, 1.0, -1.0);

    double initResidualNorm = mechResVec->L2Norm();

    AMP::pout<<"Initial Residual Norm: "<<initResidualNorm<<std::endl;

    AMP::shared_ptr<AMP::Database> mlSolver_db = input_db->getDatabase("LinearSolver"); 

    AMP::shared_ptr<AMP::Solver::SolverStrategyParameters> mlSolverParams(new AMP::Solver::SolverStrategyParameters(mlSolver_db));

    mlSolverParams->d_pOperator = bvpOperator;

    // create the ML solver interface
    AMP::shared_ptr<AMP::Solver::TrilinosMLSolver> mlSolver(new AMP::Solver::TrilinosMLSolver(mlSolverParams));

    mlSolver->setZeroInitialGuess(false);

    mlSolver->solve(mechRhsVec, mechSolVec);

    double finalSolNorm = mechSolVec->L2Norm();

    AMP::pout<<"Final Solution Norm: "<<finalSolNorm<<std::endl;

    bvpOperator->apply(mechRhsVec, mechSolVec, mechResVec, 1.0, -1.0);

    double finalResidualNorm = mechResVec->L2Norm();

    AMP::pout<<"Final Residual Norm: "<<finalResidualNorm<<std::endl;

    printSolution(meshAdapter, mechSolVec, exeName);

    if(finalResidualNorm > (1e-10*initResidualNorm)) {
      ut->failure(exeName);
    } else {
      ut->passes(exeName);
    }
  } else {
    AMP::pout << "WARNING: This is a single processor test!" << std::endl;
    ut->passes(exeName);
  }
}

int main(int argc, char *argv[])
{
  AMP::AMPManager::startup(argc, argv);
  AMP::shared_ptr<AMP::Mesh::initializeLibMesh> libmeshInit(new AMP::Mesh::initializeLibMesh(AMP_COMM_WORLD));

  AMP::UnitTest ut;

  std::vector<std::string> mesh_files;

  if(argc == 1) {
    for(int i = 1; i <= 6; i++) {
      char name[100];
      sprintf(name, "mesh2elem-%d", i);
      mesh_files.push_back(name);
    }//end for i
  } else {
    for(int i = 1; i < argc; i++) {
      char name[100];
      sprintf(name, "mesh2elem-%d", atoi(argv[i]));
      mesh_files.push_back(name);
    }//end for i
  }

  for(size_t i = 0; i < mesh_files.size(); i++) {
    for(int reduced = 0; reduced < 2; reduced++) {
      try {
        linearElasticTest(&ut, reduced, mesh_files[i]);
      } catch (std::exception &err) {
        AMP::pout << "ERROR: " << err.what() << std::endl;
        ut.failure("ERROR");
      } catch( ... ) {
        AMP::pout << "ERROR: " << "An unknown exception was thrown." << std::endl;
        ut.failure("ERROR");
      }
    }//end for reduced
  } //end for i

  ut.report();
  int num_failed = ut.NumFailGlobal();

  libmeshInit.reset();
  AMP::AMPManager::shutdown();
  return num_failed;

}   


