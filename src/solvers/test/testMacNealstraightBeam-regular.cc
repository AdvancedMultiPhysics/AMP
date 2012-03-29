#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include <iostream>
#include <string>

#include <cassert>
#include <fstream>

#include <sys/stat.h>

/* Boost files */
#include "boost/shared_ptr.hpp"

/* libMesh files */
#include "mesh.h"
#include "mesh_generation.h"
#include "mesh_communication.h"
#include "elem.h"
#include "cell_hex8.h"
#include "boundary_info.h"
#include "fe_base.h"
#include "enum_fe_family.h"
#include "enum_quadrature_type.h"
#include "quadrature.h"
#include "string_to_enum.h"

/* AMP files */
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/PIO.h"

#include "ampmesh/MeshManager.h"
#include "ampmesh/MeshVariable.h"
#include "ampmesh/MeshAdapter.h"
#include "materials/Material.h"

#include "operators/LinearBVPOperator.h"
#include "operators/OperatorBuilder.h"

#include "operators/boundary/DirichletVectorCorrection.h"


#include "vectors/Vector.h"
#include "vectors/VectorSelector.h"
#include "ampmesh/SiloIO.h"


#include "solvers/PetscKrylovSolverParameters.h"
#include "solvers/PetscKrylovSolver.h"
#include "solvers/TrilinosMLSolver.h"

#include "utils/ReadTestMesh.h"

extern "C"{
#include "petsc.h"
}

void deformMesh(AMP::Mesh::MeshManager::Adapter::shared_ptr meshAdapter,
    AMP::LinearAlgebra::Vector::shared_ptr mechSolVec) {
  AMP::Mesh::DOFMap::shared_ptr dof_map = meshAdapter->getDOFMap(mechSolVec->getVariable());

  AMP::Mesh::MeshManager::Adapter::OwnedNodeIterator nd  = meshAdapter->beginOwnedNode();
  AMP::Mesh::MeshManager::Adapter::OwnedNodeIterator end_nd   = meshAdapter->endOwnedNode();

  std::vector <unsigned int> dofIds(3);
  dofIds[0] = 0; dofIds[1] = 1; dofIds[2] = 2;

  for( ; nd != end_nd; ++nd) {
    std::vector<unsigned int> ndGlobalIds;
    dof_map->getDOFs(*nd, ndGlobalIds, dofIds);

    double xDisp = mechSolVec->getValueByGlobalID(ndGlobalIds[0]);
    double yDisp = mechSolVec->getValueByGlobalID(ndGlobalIds[1]);
    double zDisp = mechSolVec->getValueByGlobalID(ndGlobalIds[2]);

    nd->translate(xDisp, yDisp, zDisp);
  }//end for nd
}

void linearElasticTest(AMP::UnitTest *ut, std::string exeName, 
    int exampleNum) {
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName + ".txt";

  AMP::PIO::logOnlyNodeZero(log_file);

  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::AMP_MPI globalComm = AMP::AMP_MPI(AMP_COMM_WORLD);
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);

  AMP::Mesh::MeshManagerParameters::shared_ptr  meshmgrParams ( new AMP::Mesh::MeshManagerParameters ( input_db ) );
  AMP::Mesh::MeshManager::shared_ptr  manager ( new AMP::Mesh::MeshManager ( meshmgrParams ) );

  std::string mesh_file = input_db->getString("mesh_file");

  const unsigned int mesh_dim = 3;
  boost::shared_ptr< ::Mesh > mesh(new ::Mesh(mesh_dim));

  if(globalComm.getRank() == 0) {
    AMP::readTestMesh(mesh_file, mesh);
  }//end if root processor

  MeshCommunication().broadcast(*(mesh.get()));

  mesh->prepare_for_use(false);

  AMP::Mesh::MeshManager::Adapter::shared_ptr meshAdapter ( new AMP::Mesh::MeshManager::Adapter (mesh) );

  manager->addMesh(meshAdapter, "beam");

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel;
  boost::shared_ptr<AMP::Operator::LinearBVPOperator> bvpOperator =
    boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
														 "MechanicsBVPOperator",
														 input_db,
														 elementPhysicsModel));

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> dummyModel;
  boost::shared_ptr<AMP::Operator::DirichletVectorCorrection> dirichletVecOp =
    boost::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
															 "Load_Boundary",
															 input_db,
															 dummyModel));
  //This has an in-place apply. So, it has an empty input variable and
  //the output variable is the same as what it is operating on. 
  dirichletVecOp->setVariable(bvpOperator->getOutputVariable());

  AMP::LinearAlgebra::Vector::shared_ptr nullVec;

  AMP::LinearAlgebra::Vector::shared_ptr mechSolVec = meshAdapter->createVector( bvpOperator->getInputVariable() );
  AMP::LinearAlgebra::Vector::shared_ptr mechRhsVec = meshAdapter->createVector( bvpOperator->getOutputVariable() );
  AMP::LinearAlgebra::Vector::shared_ptr mechResVec = meshAdapter->createVector( bvpOperator->getOutputVariable() );

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

  boost::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase("LinearSolver"); 

  // ---- first initialize the preconditioner
  boost::shared_ptr<AMP::Database> pcSolver_db = linearSolver_db->getDatabase("Preconditioner"); 
  boost::shared_ptr<AMP::Solver::TrilinosMLSolverParameters> pcSolverParams(new AMP::Solver::TrilinosMLSolverParameters(pcSolver_db));
  pcSolverParams->d_pOperator = bvpOperator;
  boost::shared_ptr<AMP::Solver::TrilinosMLSolver> pcSolver(new AMP::Solver::TrilinosMLSolver(pcSolverParams));

  // initialize the linear solver
  boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> linearSolverParams(new
      AMP::Solver::PetscKrylovSolverParameters(linearSolver_db));
  linearSolverParams->d_pOperator = bvpOperator;
  linearSolverParams->d_comm = globalComm;
  linearSolverParams->d_pPreconditioner = pcSolver;
  boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver(new AMP::Solver::PetscKrylovSolver(linearSolverParams));

  linearSolver->setZeroInitialGuess(false);

  linearSolver->solve(mechRhsVec, mechSolVec);

  double finalSolNorm = mechSolVec->L2Norm();

  AMP::pout<<"Final Solution Norm: "<<finalSolNorm<<std::endl;

  AMP::LinearAlgebra::Vector::shared_ptr mechUvec = mechSolVec->select( AMP::LinearAlgebra::VS_Stride("U", 0, 3) , "U" );
  AMP::LinearAlgebra::Vector::shared_ptr mechVvec = mechSolVec->select( AMP::LinearAlgebra::VS_Stride("V", 1, 3) , "V" );
  AMP::LinearAlgebra::Vector::shared_ptr mechWvec = mechSolVec->select( AMP::LinearAlgebra::VS_Stride("W", 2, 3) , "W" );

  double finalMaxU = mechUvec->maxNorm();
  double finalMaxV = mechVvec->maxNorm();
  double finalMaxW = mechWvec->maxNorm();

  AMP::pout<<"Maximum U displacement: "<<finalMaxU<<std::endl;
  AMP::pout<<"Maximum V displacement: "<<finalMaxV<<std::endl;
  AMP::pout<<"Maximum W displacement: "<<finalMaxW<<std::endl;

  bvpOperator->apply(mechRhsVec, mechSolVec, mechResVec, 1.0, -1.0);

  double finalResidualNorm = mechResVec->L2Norm();

  AMP::pout<<"Final Residual Norm: "<<finalResidualNorm<<std::endl;

  if(finalResidualNorm > (1e-10*initResidualNorm)) {
    ut->failure(exeName);
  } else {
    ut->passes(exeName);
  }

#ifdef USE_SILO
  manager->registerVectorAsData ( mechSolVec , "Solution_Vector" );
  manager->registerVectorAsData ( mechResVec , "Residual_Vector" );
  char outFileName1[256];
  sprintf(outFileName1, "undeformedBeam_%d", exampleNum);
  manager->writeFile<AMP::Mesh::SiloIO>(outFileName1, 1);
  deformMesh(meshAdapter, mechSolVec);
  char outFileName2[256];
  sprintf(outFileName2, "deformedBeam_%d", exampleNum);
  manager->writeFile<AMP::Mesh::SiloIO>(outFileName2, 1);
#endif

}

int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;

  if(argc == 1) {
    exeNames.push_back("testMacNealstraightBeam-regular-X-normal-mesh0");
    exeNames.push_back("testMacNealstraightBeam-regular-X-reduced-mesh0");
    exeNames.push_back("testMacNealstraightBeam-regular-Y-normal-mesh0");
    exeNames.push_back("testMacNealstraightBeam-regular-Y-reduced-mesh0");
    exeNames.push_back("testMacNealstraightBeam-regular-Z-normal-mesh0");
    exeNames.push_back("testMacNealstraightBeam-regular-Z-reduced-mesh0");
    
    exeNames.push_back("testMacNealstraightBeam-regular-X-normal-mesh1");
    exeNames.push_back("testMacNealstraightBeam-regular-X-reduced-mesh1");
    exeNames.push_back("testMacNealstraightBeam-regular-Y-normal-mesh1");
    exeNames.push_back("testMacNealstraightBeam-regular-Y-reduced-mesh1");
    exeNames.push_back("testMacNealstraightBeam-regular-Z-normal-mesh1");
    exeNames.push_back("testMacNealstraightBeam-regular-Z-reduced-mesh1");
    
    exeNames.push_back("testMacNealstraightBeam-regular-X-normal-mesh2");
    exeNames.push_back("testMacNealstraightBeam-regular-X-reduced-mesh2");
    exeNames.push_back("testMacNealstraightBeam-regular-Y-normal-mesh2");
    exeNames.push_back("testMacNealstraightBeam-regular-Y-reduced-mesh2");
    exeNames.push_back("testMacNealstraightBeam-regular-Z-normal-mesh2");
    exeNames.push_back("testMacNealstraightBeam-regular-Z-reduced-mesh2");
  } else {
    for(int i = 1; i < argc; i+= 3) {
      char inpName[100];
      sprintf(inpName, "testMacNealstraightBeam-regular-%s-%s-mesh%d", argv[i], argv[i+1], atoi(argv[i+2]));
      exeNames.push_back(inpName);
    }//end for i
  }

    for(size_t i = 0; i < exeNames.size(); i++) {
        try {
            linearElasticTest(&ut, exeNames[i], i);
            AMP::pout<<exeNames[i]<<" had "<<ut.NumFailGlobal()<<" failures."<<std::endl;
        } catch (std::exception &err) {
            AMP::pout << "ERROR: " << err.what() << std::endl;
        } catch( ... ) {
            AMP::pout << "ERROR: " << "An unknown exception was thrown." << std::endl;
        }
    } //end for i

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;

}   


