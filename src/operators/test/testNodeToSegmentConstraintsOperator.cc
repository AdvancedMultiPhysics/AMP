
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/PIO.h"

#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Variable.h"
#include "vectors/Vector.h"
#include "vectors/VectorBuilder.h"

#include "externVars.h"

#include "ampmesh/libmesh/libMesh.h"
#include "ampmesh/Mesh.h"
#include "ampmesh/SiloIO.h"
#include "ampmesh/dendro/DendroSearch.h"

#include "operators/OperatorBuilder.h"
#include "operators/LinearBVPOperator.h"
#include "operators/ColumnOperator.h"
#include "operators/PetscMatrixShellOperator.h"
#include "operators/boundary/DirichletVectorCorrection.h"
#include "operators/mechanics/MechanicsLinearFEOperator.h"
#include "operators/contact/NodeToSegmentConstraintsOperator.h"

#include "solvers/ColumnSolver.h"
#include "solvers/PetscKrylovSolverParameters.h"
#include "solvers/PetscKrylovSolver.h"
#include "solvers/contact/MPCSolver.h"

#include <fstream>
#include <boost/lexical_cast.hpp>

double dummyFunction(const std::vector<double> &xyz, const int dof) {
  AMP_ASSERT(xyz.size() == 3);
  double x = xyz[0], y = xyz[1], z = xyz[2];
  return (1.0 + 6.0 * x) + (2.0 - 5.0 * y) + (3.0 + 4.0 * z);
}
  

void myTest(AMP::UnitTest *ut, std::string exeName) {
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName; 

  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

#ifdef USE_SILO
  AMP::Mesh::SiloIO::shared_ptr siloWriter(new AMP::Mesh::SiloIO);
#endif

  int npes = globalComm.getSize();
  int rank = globalComm.getRank();
  std::fstream fout;
  std::string fileName = "debug_driver_" + boost::lexical_cast<std::string>(rank);
  fout.open(fileName.c_str(), std::fstream::out);

  // Load the input file
  globalComm.barrier();
  double inpReadBeginTime = MPI_Wtime();
  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);
  globalComm.barrier();
  double inpReadEndTime = MPI_Wtime();
  if(!rank) {
    std::cout<<"Finished parsing the input file in "<<(inpReadEndTime - inpReadBeginTime)<<" seconds."<<std::endl;
  }

  // Load the mesh
  globalComm.barrier();
  double meshBeginTime = MPI_Wtime();
  AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
  boost::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase("Mesh");
  boost::shared_ptr<AMP::Mesh::MeshParameters> meshParams(new AMP::Mesh::MeshParameters(mesh_db));
  meshParams->setComm(globalComm);
  AMP::Mesh::Mesh::shared_ptr meshAdapter = AMP::Mesh::Mesh::buildMesh(meshParams);
  globalComm.barrier();
  double meshEndTime = MPI_Wtime();
  if(!rank) {
    std::cout<<"Finished reading the mesh in "<<(meshEndTime - meshBeginTime)<<" seconds."<<std::endl;
  }

  // Build the contact operator
  AMP_INSIST(input_db->keyExists("ContactOperator"), "Key ''ContactOperator'' is missing!");
  boost::shared_ptr<AMP::Database> contact_db = input_db->getDatabase("ContactOperator");
  boost::shared_ptr<AMP::Operator::NodeToSegmentConstraintsOperatorParameters> 
      contactOperatorParams( new AMP::Operator::NodeToSegmentConstraintsOperatorParameters(contact_db) );

  std::vector<AMP::Mesh::MeshID> meshIDs = meshAdapter->getBaseMeshIDs();
  AMP::Mesh::MeshID masterMeshID = meshIDs[contact_db->getIntegerWithDefault("MasterMeshIndex", 0)];
  AMP::Mesh::MeshID slaveMeshID = meshIDs[contact_db->getIntegerWithDefault("SlaveMeshIndex", 1)];
  contactOperatorParams->d_MasterMeshID = masterMeshID;
  contactOperatorParams->d_SlaveMeshID = slaveMeshID;
  contactOperatorParams->d_MasterBoundaryID = contact_db->getIntegerWithDefault("MasterBoundaryID", 1);
  contactOperatorParams->d_SlaveBoundaryID = contact_db->getIntegerWithDefault("SlaveBoundaryID", 2);
  
  int dofsPerNode = 3;
  int nodalGhostWidth = 1;
  bool split = true;
  AMP::Discretization::DOFManager::shared_ptr dofManager = AMP::Discretization::simpleDOFManager::create(meshAdapter,
      AMP::Mesh::Vertex, nodalGhostWidth, dofsPerNode, split);
  contactOperatorParams->d_DOFsPerNode = dofsPerNode;
  contactOperatorParams->d_DOFManager = dofManager;

  contactOperatorParams->d_GlobalComm = globalComm;
  contactOperatorParams->d_Mesh = meshAdapter;

  boost::shared_ptr<AMP::Operator::NodeToSegmentConstraintsOperator> 
      contactOperator( new AMP::Operator::NodeToSegmentConstraintsOperator(contactOperatorParams) );

if (npes==1) {
  AMP::LinearAlgebra::Variable::shared_ptr dummyVariable(new AMP::LinearAlgebra::Variable("Dummy"));
  AMP::LinearAlgebra::Vector::shared_ptr dummyXVector = createVector(dofManager, dummyVariable, split);
  AMP::LinearAlgebra::Vector::shared_ptr dummyYVector = createVector(dofManager, dummyVariable, split);
  dummyXVector->setToScalar(1.0);
  dummyYVector->setToScalar(1.0);

  size_t slaveDOFs = dofManager->numGlobalDOF() / 4;
  std::vector<size_t> slaveIndices(slaveDOFs, slaveDOFs), masterIndices(slaveDOFs * 4, slaveDOFs * 4);
  std::vector<double> coefficients(slaveDOFs * 4 / dofsPerNode, 0.0);
  for (size_t i = 0; i < slaveIndices.size()/dofsPerNode; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      coefficients[4*i+j] = static_cast<double>(j);
    } // end for j
    for (size_t k = 0; k < dofsPerNode; ++k) {
      slaveIndices[dofsPerNode*i+k] = dofsPerNode*i+k;
      double slaveValue = 0.0;
      for (size_t j = 0; j < 4; ++j) {
        masterIndices[dofsPerNode*(4*i+j)+k] = slaveDOFs+dofsPerNode*(i+j)+k;
        slaveValue += coefficients[4*i+j] * dummyXVector->getLocalValueByGlobalID(masterIndices[dofsPerNode*(4*i+j)+k]);
      } // end for j
      dummyYVector->setLocalValueByGlobalID(slaveIndices[dofsPerNode*i+k], slaveValue);
    } // end for k
  } // end for i
  contactOperator->debugSet(slaveIndices, masterIndices, coefficients);
  contactOperator->applySolutionCorrection(dummyXVector);
  dummyXVector->subtract(dummyXVector, dummyYVector);
  std::cout<<"DEBUG SET >> APPLY >> MAX OF X-Y IS "<<dummyXVector->max()<<"  L2NORM OF X IS "<<dummyYVector->L2Norm()<<std::endl;

  dummyXVector->setToScalar(1.0);
  dummyYVector->setToScalar(1.0);
  for (size_t i = 0; i < slaveIndices.size()/dofsPerNode; ++i) {
    for (size_t k = 0; k < dofsPerNode; ++k) {
      double slaveValue = dummyXVector->getLocalValueByGlobalID(slaveIndices[dofsPerNode*4+k]);
      for (size_t j = 0; j < 4; ++j) {
        double addToMasterValue = coefficients[4*i+j] * slaveValue;
        dummyYVector->addLocalValueByGlobalID(masterIndices[dofsPerNode*(4*i+j)+k], addToMasterValue);
      } // end for j
      dummyYVector->setLocalValueByGlobalID(slaveIndices[dofsPerNode*4+k], 0.0);
    } // end for k
  } // end for i
  contactOperator->applyResidualCorrection(dummyXVector);
  dummyXVector->subtract(dummyXVector, dummyYVector);
  std::cout<<"DEBUG SET >> TRANSPOSE >> MAX OF X-Y IS "<<dummyXVector->max()<<"  L2NORM OF X IS "<<dummyYVector->L2Norm()<<std::endl;;

}

  // TODO: RESET IN CONSTRUCTOR?
  contactOperator->reset(contactOperatorParams);

if (npes==1) {
  std::vector<size_t> slaveIndices, masterIndices;
  std::vector<double> coefficients;
  contactOperator->debugGet(slaveIndices, masterIndices, coefficients);
  AMP_ASSERT( masterIndices.size() == 4 * slaveIndices.size() );
  AMP_ASSERT( masterIndices.size() == dofsPerNode * coefficients.size() );
  AMP::LinearAlgebra::Variable::shared_ptr dummyVariable(new AMP::LinearAlgebra::Variable("Dummy"));
  AMP::LinearAlgebra::Vector::shared_ptr dummyXVector = createVector(dofManager, dummyVariable, split);
  AMP::LinearAlgebra::Vector::shared_ptr dummyYVector = createVector(dofManager, dummyVariable, split);
  dummyXVector->setToScalar(1.0);
  dummyYVector->setToScalar(1.0);
  // CHECK APPLY CONSTRAINT MATRIX
  for (size_t i = 0; i < slaveIndices.size()/dofsPerNode; ++i) {
    for (size_t k = 0; k < dofsPerNode; ++k) {
      double slaveValue = 0.0;
      for (size_t j = 0; j < 4; ++j) {
        slaveValue += coefficients[4*i+j] * dummyXVector->getLocalValueByGlobalID(masterIndices[dofsPerNode*(4*i+j)+k]);
      } // end for j
      dummyYVector->setLocalValueByGlobalID(slaveIndices[dofsPerNode*i+k], slaveValue);
    } // end for k
  } // end for i
  contactOperator->applySolutionCorrection(dummyXVector);
  dummyXVector->subtract(dummyXVector, dummyYVector);
  std::cout<<"DEBUG GET >> APPLY >> MAX OF X-Y IS "<<dummyXVector->max()<<"  L2NORM OF X IS "<<dummyYVector->L2Norm()<<std::endl;
  // CHECK APPLY TRANSPOSE
  dummyXVector->setToScalar(1.0);
  dummyYVector->setToScalar(1.0);
  for (size_t i = 0; i < slaveIndices.size()/dofsPerNode; ++i) {
    for (size_t k = 0; k < dofsPerNode; ++k) {
      double slaveValue = dummyXVector->getLocalValueByGlobalID(slaveIndices[dofsPerNode*4+k]);
      for (size_t j = 0; j < 4; ++j) {
        double addToMasterValue = coefficients[4*i+j] * slaveValue;
        dummyYVector->addLocalValueByGlobalID(masterIndices[dofsPerNode*(4*i+j)+k], addToMasterValue);
      } // end for j
      dummyYVector->setLocalValueByGlobalID(slaveIndices[dofsPerNode*4+k], 0.0);
    } // end for k
  } // end for i
  contactOperator->applyResidualCorrection(dummyXVector);
  dummyXVector->subtract(dummyXVector, dummyYVector);
  std::cout<<"DEBUG GET >> TRANSPOSE >> MAX OF X-Y IS "<<dummyXVector->max()<<"  L2NORM OF X IS "<<dummyYVector->L2Norm()<<std::endl;
} 

/*  nodeToSegmentConstraintsOperator->reset(nodeToSegmentConstraintsOperatorParams);

  AMP::LinearAlgebra::Variable::shared_ptr dummyVariable(new AMP::LinearAlgebra::Variable("Dummy"));
  AMP::LinearAlgebra::Vector::shared_ptr dummyInVector = createVector(dofManager, dummyVariable, split);
  AMP::LinearAlgebra::Vector::shared_ptr dummyOutVector = createVector(dofManager, dummyVariable, split);

  nodeToSegmentConstraintsOperator->apply(dummyInVector, dummyInVector, dummyOutVector);
  nodeToSegmentConstraintsOperator->applyTranspose(dummyInVector, dummyInVector, dummyOutVector);*/


  // build a column operator and a column preconditioner
  boost::shared_ptr<AMP::Operator::OperatorParameters> emptyParams;
  boost::shared_ptr<AMP::Operator::ColumnOperator> columnOperator(new AMP::Operator::ColumnOperator(emptyParams));

  boost::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase("LinearSolver"); 
  boost::shared_ptr<AMP::Database> columnPreconditioner_db = linearSolver_db->getDatabase("Preconditioner");
  boost::shared_ptr<AMP::Solver::ColumnSolverParameters> columnPreconditionerParams(new
      AMP::Solver::ColumnSolverParameters(columnPreconditioner_db));
  columnPreconditionerParams->d_pOperator = columnOperator;
  boost::shared_ptr<AMP::Solver::ColumnSolver> columnPreconditioner(new AMP::Solver::ColumnSolver(columnPreconditionerParams));

  // build the master and slave operators
  AMP::Mesh::Mesh::shared_ptr masterMeshAdapter = meshAdapter->Subset(masterMeshID);
  if (masterMeshAdapter != NULL) {
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> masterElementPhysicsModel;
    boost::shared_ptr<AMP::Operator::LinearBVPOperator> masterOperator = boost::dynamic_pointer_cast<
        AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(masterMeshAdapter,
                                                                                         "MasterBVPOperator",
                                                                                         input_db,
                                                                                         masterElementPhysicsModel));
    columnOperator->append(masterOperator);


    boost::shared_ptr<AMP::Database> masterSolver_db = columnPreconditioner_db->getDatabase("MasterSolver"); 
    boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> masterSolverParams(new
        AMP::Solver::PetscKrylovSolverParameters(masterSolver_db));
    masterSolverParams->d_pOperator = masterOperator;
    masterSolverParams->d_comm = masterMeshAdapter->getComm();
//    masterSolverParams->d_comm = globalComm;
    boost::shared_ptr<AMP::Solver::PetscKrylovSolver> masterSolver(new AMP::Solver::PetscKrylovSolver(masterSolverParams));
    columnPreconditioner->append(masterSolver);
  }

  boost::shared_ptr<AMP::Operator::DirichletVectorCorrection> slaveLoadOperator;
  AMP::Mesh::Mesh::shared_ptr slaveMeshAdapter = meshAdapter->Subset(slaveMeshID);
  if (slaveMeshAdapter != NULL) {
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> slaveElementPhysicsModel;
    boost::shared_ptr<AMP::Operator::MechanicsLinearFEOperator> slaveOperator = boost::dynamic_pointer_cast<
        AMP::Operator::MechanicsLinearFEOperator>(AMP::Operator::OperatorBuilder::createOperator(slaveMeshAdapter,
                                                                                                 "MechanicsLinearFEOperator",
                                                                                                 input_db,
                                                                                                 slaveElementPhysicsModel));

    columnOperator->append(slaveOperator);

    boost::shared_ptr<AMP::Database> slaveSolver_db = columnPreconditioner_db->getDatabase("SlaveSolver"); 
    boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> slaveSolverParams(new
        AMP::Solver::PetscKrylovSolverParameters(slaveSolver_db));
    slaveSolverParams->d_pOperator = slaveOperator;
//    slaveSolverParams->d_comm = globalComm;
    slaveSolverParams->d_comm = slaveMeshAdapter->getComm();
    boost::shared_ptr<AMP::Solver::PetscKrylovSolver> slaveSolver(new AMP::Solver::PetscKrylovSolver(slaveSolverParams));
    columnPreconditioner->append(slaveSolver);


    slaveLoadOperator = boost::dynamic_pointer_cast<
        AMP::Operator::DirichletVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(slaveMeshAdapter, 
                                                                                                 "SlaveLoadOperator", 
                                                                                                 input_db, 
                                                                                                 slaveElementPhysicsModel));

    AMP::LinearAlgebra::Variable::shared_ptr slaveVar= slaveOperator->getOutputVariable();
    slaveLoadOperator->setVariable(slaveVar);
  }

  columnOperator->append(contactOperator);

  boost::shared_ptr<AMP::Database> contactPreconditioner_db = columnPreconditioner_db->getDatabase("ContactPreconditioner"); 
  boost::shared_ptr<AMP::Solver::MPCSolverParameters> contactPreconditionerParams(new 
      AMP::Solver::MPCSolverParameters(contactPreconditioner_db));
  contactPreconditionerParams->d_pOperator = contactOperator;
  boost::shared_ptr<AMP::Solver::MPCSolver> contactPreconditioner(new AMP::Solver::MPCSolver(contactPreconditionerParams));
  columnPreconditioner->append(contactPreconditioner);


  // Build a matrix shell operator to use the column operator with the petsc krylov solvers
  boost::shared_ptr<AMP::Database> matrixShellDatabase = input_db->getDatabase("MatrixShellOperator");
  boost::shared_ptr<AMP::Operator::OperatorParameters> matrixShellParams(new
      AMP::Operator::OperatorParameters(matrixShellDatabase));
  boost::shared_ptr<AMP::Operator::PetscMatrixShellOperator> matrixShellOperator(new
      AMP::Operator::PetscMatrixShellOperator(matrixShellParams));

  int numMasterLocalNodes = 0;
  int numSlaveLocalNodes = 0;
  if (masterMeshAdapter != NULL) { numMasterLocalNodes = masterMeshAdapter->numLocalElements(AMP::Mesh::Vertex); }
  if (slaveMeshAdapter != NULL) { numSlaveLocalNodes = slaveMeshAdapter->numLocalElements(AMP::Mesh::Vertex); }
  int matLocalSize = dofsPerNode * (numMasterLocalNodes + numSlaveLocalNodes);
  AMP_ASSERT( matLocalSize == dofManager->numLocalDOF() );
  if(!rank) {
    std::cout<<"numMasterNodes = "<<numMasterLocalNodes<<std::endl;
    std::cout<<"numSlaveNodes = "<<numSlaveLocalNodes<<std::endl;
    std::cout<<"matLocalSize = "<<matLocalSize<<std::endl;
  }
  matrixShellOperator->setComm(globalComm);
  matrixShellOperator->setMatLocalRowSize(matLocalSize);
  matrixShellOperator->setMatLocalColumnSize(matLocalSize);
  matrixShellOperator->setOperator(columnOperator); 



  AMP::LinearAlgebra::Variable::shared_ptr columnVar = columnOperator->getOutputVariable();

  AMP::LinearAlgebra::Vector::shared_ptr nullVec;
  AMP::LinearAlgebra::Vector::shared_ptr columnSolVec = createVector(dofManager, columnVar, split);
  AMP::LinearAlgebra::Vector::shared_ptr columnRhsVec = createVector(dofManager, columnVar, split);
  AMP::LinearAlgebra::Vector::shared_ptr columnResVec = createVector(dofManager, columnVar, split);

  columnSolVec->zero();
  columnRhsVec->zero();
  if (slaveLoadOperator != NULL) { slaveLoadOperator->apply(nullVec, nullVec, columnRhsVec, 1.0, 0.0); }
  double rhsNormBefore = columnRhsVec->L2Norm();
  contactOperator->applyResidualCorrection(columnRhsVec);
  double rhsNormAfter = columnRhsVec->L2Norm();

  std::cout<<"rhsNormBefore = "<<std::setprecision(15)<<rhsNormBefore
    <<" rhsNormAfter = "<<std::setprecision(15)<<rhsNormAfter<<std::endl;


  columnSolVec->setToScalar(1.0);
  contactOperator->debugSetSlaveToZero(columnSolVec);
  std::cout<<"ZEROING SLAVE USING CONTACT OP PRIOR APPLY SOL L2NORM IS "<<columnSolVec->L2Norm()<<std::endl;
  columnSolVec->setToScalar(1.0);
  if (slaveMeshAdapter != NULL) {
    AMP::Mesh::MeshIterator slaveBoundaryIDIterator = slaveMeshAdapter->getBoundaryIDIterator(AMP::Mesh::Vertex, contactOperatorParams->d_SlaveBoundaryID);
    AMP::Mesh::MeshIterator boundaryIterator = slaveBoundaryIDIterator.begin(),
        boundaryIterator_end = slaveBoundaryIDIterator.end();
    for ( ; boundaryIterator != boundaryIterator_end; ++boundaryIterator) {
      std::vector<size_t> dofs;
      dofManager->getDOFs(boundaryIterator->globalID(), dofs);
      AMP_ASSERT( dofs.size() == dofsPerNode );
      std::vector<double> zeros(dofsPerNode, 0.0);
      columnSolVec->setLocalValuesByGlobalID(dofsPerNode, &(dofs[0]), &(zeros[0]));
    } // end for
  } // end if
  std::cout<<"ZEROING SLAVE USING MESH ITERATOR PRIOR APPLY SOL L2NORM IS "<<columnSolVec->L2Norm()<<std::endl;
  columnSolVec->setToScalar(1.0);
  columnOperator->apply(columnRhsVec, columnSolVec, columnResVec, -1.0, 1.0);
  std::cout<<"APPLY COLUMN OP RES L2NORM IS "<<columnResVec->L2Norm()<<std::endl;
  std::cout<<"MESH GLOBAL NUMBER OF VERTICES IS "<<meshAdapter->numGlobalElements(AMP::Mesh::Vertex)<<std::endl;
  std::cout<<"MESH GLOBAL NUMBER OF ELEMENTS IS "<<meshAdapter->numGlobalElements(AMP::Mesh::Volume)<<std::endl;

  AMP::Mesh::MeshIterator meshIterator = meshAdapter->getIterator(AMP::Mesh::Vertex);
  for (meshIterator = meshIterator.begin(); meshIterator != meshIterator.end(); ++meshIterator) {
    std::vector<size_t> dofs;
    dofManager->getDOFs(meshIterator->globalID(), dofs);
    AMP_ASSERT( dofs.size() == dofsPerNode );
    std::vector<double> nodeCoordinates = meshIterator->coord();
    std::vector<double> dummyValues(dofsPerNode);
    for (size_t i = 0; i < dofsPerNode; ++i) {
      dummyValues[i] = dummyFunction(nodeCoordinates, i); 
    } // end for i
    columnSolVec->setLocalValuesByGlobalID(dofsPerNode, &(dofs[0]), &(dummyValues[0]));
  } // end for
  columnRhsVec->copy(columnSolVec);
  contactOperator->applySolutionCorrection(columnRhsVec);
  columnRhsVec->subtract(columnRhsVec, columnSolVec);
  std::cout<<"TESTING APPLY SOLUTION CORRECTION MAX INTERPOLATION ERROR IS "<<columnRhsVec->max()<<std::endl;

  columnSolVec->zero();
  columnRhsVec->zero();

//  contactOperator->applyResidualCorrection(columnResVec);
//  contactOperator->applySolutionConstraints(columnResVec);
//  contactOperator->getShift(columnResVec);
//  globalComm.barrier();
//  fout<<"contact op worked just fine"<<std::endl;

  boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> linearSolverParams(new
      AMP::Solver::PetscKrylovSolverParameters(linearSolver_db));
  linearSolverParams->d_pOperator = matrixShellOperator;
  linearSolverParams->d_comm = globalComm;
  linearSolverParams->d_pPreconditioner = columnPreconditioner;
  boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver(new AMP::Solver::PetscKrylovSolver(linearSolverParams));
  linearSolver->setZeroInitialGuess(true);

  linearSolver->solve(columnRhsVec, columnSolVec);


#ifdef USE_SILO
  siloWriter->registerVector(columnSolVec, meshAdapter, AMP::Mesh::Vertex, "Solution");
  char outFileName[256];
  sprintf(outFileName, "MPC_%d", 0);
  siloWriter->writeFile(outFileName, 0);
#endif
  fout.close();

  ut->passes(exeName);
}



void myTest2(AMP::UnitTest *ut, std::string exeName) {
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName; 

  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

#ifdef USE_SILO
  AMP::Mesh::SiloIO::shared_ptr siloWriter(new AMP::Mesh::SiloIO);
#endif

  int rank = globalComm.getRank();
  std::fstream fout;
  std::string fileName = "debug_driver_" + boost::lexical_cast<std::string>(rank);
  fout.open(fileName.c_str(), std::fstream::out);

  // Load the input file
  globalComm.barrier();
  double inpReadBeginTime = MPI_Wtime();
  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);
  globalComm.barrier();
  double inpReadEndTime = MPI_Wtime();
  if(!rank) {
    std::cout<<"Finished parsing the input file in "<<(inpReadEndTime - inpReadBeginTime)<<" seconds."<<std::endl;
  }

  fout<<"input file loaded"<<std::endl;
  // Load the mesh
  globalComm.barrier();
  double meshBeginTime = MPI_Wtime();
  AMP_INSIST(input_db->keyExists("FusedMesh"), "Key ''FusedMesh'' is missing!");
  boost::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase("FusedMesh");
  boost::shared_ptr<AMP::Mesh::MeshParameters> meshParams(new AMP::Mesh::MeshParameters(mesh_db));
  meshParams->setComm(globalComm);
  AMP::Mesh::Mesh::shared_ptr meshAdapter = AMP::Mesh::Mesh::buildMesh(meshParams);
  globalComm.barrier();
  double meshEndTime = MPI_Wtime();
  if(!rank) {
    std::cout<<"Finished reading the mesh in "<<(meshEndTime - meshBeginTime)<<" seconds."<<std::endl;
  }
  fout<<"mesh loaded"<<std::endl;

  int dofsPerNode = 3;
  int nodalGhostWidth = 1;
  bool split = true;
  AMP::Discretization::DOFManager::shared_ptr dofManager = AMP::Discretization::simpleDOFManager::create(meshAdapter,
      AMP::Mesh::Vertex, nodalGhostWidth, dofsPerNode, split);


  // build a column operator and a column preconditioner
  boost::shared_ptr<AMP::Operator::OperatorParameters> emptyParams;
  boost::shared_ptr<AMP::Operator::ColumnOperator> columnOperator(new AMP::Operator::ColumnOperator(emptyParams));

  boost::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase("LinearSolver"); 
  boost::shared_ptr<AMP::Database> columnPreconditioner_db = linearSolver_db->getDatabase("Preconditioner");
  boost::shared_ptr<AMP::Solver::ColumnSolverParameters> columnPreconditionerParams(new
      AMP::Solver::ColumnSolverParameters(columnPreconditioner_db));
  columnPreconditionerParams->d_pOperator = columnOperator;
  boost::shared_ptr<AMP::Solver::ColumnSolver> columnPreconditioner(new AMP::Solver::ColumnSolver(columnPreconditionerParams));

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> masterElementPhysicsModel;
  boost::shared_ptr<AMP::Operator::LinearBVPOperator> masterOperator = boost::dynamic_pointer_cast<
      AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
                                                                                       "MasterBVPOperator",
                                                                                       input_db,
                                                                                       masterElementPhysicsModel));
  columnOperator->append(masterOperator);

  boost::shared_ptr<AMP::Database> masterSolver_db = columnPreconditioner_db->getDatabase("MasterSolver"); 
  boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> masterSolverParams(new
      AMP::Solver::PetscKrylovSolverParameters(masterSolver_db));
  masterSolverParams->d_pOperator = masterOperator;
  masterSolverParams->d_comm = globalComm;
  boost::shared_ptr<AMP::Solver::PetscKrylovSolver> masterSolver(new AMP::Solver::PetscKrylovSolver(masterSolverParams));
  columnPreconditioner->append(masterSolver);

  boost::shared_ptr<AMP::Operator::DirichletVectorCorrection> slaveLoadOperator = boost::dynamic_pointer_cast<
      AMP::Operator::DirichletVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter, 
                                                                                               "SlaveLoadOperator", 
                                                                                               input_db, 
                                                                                               masterElementPhysicsModel));

  AMP::LinearAlgebra::Variable::shared_ptr slaveVar= masterOperator->getOutputVariable();
  slaveLoadOperator->setVariable(slaveVar);
  

  // Build a matrix shell operator to use the column operator with the petsc krylov solvers
  boost::shared_ptr<AMP::Database> matrixShellDatabase = input_db->getDatabase("MatrixShellOperator");
  boost::shared_ptr<AMP::Operator::OperatorParameters> matrixShellParams(new
      AMP::Operator::OperatorParameters(matrixShellDatabase));
  boost::shared_ptr<AMP::Operator::PetscMatrixShellOperator> matrixShellOperator(new
      AMP::Operator::PetscMatrixShellOperator(matrixShellParams));

  int matLocalSize = dofsPerNode * meshAdapter->numLocalElements(AMP::Mesh::Vertex);
  if(!rank) {
    std::cout<<"matLocalSize = "<<matLocalSize<<std::endl;
  }
  matrixShellOperator->setComm(globalComm);
  matrixShellOperator->setMatLocalRowSize(matLocalSize);
  matrixShellOperator->setMatLocalColumnSize(matLocalSize);
  matrixShellOperator->setOperator(columnOperator); 

  AMP::LinearAlgebra::Variable::shared_ptr columnVar = columnOperator->getOutputVariable();

  AMP::LinearAlgebra::Vector::shared_ptr nullVec;
  AMP::LinearAlgebra::Vector::shared_ptr columnSolVec = createVector(dofManager, columnVar, split);
  AMP::LinearAlgebra::Vector::shared_ptr columnRhsVec = createVector(dofManager, columnVar, split);
  AMP::LinearAlgebra::Vector::shared_ptr columnResVec = createVector(dofManager, columnVar, split);

  columnSolVec->zero();
  columnRhsVec->zero();
  slaveLoadOperator->apply(nullVec, nullVec, columnRhsVec, 1.0, 0.0);
  double fusedRhsNorm = columnRhsVec->L2Norm();
  std::cout<<"FusedRhsNorm = "<<std::setprecision(15)<<fusedRhsNorm<<std::endl;


  columnSolVec->setToScalar(1.0);
  std::cout<<"PRIOR APPLY SOL L2NORM IS "<<columnSolVec->L2Norm()<<std::endl;
  columnOperator->apply(columnRhsVec, columnSolVec, columnResVec, -1.0, 1.0);
  std::cout<<"APPLY COLUMN OP RES L2NORM IS "<<columnResVec->L2Norm()<<std::endl;
  columnSolVec->zero();
  std::cout<<"MESH GLOBAL NUMBER OF VERTICES IS "<<meshAdapter->numGlobalElements(AMP::Mesh::Vertex)<<std::endl;
  std::cout<<"MESH GLOBAL NUMBER OF ELEMENTS IS "<<meshAdapter->numGlobalElements(AMP::Mesh::Volume)<<std::endl;

  boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> linearSolverParams(new
      AMP::Solver::PetscKrylovSolverParameters(linearSolver_db));
  linearSolverParams->d_pOperator = matrixShellOperator;
  linearSolverParams->d_comm = globalComm;
  linearSolverParams->d_pPreconditioner = columnPreconditioner;
  boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver(new AMP::Solver::PetscKrylovSolver(linearSolverParams));
  linearSolver->setZeroInitialGuess(true);

  linearSolver->solve(columnRhsVec, columnSolVec);


#ifdef USE_SILO
  siloWriter->registerVector(columnSolVec, meshAdapter, AMP::Mesh::Vertex, "Solution");
  char outFileName[256];
  sprintf(outFileName, "MPC_%d", 0);
  siloWriter->writeFile(outFileName, 0);
#endif
  fout.close();

  ut->passes(exeName);
}

int main(int argc, char *argv[])
{
  AMP::AMPManager::startup(argc, argv);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);
//  boost::shared_ptr<AMP::Mesh::initializeLibMesh> libmeshInit( new AMP::Mesh::initializeLibMesh(globalComm) );
  AMP::UnitTest ut;

  std::string exeName = "testNodeToSegmentConstraintsOperator";

  try {
    myTest(&ut, exeName);
    myTest2(&ut, exeName);
  } catch (std::exception &err) {
    std::cout << "ERROR: While testing "<<argv[0] << err.what() << std::endl;
    ut.failure("ERROR: While testing");
  } catch( ... ) {
    std::cout << "ERROR: While testing "<<argv[0] << "An unknown exception was thrown." << std::endl;
    ut.failure("ERROR: While testing");
  }

  ut.report();
  int num_failed = ut.NumFailGlobal();

//  libmeshInit.reset();
  AMP::AMPManager::shutdown();
  return num_failed;
}  



