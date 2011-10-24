
#include "utils/InputManager.h"
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/WriteSolutionToFile.h"

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "ampmesh/MeshManager.h"
#include "ampmesh/MeshAdapter.h"

#include "operators/LinearBVPOperator.h"
#include "operators/OperatorBuilder.h"
#include "operators/boundary/DirichletVectorCorrection.h"

#include "PetscKrylovSolverParameters.h"
#include "PetscKrylovSolver.h"
#include "TrilinosMLSolver.h"

#include "boost/shared_ptr.hpp"

void myTest(AMP::UnitTest *ut, std::string exeName) {
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName;

  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

  //Read the input file
  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);

  //Read the mesh
  AMP::Mesh::MeshManagerParameters::shared_ptr  meshmgrParams ( new AMP::Mesh::MeshManagerParameters ( input_db ) );
  AMP::Mesh::MeshManager::shared_ptr  manager ( new AMP::Mesh::MeshManager ( meshmgrParams ) );
  AMP::Mesh::MeshManager::Adapter::shared_ptr meshAdapter = manager->getMesh ( "mesh" );

  std::cout<<"Mesh has "<<(meshAdapter->numLocalNodes())<<" nodes."<<std::endl;

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

  AMP::LinearAlgebra::Vector::shared_ptr mechSolVec = meshAdapter->createVector( bvpOperator->getOutputVariable() );
  AMP::LinearAlgebra::Vector::shared_ptr mechRhsVec = meshAdapter->createVector( bvpOperator->getOutputVariable() );
  AMP::LinearAlgebra::Vector::shared_ptr mechResVec = meshAdapter->createVector( bvpOperator->getOutputVariable() );

  mechRhsVec->zero();
  mechResVec->zero();

  dirichletVecOp->apply(nullVec, nullVec, mechRhsVec, 1.0, 0.0);

  for(int type = 1; type < 4; type++) {
    if(type == 0) {
      std::cout<<"Solving using CG algorithm (Own Implementation)..."<<std::endl;

      boost::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase("CGsolver"); 

      int maxIters = linearSolver_db->getInteger("max_iterations");

      AMP::LinearAlgebra::Vector::shared_ptr matOutVec = mechSolVec->cloneVector();
      AMP::LinearAlgebra::Vector::shared_ptr pVec = mechSolVec->cloneVector();

      mechSolVec->zero();

      bvpOperator->apply(nullVec, mechSolVec, matOutVec, 1.0, 0.0);

      mechResVec->subtract(mechRhsVec, matOutVec);

      pVec->copyVector(mechResVec);

      for(int iter = 0; iter <= maxIters; iter++) {
        double resNorm = mechResVec->L2Norm();
        std::cout<<"Iter = "<<iter<<" ResNorm2 = "<<std::setprecision(15)<<resNorm<<std::endl;

        bvpOperator->apply(nullVec, pVec, matOutVec, 1.0, 0.0);

        double matOutNorm = matOutVec->L2Norm();
        std::cout<<"CG-Iter = "<<iter<<" MatOutNorm2 = "<<std::setprecision(15)<<matOutNorm<<std::endl;

        double resOldDot = mechResVec->dot(mechResVec);

        double alphaDenom = matOutVec->dot(pVec);

        double alpha = resOldDot/alphaDenom;

        mechSolVec->axpy(alpha, pVec, mechSolVec);

        mechResVec->axpy(-alpha, matOutVec, mechResVec);

        double resNewDot = mechResVec->dot(mechResVec);

        double beta = resNewDot/resOldDot;

        std::cout<<"Iter = "<<iter
          <<" resOldDot = "<<std::setprecision(15)<<resOldDot
          <<" alphaDenom = "<<std::setprecision(15)<<alphaDenom
          <<" alpha = "<<std::setprecision(15)<<alpha
          <<" resNewDot = "<<std::setprecision(15)<<resNewDot
          <<" beta = "<<std::setprecision(15)<<beta
          <<std::endl<<std::endl;

        pVec->axpy(beta, pVec, mechResVec);
      }

      std::cout<<std::endl<<std::endl;
    } else if(type == 1) {
      std::cout<<"Solving using CG algorithm (Petsc Implementation)..."<<std::endl;

      boost::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase("CGsolver"); 

      // initialize the linear solver
      boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> linearSolverParams(new
          AMP::Solver::PetscKrylovSolverParameters(linearSolver_db));
      linearSolverParams->d_pOperator = bvpOperator;
      linearSolverParams->d_comm = globalComm;
      boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver(new AMP::Solver::PetscKrylovSolver(linearSolverParams));

      linearSolver->solve(mechRhsVec, mechSolVec);

      std::cout<<std::endl<<std::endl;
    } else if(type == 2) {
      std::cout<<"Solving using Jacobi preconditioned CG algorithm..."<<std::endl;

      boost::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase("JacobiCGsolver"); 

      // initialize the linear solver
      boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> linearSolverParams(new
          AMP::Solver::PetscKrylovSolverParameters(linearSolver_db));
      linearSolverParams->d_pOperator = bvpOperator;
      linearSolverParams->d_comm = globalComm;
      boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver(new AMP::Solver::PetscKrylovSolver(linearSolverParams));

      linearSolver->solve(mechRhsVec, mechSolVec);

      std::cout<<std::endl<<std::endl;
    } else {
      std::cout<<"Solving using ML preconditioned CG algorithm..."<<std::endl;

      boost::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase("MLCGsolver"); 

      // ---- first initialize the preconditioner
      boost::shared_ptr<AMP::Database> pcSolver_db = linearSolver_db->getDatabase("MLsolver"); 
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

      linearSolver->solve(mechRhsVec, mechSolVec);

      std::cout<<std::endl<<std::endl;
    }
  }

  ut->passes(exeName);
}


int main(int argc, char *argv[])
{
  AMP::AMPManager::startup(argc, argv);
  AMP::UnitTest ut;

  std::string exeName = "testJacobiVsML";

  try {
    myTest(&ut, exeName);
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
