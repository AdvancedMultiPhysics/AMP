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


#include "ampmesh/MeshVariable.h"
#include "ampmesh/SiloIO.h"


#include "operators/mechanics/ThermalVonMisesMatModel.h"
#include "operators/mechanics/MechanicsNonlinearElement.h"
#include "operators/mechanics/MechanicsLinearElement.h"
#include "operators/mechanics/MechanicsLinearFEOperator.h"
#include "operators/mechanics/MechanicsNonlinearFEOperator.h"

#include "operators/boundary/DirichletMatrixCorrection.h"
#include "operators/boundary/DirichletVectorCorrection.h"

#include "operators/BVPOperatorParameters.h"
#include "operators/LinearBVPOperator.h"
#include "operators/NonlinearBVPOperator.h"
#include "operators/OperatorBuilder.h"

#include "../PetscKrylovSolverParameters.h"
#include "../PetscKrylovSolver.h"
#include "../PetscSNESSolverParameters.h"
#include "../PetscSNESSolver.h"

#include "../TrilinosMLSolver.h"



void myTest(AMP::UnitTest *ut)
{
  //std::string exeName("testPetscSNESSolver-NonlinearMechanics-2_COMPARISON");
  //std::string exeName("testPetscSNESSolver-NonlinearMechanics-2_COMPARISON-1");
  //std::string exeName("testPetscSNESSolver-NonlinearMechanics-2_COMPARISON-2");
  std::string exeName("testPetscSNESSolver-NonlinearMechanics-2_COMPARISON-3");
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName;

  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);

  AMP::Mesh::MeshManagerParameters::shared_ptr  meshmgrParams ( new AMP::Mesh::MeshManagerParameters ( input_db ) );
  AMP::Mesh::MeshManager::shared_ptr  manager ( new AMP::Mesh::MeshManager ( meshmgrParams ) );
  AMP::Mesh::MeshManager::Adapter::shared_ptr meshAdapter = manager->getMesh ( "brick" );

  AMP_INSIST(input_db->keyExists("NumberOfLoadingSteps"), "Key ''NumberOfLoadingSteps'' is missing!");
  int NumberOfLoadingSteps = input_db->getInteger("NumberOfLoadingSteps");

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel;
  boost::shared_ptr<AMP::Operator::NonlinearBVPOperator> nonlinBvpOperator = 
    boost::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
														    "nonlinearMechanicsBVPOperator",
														    input_db,
														    elementPhysicsModel));

  AMP::LinearAlgebra::Variable::shared_ptr displacementVariable = boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
      nonlinBvpOperator->getVolumeOperator())->getInputVariable(AMP::Operator::Mechanics::DISPLACEMENT); 
  AMP::LinearAlgebra::Variable::shared_ptr temperatureVariable = boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
      nonlinBvpOperator->getVolumeOperator())->getInputVariable(AMP::Operator::Mechanics::TEMPERATURE); 
  AMP::LinearAlgebra::Variable::shared_ptr residualVariable = nonlinBvpOperator->getOutputVariable();

  AMP::LinearAlgebra::Vector::shared_ptr initTempVec = meshAdapter->createVector( temperatureVariable );
  AMP::LinearAlgebra::Vector::shared_ptr finalTempVec = meshAdapter->createVector( temperatureVariable );

  double Temp_0 = 400.0;
  double Temp_1 = 2000.0;
  initTempVec->setToScalar(Temp_0);
  initTempVec->abs ( initTempVec );
  double initTempConst = input_db->getDoubleWithDefault("INIT_TEMP_CONST", 1.0);
  initTempVec->scale(initTempConst);

  bool setFinalTempEqualsInitialTemp = input_db->getBoolWithDefault("SET_FINAL_TEMP_EQUALS_INIT_TEMP", false);

  if(setFinalTempEqualsInitialTemp) {
    finalTempVec->copyVector(initTempVec);
  } else {
    double Temp_n = Temp_0 + ((Temp_1 - Temp_0) / ((double)(NumberOfLoadingSteps)));
    AMP::pout << "Temp_n = " << Temp_n << std::endl;
    finalTempVec->setToScalar(Temp_n);
    double finalTempConst = input_db->getDoubleWithDefault("FINAL_TEMP_CONST", 1.0);
    finalTempVec->scale(finalTempConst);
  }

  //  manager->registerVectorAsData( initTempVec, "InitialTemperature" );
  // manager->registerVectorAsData( finalTempVec, "FinalTemperature" );

  (boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->
                                                                  getVolumeOperator()))->setReferenceTemperature(initTempVec);
  (boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->
                                                                  getVolumeOperator()))->setVector(AMP::Operator::Mechanics::TEMPERATURE, finalTempVec); 
  (boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->
                                                                  getVolumeOperator()))->init();

  boost::shared_ptr<AMP::Operator::LinearBVPOperator> linBvpOperator =
    boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
														 "linearMechanicsBVPOperator",
														 input_db,
														 elementPhysicsModel));

  //For RHS (Point Forces)
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> dummyModel;
  boost::shared_ptr<AMP::Operator::DirichletVectorCorrection> dirichletLoadVecOp =
    boost::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
															 "Load_Boundary",
															 input_db,
															 dummyModel));
  dirichletLoadVecOp->setVariable(residualVariable);

  //For Initial-Guess
  boost::shared_ptr<AMP::Operator::DirichletVectorCorrection> dirichletDispInVecOp =
    boost::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
															 "Displacement_Boundary",
															 input_db,
															 dummyModel));
  dirichletDispInVecOp->setVariable(displacementVariable);

  AMP::LinearAlgebra::Vector::shared_ptr nullVec;

  AMP::LinearAlgebra::Vector::shared_ptr mechNlSolVec = meshAdapter->createVector( displacementVariable );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlRhsVec = meshAdapter->createVector( residualVariable );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlResVec = meshAdapter->createVector( residualVariable );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlScaledRhsVec = meshAdapter->createVector( residualVariable );

  manager->registerVectorAsData ( mechNlSolVec, "MechanicsSolution" );
  //  manager->registerVectorAsData ( mechNlRhsVec, "RHS" );
  //  manager->registerVectorAsData ( mechNlResVec, "Residual" );

  //Initial guess for NL solver must satisfy the displacement boundary conditions
  mechNlSolVec->setToScalar(0.0);
  dirichletDispInVecOp->apply(nullVec, nullVec, mechNlSolVec, 1.0, 0.0);

  nonlinBvpOperator->apply(nullVec, mechNlSolVec, mechNlResVec, 1.0, 0.0);
  linBvpOperator->reset(nonlinBvpOperator->getJacobianParameters(mechNlSolVec));

  mechNlRhsVec->setToScalar(0.0);
  dirichletLoadVecOp->apply(nullVec, nullVec, mechNlRhsVec, 1.0, 0.0);

  double initSolNorm = mechNlSolVec->L2Norm();

  std::cout<<"Initial Solution Norm: "<<initSolNorm<<std::endl;

  boost::shared_ptr<AMP::Database> nonlinearSolver_db = input_db->getDatabase("NonlinearSolver"); 

  boost::shared_ptr<AMP::Database> linearSolver_db = nonlinearSolver_db->getDatabase("LinearSolver"); 

  // ---- first initialize the preconditioner
  boost::shared_ptr<AMP::Database> pcSolver_db = linearSolver_db->getDatabase("Preconditioner"); 
  boost::shared_ptr<AMP::Solver::SolverStrategyParameters> pcSolverParams(new AMP::Solver::SolverStrategyParameters(pcSolver_db));
  pcSolverParams->d_pOperator = linBvpOperator;
  boost::shared_ptr<AMP::Solver::TrilinosMLSolver> pcSolver(new AMP::Solver::TrilinosMLSolver(pcSolverParams));

  //HACK to prevent a double delete on Petsc Vec
  boost::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver;

  // initialize the linear solver
  boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> linearSolverParams(new
      AMP::Solver::PetscKrylovSolverParameters(linearSolver_db));

  linearSolverParams->d_pOperator = linBvpOperator;
  linearSolverParams->d_comm = globalComm;
  linearSolverParams->d_pPreconditioner = pcSolver;

  boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver(new AMP::Solver::PetscKrylovSolver(linearSolverParams));

  boost::shared_ptr<AMP::Solver::PetscSNESSolverParameters> nonlinearSolverParams(new
      AMP::Solver::PetscSNESSolverParameters(nonlinearSolver_db));

  // change the next line to get the correct communicator out
  nonlinearSolverParams->d_comm = globalComm;
  nonlinearSolverParams->d_pOperator = nonlinBvpOperator;
  nonlinearSolverParams->d_pKrylovSolver = linearSolver;
  nonlinearSolverParams->d_pInitialGuess = mechNlSolVec;

  nonlinearSolver.reset(new AMP::Solver::PetscSNESSolver(nonlinearSolverParams));

  nonlinearSolver->setZeroInitialGuess(false);

  for (int step=0;step<NumberOfLoadingSteps; step++)
  {
    AMP::pout << "########################################" << std::endl;
    AMP::pout << "The current loading step is " << (step+1) << std::endl;

    if(step > 0) {
      double Temp_n = Temp_0 + (((double)(step + 1)) * ((Temp_1 - Temp_0) / ((double)(NumberOfLoadingSteps))));
      AMP::pout << "Temp_n = " << Temp_n << std::endl;
      finalTempVec->setToScalar(Temp_n);
      (boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->
                                                                      getVolumeOperator()))->setVector(AMP::Operator::Mechanics::TEMPERATURE, finalTempVec);
    }

    double scaleValue  = ((double)step+1.0)/NumberOfLoadingSteps;
    mechNlScaledRhsVec->scale(scaleValue, mechNlRhsVec);
    AMP::pout << "L2 Norm at loading step " << (step+1) << " is " << mechNlScaledRhsVec->L2Norm() << std::endl;

    nonlinBvpOperator->apply(mechNlScaledRhsVec, mechNlSolVec, mechNlResVec, 1.0, -1.0);
    double initialResidualNorm  = mechNlResVec->L2Norm();
    AMP::pout<<"Initial Residual Norm for loading step "<<(step+1)<<" is "<<initialResidualNorm<<std::endl;

    if(initialResidualNorm < 1.0e-2) {
      ut->passes("Nonlinear solve for current loading step");
    } else {
      AMP::pout<<"Starting Nonlinear Solve..."<<std::endl;
      nonlinearSolver->solve(mechNlScaledRhsVec, mechNlSolVec);

      nonlinBvpOperator->apply(mechNlScaledRhsVec, mechNlSolVec, mechNlResVec, 1.0, -1.0);
      double finalResidualNorm  = mechNlResVec->L2Norm();
      AMP::pout<<"Final Residual Norm for loading step "<<(step+1)<<" is "<<finalResidualNorm<<std::endl;
      AMP::pout<<"Maxx value in the final sol for step "<<(step+1)<<": "<<mechNlSolVec->max()<<std::endl;

      if( finalResidualNorm > (1.0e-8*initialResidualNorm) ) {
        ut->failure("Nonlinear solve for current loading step");
      } else {
        ut->passes("Nonlinear solve for current loading step");
      }
    }

    AMP::LinearAlgebra::Vector::shared_ptr mechUvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride("U", 0, 3) , "U" );
    AMP::LinearAlgebra::Vector::shared_ptr mechVvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride("V", 1, 3) , "V" );
    AMP::LinearAlgebra::Vector::shared_ptr mechWvec = mechNlSolVec->select( AMP::LinearAlgebra::VS_Stride("W", 2, 3) , "W" );

    double finalMaxU = mechUvec->maxNorm();
    double finalMaxV = mechVvec->maxNorm();
    double finalMaxW = mechWvec->maxNorm();

    AMP::pout<<"Maximum U displacement: "<<finalMaxU<<std::endl;
    AMP::pout<<"Maximum V displacement: "<<finalMaxV<<std::endl;
    AMP::pout<<"Maximum W displacement: "<<finalMaxW<<std::endl;

    boost::shared_ptr<AMP::InputDatabase> tmp_db (new AMP::InputDatabase("Dummy"));
    boost::shared_ptr<AMP::Operator::MechanicsNonlinearFEOperatorParameters> tmpParams(new
        AMP::Operator::MechanicsNonlinearFEOperatorParameters(tmp_db));
    (nonlinBvpOperator->getVolumeOperator())->reset(tmpParams);
    nonlinearSolver->setZeroInitialGuess(false);
  }

  double finalSolNorm = mechNlSolVec->L2Norm();
  AMP::pout<<"Final Solution Norm: "<<finalSolNorm<<std::endl;
  AMP::pout<<"Maxx value in the final sol: "<<mechNlSolVec->max()<<std::endl;

#ifdef USE_SILO
  manager->writeFile<AMP::Mesh::SiloIO> ( exeName , 1 );
#endif

  ut->passes(exeName);

}

int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    try {
        myTest(&ut);
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


