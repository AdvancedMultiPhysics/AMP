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
#include "ampmesh/MeshOperations.h"
#include "ampmesh/MeshUtils.h"
#include "ampmesh/SiloIO.h"


#include "operators/ThermalVonMisesMatModel.h"
#include "operators/MechanicsNonlinearElement.h"
#include "operators/MechanicsLinearElement.h"
#include "operators/MechanicsLinearFEOperator.h"
#include "operators/MechanicsNonlinearFEOperator.h"
#include "operators/DirichletMatrixCorrection.h"
#include "operators/DirichletVectorCorrection.h"
#include "operators/BVPOperatorParameters.h"
#include "operators/LinearBVPOperator.h"
#include "operators/NonlinearBVPOperator.h"
#include "operators/OperatorBuilder.h"
#include "operators/diffusion/DiffusionLinearFEOperator.h"
#include "operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "operators/VolumeIntegralOperator.h"
#include "operators/NeutronicsRhs.h"

#include "../PetscKrylovSolverParameters.h"
#include "../PetscKrylovSolver.h"
#include "../PetscSNESSolverParameters.h"
#include "../PetscSNESSolver.h"

#include "../TrilinosMLSolver.h"



boost::shared_ptr<AMP::Solver::PetscSNESSolver> 
       buildMechanicsSolver ( boost::shared_ptr<AMP::Database>  input_db , 
                              AMP::Mesh::MeshManager::Adapter::shared_ptr meshAdapter , 
                              AMP::LinearAlgebra::Vector::shared_ptr  referenceTemperatureVec ,
                              AMP::LinearAlgebra::Vector::shared_ptr  temperatureVec , 
                              AMP::LinearAlgebra::Vector::shared_ptr &mechNlSolVec ,
                              AMP::LinearAlgebra::Vector::shared_ptr &mechNlRhsVec , 
                              AMP::LinearAlgebra::Vector::shared_ptr &mechNlResVec ,
                              boost::shared_ptr<AMP::Operator::Operator>  &mechVolOp )
{
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel;
  boost::shared_ptr<AMP::Database> nonlinOpDatabase = input_db->getDatabase("nonlinearMechanicsBVPOperator");
  boost::shared_ptr<AMP::Operator::NonlinearBVPOperator> nonlinBvpOperator = 
    boost::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter, 
          nonlinOpDatabase, elementPhysicsModel));

  AMP::LinearAlgebra::Variable::shared_ptr displacementVariable = boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
      nonlinBvpOperator->getVolumeOperator())->getInputVariable(AMP::Operator::Mechanics::DISPLACEMENT); 

  AMP::LinearAlgebra::Variable::shared_ptr residualVariable = nonlinBvpOperator->getOutputVariable();


  (boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->
                                                                  getVolumeOperator()))->setReferenceTemperature(referenceTemperatureVec);
  (boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->
                                                                  getVolumeOperator()))->setVector(AMP::Operator::Mechanics::TEMPERATURE, temperatureVec); 
  (boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->
                                                                  getVolumeOperator()))->init();

  boost::shared_ptr<AMP::Database> linOpDatabase = input_db->getDatabase("linearMechanicsBVPOperator");
  boost::shared_ptr<AMP::Operator::LinearBVPOperator> linBvpOperator =
    boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
          linOpDatabase, elementPhysicsModel));

  //For RHS (Point Forces)
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> dummyModel;
  boost::shared_ptr<AMP::Database> load_db = input_db->getDatabase("Load_Boundary");
  boost::shared_ptr<AMP::Operator::DirichletVectorCorrection> dirichletLoadVecOp =
    boost::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
          load_db, dummyModel));
  dirichletLoadVecOp->setVariable(residualVariable);

  //For Initial-Guess
  boost::shared_ptr<AMP::Database> disp_db = input_db->getDatabase("Displacement_Boundary");
  boost::shared_ptr<AMP::Operator::DirichletVectorCorrection> dirichletDispInVecOp =
    boost::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
          disp_db, dummyModel));
  dirichletDispInVecOp->setVariable(displacementVariable);

  AMP::LinearAlgebra::Vector::shared_ptr nullVec;

  mechNlSolVec = meshAdapter->createVector( displacementVariable );
  mechNlRhsVec = meshAdapter->createVector( residualVariable );
  mechNlResVec = meshAdapter->createVector( residualVariable );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlScaledRhsVec = meshAdapter->createVector( residualVariable );

  //Initial guess for NL solver must satisfy the displacement boundary conditions
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
  mechVolOp = nonlinBvpOperator->getVolumeOperator();
  return nonlinearSolver;
}


boost::shared_ptr<AMP::Solver::PetscSNESSolver> 
       buildThermalSolver( boost::shared_ptr<AMP::Database> input_db ,
                           AMP::Mesh::MeshManager::Adapter::shared_ptr meshAdapter ,
                           AMP::LinearAlgebra::Vector::shared_ptr &solVec , 
                           AMP::LinearAlgebra::Vector::shared_ptr &rhsVec ,
                           AMP::LinearAlgebra::Vector::shared_ptr &resVec ,
                           boost::shared_ptr<AMP::Operator::Operator>  &thermVolOp )
{

  //----------------------------------------------------------------------------------------------------------------------------------------------//
  // create a nonlinear BVP operator for nonlinear thermal diffusion
  AMP_INSIST( input_db->keyExists("testNonlinearThermalOperator"), "key missing!" );

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> thermalTransportModel;
  boost::shared_ptr<AMP::Database> nonlinearThermalDatabase = input_db->getDatabase("testNonlinearThermalOperator");
  boost::shared_ptr<AMP::Operator::NonlinearBVPOperator> nonlinearThermalOperator = boost::dynamic_pointer_cast<
    AMP::Operator::NonlinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter, nonlinearThermalDatabase, thermalTransportModel));

  //----------------------------------------------------------------------------------------------------------------------------------------------//
  // initialize the input variable
  boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> thermalVolumeOperator =
		  boost::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(nonlinearThermalOperator->getVolumeOperator());

  boost::shared_ptr<AMP::LinearAlgebra::Variable> thermalVariable = thermalVolumeOperator->getOutputVariable();

  // create solution, rhs, and residual vectors
  solVec = meshAdapter->createVector( thermalVariable );
  rhsVec = meshAdapter->createVector( thermalVariable );
  resVec = meshAdapter->createVector( thermalVariable );

  // create the following shared pointers for ease of use
  AMP::LinearAlgebra::Vector::shared_ptr nullVec;

  //----------------------------------------------------------------------------------------------------------------------------------------------//
  // now construct the linear BVP operator for thermal
  AMP_INSIST( input_db->keyExists("testLinearThermalOperator"), "key missing!" );
  boost::shared_ptr<AMP::Database> linearThermalDatabase = input_db->getDatabase("testLinearThermalOperator");
  boost::shared_ptr<AMP::Operator::LinearBVPOperator> linearThermalOperator = boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
      AMP::Operator::OperatorBuilder::createOperator(meshAdapter, linearThermalDatabase, thermalTransportModel));

  //----------------------------------------------------------------------------------------------------------------------------------------------//
  //Initial guess

  solVec->setToScalar(400.);
  double initialGuessNorm  = solVec->L2Norm();
  std::cout << "initial guess norm = " << initialGuessNorm <<"\n";

  nonlinearThermalOperator->modifyInitialSolutionVector(solVec);

  initialGuessNorm  = solVec->L2Norm();
  std::cout << "initial guess norm  after apply = " << initialGuessNorm <<"\n";

////////////////////////////////////
//  CREATE THE NEUTRONICS SOURCE  //
////////////////////////////////////
  AMP_INSIST(input_db->keyExists("NeutronicsOperator"), "Key ''NeutronicsOperator'' is missing!");
  boost::shared_ptr<AMP::Database>  neutronicsOp_db = input_db->getDatabase("NeutronicsOperator");
  boost::shared_ptr<AMP::Operator::NeutronicsRhsParameters> neutronicsParams(new AMP::Operator::NeutronicsRhsParameters( neutronicsOp_db ));
  neutronicsParams->d_MeshAdapter = meshAdapter;
  boost::shared_ptr<AMP::Operator::NeutronicsRhs> neutronicsOperator(new AMP::Operator::NeutronicsRhs( neutronicsParams ));

  AMP::LinearAlgebra::Variable::shared_ptr SpecificPowerVar = neutronicsOperator->getOutputVariable();
  AMP::LinearAlgebra::Vector::shared_ptr   SpecificPowerVec = meshAdapter->createVector( SpecificPowerVar );

  neutronicsOperator->apply(nullVec, nullVec, SpecificPowerVec, 1., 0.);

/////////////////////////////////////////////////////
//  Integrate Nuclear Rhs over Desnity * Volume //
/////////////////////////////////////////////////////

  AMP_INSIST( input_db->keyExists("VolumeIntegralOperator"), "key missing!" );

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> stransportModel;
  boost::shared_ptr<AMP::Database> sourceDatabase = input_db->getDatabase("VolumeIntegralOperator");
  boost::shared_ptr<AMP::Operator::VolumeIntegralOperator> sourceOperator = boost::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter, sourceDatabase, stransportModel));

  // Create the power (heat source) vector.
  AMP::LinearAlgebra::Variable::shared_ptr PowerInWattsVar = sourceOperator->getOutputVariable();
  AMP::LinearAlgebra::Vector::shared_ptr   PowerInWattsVec = meshAdapter->createVector( PowerInWattsVar );
  PowerInWattsVec->zero();

  // convert the vector of specific power to power for a given basis.
  sourceOperator->apply(nullVec, SpecificPowerVec, PowerInWattsVec, 1., 0.);

  rhsVec->copyVector(PowerInWattsVec);

  nonlinearThermalOperator->modifyRHSvector(rhsVec);

  double rhsNorm  = rhsVec->L2Norm();
  std::cout << "rhs norm  after apply = " << rhsNorm <<"\n";

  //----------------------------------------------------------------------------------------------------------------------------------------------/

  boost::shared_ptr<AMP::Database> nonlinearSolver_db = input_db->getDatabase("NonlinearSolver"); 
  boost::shared_ptr<AMP::Database> linearSolver_db = nonlinearSolver_db->getDatabase("LinearSolver"); 

  //----------------------------------------------------------------------------------------------------------------------------------------------//
  // initialize the nonlinear solver
  boost::shared_ptr<AMP::Solver::PetscSNESSolverParameters> nonlinearSolverParams(new
      AMP::Solver::PetscSNESSolverParameters(nonlinearSolver_db));

  // change the next line to get the correct communicator out
  nonlinearSolverParams->d_comm = globalComm;
  nonlinearSolverParams->d_pOperator = nonlinearThermalOperator;
  nonlinearSolverParams->d_pInitialGuess = solVec;

  boost::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver(new AMP::Solver::PetscSNESSolver(nonlinearSolverParams));

  //----------------------------------------------------------------------------------------------------------------------------------------------//
  boost::shared_ptr<AMP::Database> thermalPreconditioner_db = linearSolver_db->getDatabase("Preconditioner");
  boost::shared_ptr<AMP::Solver::SolverStrategyParameters> thermalPreconditionerParams(new AMP::Solver::SolverStrategyParameters(thermalPreconditioner_db));
  thermalPreconditionerParams->d_pOperator = linearThermalOperator;
  boost::shared_ptr<AMP::Solver::TrilinosMLSolver> linearThermalPreconditioner(new AMP::Solver::TrilinosMLSolver(thermalPreconditionerParams));

  //----------------------------------------------------------------------------------------------------------------------------------------------//
  // register the preconditioner with the Jacobian free Krylov solver
  boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver = nonlinearSolver->getKrylovSolver();

  linearSolver->setPreconditioner(linearThermalPreconditioner);

  nonlinearThermalOperator->apply(rhsVec, solVec, resVec, 1.0, -1.0);
  double initialResidualNorm  = resVec->L2Norm();

  AMP::pout<<"Initial Residual Norm: "<<initialResidualNorm<<std::endl;

  nonlinearSolver->setZeroInitialGuess(false);

  return nonlinearSolver;
}

/*
void myTest ( AMP::UnitTest *ut )
{
  for (int step=0;step<NumberOfLoadingSteps; step++)
  {
    AMP::pout << "########################################" << std::endl;
    AMP::pout << "The current loading step is " << (step+1) << std::endl;

    double scaleValue  = ((double)step+1.0)/NumberOfLoadingSteps;
    mechNlScaledRhsVec->scale(scaleValue, mechNlRhsVec);
    AMP::pout << "L2 Norm at loading step " << (step+1) << " is " << mechNlScaledRhsVec->L2Norm() << std::endl;

    nonlinBvpOperator->apply(mechNlScaledRhsVec, mechNlSolVec, mechNlResVec, 1.0, -1.0);
    double initialResidualNorm  = mechNlResVec->L2Norm();
    AMP::pout<<"Initial Residual Norm for loading step "<<(step+1)<<" is "<<initialResidualNorm<<std::endl;

    if(initialResidualNorm < 1.0e-2) {
      ut.passes("Nonlinear solve for current loading step");
    }    else {
      AMP::pout<<"Starting Nonlinear Solve..."<<std::endl;
      nonlinearSolver->solve(mechNlScaledRhsVec, mechNlSolVec);

      nonlinBvpOperator->apply(mechNlScaledRhsVec, mechNlSolVec, mechNlResVec, 1.0, -1.0);
      double finalResidualNorm  = mechNlResVec->L2Norm();
      AMP::pout<<"Final Residual Norm for loading step "<<(step+1)<<" is "<<finalResidualNorm<<std::endl;

      if( finalResidualNorm > (1.0e-8*initialResidualNorm) ) {
        ut.numFails++;
        ut.failure("Nonlinear solve for current loading step");
      } else {
        ut.passes("Nonlinear solve for current loading step");
      }
    }

    boost::shared_ptr<AMP::InputDatabase> tmp_db (new AMP::InputDatabase("Dummy"));
    boost::shared_ptr<AMP::Operator::MechanicsNonlinearFEOperatorParameters> tmpParams(new
        AMP::Operator::MechanicsNonlinearFEOperatorParameters(tmp_db));
    (nonlinBvpOperator->getVolumeOperator())->reset(tmpParams);
    nonlinearSolver->setZeroInitialGuess(false);
  }

  double finalSolNorm = mechNlSolVec->L2Norm();
  AMP::pout<<"Final Solution Norm: "<<finalSolNorm<<std::endl;

#ifdef USE_SILO
  // manager->writeFile<AMP::Mesh::SiloIO> ( "FrozenTemp_NonlinearMechExample" , 1 );
#endif

  ut.passes(exeName);

  AMP::AMPManager::shutdown();

}
*/


void  myTest ( AMP::UnitTest *ut )
{
  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile("OperatorSplit", input_db);
  input_db->printClassData(AMP::plog);

  AMP::Mesh::MeshManagerParameters::shared_ptr  meshmgrParams ( new AMP::Mesh::MeshManagerParameters ( input_db ) );
  AMP::Mesh::MeshManager::shared_ptr  manager ( new AMP::Mesh::MeshManager ( meshmgrParams ) );
  AMP::Mesh::MeshManager::Adapter::shared_ptr meshAdapter = *(manager->beginMeshes()); 

  boost::shared_ptr<AMP::Solver::SolverStrategy>  mechanicsSolver;
  boost::shared_ptr<AMP::Solver::SolverStrategy>  thermalSolver;
  boost::shared_ptr<AMP::Operator::Operator>        mechanicsVolumeOperator;
  boost::shared_ptr<AMP::Operator::Operator>        thermalVolumeOperator;
  AMP::LinearAlgebra::Vector::shared_ptr   mechanicsSolution;
  AMP::LinearAlgebra::Vector::shared_ptr   mechanicsSolutionLast;
  AMP::LinearAlgebra::Vector::shared_ptr   mechanicsResidual;
  AMP::LinearAlgebra::Vector::shared_ptr   mechanicsRHS;
  AMP::LinearAlgebra::Vector::shared_ptr   mechanicsScaledRHS;
  AMP::LinearAlgebra::Vector::shared_ptr   thermalSolution;
  AMP::LinearAlgebra::Vector::shared_ptr   thermalSolutionLast;
  AMP::LinearAlgebra::Vector::shared_ptr   thermalResidual;
  AMP::LinearAlgebra::Vector::shared_ptr   thermalRHS;
  AMP::LinearAlgebra::Vector::shared_ptr   thermalReference;

  thermalSolver = buildThermalSolver ( input_db->getDatabase ( "ThermalDB" ) ,
                                       meshAdapter ,
                                       thermalSolution ,
                                       thermalRHS ,
                                       thermalResidual ,
                                       thermalVolumeOperator );

  thermalReference = thermalResidual->cloneVector ();
  thermalReference->setToScalar ( 301.0 );

  mechanicsSolver = buildMechanicsSolver ( input_db->getDatabase ( "MechanicsDB" ) ,
                                           meshAdapter ,
                                           thermalReference ,
                                           thermalSolution ,
                                           mechanicsSolution ,
                                           mechanicsRHS ,
                                           mechanicsResidual ,
                                           mechanicsVolumeOperator );

  mechanicsScaledRHS = mechanicsRHS->cloneVector(); 

  AMP_INSIST(input_db->keyExists("NumberOfLoadingSteps"), "Key ''NumberOfLoadingSteps'' is missing!");
  int NumberOfLoadingSteps = input_db->getInteger("NumberOfLoadingSteps");

  mechanicsSolution->setToScalar(0.0);
  mechanicsSolutionLast = mechanicsSolution->cloneVector();
  thermalSolution->setToScalar(400.);
  thermalSolutionLast = thermalSolution->cloneVector();

  typedef AMP::DeformMesh<AMP::Mesh::MeshManager::Adapter>  Deformer;
  Deformer deform ( mechanicsSolution , meshAdapter , 1.0 );
  Deformer undeform ( mechanicsSolution , meshAdapter , -1.0 );

  for (int step=0;step<NumberOfLoadingSteps; step++)
  {
    AMP::pout << "########################################" << std::endl;
    AMP::pout << "The current loading step is " << (step+1) << std::endl;

    double scaleValue  = ((double)step+1.0)/NumberOfLoadingSteps;
    mechanicsScaledRHS->scale( scaleValue, mechanicsRHS );
    bool converged = false;
    while ( !converged )
    {
      mechanicsSolutionLast->copyVector ( mechanicsSolution );
      thermalSolutionLast->copyVector ( thermalSolution );
      mechanicsSolver->setZeroInitialGuess(false);
      thermalSolver->setZeroInitialGuess(false);

      mechanicsSolver->solve( mechanicsScaledRHS , mechanicsSolution );

      AMP::MeshApply<Deformer> ( meshAdapter->beginNode() , 
                                 meshAdapter->endNode () ,
                                 deform );

      thermalSolver->solve(thermalRHS , thermalSolution);

      AMP::MeshApply<Deformer> ( meshAdapter->beginNode() , 
                                 meshAdapter->endNode () ,
                                 undeform );

      mechanicsSolutionLast->subtract ( mechanicsSolutionLast , mechanicsSolution );
      thermalSolutionLast->subtract ( thermalSolutionLast , thermalSolution );
      double diff = mechanicsSolutionLast->maxNorm();
      diff += thermalSolutionLast->maxNorm();
      converged = ( diff < 0.00000001 );
    }
    boost::shared_ptr<AMP::InputDatabase> tmp_db (new AMP::InputDatabase("Dummy"));
    boost::shared_ptr<AMP::Operator::MechanicsNonlinearFEOperatorParameters> tmpParams(new
        AMP::Operator::MechanicsNonlinearFEOperatorParameters(tmp_db));
    mechanicsVolumeOperator->reset(tmpParams);

  }
}

int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    try {
    myTest(ut);
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


