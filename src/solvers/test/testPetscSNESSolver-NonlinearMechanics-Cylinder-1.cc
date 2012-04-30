#include <iostream>
#include <string>

#include "boost/shared_ptr.hpp"

#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/PIO.h"
#include "materials/Material.h"

#include "ampmesh/Mesh.h"
#include "ampmesh/SiloIO.h"

#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Variable.h"
#include "vectors/VectorBuilder.h"

#include "operators/mechanics/MechanicsLinearFEOperator.h"
#include "operators/mechanics/MechanicsNonlinearFEOperator.h"

#include "operators/boundary/DirichletVectorCorrection.h"
#include "operators/boundary/PressureBoundaryVectorCorrection.h"

#include "operators/BVPOperatorParameters.h"
#include "operators/LinearBVPOperator.h"
#include "operators/NonlinearBVPOperator.h"
#include "operators/OperatorBuilder.h"
#include "operators/TrilinosMatrixShellOperator.h"

#include "solvers/PetscKrylovSolverParameters.h"
#include "solvers/PetscKrylovSolver.h"
#include "solvers/PetscSNESSolverParameters.h"
#include "solvers/PetscSNESSolver.h"

#include "solvers/TrilinosMLSolver.h"

AMP::LinearAlgebra::Matrix::shared_ptr myMatrix;

void myGetRow(int row, std::vector<unsigned int> &cols, std::vector<double> &values) {
  myMatrix->getRowByGlobalID(row, cols, values);
}

void myTest(AMP::UnitTest *ut, std::string exeName)
{
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName;
  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

  boost::shared_ptr<AMP::InputDatabase>  input_db ( new AMP::InputDatabase ( "input_db" ) );
  AMP::InputManager::getManager()->parseInputFile ( input_file , input_db );
  input_db->printClassData(AMP::plog);

  // Get the Mesh database and create the mesh parameters
  boost::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
  boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
  params->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));

  // Create the meshes from the input database
  AMP::Mesh::Mesh::shared_ptr  mesh = AMP::Mesh::Mesh::buildMesh(params);

  // Create the DOFManagers
  AMP::Discretization::DOFManager::shared_ptr NodalVectorDOF = 
    AMP::Discretization::simpleDOFManager::create(mesh,AMP::Mesh::Vertex,1,3);

  AMP_INSIST(input_db->keyExists("NumberOfLoadingSteps"), "Key ''NumberOfLoadingSteps'' is missing!");
  int NumberOfLoadingSteps = input_db->getInteger("NumberOfLoadingSteps");

  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel;
  boost::shared_ptr<AMP::Operator::NonlinearBVPOperator> nonlinBvpOperator = 
    boost::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(mesh,
														    "nonlinearMechanicsBVPOperator",
														    input_db,
														    elementPhysicsModel));
  (boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->getVolumeOperator()))->init();

  boost::shared_ptr<AMP::Operator::LinearBVPOperator> linBvpOperator =
    boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(mesh,
														 "linearMechanicsBVPOperator",
														 input_db,
														 elementPhysicsModel));
  
  boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> multivariable = boost::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(
      nonlinBvpOperator->getVolumeOperator()->getInputVariable()); 
  AMP::LinearAlgebra::Variable::shared_ptr displacementVariable = multivariable->getVariable(AMP::Operator::Mechanics::DISPLACEMENT); 
  AMP::LinearAlgebra::Variable::shared_ptr residualVariable = nonlinBvpOperator->getOutputVariable();

  //For RHS (Point Forces)
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> dummyModel;
  boost::shared_ptr<AMP::Operator::DirichletVectorCorrection> dirichletLoadVecOp =
    boost::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(mesh,
															 "Load_Boundary",
															 input_db,
															 dummyModel));
  dirichletLoadVecOp->setVariable(residualVariable);

  //Pressure RHS
  boost::shared_ptr<AMP::Operator::PressureBoundaryVectorCorrection> pressureLoadVecOp =
    boost::dynamic_pointer_cast<AMP::Operator::PressureBoundaryVectorCorrection>(AMP::Operator::OperatorBuilder::createOperator(mesh,
																"Pressure_Boundary",
																input_db,
																dummyModel));
  //This has an in-place apply. So, it has an empty input variable and
  //the output variable is the same as what it is operating on. 
  pressureLoadVecOp->setVariable(nonlinBvpOperator->getOutputVariable());

  AMP::LinearAlgebra::Vector::shared_ptr nullVec;

  AMP::LinearAlgebra::Vector::shared_ptr mechNlSolVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlRhsVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
  AMP::LinearAlgebra::Vector::shared_ptr mechPressureVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlResVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlScaledRhsVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, displacementVariable );

  // Create the silo writer and register the data
  #ifdef USE_SILO
    AMP::Mesh::SiloIO::shared_ptr  siloWriter( new AMP::Mesh::SiloIO);
    siloWriter->registerVector( mechNlSolVec, mesh, AMP::Mesh::Vertex, "Solution_Vector" );
    siloWriter->registerVector( mechNlResVec, mesh, AMP::Mesh::Vertex, "Residual_Vector" );
#endif

  //Initial guess for NL solver must satisfy the displacement boundary conditions
  mechNlSolVec->setToScalar(0.0);

  nonlinBvpOperator->apply(nullVec, mechNlSolVec, mechNlResVec, 1.0, 0.0);
  linBvpOperator->reset(nonlinBvpOperator->getJacobianParameters(mechNlSolVec));

  //Point forces
  mechNlRhsVec->setToScalar(0.0);
  dirichletLoadVecOp->apply(nullVec, nullVec, mechNlRhsVec, 1.0, 0.0);

  //Applying the pressure load
  pressureLoadVecOp->apply(nullVec, nullVec, mechPressureVec, 1.0, 0.0);
  mechNlRhsVec->add(mechNlRhsVec, mechPressureVec);

  boost::shared_ptr<AMP::Database> nonlinearSolver_db = input_db->getDatabase("NonlinearSolver"); 
  boost::shared_ptr<AMP::Database> linearSolver_db = nonlinearSolver_db->getDatabase("LinearSolver"); 

  // ---- first initialize the preconditioner
  boost::shared_ptr<AMP::Database> pcSolver_db = linearSolver_db->getDatabase("Preconditioner"); 
  boost::shared_ptr<AMP::Solver::TrilinosMLSolverParameters> pcSolverParams(new AMP::Solver::TrilinosMLSolverParameters(pcSolver_db));
  pcSolverParams->d_pOperator = linBvpOperator;
  boost::shared_ptr<AMP::Solver::TrilinosMLSolver> pcSolver(new AMP::Solver::TrilinosMLSolver(pcSolverParams));

  // initialize the nonlinear solver
  boost::shared_ptr<AMP::Solver::PetscSNESSolverParameters> nonlinearSolverParams(new
      AMP::Solver::PetscSNESSolverParameters(nonlinearSolver_db));
  // change the next line to get the correct communicator out
  nonlinearSolverParams->d_comm = globalComm;
  nonlinearSolverParams->d_pOperator = nonlinBvpOperator;
  nonlinearSolverParams->d_pInitialGuess = mechNlSolVec;
  boost::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver(new AMP::Solver::PetscSNESSolver(nonlinearSolverParams));
  nonlinearSolver->setZeroInitialGuess(false);

  boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver = nonlinearSolver->getKrylovSolver();
  linearSolver->setPreconditioner(pcSolver);

  for (int step=0;step<NumberOfLoadingSteps; step++)
  {
    AMP::pout << "########################################" << std::endl;
    AMP::pout << "The current loading step is " << (step+1) << std::endl;

    double scaleValue  = ((double)step+1.0)/NumberOfLoadingSteps;
    mechNlScaledRhsVec->scale(scaleValue, mechNlRhsVec);
    AMP::pout << "L2 Norm of RHS at loading step " << (step+1) << " is " << mechNlScaledRhsVec->L2Norm() << std::endl;

    nonlinBvpOperator->apply(mechNlScaledRhsVec, mechNlSolVec, mechNlResVec, 1.0, -1.0);
    double initialResidualNorm  = mechNlResVec->L2Norm();
    AMP::pout<<"Initial Residual Norm for loading step "<<(step+1)<<" is "<<initialResidualNorm<<std::endl;

    AMP::pout<<"Starting Nonlinear Solve..."<<std::endl;
    nonlinearSolver->solve(mechNlScaledRhsVec, mechNlSolVec);

    nonlinBvpOperator->apply(mechNlScaledRhsVec, mechNlSolVec, mechNlResVec, 1.0, -1.0);
    double finalResidualNorm  = mechNlResVec->L2Norm();
    AMP::pout<<"Final Residual Norm for loading step "<<(step+1)<<" is "<<finalResidualNorm<<std::endl;

    if( finalResidualNorm > (1.0e-10*initialResidualNorm) ) {
      ut->failure("Nonlinear solve for current loading step");
    } else {
      ut->passes("Nonlinear solve for current loading step");
    }

    double finalSolNorm = mechNlSolVec->L2Norm();

    AMP::pout<<"Final Solution Norm: "<<finalSolNorm<<std::endl;

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

  #ifdef USE_SILO
    siloWriter->writeFile( exeName, 1 );
  #endif

  ut->passes(exeName);

}

int main(int argc, char *argv[])
{
  AMP::AMPManager::startup(argc, argv);
  AMP::UnitTest ut;

  std::vector<std::string> exeNames;
  exeNames.push_back("testPetscSNESSolver-NonlinearMechanics-Cylinder-1");
  //exeNames.push_back("testPetscSNESSolver-NonlinearMechanics-PlateWithHole-1");

  for(size_t i = 0; i < exeNames.size(); i++) {
    try {
      myTest(&ut, exeNames[i]);
    } catch (std::exception &err) {
      std::cout << "ERROR: While testing "<<argv[0] << err.what() << std::endl;
      ut.failure("ERROR: While testing");
    } catch( ... ) {
      std::cout << "ERROR: While testing "<<argv[0] << "An unknown exception was thrown." << std::endl;
      ut.failure("ERROR: While testing");
    }
  }

  ut.report();

  int num_failed = ut.NumFailGlobal();
  AMP::AMPManager::shutdown();
  return num_failed;
}   


