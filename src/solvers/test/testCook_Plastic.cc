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

#include "ampmesh/Mesh.h"
#include "ampmesh/libmesh/initializeLibMesh.h"
#include "ampmesh/libmesh/libMesh.h"
#include "materials/Material.h"

#include "operators/LinearBVPOperator.h"
#include "operators/NonlinearBVPOperator.h"
#include "operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "operators/ElementPhysicsModel.h"
#include "operators/OperatorBuilder.h"
#include "operators/boundary/DirichletVectorCorrection.h"


#include "vectors/Vector.h"
#include "vectors/VectorSelector.h"
#include "utils/Writer.h"
#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Variable.h"
#include "vectors/MultiVariable.h"
#include "vectors/VectorBuilder.h"

#include "solvers/PetscKrylovSolverParameters.h"
#include "solvers/PetscKrylovSolver.h"
#include "solvers/PetscSNESSolver.h"
#include "solvers/trilinos/TrilinosMLSolver.h"

#include "utils/ReadTestMesh.h"

extern "C"{
#include "petsc.h"
}


void myTest(AMP::UnitTest *ut, std::string exeName)
{
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName + ".txt";

  AMP::PIO::logOnlyNodeZero(log_file);

  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::AMP_MPI globalComm = AMP::AMP_MPI(AMP_COMM_WORLD);
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);

  boost::shared_ptr<AMP::Mesh::initializeLibMesh> libmeshInit(new AMP::Mesh::initializeLibMesh(globalComm));

  const unsigned int mesh_dim = 3;
  boost::shared_ptr< ::Mesh > mesh(new ::Mesh(mesh_dim));

  std::string mesh_file = input_db->getString("mesh_file");
  if(globalComm.getRank() == 0) {
    AMP::readTestMesh(mesh_file, mesh);
  }//end if root processor

  MeshCommunication().broadcast(*(mesh.get()));
  mesh->prepare_for_use(false);
  AMP::Mesh::Mesh::shared_ptr meshAdapter ( new AMP::Mesh::libMesh(mesh,"cook") );

  AMP_INSIST(input_db->keyExists("NumberOfLoadingSteps"), "Key ''NumberOfLoadingSteps'' is missing!");
  int NumberOfLoadingSteps = input_db->getInteger("NumberOfLoadingSteps");

  boost::shared_ptr<AMP::Operator::NonlinearBVPOperator> nonlinBvpOperator = 
    boost::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
    AMP::Operator::OperatorBuilder::createOperator(meshAdapter,"nonlinearMechanicsBVPOperator",input_db));
  boost::shared_ptr<AMP::Operator::MechanicsNonlinearFEOperator> nonlinearMechanicsVolumeOperator = 
    boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(nonlinBvpOperator->getVolumeOperator());
  boost::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel = nonlinearMechanicsVolumeOperator->getMaterialModel();

  boost::shared_ptr<AMP::Operator::LinearBVPOperator> linBvpOperator =
    boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
														 "linearMechanicsBVPOperator",
														 input_db,
														 elementPhysicsModel));

  boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> tmpVar = boost::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(
     boost::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>( nonlinBvpOperator->getVolumeOperator())->getInputVariable() );
  AMP::LinearAlgebra::Variable::shared_ptr displacementVariable = tmpVar->getVariable(AMP::Operator::Mechanics::DISPLACEMENT); 
  AMP::LinearAlgebra::Variable::shared_ptr residualVariable = nonlinBvpOperator->getOutputVariable();

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

  AMP::Discretization::DOFManager::shared_ptr DOF_vector = AMP::Discretization::simpleDOFManager::create(meshAdapter,AMP::Mesh::Vertex,1,3,true);
  AMP::LinearAlgebra::Vector::shared_ptr mechNlSolVec = AMP::LinearAlgebra::createVector( DOF_vector, residualVariable, true );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlRhsVec = AMP::LinearAlgebra::createVector( DOF_vector, residualVariable, true );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlResVec = AMP::LinearAlgebra::createVector( DOF_vector, residualVariable, true );
  AMP::LinearAlgebra::Vector::shared_ptr mechNlScaledRhsVec = AMP::LinearAlgebra::createVector( DOF_vector, residualVariable, true );

#ifdef USE_EXT_SILO
  AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter("Silo");
  siloWriter->registerVector( mechNlSolVec, meshAdapter, AMP::Mesh::Vertex, "Solution_Vector" );
  siloWriter->registerVector( mechNlResVec, meshAdapter, AMP::Mesh::Vertex, "Residual_Vector" );
#endif

  //Initial guess for NL solver must satisfy the displacement boundary conditions
  mechNlSolVec->setToScalar(0.0);
  dirichletDispInVecOp->apply(nullVec, nullVec, mechNlSolVec, 1.0, 0.0);
  mechNlSolVec->makeConsistent(AMP::LinearAlgebra::Vector::CONSISTENT_SET);

  nonlinBvpOperator->apply(nullVec, mechNlSolVec, mechNlResVec, 1.0, 0.0);
  linBvpOperator->reset(nonlinBvpOperator->getJacobianParameters(mechNlSolVec));

  //Point forces
  mechNlRhsVec->setToScalar(0.0);
  dirichletLoadVecOp->apply(nullVec, nullVec, mechNlRhsVec, 1.0, 0.0);

  boost::shared_ptr<AMP::Database> nonlinearSolver_db = input_db->getDatabase("NonlinearSolver"); 
  boost::shared_ptr<AMP::Database> linearSolver_db = nonlinearSolver_db->getDatabase("LinearSolver"); 

  // ---- first initialize the preconditioner
  boost::shared_ptr<AMP::Database> pcSolver_db = linearSolver_db->getDatabase("Preconditioner"); 
  boost::shared_ptr<AMP::Solver::TrilinosMLSolverParameters> pcSolverParams(new AMP::Solver::TrilinosMLSolverParameters(pcSolver_db));
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

  // initialize the nonlinear solver
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

    double scaleValue  = ((double)step+1.0)/NumberOfLoadingSteps;
    mechNlScaledRhsVec->scale(scaleValue, mechNlRhsVec);
    mechNlScaledRhsVec->makeConsistent(AMP::LinearAlgebra::Vector::CONSISTENT_SET);
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

    boost::shared_ptr<AMP::InputDatabase> tmp_db (new AMP::InputDatabase("Dummy"));
    boost::shared_ptr<AMP::Operator::MechanicsNonlinearFEOperatorParameters> tmpParams(new
        AMP::Operator::MechanicsNonlinearFEOperatorParameters(tmp_db));
    (nonlinBvpOperator->getVolumeOperator())->reset(tmpParams);
    nonlinearSolver->setZeroInitialGuess(false);
  }

  double finalSolNorm = mechNlSolVec->L2Norm();
  AMP::pout<<"Final Solution Norm: "<<finalSolNorm<<std::endl;

#ifdef USE_EXT_SILO
  siloWriter->writeFile( exeName, 0 );
#endif

  ut->passes(exeName);

}

int main(int argc, char *argv[])
{

    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.push_back("testCook_Plastic_normal");
    //exeNames.push_back("testCook_Plastic_reduced");

    for(size_t i=0; i<exeNames.size(); i++) {
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


