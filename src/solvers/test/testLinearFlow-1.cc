#include <string>
#include "boost/shared_ptr.hpp"

#include "utils/AMPManager.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/Database.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/PIO.h"

#include "ampmesh/Mesh.h"
#include "ampmesh/SiloIO.h"

#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"

#include "vectors/Vector.h"
#include "vectors/MultiVector.h"
#include "vectors/Variable.h"
#include "vectors/VectorBuilder.h"

#include "operators/NeutronicsRhs.h"
#include "vectors/Variable.h"
#include "operators/VolumeIntegralOperator.h"
#include "materials/Material.h"

#include "matrices/Matrix.h"
#include "matrices/MatrixBuilder.h"
#include "matrices/trilinos/EpetraMatrix.h"

/*
#include "flow/ConsMomentumGalWFLinearElement.h"
#include "flow/ConsMomentumGalWFLinearFEOperator.h"
#include "flow/ConsMassGalWFLinearElement.h"
#include "flow/ConsMassGalWFLinearFEOperator.h"
*/

#include "operators/EpetraMatrixOperator.h"
#include "operators/EpetraMatrixOperatorParameters.h"
#include "operators/LinearBVPOperator.h"
#include "operators/OperatorBuilder.h"

#include "operators/BlockOperator.h"
#include "operators/TrilinosMatrixShellOperator.h"
#include "operators/PetscMatrixShellOperator.h"

#include "solvers/TrilinosMLSolver.h"
#include "solvers/ColumnSolver.h"
#include "solvers/PetscKrylovSolverParameters.h"
#include "solvers/PetscKrylovSolver.h"
#include "solvers/PetscSNESSolverParameters.h"
#include "solvers/PetscSNESSolver.h"


#define ITFAILS ut.failure(__LINE__);
#define UNIT_TEST(a) if (!(a)) ut.failure(__LINE__);

double h_Size = 1.46286; 

void myTest(AMP::UnitTest *ut, std::string exeName)

{

	std::string input_file = "input_" + exeName;
    std::string silo_name = exeName;

    // Read the input file
    boost::shared_ptr<AMP::InputDatabase>  input_db ( new AMP::InputDatabase ( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile ( input_file , input_db );
	AMP::AMP_MPI globalComm = AMP::AMP_MPI(AMP_COMM_WORLD);
    input_db->printClassData (AMP::plog);

    // Get the Mesh database and create the mesh parameters
    boost::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
    boost::shared_ptr<AMP::Mesh::MeshParameters> params(new AMP::Mesh::MeshParameters(database));
    params->setComm(globalComm);

    // Create the meshes from the input database
    AMP::Mesh::Mesh::shared_ptr manager = AMP::Mesh::Mesh::buildMesh(params);
	AMP::Mesh::Mesh::shared_ptr meshAdapterH27 = manager->Subset ("cubeH27");
	AMP::Mesh::Mesh::shared_ptr meshAdapterH08 = manager->Subset ("cubeH08");
	
    /////////////////////////////////////////////////
    //   CREATE THE Conservation of Momentum Operator  //
    /////////////////////////////////////////////////

	boost::shared_ptr<AMP::Operator::ElementPhysicsModel> FlowTransportModel; 
	AMP_INSIST ( input_db->keyExists("ConsMomentumLinearFEOperator"),"key missing!");
	boost::shared_ptr<AMP::Operator::LinearBVPOperator> ConsMomentumOperator = boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapterH27, "ConsMomentumLinearBVPOperator", input_db, FlowTransportModel));

    /////////////////////////////////////////////////
    //   CREATE THE Conservation of Mass Operator  //
    /////////////////////////////////////////////////

	AMP_INSIST ( input_db->keyExists("ConsMassLinearFEOperator"),"key missing!");
	boost::shared_ptr<AMP::Operator::LinearBVPOperator> ConsMassOperator = boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapterH08, "ConsMassLinearBVPOperator", input_db, FlowTransportModel));

        AMP::pout << "Finished creating Mass Operator" << std::endl;
	
    // Create the variables
    AMP::LinearAlgebra::Variable::shared_ptr velocityVar   = ConsMomentumOperator->getOutputVariable();
    AMP::LinearAlgebra::Variable::shared_ptr pressureVar   = ConsMassOperator->getOutputVariable();

    // Create the DOF managers
    AMP::Discretization::DOFManager::shared_ptr DOF_scalar = AMP::Discretization::simpleDOFManager::create(manager,AMP::Mesh::Vertex,1,1,true);
    AMP::Discretization::DOFManager::shared_ptr DOF_vector = AMP::Discretization::simpleDOFManager::create(manager,AMP::Mesh::Vertex,1,3,true);

    // Create the vectors
    boost::shared_ptr<AMP::LinearAlgebra::MultiVector>  globalSolVec =
        AMP::LinearAlgebra::MultiVector::create( "globalSolVec", manager->getComm() );
    globalSolVec->addVector( AMP::LinearAlgebra::createVector( DOF_scalar, pressureVar ) );
    globalSolVec->addVector( AMP::LinearAlgebra::createVector( DOF_vector, velocityVar ) );
    AMP::LinearAlgebra::Vector::shared_ptr globalRhsVec = globalSolVec->cloneVector("globalRhsVec");
    // AMP::LinearAlgebra::Vector::shared_ptr globalResVec = globalSolVec->cloneVector("globalResVec");

    // Get the matricies
    AMP::LinearAlgebra::Matrix::shared_ptr FMat = ConsMomentumOperator->getMatrix();
    AMP::LinearAlgebra::Matrix::shared_ptr BMat = ConsMassOperator->getMatrix();

    AMP::LinearAlgebra::Matrix::shared_ptr BtMat =  BMat->transpose();

    // Create a zero matrix over meshAdapterH08
    AMP::Discretization::DOFManager::shared_ptr DOF_H08_scalar = 
        AMP::Discretization::simpleDOFManager::create(meshAdapterH08,AMP::Mesh::Vertex,1,1,true);
    AMP::LinearAlgebra::Matrix::shared_ptr zeroMat = AMP::LinearAlgebra::createMatrix( 
        AMP::LinearAlgebra::createVector( DOF_H08_scalar, pressureVar ),
        AMP::LinearAlgebra::createVector( DOF_H08_scalar, pressureVar ) );
    zeroMat->zero();



	boost::shared_ptr<AMP::Database> dummy_db;
        boost::shared_ptr<AMP::Operator::EpetraMatrixOperatorParameters> dummyParams1(new AMP::Operator::EpetraMatrixOperatorParameters( dummy_db ));
        dummyParams1->d_Matrix = &(boost::dynamic_pointer_cast<AMP::LinearAlgebra::EpetraMatrix>(BtMat)->getEpetra_CrsMatrix());
        boost::shared_ptr<AMP::Operator::EpetraMatrixOperator> bTOperator ( new AMP::Operator::EpetraMatrixOperator (dummyParams1) );
        bTOperator->setVariables(ConsMassOperator->getOutputVariable(), ConsMassOperator->getInputVariable());

        boost::shared_ptr<AMP::Operator::EpetraMatrixOperatorParameters> dummyParams2(new AMP::Operator::EpetraMatrixOperatorParameters( dummy_db ));
        dummyParams2->d_Matrix = &(boost::dynamic_pointer_cast<AMP::LinearAlgebra::EpetraMatrix>(zeroMat)->getEpetra_CrsMatrix());
         boost::shared_ptr<AMP::Operator::EpetraMatrixOperator> zeroOperator ( new AMP::Operator::EpetraMatrixOperator (dummyParams2) );
        zeroOperator->setVariables(ConsMassOperator->getOutputVariable(), ConsMassOperator->getOutputVariable());

        AMP_INSIST(input_db->keyExists("LinearSolver"),   "Key ''LinearSolver'' is missing!");
        boost::shared_ptr<AMP::Database>  mlSolver_db  = input_db->getDatabase("LinearSolver"); 
       
        boost::shared_ptr<AMP::Solver::SolverStrategyParameters> convdiffSolverParams(new AMP::Solver::SolverStrategyParameters(mlSolver_db));
        convdiffSolverParams->d_pOperator = ConsMomentumOperator;
        boost::shared_ptr<AMP::Solver::TrilinosMLSolver>  convdiffSolver(new AMP::Solver::TrilinosMLSolver(convdiffSolverParams));
        convdiffSolver->setZeroInitialGuess(false);

        AMP::LinearAlgebra::Vector::shared_ptr diagonalVec = FMat->extractDiagonal();
        AMP::LinearAlgebra::Vector::shared_ptr diagonalInvVec = diagonalVec->cloneVector();
        diagonalInvVec->reciprocal(diagonalVec);

        AMP::LinearAlgebra::Matrix::shared_ptr DMat = FMat->cloneMatrix() ;
        DMat->zero();
        DMat->setDiagonal(diagonalVec);

        AMP::LinearAlgebra::Matrix::shared_ptr DInvMat = FMat->cloneMatrix() ;
        DInvMat->zero();
        DInvMat->setDiagonal(diagonalInvVec);

        AMP::LinearAlgebra::Matrix::shared_ptr schurMat = zeroMat->cloneMatrix() ;

        AMP::LinearAlgebra::Matrix::shared_ptr DInvBtMat = AMP::LinearAlgebra::Matrix::matMultiply(DInvMat, BtMat);

        schurMat = AMP::LinearAlgebra::Matrix::matMultiply(BtMat, DInvBtMat);

        boost::shared_ptr<AMP::Operator::EpetraMatrixOperatorParameters> dummyParams3(new AMP::Operator::EpetraMatrixOperatorParameters( dummy_db ));
        dummyParams3->d_Matrix = &(boost::dynamic_pointer_cast<AMP::LinearAlgebra::EpetraMatrix>(schurMat)->getEpetra_CrsMatrix());
        boost::shared_ptr<AMP::Operator::EpetraMatrixOperator> schurMatOperator ( new AMP::Operator::EpetraMatrixOperator (dummyParams3) );
        schurMatOperator->setVariables(ConsMassOperator->getOutputVariable(), ConsMassOperator->getOutputVariable());

        boost::shared_ptr<AMP::Solver::SolverStrategyParameters> schurMatSolverParams(new AMP::Solver::SolverStrategyParameters(mlSolver_db));
        schurMatSolverParams->d_pOperator = schurMatOperator;
        boost::shared_ptr<AMP::Solver::TrilinosMLSolver>  schurMatSolver(new AMP::Solver::TrilinosMLSolver(schurMatSolverParams));
        schurMatSolver->setZeroInitialGuess(false);

        AMP::LinearAlgebra::Vector::shared_ptr velocityRhsVec = globalRhsVec->subsetVectorForVariable(velocityVar); 
        AMP::LinearAlgebra::Vector::shared_ptr pressureRhsVec = globalRhsVec->subsetVectorForVariable(pressureVar);

        AMP::LinearAlgebra::Vector::shared_ptr velocitySolVec = globalSolVec->subsetVectorForVariable(velocityVar); 
        AMP::LinearAlgebra::Vector::shared_ptr pressureSolVec = globalSolVec->subsetVectorForVariable(pressureVar);
       
        AMP::LinearAlgebra::Vector::shared_ptr pressureUpdateVec = pressureSolVec->cloneVector();
        AMP::LinearAlgebra::Vector::shared_ptr velocityUpdateVec = velocitySolVec->cloneVector();
        
//        AMP::LinearAlgebra::Vector::shared_ptr pressurePrimeVec = pressureSolVec->cloneVector();
        AMP::LinearAlgebra::Vector::shared_ptr velocityPrimeVec = velocitySolVec->cloneVector();

        //SIMPLE(Semi Implicit Method for Pressure Linked Equations) ALGORITHM

        // STEP 1 :
        BtMat->mult( pressureSolVec , velocityRhsVec );
        convdiffSolver->solve( velocityRhsVec , velocityPrimeVec );

        // STEP 2 :
        BMat->mult( velocityPrimeVec , pressureRhsVec );
        schurMatSolver->solve( pressureRhsVec, pressureUpdateVec );

        // STEP 3 :
        DInvBtMat->mult(pressureUpdateVec, velocityUpdateVec); 
        velocitySolVec->subtract( velocityPrimeVec, velocityUpdateVec);

        // STEP 4 :
        pressureSolVec->add(velocityPrimeVec, pressureUpdateVec );

}

int main(int argc, char *argv[])
{
  AMP::AMPManager::startup(argc, argv);
  AMP::UnitTest ut;
  std::vector<std::string> exeNames;
  exeNames.push_back("testLinearFlow-1");

  for(unsigned int i=0; i < exeNames.size(); i++)
  {
    try
    {
      myTest(&ut, exeNames[i]);
    } catch (std::exception &err) {
      std::cout << "ERROR: While testing " << argv[0] << err.what() << std::endl;
      ut.failure("ERROR: While testing");
    } catch( ... ){
      std::cout << "ERROR: While testing " << argv[0] << "An unknown exception was thrown." << std::endl;
      ut.failure("ERROR: While testing");
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;

  }

}




























