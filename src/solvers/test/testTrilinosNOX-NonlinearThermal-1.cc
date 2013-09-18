#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include <iostream>
#include <string>

#include "boost/shared_ptr.hpp"

#include "operators/VolumeIntegralOperator.h"
#include "operators/NeutronicsRhs.h"

#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/PIO.h"
#include "materials/Material.h"

#include "utils/Writer.h"

#include "ampmesh/Mesh.h"
#include "vectors/VectorBuilder.h"
#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"

#include "operators/mechanics/MechanicsLinearFEOperator.h"
#include "operators/mechanics/MechanicsNonlinearFEOperator.h"

#include "operators/diffusion/DiffusionLinearFEOperator.h"
#include "operators/diffusion/DiffusionNonlinearFEOperator.h"

#include "operators/boundary/DirichletVectorCorrection.h"

#include "operators/BVPOperatorParameters.h"
#include "operators/LinearBVPOperator.h"
#include "operators/NonlinearBVPOperator.h"
#include "operators/ColumnOperator.h"
#include "operators/OperatorBuilder.h"

#include "solvers/ColumnSolver.h"
#include "solvers/trilinos/nox/TrilinosNOXSolverParameters.h"
#include "solvers/trilinos/nox/TrilinosNOXSolver.h"

#include "solvers/trilinos/TrilinosMLSolver.h"


void myTest(AMP::UnitTest *ut, std::string exeName)
{
    std::string input_file = "input_" + exeName;
    //std::string log_file = "output_" + exeName;

    //  AMP::PIO::logOnlyNodeZero(log_file);
    AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

    size_t N_error0 = ut->NumFailLocal();;

    boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
    AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
    input_db->printClassData(AMP::plog);

    //--------------------------------------------------
    //   Create the Mesh.
    //--------------------------------------------------
    AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
    boost::shared_ptr<AMP::Database>  mesh_db = input_db->getDatabase("Mesh");
    boost::shared_ptr<AMP::Mesh::MeshParameters> mgrParams(new AMP::Mesh::MeshParameters(mesh_db));
    mgrParams->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));
    boost::shared_ptr<AMP::Mesh::Mesh> meshAdapter = AMP::Mesh::Mesh::buildMesh(mgrParams);
    //--------------------------------------------------

    //--------------------------------------------------
    // Create a DOF manager for a nodal vector 
    //--------------------------------------------------
    int DOFsPerNode = 1;
    int DOFsPerElement = 8;
    int nodalGhostWidth = 1;
    int gaussPointGhostWidth = 1;
    bool split = true;
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap = 
        AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, nodalGhostWidth,      DOFsPerNode,    split);
    AMP::Discretization::DOFManager::shared_ptr gaussPointDofMap = 
        AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Volume, gaussPointGhostWidth, DOFsPerElement, split);

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // create a nonlinear BVP operator for nonlinear thermal diffusion
    AMP_INSIST( input_db->keyExists("testNonlinearThermalOperator"), "key missing!" );
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> thermalTransportModel;
    boost::shared_ptr<AMP::Operator::NonlinearBVPOperator> nonlinearThermalOperator = 
        boost::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
										    "testNonlinearThermalOperator",
										    input_db,
										    thermalTransportModel));

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // initialize the input variable
    boost::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> thermalVolumeOperator =
	    boost::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(nonlinearThermalOperator->getVolumeOperator());

    boost::shared_ptr<AMP::LinearAlgebra::Variable> thermalVariable = thermalVolumeOperator->getOutputVariable();

    // create solution, rhs, and residual vectors
    AMP::LinearAlgebra::Vector::shared_ptr solVec = AMP::LinearAlgebra::createVector( nodalDofMap, thermalVariable );
    AMP::LinearAlgebra::Vector::shared_ptr rhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, thermalVariable );
    AMP::LinearAlgebra::Vector::shared_ptr resVec = AMP::LinearAlgebra::createVector( nodalDofMap, thermalVariable );

    // create the following shared pointers for ease of use
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // now construct the linear BVP operator for thermal
    AMP_INSIST( input_db->keyExists("testLinearThermalOperator"), "key missing!" );
    boost::shared_ptr<AMP::Operator::LinearBVPOperator> linearThermalOperator = 
        boost::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
		AMP::Operator::OperatorBuilder::createOperator( meshAdapter,
		"testLinearThermalOperator", input_db, thermalTransportModel ) );

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
    boost::shared_ptr<AMP::Operator::NeutronicsRhs> neutronicsOperator(new AMP::Operator::NeutronicsRhs( neutronicsParams ));

    AMP::LinearAlgebra::Variable::shared_ptr SpecificPowerVar = neutronicsOperator->getOutputVariable();
    AMP::LinearAlgebra::Vector::shared_ptr   SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, SpecificPowerVar );

    neutronicsOperator->apply(nullVec, nullVec, SpecificPowerVec, 1., 0.);

    /////////////////////////////////////////////////////
    //  Integrate Nuclear Rhs over Desnity * Volume //
    /////////////////////////////////////////////////////

    AMP_INSIST( input_db->keyExists("VolumeIntegralOperator"), "key missing!" );

    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> stransportModel;
    boost::shared_ptr<AMP::Operator::VolumeIntegralOperator> sourceOperator = 
        boost::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
																						          "VolumeIntegralOperator",
																						          input_db,
																						          stransportModel));

    // Create the power (heat source) vector.
    AMP::LinearAlgebra::Variable::shared_ptr PowerInWattsVar = sourceOperator->getOutputVariable();
    AMP::LinearAlgebra::Vector::shared_ptr   PowerInWattsVec = AMP::LinearAlgebra::createVector( nodalDofMap, PowerInWattsVar );
    PowerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply(nullVec, SpecificPowerVec, PowerInWattsVec, 1., 0.);

    rhsVec->copyVector(PowerInWattsVec);

    nonlinearThermalOperator->modifyRHSvector(rhsVec);

    double initialRhsNorm  = rhsVec->L2Norm();
    std::cout << "rhs norm  after modifyRHSvector = " << initialRhsNorm <<"\n";
    double expectedVal = 0.688628;
    if( !AMP::Utilities::approx_equal( expectedVal, initialRhsNorm, 1e-5) )
        ut->failure("the rhs norm after modifyRHSvector has changed.");

    // Get the solver databases
    boost::shared_ptr<AMP::Database> nonlinearSolver_db = input_db->getDatabase("NonlinearSolver"); 
    boost::shared_ptr<AMP::Database> linearSolver_db = nonlinearSolver_db->getDatabase("LinearSolver"); 

    // Create the preconditioner
    boost::shared_ptr<AMP::Database> thermalPreconditioner_db = linearSolver_db->getDatabase("Preconditioner");
    boost::shared_ptr<AMP::Solver::SolverStrategyParameters> thermalPreconditionerParams(
        new AMP::Solver::SolverStrategyParameters(thermalPreconditioner_db));
    thermalPreconditionerParams->d_pOperator = linearThermalOperator;
    boost::shared_ptr<AMP::Solver::TrilinosMLSolver> linearThermalPreconditioner(new AMP::Solver::TrilinosMLSolver(thermalPreconditionerParams));

    // Crete the solvers
    boost::shared_ptr<AMP::Solver::TrilinosNOXSolverParameters> nonlinearSolverParams(new
        AMP::Solver::TrilinosNOXSolverParameters(nonlinearSolver_db));
    nonlinearSolverParams->d_comm = globalComm;
    nonlinearSolverParams->d_pOperator = nonlinearThermalOperator;
    nonlinearSolverParams->d_pLinearOperator = nonlinearThermalOperator;
    nonlinearSolverParams->d_pInitialGuess = solVec;
    nonlinearSolverParams->d_preconditioner = linearThermalPreconditioner;
    boost::shared_ptr<AMP::Solver::TrilinosNOXSolver> nonlinearSolver(new AMP::Solver::TrilinosNOXSolver(nonlinearSolverParams));

    nonlinearThermalOperator->apply(rhsVec, solVec, resVec, 1.0, -1.0);
    double initialResidualNorm  = resVec->L2Norm();

    AMP::pout<<"Initial Residual Norm: "<<initialResidualNorm<<std::endl;
    expectedVal = 3625.84;
    if( !AMP::Utilities::approx_equal( expectedVal, initialResidualNorm, 1e-5) ) {
        ut->failure("the Initial Residual Norm has changed."); }

    nonlinearSolver->setZeroInitialGuess(false);

    nonlinearSolver->solve(rhsVec, solVec);

    solVec->makeConsistent ( AMP::LinearAlgebra::Vector::CONSISTENT_SET );
    resVec->makeConsistent ( AMP::LinearAlgebra::Vector::CONSISTENT_SET );

    nonlinearThermalOperator->apply(rhsVec, solVec, resVec, 1.0, -1.0);

    double finalResidualNorm  = resVec->L2Norm();
    double finalSolutionNorm  = solVec->L2Norm();
    double finalRhsNorm  = rhsVec->L2Norm();

    std::cout << "Final Residual Norm: " << finalResidualNorm << std::endl;
    std::cout << "Final Solution Norm: " << solVec->L2Norm()  << std::endl;
    std::cout << "Final Rhs Norm: "      << finalRhsNorm      << std::endl;

    if( fabs(finalResidualNorm) > 1e-9 )
        ut->failure("the Final Residual is larger than the tolerance");
    if( !AMP::Utilities::approx_equal( 45431.3, solVec->L2Norm(), 1e-5) )
        ut->failure("the Final Solution Norm has changed.");
    if( !AMP::Utilities::approx_equal( initialRhsNorm, finalRhsNorm, 1e-12) )
        ut->failure("the Final Rhs Norm has changed.");

    #ifdef USE_EXT_SILO
        AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter("Silo");
        siloWriter->registerMesh( meshAdapter );
        siloWriter->registerVector( solVec, meshAdapter, AMP::Mesh::Vertex, "Solution" );
        siloWriter->registerVector( resVec, meshAdapter, AMP::Mesh::Vertex, "Residual" );
        siloWriter->writeFile( input_file , 0 );
    #endif

    if ( N_error0 == ut->NumFailLocal() )
        ut->passes(exeName);
    else
        ut->failure(exeName);

}

int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.push_back("testTrilinosNOX-NonlinearThermal-cylinder_MATPRO");

    for(unsigned int i = 0; i < exeNames.size(); i++) {
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

