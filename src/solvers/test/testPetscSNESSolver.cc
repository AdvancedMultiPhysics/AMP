// This tests checks the creation of a PetscSNESSolver
// Note: the comm used should NOT be comm_world as there are cleanup issues for other comms when using the monitor option
#include <iostream>
#include <string>
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/PIO.h"

#include "ampmesh/Mesh.h"
#include "vectors/NullVector.h"
#include "solvers/PetscSNESSolver.h"



void myTest(AMP::UnitTest *ut, std::string exeName)
{
    std::string input_file = "input_" + exeName;
    std::string log_file = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero(log_file);
    AMP::AMP_MPI globalComm(AMP_COMM_WORLD);
    AMP::AMP_MPI solverComm = globalComm.dup();     // Create a unique solver comm to test proper cleanup

    boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
    AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
    input_db->printClassData(AMP::plog);

    // Create a null vector for the initial guess
    AMP::LinearAlgebra::Vector::shared_ptr  nullVec = AMP::LinearAlgebra::NullVector::create("null");

    // Get the databases for the nonlinear and linear solvers
    boost::shared_ptr<AMP::Database> nonlinearSolver_db = input_db->getDatabase("NonlinearSolver"); 
    //boost::shared_ptr<AMP::Database> linearSolver_db = nonlinearSolver_db->getDatabase("LinearSolver"); 

    // initialize the nonlinear solver parameters
    boost::shared_ptr<AMP::Solver::PetscSNESSolverParameters> nonlinearSolverParams(new
       AMP::Solver::PetscSNESSolverParameters(nonlinearSolver_db));
    nonlinearSolverParams->d_comm = solverComm;
    nonlinearSolverParams->d_pInitialGuess = nullVec;

    // Create the nonlinear solver
    boost::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver(new AMP::Solver::PetscSNESSolver(nonlinearSolverParams));

    ut->passes("PetscSNESSolver created");
}



int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;

    myTest( &ut, "testPetscSNESSolver" );
   
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}   
