#include "AMP/IO/PIO.h"
#include "AMP/IO/AsciiWriter.h"
#include "AMP/utils/AMPManager.h"

#include "AMP/vectors/CommunicationList.h"
#include "AMP/matrices/petsc/NativePetscMatrix.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/data/VectorDataNull.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"
#include "AMP/vectors/VectorBuilder.h"

#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/structured/BoxMesh.h"

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixBuilder.h"

#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/petsc/PetscMatrixShellOperator.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/operators/NullOperator.h"

#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"

#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/TimeOperator.h"
#include "AMP/time_integrators/ImplicitIntegrator.h"

#include <iostream>
#include <iomanip>
#include <filesystem>

#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionModel.h"

#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDBEWrappers.h"
#include "AMP/operators/radiationDiffusionFD/RDUtils.h" // oktodo: delete

//#include "AMP/operators/radiationDiffusionFD/RDFDMonolithicPrec.h"

#include "AMP/operators/testHelpers/testDiffusionFDHelper.h"

#include "AMP/solvers/radiationDiffusionFDOpSplitPrec/RadiationDiffusionFDOpSplitPrec.h"

// oktodo: clean up includes


/*
Try turning on the auto component scaling once everything is working. this means I have to change the apply functions in my BEOperators. Bobby mentioned that since one component is much bigger than the other, if you look at their relative convergence, one drops much more than the other.

I'll also need to understand what exactly this is doing because if there are constants being scaling introduced, does that influence the linearization?
*/




/* In a multiVector, DOFs are ordered by variable on process.

Test routines to map between different orderings
*/
#if 0
void orderingTest( std::shared_ptr<AMP::Discretization::DOFManager> scalarDOF, 
                    std::shared_ptr<AMP::Discretization::multiDOFManager> multiDOF ) {

    std::vector<size_t> variableDOFs;

    // This shows how global DOFs are ordered in a multiVector
    std::cout << "Ordering of multiVector:" << "\n";
    std::cout << "------------------------" << "\n";
    for ( auto dof = scalarDOF->beginDOF(); dof != scalarDOF->endDOF(); dof++ ) {
        std::vector<size_t> dof_ = { dof };
        auto E_dof = multiDOF->getGlobalDOF( 0, dof_ );
        auto T_dof = multiDOF->getGlobalDOF( 1, dof_ );
        std::cout << "scalar_dof=" << dof << "\tglobal DOFs: E=" << E_dof[0] << ", T=" << T_dof[0] << std::endl; 

        variableDOFs.push_back( E_dof[0] );
        variableDOFs.push_back( T_dof[0] );
    }


    auto n = multiDOF->numLocalDOF() / 2;
    std::vector<size_t> nodalDOFs;
    std::cout << "\n\nTest: variable ordering --> nodal ordering\n";
    std::cout << "-----------------------------------------\n";
    variableOrderingToNodalOrdering( n, variableDOFs, nodalDOFs );
    for ( auto i = 0; i < variableDOFs.size(); i++ ) {
        std::cout << "var_idx=" << variableDOFs[i] << " --> ndl_idx=" << nodalDOFs[i] << "\n";
    }

    variableDOFs.clear();
    std::cout << "\n\nTest: nodal ordering --> variable ordering\n";
    std::cout << "-----------------------------------------\n";
    nodalOrderingToVariableOrdering( n, nodalDOFs, variableDOFs );
    for ( auto i = 0; i < nodalDOFs.size(); i++ ) {
        std::cout << "ndl_idx=" << nodalDOFs[i] << " --> var_idx=" << variableDOFs[i] << "\n";
    }

}
#endif

/*
I'm going to need to map multiVectors to Vectors that are nodally ordered. 

Building the matrix is going to be a pain... Alternatively, I could try to set up an interface to HYPRE's "HYPRE_Int HYPRE_BoomerAMGSetDofFunc(HYPRE_Solver solver, HYPRE_Int *dof_func)"
Hmm. That ordering is going to be weird though, because it's blocked by variable only on process...

What if I have a mesh with 2 DOFs per node? But I'd be duplicating the mesh, and I don't know for sure that it'd be parallelized identially... Well actually. It's the DOFManager that's made aware of the number of DOFs per element, not the mesh...  
*/



void driver(AMP::AMP_MPI comm, 
            std::shared_ptr<AMP::Database> input_db ) {

    /****************************************************************
    * Re-organize database input                                    *
    ****************************************************************/
    // Basic error check the input has required things
    AMP_INSIST( input_db, "Non-null input_db required" );
    AMP_INSIST( input_db->getDatabase( "PDE" ), "PDE is null" );
    AMP_INSIST( input_db->getDatabase( "Mesh" ), "Mesh is null" );
    AMP_INSIST( input_db->getDatabase( "TimeIntegrator" ), "TimeIntegrator is null" );
    //AMP_INSIST( input_db->getDatabase( "Solver" ), "Solver is null" );
    
    // Create a discretization and solver DB
    auto PDE_basic_db = input_db->getDatabase( "PDE" );
    auto mesh_db      = input_db->getDatabase( "Mesh" )->cloneDatabase();
    auto ti_db        = input_db->getDatabase( "TimeIntegrator" );
    
    // Get PDE model-specific parameter database
    auto problemID        = input_db->getDatabase( "PDE" )->getScalar<std::string>( "problemID" );
    auto PDE_mspecific_db = input_db->getDatabase( problemID + "_Parameters" );
    AMP_INSIST( PDE_mspecific_db, "Input must have the model-specific database: '" + problemID + "_Parameters'" );
        

    /****************************************************************
    * Create radiation-diffusion model                              *
    ****************************************************************/
    std::shared_ptr<AMP::Operator::RadDifModel> myRadDifModel;

    if ( problemID == "Mousseau_etal_2000" ) {
        auto myRadDifModel_ = std::make_shared<AMP::Operator::Mousseau_etal_2000_RadDifModel>( PDE_basic_db, PDE_mspecific_db );
        myRadDifModel       = myRadDifModel_;

    } else if ( problemID == "Manufactured" ) {
        auto myRadDifModel_ = std::make_shared<AMP::Operator::Manufactured_RadDifModel>( PDE_basic_db, PDE_mspecific_db );
        myRadDifModel = myRadDifModel_; 

    } else {
        AMP_ERROR( "Invalid problemID" );
    }

    // Get parameters needed to build the RadDifOp
    auto RadDifOp_db = myRadDifModel->getRadiationDiffusionFD_input_db( );

    /****************************************************************
    * Create a mesh                                                 *
    ****************************************************************/
    // Put variable "dim" into mesh Database 
    mesh_db->putScalar<int>( "dim", RadDifOp_db->getScalar<int>( "dim" ) );
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh = createBoxMesh( comm, mesh_db );
    
    AMP::pout << "The discretization database is" << std::endl;
    AMP::pout << "------------------------------" << std::endl;
    RadDifOp_db->print( AMP::pout );
    AMP::pout << "------------------------------" << std::endl;
    
    

    /****************************************************************
    * Create a BERadDifOperator                                     *
    ****************************************************************/
    // Create an OperatorParameters object, from a Database.
    auto Op_db = std::make_shared<AMP::Database>( "Op_db" );
    auto OpParams = std::make_shared<AMP::Operator::OperatorParameters>( Op_db );
    // Operator parameters has: a mesh, an operator, and a memory location. We just set the mesh 
    OpParams->d_Mesh = mesh;
    OpParams->d_db   = RadDifOp_db; // Set DataBase of parameters.

    // Create BERadDifOp 
    auto myBERadDifOp = std::make_shared<AMP::Operator::BERadDifOp>( OpParams );  
    // Extract the underlying RadDifOp
    auto myRadDifOp = myBERadDifOp->d_RadDifOp; 

    // orderingTest( myRadDifOp->d_scalarDOFMan, myRadDifOp->d_multiDOFMan );
    // AMP_ERROR( "Halt" );

    // Create an OperatorFactory and register Jacobian of BERadDifOp in it 
    auto & operatorFactory = AMP::Operator::OperatorFactory::getFactory();
    operatorFactory.registerFactory( "BERadDifOpPJac", AMP::Operator::BERadDifOpPJac::create );

    // Create a SolverFactory and register preconditioner(s) of the above operator in it 
    auto & solverFactory = AMP::Solver::SolverFactory::getFactory();
    solverFactory.registerFactory( "BERadDifOpPJacOpSplitPrec", AMP::Solver::BERadDifOpPJacOpSplitPrec::create );
    
    // Create hassle-free wrappers around ic, source term and exact solution
    auto icFun        = std::bind( &AMP::Operator::RadDifModel::initialCondition, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 );
    auto PDESourceFun = std::bind( &AMP::Operator::RadDifModel::sourceTerm, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 );
    auto uexactFun    = std::bind( &AMP::Operator::RadDifModel::exactSolution, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 );


    // If using a manufactured model, overwrite the default RadDifOp boundary condition functions to point to those of the Manufactured model
    if ( problemID == "Manufactured" ) {
        AMP::pout << "Manufactured RadDif model BCs are being used" << std::endl;
        auto myManufacturedRadDifModel = std::dynamic_pointer_cast<AMP::Operator::Manufactured_RadDifModel>( myRadDifModel );
        AMP_INSIST( myManufacturedRadDifModel, "Model is null" );
        
        // Point the Robin E BC values in the RadDifOp to those given by the manufactured problem
        myRadDifOp->setRobinFunctionE( std::bind( &AMP::Operator::Manufactured_RadDifModel::getRobinValueE, &( *myManufacturedRadDifModel ), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4 ) );

        // Point the pseudo Neumann T BC values in the radDifOp to those given by the manufactured problem
        myRadDifOp->setPseudoNeumannFunctionT( std::bind( &AMP::Operator::Manufactured_RadDifModel::getPseudoNeumannValueT, &( *myManufacturedRadDifModel ), std::placeholders::_1, std::placeholders::_2 ) );
    }


    /****************************************************************
    * Set up relevant vectors                                       *
    ****************************************************************/
    // Create required vectors over the mesh
    auto numSolVec    = myRadDifOp->createInputVector();
    auto manSolVec    = myRadDifOp->createInputVector();
    auto errorVec     = myRadDifOp->createInputVector();
    auto BDFSourceVec = myRadDifOp->createInputVector();

    // Create initial condition vector
    auto ic = myRadDifOp->createInputVector();
    myRadDifOp->fillMultiVectorWithFunction( ic, icFun );
    ic->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Create vectors to hold current and new solution (when integrating)
    auto sol_old = ic->clone( );
    sol_old->copyVector( ic ); 
    auto sol_new = ic->clone( );
    sol_new->copyVector( ic ); 

    /****************************************************************
    * Set up implicit time integrator                               *
    ****************************************************************/
    // Ensure BDF integrator is being used
    auto bdf_ti = { "Backward Euler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    auto ti_name   = ti_db->getScalar<std::string>( "name" );
    auto is_bdf = ( std::find( bdf_ti.begin(), bdf_ti.end(), ti_name ) != bdf_ti.end() );
    AMP_INSIST( is_bdf, "Implementation assumes BDF integrator" );

    // Parameters for time integrator
    auto h = myRadDifOp->getMeshSize()[0];
    //double dt = 1.0 * h * h;
    //double dt = 0.5 * h;
    double dt = ti_db->getScalar<double>( "initial_dt" );
    auto tiParams           = std::make_shared<AMP::TimeIntegrator::TimeIntegratorParameters>( ti_db );
    tiParams->d_ic_vector   = ic;
    tiParams->d_operator    = myBERadDifOp;
    tiParams->d_pSourceTerm = BDFSourceVec; // Point source vector to our source vector

    // Create timeIntegrator from factory
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegrator> timeIntegrator = AMP::TimeIntegrator::TimeIntegratorFactory::create( tiParams );
    
    // Cast to implicit integrator
    auto implicitIntegrator =
            std::dynamic_pointer_cast<AMP::TimeIntegrator::ImplicitIntegrator>( timeIntegrator );

    // Tell implicitIntegrator how to tell our operator what the time step is
    implicitIntegrator->setTimeScalingFunction(
        std::bind( &AMP::Operator::BERadDifOp::setGamma, &( *myBERadDifOp ), std::placeholders::_1 ) );

    
    int step = 0;
    int n = -1;
    //int n = disc_db->getDatabase( "mesh" )->getScalar<int>( "n" );
    std::string out_dir = "out/n" + std::to_string(n) + "_" + std::to_string(RadDifOp_db->getScalar<int>( "dim" )) + "D/";
    std::string num_dir = out_dir + "ETnum";
    std::string man_dir = out_dir + "ETman";   

    // Only output solution data if requested and on single process
    bool outputSolution = ti_db->getWithDefault<bool>( "outputSolution", false );
    outputSolution      = outputSolution && (comm.getSize() == 1);
    #if 1
    if ( outputSolution ) {
        // Remove outdir and its contents if it already exists
        if (std::filesystem::is_directory(out_dir)) {
            std::filesystem::remove_all( out_dir ); 
        }
        //create outdir
        std::filesystem::create_directory( out_dir );
        // Write IC
        {
            double T = 0.0;
            AMP::IO::AsciiWriter vecWriter_man;
            std::string name = std::to_string( T );
            manSolVec->setName( name );
            sol_new->setName( name );
            vecWriter_man.registerVector( ic );
            vecWriter_man.writeFile( man_dir, step, T  );
            AMP::IO::AsciiWriter vecWriter_num;
            vecWriter_num.registerVector( sol_new );
            vecWriter_num.writeFile( num_dir, step, T  );
        }
    }
    #endif


    // Integrate!
    double finalTime = timeIntegrator->getFinalTime();
    double T = 0.0;
    timeIntegrator->setInitialDt( dt );
    AMP::pout << "--------------------------" << std::endl;
    AMP::pout << "Beginning time integration" << std::endl;
    AMP::pout << "--------------------------" << std::endl;

    // Step from T to T + dt, so long as T is smaller than the final time.
    while ( T < finalTime ) {

        // Try to advance the solution with the current dt; if that fails (for whatever reason) we'll try again with a different dt 
        bool good_solution = false;
        while ( !good_solution ) {

            // Set the solution-independent source term; note that this approach only works for implicit multistep methods
            myRadDifModel->setCurrentTime( T + dt ); // Set model to new time---this ensures the source term and Robin values are sampled at the new time.
            // Fill BDF source vector with sol-independent PDE source term
            myRadDifOp->fillMultiVectorWithFunction( BDFSourceVec, PDESourceFun );

            // Attempt to advance the solution with the current dt, getting return code from solver.
            int solver_retcode = timeIntegrator->advanceSolution( dt, T == 0.0, sol_old, sol_new );
            //AMP::pout << "step=" << step << ": solver_retcode=" << solver_retcode << "\n";

            // Check the computed solution (returns true if it is acceptable, and false otherwise) 
            good_solution = timeIntegrator->checkNewSolution( );

            // If step succeeded, update solution, time, step counter, etc.
            if ( good_solution ) {
                timeIntegrator->updateSolution( );
                sol_old->copyVector( sol_new );
                T += dt; 
                step++;
            } 
            // else {
            //     AMP_WARNING( "Something didn't pass with current dt=" + std::to_string(dt) + ", next dt=" + std::to_string(dt_next) + "\n" );
            // }

            /** Note that ImplicitIntegrator's getNextDt() will call BDF's integratorSpecificGetNextDt(), and parse it the solver_retcode
             * Return the next time increment through which to advance the solution.
             * The good_solution is the value returned by a call to checkNewSolution(),
             * which determines whether the computed solution is acceptable or not.
             * The integer solver_retcode is the return code generated by the
             * nonlinear solver.   This value must be interpreted in a manner
             * consistant with the solver in use.
             */
            double dt_next = implicitIntegrator->getNextDt( good_solution );

            // Set dt for the next step
            dt = dt_next;
        }
        

        /* Compare numerical solution with manufactured solution */
        if ( myRadDifModel->d_exactSolutionAvailable ) {
            myRadDifModel->setCurrentTime( T );
            myRadDifOp->fillMultiVectorWithFunction( manSolVec, uexactFun );
            errorVec->subtract( *sol_new, *manSolVec );
            AMP::pout << "----------------------------------------" << std::endl;
            AMP::pout << "Manufactured discretization error norms:" << std::endl;
            auto enorms = getDiscreteNorms( myRadDifOp->getMeshSize(), errorVec );
            AMP::pout << "||e||=(" << enorms[0] << "," << enorms[1] << "," << enorms[2] << ")" << std::endl;
            AMP::pout << "----------------------------------------" << std::endl;

            if ( outputSolution ) {
                std::string name = std::to_string( T );
                manSolVec->setName( name );
                AMP::IO::AsciiWriter vecWriter_man;
                vecWriter_man.registerVector( manSolVec );
                vecWriter_man.writeFile( man_dir, step, T  );
            }
        }

        // Write numerical solution to file.
        #if 1
        if ( outputSolution ) {
            std::string name = std::to_string( T );
            sol_new->setName( name );
            AMP::IO::AsciiWriter vecWriter_num;
            vecWriter_num.registerVector( sol_new );
            vecWriter_num.writeFile( num_dir, step, T  );
        }
        #endif
        

        // Drop out if we've exceeded max steps
        if ( !timeIntegrator->stepsRemaining() ) {
            AMP_WARNING( "max_integrator_steps has been reached, dropping out of loop now..." );
            break;
        }
    }
    // End of ti loop

    timeIntegrator->printClassData( AMP::pout );
}
// end of driver()




/*  Input usage is: >> <input_file>
    e.g., >> mpirun -n 1 FD input_db
*/
int main( int argc, char **argv )
{
    if (argc != 2) {
        AMP_ERROR( "An input file must be specified through the command line" );
    }

    AMP::AMPManager::startup( argc, argv );

    // Create a global communicator
    AMP::AMP_MPI comm( AMP_COMM_WORLD );
    int myRank   = comm.getRank();
    int numRanks = comm.getSize();


    // TODO: Add test file here that we want to parse in and put guards around it depending on hypre being included since boomer is used.

    // Unpack inputs
    //int n = atoi(argv[1]); // Grid size in each dimension
    
    std::string input_db_name = argv[1]; 

    //auto input_db = AMP::Database::parseInputFile( "../src/input_db" );
    auto input_db = AMP::Database::parseInputFile( input_db_name );

    // Driver
    driver( comm, input_db );

    AMP::AMPManager::shutdown();

    return 0;
}