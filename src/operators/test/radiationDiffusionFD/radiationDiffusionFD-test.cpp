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

#include "AMP/operators/radiationDiffusionFD/discretization.hpp"
#include "AMP/operators/radiationDiffusionFD/utils.hpp"
#include "AMP/operators/radiationDiffusionFD/solver.hpp"
#include "AMP/operators/radiationDiffusionFD/model.hpp"


/*
Try turning on the auto component scaling once everything is working. this means I have to change the apply functions in my BEOperators. Bobby mentioned that since one component is much bigger than the other, if you look at their relative convergence, one drops much more than the other.

I'll also need to understand what exactly this is doing because if there are constants being scaling introduced, does that influence the linearization?
*/




/* In a multiVector, DOFs are ordered by variable on process.

Test routines to map between different orderings
*/
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
    std::shared_ptr<RadDifModel> myRadDifModel;

    if ( problemID == "Mousseau_etal_2000" ) {
        auto myRadDifModel_ = std::make_shared<Mousseau_etal_2000_RadDifModel>( PDE_basic_db, PDE_mspecific_db );
        myRadDifModel       = myRadDifModel_;

    } else if ( problemID == "Manufactured" ) {
        auto myRadDifModel_ = std::make_shared<Manufactured_RadDifModel>( PDE_basic_db, PDE_mspecific_db );
        myRadDifModel = myRadDifModel_; 

    } else {
        AMP_ERROR( "Invalid problemID" );
    }

    // Get general PDE model parameters to build the RadDifOp
    auto PDE_general_db = myRadDifModel->getGeneralPDEModelParameters( );
    auto PDE_general_db_ = std::make_unique<AMP::Database>( *PDE_general_db );

    /****************************************************************
    * Create a mesh                                                 *
    ****************************************************************/
    // Put variable "dim" into mesh Database 
    mesh_db->putScalar<int>( "dim", PDE_general_db->getScalar<int>( "dim" ) );
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh = createBoxMesh( comm, mesh_db );
    
    // Package PDE and mesh dbs into a discretization db
    auto disc_db = std::make_shared<AMP::Database>( "disc_db" );
    disc_db->putDatabase( "PDE",  std::move( PDE_general_db_ ) );
    disc_db->putDatabase( "mesh", std::move( mesh_db ) );
    //disc_db->putScalar<int>( "print_info_level", 1 );
    disc_db->putScalar<int>( "print_info_level", disc_db->getDatabase( "PDE" )->getScalar<int>( "print_info_level" ) );


    AMP::pout << "The discretization database is" << std::endl;
    AMP::pout << "------------------------------" << std::endl;
    disc_db->print( AMP::pout );
    AMP::pout << "------------------------------" << std::endl;
    
    

    /****************************************************************
    * Create a BERadDifOperator                                     *
    ****************************************************************/
    // Create an OperatorParameters object, from a Database.
    auto Op_db = std::make_shared<AMP::Database>( "Op_db" );
    auto OpParams = std::make_shared<AMP::Operator::OperatorParameters>( Op_db );
    // Operator parameters has: a mesh, an operator, and a memory location. We just set the mesh 
    OpParams->d_Mesh = mesh;
    OpParams->d_db   = disc_db; // Set DataBase of parameters.

    // Create BERadDifOp 
    auto myBERadDifOp = std::make_shared<BERadDifOp>( OpParams );  
    // Extract the underlying RadDifOp
    auto myRadDifOp = myBERadDifOp->d_RadDifOp; 

    // orderingTest( myRadDifOp->d_scalarDOFMan, myRadDifOp->d_multiDOFMan );
    // AMP_ERROR( "Halt" );

    // Create an OperatorFactory and register Jacobian of BERadDifOp in it 
    auto & operatorFactory = AMP::Operator::OperatorFactory::getFactory();
    operatorFactory.registerFactory( "BERadDifOpJac", BERadDifOpJac::create );

    // Create a SolverFactory and register preconditioner(s) of the above operator in it 
    auto & solverFactory = AMP::Solver::SolverFactory::getFactory();
    solverFactory.registerFactory( "BERadDifOpJacOpSplitPrec", BERadDifOpJacOpSplitPrec::create );
    solverFactory.registerFactory( "BERadDifOpJacMonolithic", BERadDifOpJacMonolithic::create );
    
    // Create hassle-free wrappers around ic, source term and exact solution
    auto icFun        = std::bind( &RadDifModel::initialCondition, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 );
    auto PDESourceFun = std::bind( &RadDifModel::sourceTerm, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 );
    auto uexactFun    = std::bind( &RadDifModel::exactSolution, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 );


    // If using a manufactured model, overwrite the default RadDifOp boundary condition functions to point to those of the Manufactured model
    if ( problemID == "Manufactured" ) {
        AMP::pout << "Manufactured RadDif model BCs are being used" << std::endl;
        auto myManufacturedRadDifModel = std::dynamic_pointer_cast<Manufactured_RadDifModel>( myRadDifModel );
        AMP_INSIST( myManufacturedRadDifModel, "Model is null" );
        
        // Point the Robin E BC values in the RadDifOp to those given by the manufactured problem
        myRadDifOp->setRobinFunctionE( std::bind( &Manufactured_RadDifModel::getRobinValueE, &( *myManufacturedRadDifModel ), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4 ) );

        // Point the pseudo Neumann T BC values in the radDifOp to those given by the manufactured problem
        myRadDifOp->setPseudoNeumannFunctionT( std::bind( &Manufactured_RadDifModel::getPseudoNeumannValueT, &( *myManufacturedRadDifModel ), std::placeholders::_1, std::placeholders::_2 ) );
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
    auto h = disc_db->getDatabase( "mesh" )->getScalar<double>( "h" );
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
        std::bind( &BERadDifOp::setGamma, &( *myBERadDifOp ), std::placeholders::_1 ) );

    
    int step = 0;
    int n = disc_db->getDatabase( "mesh" )->getScalar<int>( "n" );
    std::string out_dir = "out/n" + std::to_string(n) + "_" + std::to_string(disc_db->getDatabase( "mesh" )->getScalar<int>( "dim" )) + "D/";
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
            auto enorms = getDiscreteNorms( disc_db->getDatabase( "mesh" )->getScalar<double>( "h" ), errorVec );
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

    e.g., >> mpirun -n 1 FD ../src/data/input_db
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

    // Unpack inputs
    //int n = atoi(argv[1]); // Grid size in each dimension
    
    std::string input_db_name = argv[1]; 

    //auto input_db = AMP::Database::parseInputFile( "../src/input_db" );
    auto input_db = AMP::Database::parseInputFile( input_db_name );
     
    // // Create DB with PDE- and mesh-related quantities
    // auto PDE_db = input_db->getDatabase( "PDE" );
    // PDE_db->putScalar( "n",   n );


    // // Store the mesh size: There are n+1 points on the mesh covering the domain [-h/2, 1+h/2].
    // // Eastiest to think about the case of n cells...
    
    // PDE_db->putScalar( "h",   h );

    // // // Constant in PDE
    // double z = 5.0; // todo: this is allowed to vary in the domain...
    // PDE_db->putScalar( "z",   z );

    //double h = 1.0 / (n-1); // n+1 points between -h/2,1+h/2
    //auto mesh_db = AMP::Database::create( "n", n, "h", h );
    //mesh_db->print( AMP::pout );

    //input_db->putDatabase( "mesh", std::move(mesh_db) );

    // // Print out input database
    // AMP::pout << "Input database is:" << std::endl;
    // AMP::pout << "-----------------" << std::endl;
    // input_db->print( AMP::pout );
    // AMP::pout << "-----------------" << std::endl;


    // Driver
    driver( comm, input_db );

    AMP::AMPManager::shutdown();

    return 0;
}


// Example of how to generate a multivector
// ----------------------------------------
// // Create vectors for E and T
// auto E_var = std::make_shared<AMP::LinearAlgebra::Variable>( "E" );
// auto T_var = std::make_shared<AMP::LinearAlgebra::Variable>( "T" );
// auto E_vec = AMP::LinearAlgebra::createVector( this->d_multiDOFMan->getDOFManager(0), E_var );
// auto T_vec = AMP::LinearAlgebra::createVector( this->d_multiDOFMan->getDOFManager(1), T_var );
// // Create a multivector consisting of E and T vectors
// auto tmp_var_in = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "inputVariable" );
// auto ET_vec = AMP::LinearAlgebra::MultiVector::create( tmp_var_in, this->getMesh()->getComm() );
// ET_vec->addVector( E_vec );
// ET_vec->addVector( T_vec );
// ET_vec->setToScalar( 1.0 );
// ET_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );


// Some other code:

// --- Example of how to read in mesh file from input
// AMP_INSIST( argc == 2, "Usage is:  createmesh  inputFile" );
// std::string filename( argv[1] );
// AMP::pout << "the filename is " << filename << "\n";
//createBoxMesh( filename );
//void createBoxMesh( const std::string &input_file )
// // Read the input file
// auto input_db = AMP::Database::parseInputFile( input_file );
// // Get the Mesh database and create the mesh parameters
// auto database = input_db->getDatabase( "Mesh" )->getDatabase( "Mesh_1" );
//auto params = std::make_shared<AMP::Mesh::MeshParameters>( database );
//params->setComm( globalComm );
// // Create the meshes from the input database
// //auto mesh = AMP::Mesh::MeshFactory::create( params );


// // --- Example of how to create a DB in on pass. 
// std::vector<int> size = {4, 4};
// std::vector<int> range = {0, 1, 0, 1};
// auto myMeshDataBase = AMP::Database::create( 
//     "name", "boxMeshData", 
//     "MeshName", "mesh",
//     "dim", 2,
//     "Generator", "cube",
//     "Size", size,
//     "Range", range);
// AMP::pout << "myDataBase = \n" << myMeshDataBase->print() << std::endl;
// // Get the Mesh database and create the mesh parameters
// //auto database = myMeshDataBase->getDatabase( "boxMeshData" );
// // Create MeshParameters; note the input has to be a std::make_shared<AMP::DataBase>
// auto myMeshParams   = std::make_shared<AMP::Mesh::MeshParameters>( database );
// myMeshParams->setComm( comm );
// // Create the mesh from the meshParameters
// auto mesh = AMP::Mesh::BoxMesh( myMeshParams );


// //--- Example showing how to iterate over elements on the mesh
// int meshDim = static_cast<int>( mesh->getDim() );
// elem is a pointer to a AMP::Mesh::MeshElement
// for (auto elem = myMeshIterator.begin(); elem != myMeshIterator.end(); elem++) {

//     //std::cout << "element class = " << elem->elementClass() << std::endl;

//     // Get structure used to identify the mesh element
//     AMP::Mesh::MeshElementID id = elem->globalID();

//     // Understanding mesh object
//     if (meshDim == 1) {
//         std::cout << "Element " << id.local_id() << " on process " << elem->globalOwnerRank() << " has coord x=" << elem->coord(0) << std::endl;
//     } else if (meshDim == 2) {
//         std::cout << "Element " << id.local_id() << " on process " << elem->globalOwnerRank() << " has coords (x,y)=(" << elem->coord(0) << "," << elem->coord(1) << ")" << std::endl;
//     }

//     // TODO: I don't understand what this does... and it's influenced by gcw
//     // std::cout << "\t rowDOFs are: ";
//     // auto myRowDOFs = myDOFManager->getRowDOFs(id);
//     // for (size_t DOF : myRowDOFs) {
//     //    std::cout << DOF << " ";
//     // }
//     // std::cout << std::endl;
// }



// Example of function that returns col ids for a given row.. Note this isn't too robust because it relies on the assumption of how the underlying DOFs are ordered, and I don't really know this
//auto getColumnIDs = std::bind(Laplacian1DColIDs, std::placeholders::_1, n);
// Suppose there are n+1 DOFs, indexed 0, 1, ..., n. Here, DOF 0 and n are the boundary DOFs and will just have a non-zero diagonal.
// std::vector<size_t> Laplacian1DColIDs(size_t row, size_t n) {
//     std::vector<size_t> cols;
//     if (row == 0) {
//         cols.push_back(0);
//     } else if (row == n) {
//         cols.push_back(n);
//     } else {
//         cols.push_back(row-1);
//         cols.push_back(row);
//         cols.push_back(row+1);
//     }
//     return cols;
// }


// void fillWithPseudoLaplacian( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix,
//                               std::shared_ptr<AMP::Discretization::DOFManager> dofmap )
// {
//     // Iterate through rows
//     for ( size_t i = dofmap->beginDOF(); i != dofmap->endDOF(); i++ ) {
//         std::cout << "i = " << i << std::endl;
//         // Get pointer to cols
//         auto cols        = matrix->getColumnIDs( i );
//         const auto ncols = cols.size();
//         std::vector<double> vals( ncols );
//         for ( size_t j = 0; j != ncols; j++ ) {
//             std::cout << "\tj = " << j << std::endl;
//             if ( cols[j] == i )
//                 vals[j] = static_cast<double>( ncols );
//             else
//                 vals[j] = -1;
//         }
//         if ( ncols ) {
//             matrix->setValuesByGlobalID<double>( 1, ncols, &i, cols.data(), vals.data() );
//         }
//     }
// }


// // Create vectors for E and T
    // auto E_var = std::make_shared<AMP::LinearAlgebra::Variable>( "E" );
    // auto T_var = std::make_shared<AMP::LinearAlgebra::Variable>( "T" );
    // auto E_vec = AMP::LinearAlgebra::createVector( DOFManagersVec[0], E_var );
    // auto T_vec = AMP::LinearAlgebra::createVector( DOFManagersVec[1], T_var );
    // // Ditto for manufactured solution
    // auto Eman_vec = AMP::LinearAlgebra::createVector( DOFManagersVec[0], E_var ); 
    // auto Tman_vec = AMP::LinearAlgebra::createVector( DOFManagersVec[1], T_var );
    // // Ditto for residual vectors
    // auto rE_var = std::make_shared<AMP::LinearAlgebra::Variable>( "rE" );
    // auto rT_var = std::make_shared<AMP::LinearAlgebra::Variable>( "rT" );
    // auto rE_vec = AMP::LinearAlgebra::createVector( DOFManagersVec[0], rE_var );
    // auto rT_vec = AMP::LinearAlgebra::createVector( DOFManagersVec[1], rT_var );
    // // Ditto for source vectors
    // auto sE_var = std::make_shared<AMP::LinearAlgebra::Variable>( "sE" );
    // auto sT_var = std::make_shared<AMP::LinearAlgebra::Variable>( "sT" );
    // auto sE_vec = AMP::LinearAlgebra::createVector( DOFManagersVec[0], sE_var );
    // auto sT_vec = AMP::LinearAlgebra::createVector( DOFManagersVec[1], sT_var );

    // // Create a multivector consisting of E and T vectors
    // auto tmp_var_in = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "inputVariable" );
    // auto ET_vec = AMP::LinearAlgebra::MultiVector::create( tmp_var_in, comm );
    // ET_vec->addVector( E_vec );
    // ET_vec->addVector( T_vec );
    // auto ETman_vec = AMP::LinearAlgebra::MultiVector::create( tmp_var_in, comm );
    // ETman_vec->addVector( Eman_vec );
    // ETman_vec->addVector( Tman_vec );
    // // Ditto for residual vectors
    // auto tmp_var_out = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "outputVariable" );
    // auto rET_vec = AMP::LinearAlgebra::MultiVector::create( tmp_var_out, comm );
    // rET_vec->addVector( rE_vec );
    // rET_vec->addVector( rT_vec );
    // // Ditto for source vectors
    // auto tmp_var_source = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "sourceVariable" );
    // auto sET_vec = AMP::LinearAlgebra::MultiVector::create( tmp_var_source, comm );
    // sET_vec->addVector( sE_vec );
    // sET_vec->addVector( sT_vec );