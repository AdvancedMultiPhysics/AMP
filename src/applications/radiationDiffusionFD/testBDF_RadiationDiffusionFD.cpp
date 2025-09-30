#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDBDFWrappers.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionModel.h"
#include "AMP/operators/testHelpers/FDHelper.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/radiationDiffusionFDOpSplitPrec/RadiationDiffusionFDOpSplitPrec.h"
#include "AMP/time_integrators/ImplicitIntegrator.h"
#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include <filesystem>
#include <iomanip>
#include <iostream>


/** This test applies BDF integration to a radiation-diffusion problem discretized with finite
 * differences. There is a manufactured solution available, which can be used to measure the
 * discretization error at each time step.
 */

void driver( AMP::AMP_MPI comm, AMP::UnitTest *ut, const std::string &inputFileName )
{

    // Input and output file names
    std::string input_file = inputFileName;
    std::string log_file   = "output_" + inputFileName;

    AMP::logOnlyNodeZero( log_file );
    AMP::pout << "Running driver with input " << input_file << std::endl;

    auto input_db = AMP::Database::parseInputFile( input_file );
    AMP::plog << "Input database:" << std::endl;
    AMP::plog << "---------------" << std::endl;
    input_db->print( AMP::plog );


    /****************************************************************
     * Re-organize database input                                    *
     ****************************************************************/
    // Unpack databases
    auto PDE_basic_db = input_db->getDatabase( "PDE" );
    auto mesh_db      = input_db->getDatabase( "Mesh" );
    auto ti_db        = input_db->getDatabase( "TimeIntegrator" );

    // Basic error check the input has required things
    AMP_INSIST( PDE_basic_db, "PDE is null" );
    AMP_INSIST( mesh_db, "Mesh is null" );
    AMP_INSIST( ti_db, "TimeIntegrator is null" );

    // Get PDE model-specific parameter database
    auto modelID          = PDE_basic_db->getScalar<std::string>( "modelID" );
    auto PDE_mspecific_db = input_db->getDatabase( modelID + "_Parameters" );
    AMP_INSIST( PDE_mspecific_db,
                "Input must have the model-specific database: '" + modelID + "_Parameters'" );

    // Push problem dimension into PDE_basic_db
    PDE_basic_db->putScalar<int>( "dim", mesh_db->getScalar<int>( "dim" ) );


    /****************************************************************
     * Create radiation-diffusion model                              *
     ****************************************************************/
    std::shared_ptr<AMP::Operator::RadDifModel> myRadDifModel;

    if ( modelID == "Mousseau_etal_2000" ) {
        auto myRadDifModel_ = std::make_shared<AMP::Operator::Mousseau_etal_2000_RadDifModel>(
            PDE_basic_db, PDE_mspecific_db );
        myRadDifModel = myRadDifModel_;

    } else if ( modelID == "Manufactured" ) {
        auto myRadDifModel_ = std::make_shared<AMP::Operator::Manufactured_RadDifModel>(
            PDE_basic_db, PDE_mspecific_db );
        myRadDifModel = myRadDifModel_;

    } else {
        AMP_ERROR( "Invalid modelID" );
    }

    // Get parameters needed to build the RadDifOp
    auto RadDifOp_db = myRadDifModel->getRadiationDiffusionFD_input_db();


    /****************************************************************
     * Create a mesh                                                 *
     ****************************************************************/
    // Create MeshParameters
    auto mesh_params = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mesh_params->setComm( comm );
    // Create Mesh
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh = AMP::Mesh::BoxMesh::generate( mesh_params );


    /****************************************************************
     * Create a BDFRadDifOperator                                     *
     ****************************************************************/
    // Create an OperatorParameters object, from a Database.
    auto Op_db    = std::make_shared<AMP::Database>( "Op_db" );
    auto OpParams = std::make_shared<AMP::Operator::OperatorParameters>( Op_db );
    // Operator parameters has: a mesh, an operator, and a memory location. We just set the mesh
    OpParams->d_Mesh = mesh;
    OpParams->d_db   = RadDifOp_db; // Set DataBase of parameters.

    // Create BDFRadDifOp
    auto myBDFRadDifOp = std::make_shared<AMP::Operator::BDFRadDifOp>( OpParams );
    // Extract the underlying RadDifOp
    auto myRadDifOp = myBDFRadDifOp->d_RadDifOp;

    // Create an OperatorFactory and register Jacobian of BDFRadDifOp in it
    auto &operatorFactory = AMP::Operator::OperatorFactory::getFactory();
    operatorFactory.registerFactory( "BDFRadDifOpPJac", AMP::Operator::BDFRadDifOpPJac::create );

    // Create a SolverFactory and register preconditioner(s) of the above operator in it
    auto &solverFactory = AMP::Solver::SolverFactory::getFactory();
    solverFactory.registerFactory( "BDFRadDifOpPJacOpSplitPrec",
                                   AMP::Solver::BDFRadDifOpPJacOpSplitPrec::create );

    // Create hassle-free wrappers around ic, source term and exact solution
    auto icFun        = std::bind( &AMP::Operator::RadDifModel::initialCondition,
                            &( *myRadDifModel ),
                            std::placeholders::_1,
                            std::placeholders::_2 );
    auto PDESourceFun = std::bind( &AMP::Operator::RadDifModel::sourceTerm,
                                   &( *myRadDifModel ),
                                   std::placeholders::_1,
                                   std::placeholders::_2 );
    auto uexactFun    = std::bind( &AMP::Operator::RadDifModel::exactSolution,
                                &( *myRadDifModel ),
                                std::placeholders::_1,
                                std::placeholders::_2 );


    // If using a manufactured model, overwrite the default RadDifOp boundary condition functions to
    // point to those of the Manufactured model
    if ( modelID == "Manufactured" ) {
        AMP::pout << "Manufactured RadDif model BCs are being used" << std::endl;
        auto myManufacturedRadDifModel =
            std::dynamic_pointer_cast<AMP::Operator::Manufactured_RadDifModel>( myRadDifModel );
        AMP_INSIST( myManufacturedRadDifModel, "Model is null" );

        // Point the Robin E BC values in the RadDifOp to those given by the manufactured problem
        myRadDifOp->setBoundaryFunctionE(
            std::bind( &AMP::Operator::Manufactured_RadDifModel::getBoundaryFunctionValueE,
                       &( *myManufacturedRadDifModel ),
                       std::placeholders::_1,
                       std::placeholders::_2 ) );

        // Point the pseudo Neumann T BC values in the radDifOp to those given by the manufactured
        // problem
        myRadDifOp->setBoundaryFunctionT(
            std::bind( &AMP::Operator::Manufactured_RadDifModel::getBoundaryFunctionValueT,
                       &( *myManufacturedRadDifModel ),
                       std::placeholders::_1,
                       std::placeholders::_2 ) );
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
    fillMultiVectorWithFunction( myRadDifOp->getMesh(),
                                 myRadDifOp->getGeomType(),
                                 myRadDifOp->getScalarDOFManager(),
                                 ic,
                                 icFun );
    ic->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Create vectors to hold current and new solution (when integrating)
    auto sol_old = ic->clone();
    sol_old->copyVector( ic );
    auto sol_new = ic->clone();
    sol_new->copyVector( ic );


    /****************************************************************
     * Set up implicit time integrator                               *
     ****************************************************************/
    // Ensure BDF integrator is being used
    auto bdf_ti  = { "Backward Euler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    auto ti_name = ti_db->getScalar<std::string>( "name" );
    auto is_bdf  = ( std::find( bdf_ti.begin(), bdf_ti.end(), ti_name ) != bdf_ti.end() );
    AMP_INSIST( is_bdf, "Implementation assumes BDF integrator" );

    // Parameters for time integrator
    // auto h = myRadDifOp->getMeshSize()[0];
    // double dt = 1.0 * h * h;
    // double dt = 0.5 * h;
    double dt     = ti_db->getScalar<double>( "initial_dt" );
    auto tiParams = std::make_shared<AMP::TimeIntegrator::TimeIntegratorParameters>( ti_db );
    tiParams->d_ic_vector   = ic;
    tiParams->d_operator    = myBDFRadDifOp;
    tiParams->d_pSourceTerm = BDFSourceVec; // Point source vector to our source vector

    // Create timeIntegrator from factory
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegrator> timeIntegrator =
        AMP::TimeIntegrator::TimeIntegratorFactory::create( tiParams );

    // Cast to implicit integrator
    auto implicitIntegrator =
        std::dynamic_pointer_cast<AMP::TimeIntegrator::ImplicitIntegrator>( timeIntegrator );

    // Tell implicitIntegrator how to tell our operator what the time step is
    implicitIntegrator->setTimeScalingFunction( std::bind(
        &AMP::Operator::BDFRadDifOp::setGamma, &( *myBDFRadDifOp ), std::placeholders::_1 ) );

    // Tell implicitIntegrator how to tell our operator what the component scalings are
    implicitIntegrator->setComponentScalingFunction(
        std::bind( &AMP::Operator::BDFRadDifOp::setComponentScalings,
                   &( *myBDFRadDifOp ),
                   std::placeholders::_1,
                   std::placeholders::_2 ) );


    int step = 0;
    // int n = mesh_db->getScalar<int>( "Size" );
    int n               = mesh_db->getArray<int>( "Size" )[0];
    std::string out_dir = "out/n" + std::to_string( n ) + "_" +
                          std::to_string( mesh_db->getScalar<int>( "dim" ) ) + "D/";
    std::string num_dir = out_dir + "ETnum";
    std::string man_dir = out_dir + "ETman";

    // Only output solution data if requested and on single process
    bool outputSolution = ti_db->getWithDefault<bool>( "outputSolution", false );
    outputSolution      = outputSolution && ( comm.getSize() == 1 );
#if 1
    if ( outputSolution ) {
        // Remove outdir and its contents if it already exists
        if ( std::filesystem::is_directory( out_dir ) ) {
            std::filesystem::remove_all( out_dir );
        }
        // create outdir
        std::filesystem::create_directory( out_dir );
        // Write IC
        {
            double T = 0.0;
            AMP::IO::AsciiWriter vecWriter_man;
            std::string name = std::to_string( T );
            manSolVec->setName( name );
            sol_new->setName( name );
            vecWriter_man.registerVector( ic );
            vecWriter_man.writeFile( man_dir, step, T );
            AMP::IO::AsciiWriter vecWriter_num;
            vecWriter_num.registerVector( sol_new );
            vecWriter_num.writeFile( num_dir, step, T );
        }
    }
#endif


    // Integrate!
    double finalTime = timeIntegrator->getFinalTime();
    double T         = 0.0;
    timeIntegrator->setInitialDt( dt );
    AMP::pout << "--------------------------" << std::endl;
    AMP::pout << "Beginning time integration" << std::endl;
    AMP::pout << "--------------------------" << std::endl;

    // Step from T to T + dt, so long as T is smaller than the final time.
    while ( T < finalTime ) {

        // Try to advance the solution with the current dt; if that fails (for whatever reason)
        // we'll try again with a different dt
        bool good_solution = false;
        while ( !good_solution ) {

            // Set the solution-independent source term; note that this approach only works for
            // implicit multistep methods
            myRadDifModel->setCurrentTime(
                T + dt ); // Set model to new time---this ensures the source term and Robin values
                          // are sampled at the new time.
            // Fill BDF source vector with sol-independent PDE source term
            fillMultiVectorWithFunction( myRadDifOp->getMesh(),
                                         myRadDifOp->getGeomType(),
                                         myRadDifOp->getScalarDOFManager(),
                                         BDFSourceVec,
                                         PDESourceFun );

            // Attempt to advance the solution with the current dt, getting return code from solver.
            auto solver_retcode = timeIntegrator->advanceSolution( dt, T == 0.0, sol_old, sol_new );

            // Check the computed solution (returns true if it is acceptable, and false otherwise)
            good_solution = timeIntegrator->checkNewSolution();

            // If step succeeded, update solution, time, step counter, etc.
            if ( good_solution ) {
                timeIntegrator->updateSolution();
                sol_old->copyVector( sol_new );
                T += dt;
                step++;
            }

            double dt_next = implicitIntegrator->getNextDt( good_solution );

            // Set dt for the next step
            dt = dt_next;
        }


        /* Compare numerical solution with manufactured solution */
        if ( myRadDifModel->exactSolutionAvailable() ) {
            myRadDifModel->setCurrentTime( T );
            fillMultiVectorWithFunction( myRadDifOp->getMesh(),
                                         myRadDifOp->getGeomType(),
                                         myRadDifOp->getScalarDOFManager(),
                                         manSolVec,
                                         uexactFun );
            errorVec->subtract( *sol_new, *manSolVec );
            AMP::pout << "----------------------------------------" << std::endl;
            AMP::pout << "Manufactured discretization error norms:" << std::endl;
            auto enorms = getDiscreteNorms( myRadDifOp->getMeshSize(), errorVec );
            AMP::pout.precision( 3 );
            AMP::pout << "||e||=(" << enorms[0] << "," << enorms[1] << "," << enorms[2] << ")"
                      << std::endl;
            AMP::pout << "----------------------------------------" << std::endl;

            if ( outputSolution ) {
                std::string name = std::to_string( T );
                manSolVec->setName( name );
                AMP::IO::AsciiWriter vecWriter_man;
                vecWriter_man.registerVector( manSolVec );
                vecWriter_man.writeFile( man_dir, step, T );
            }
        }

// Write numerical solution to file.
#if 1
        if ( outputSolution ) {
            std::string name = std::to_string( T );
            sol_new->setName( name );
            AMP::IO::AsciiWriter vecWriter_num;
            vecWriter_num.registerVector( sol_new );
            vecWriter_num.writeFile( num_dir, step, T );
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


    ut->passes( inputFileName + ": time integration" );
}
// end of driver()


int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Create a global communicator
    AMP::AMP_MPI comm( AMP_COMM_WORLD );

    std::vector<std::string> exeNames;
#ifdef AMP_USE_HYPRE
    exeNames.emplace_back( "input_testBDF_RadiationDiffusionFD-1D-BoomerAMG" );
    exeNames.emplace_back( "input_testBDF_RadiationDiffusionFD-2D-BoomerAMG" );
#endif

    for ( auto &exeName : exeNames ) {
        PROFILE_ENABLE();

        driver( comm, &ut, exeName );

        // build unique profile name to avoid collisions
        std::ostringstream ss;
        ss << exeName << std::setw( 3 ) << std::setfill( '0' )
           << AMP::AMPManager::getCommWorld().getSize();
        PROFILE_SAVE( ss.str() );
    }
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}