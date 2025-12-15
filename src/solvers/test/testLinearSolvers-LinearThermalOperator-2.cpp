#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"


void linearThermalTest( AMP::UnitTest *ut,
                        const std::string &input_file,
                        std::shared_ptr<AMP::Database> input_db,
                        std::shared_ptr<AMP::Mesh::Mesh> mesh,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> powerVec )
{

    AMP_ASSERT( input_db && mesh && powerVec );

    // create the Thermal Operator
    auto diffusionOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "DiffusionBVPOperator", input_db ) );

    auto nodalDofMap    = powerVec->getDOFManager();
    auto inputVariable  = diffusionOperator->getInputVariable();
    auto outputVariable = diffusionOperator->getOutputVariable();
    auto memoryLocation = diffusionOperator->getMemoryLocation();

    auto TemperatureInKelvinVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, inputVariable, true, memoryLocation );
    auto RightHandSideVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable, true, memoryLocation );
    auto ResidualVec = RightHandSideVec->clone();

    //   Add the boundary conditions corrections //
    RightHandSideVec->copyVector( powerVec );
    diffusionOperator->modifyRHSvector( RightHandSideVec );

    AMP::pout << "RHS Max: " << RightHandSideVec->max() << std::endl;
    AMP::pout << "RHS Min: " << RightHandSideVec->min() << std::endl;
    AMP::pout << "RHS L1-norm: " << RightHandSideVec->L1Norm() << std::endl;
    AMP::pout << "RHS L2-norm: " << RightHandSideVec->L2Norm() << std::endl;

    // Set initial guess
    TemperatureInKelvinVec->setToScalar( 1.0 );
    AMP::pout << "Initial Solution L2-norm: " << TemperatureInKelvinVec->L2Norm() << std::endl;

    // Construct the solver
    auto &comm        = mesh->getComm();
    auto linearSolver = AMP::Solver::Test::buildSolver(
        "LinearSolver", input_db, comm, nullptr, diffusionOperator );

    // Solve the problem.
    linearSolver->setZeroInitialGuess( false );
    linearSolver->apply( RightHandSideVec, TemperatureInKelvinVec );

    AMP::pout << "Solution Max: " << TemperatureInKelvinVec->max() << std::endl;
    AMP::pout << "Solution Min: " << TemperatureInKelvinVec->min() << std::endl;
    AMP::pout << "Solution L1-norm: " << TemperatureInKelvinVec->L1Norm() << std::endl;
    AMP::pout << "Solution L2-norm: " << TemperatureInKelvinVec->L2Norm() << std::endl;
    AMP::pout << "======================================================================="
              << std::endl;

    //    checkConvergence( linearSolver.get(), input_file, *ut );

    // Compute the residual
    diffusionOperator->residual( RightHandSideVec, TemperatureInKelvinVec, ResidualVec );

    // Check the L2 norm of the final residual.
    auto finalResidualNorm = static_cast<double>( ResidualVec->L2Norm() );
    AMP::pout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    auto combo        = linearSolver->type();
    auto nestedSolver = linearSolver->getNestedSolver();
    if ( nestedSolver )
        combo += " with PC " + nestedSolver->type();

    if ( finalResidualNorm < 10 ) {
        ut->passes( combo + " passes linear thermal problem with input " + input_file );
    } else {
        ut->failure( combo + " fails linear thermal problem with input " + input_file );
    }
}

void linearThermalTest( AMP::UnitTest *ut, const std::string &inputFile, bool all_solvers )
{
    // Input and output file names
    std::string input_file = inputFile;
    std::string log_file   = "output_" + inputFile;

    AMP::pout << "Running with input " << input_file << std::endl;

    // Fill the database from the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    AMP::logAllNodes( log_file );

    //   Create the Mesh.
    const auto mesh      = createMesh( input_db );
    auto PowerInWattsVec = constructNeutronicsPowerSource( input_db, mesh );

    if ( all_solvers ) {
        std::vector<std::pair<std::string, std::string>> solvers{
#ifdef AMP_USE_HYPRE
            { "CG", "BoomerAMG" },
            { "GMRES", "BoomerAMG" },
            { "FGMRES", "BoomerAMG" },
            { "BiCGSTAB", "BoomerAMG" },
            { "TFQMR", "BoomerAMG" },
            { "BoomerAMG", "NoPC" },
            { "HyprePCG", "NoPC" },
            { "HyprePCG", "BoomerAMG" },
            { "HypreGMRES", "NoPC" },
            { "HypreGMRES", "BoomerAMG" },
            { "HypreBiCGSTAB", "NoPC" },
            { "HypreBiCGSTAB", "BoomerAMG" },
    #ifdef AMP_USE_PETSC
            { "PetscFGMRES", "BoomerAMG" },
    #endif
#endif
#ifdef AMP_USE_TRILINOS_ML
            { "CG", "ML" },
            { "GMRES", "ML" },
            { "FGMRES", "ML" },
            { "BiCGSTAB", "ML" },
            { "TFQMR", "ML" },
    #ifdef AMP_USE_PETSC
            { "PetscFGMRES", "ML" },
    #endif
            { "ML", "NoPC" },
#endif
#ifdef AMP_USE_TRILINOS_MUELU
            { "CG", "MueLu" },
            { "GMRES", "MueLu" },
            { "FGMRES", "MueLu" },
            { "BiCGSTAB", "MueLu" },
            { "TFQMR", "MueLu" },
    #ifdef AMP_USE_PETSC
            { "PetscFGMRES", "MueLu" },
    #endif
            { "MueLu", "NoPC" },
#endif
#ifdef AMP_USE_PETSC
            { "PetscFGMRES", "NoPC" },
#endif
            { "CG", "NoPC" },
            { "GMRES", "NoPC" },
            { "FGMRES", "NoPC" },
            { "BiCGSTAB", "NoPC" }
        };

        for ( auto &[primary, nested] : solvers ) {
            std::shared_ptr<AMP::Database> db = input_db->cloneDatabase();
            auto use_nested                   = ( nested == "NoPC" ) ? false : true;
            db->putDatabase(
                "LinearSolver",
                AMP::Solver::Test::SolverParameters::getParameters( primary, use_nested ) );
            if ( use_nested ) {
                db->putDatabase(
                    "Preconditioner",
                    AMP::Solver::Test::SolverParameters::getParameters( nested, use_nested ) );
            }

            std::string banner;
            if ( use_nested )
                banner = "Running " + primary + " with PC " + nested + " on " + input_file;
            else
                banner = "Running " + primary + " on " + input_file;
            AMP::pout << banner << std::endl;
            linearThermalTest(
                ut, input_file + " with " + primary + "+" + nested, db, mesh, PowerInWattsVec );
        }
    } else {
        linearThermalTest( ut, input_file, input_db, mesh, PowerInWattsVec );
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> inputFiles;

    if ( argc > 1 ) {

        std::string inputFile{ argv[1] };
        linearThermalTest( &ut, inputFile, false );

    } else {

        inputFiles.emplace_back( "input_LinearThermalOperator-2_HALDEN" );
        inputFiles.emplace_back( "input_LinearThermalOperator-2-cylinder" );
        inputFiles.emplace_back( "input_LinearThermalOperator-2-shell" );
        //    inputFiles.push_back( "input_testBoomerAMGSolver-LinearThermalOperator-2_HALDEN_clad"
        //    );
        for ( auto &inputFile : inputFiles )
            linearThermalTest( &ut, inputFile, true );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
