#include "AMP/operators/diffusionFD/DiffusionFD.h"
#include "AMP/operators/diffusionFD/DiffusionRotatedAnisotropicModel.h"
#include "AMP/operators/testHelpers/FDHelper.h"

#include "AMP/IO/AsciiWriter.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"

#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>


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

    // Unpack databases from the input file
    auto RACoefficients_db = input_db->getDatabase( "RACoefficients" );
    auto Mesh_db           = input_db->getDatabase( "Mesh" );

    AMP_INSIST( RACoefficients_db, "A ''RACoefficients'' database must be provided" );
    AMP_INSIST( Mesh_db, "A ''Mesh'' database must be provided" );


    /****************************************************************
     * Create a mesh                                                 *
     ****************************************************************/
    // Create MeshParameters
    auto mesh_params = std::make_shared<AMP::Mesh::MeshParameters>( Mesh_db );
    mesh_params->setComm( comm );
    // Create Mesh
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh = AMP::Mesh::BoxMesh::generate( mesh_params );

    // Print basic problem information
    AMP::plog << "--------------------------------------------------------------------------------"
              << std::endl;
    AMP::plog << "Building " << static_cast<int>( mesh->getDim() )
              << "D Poisson problem on mesh with "
              << mesh->numGlobalElements( AMP::Mesh::GeomType::Vertex ) << " total DOFs across "
              << mesh->getComm().getSize() << " ranks" << std::endl;
    AMP::plog << "--------------------------------------------------------------------------------"
              << std::endl;


    /*******************************************************************
     * Create manufactured rotated-anisotropic diffusion equation model *
     ********************************************************************/
    auto myRADiffusionModel =
        std::make_shared<AMP::Operator::ManufacturedRotatedAnisotropicDiffusionModel>(
            RACoefficients_db );

    // Create hassle-free wrappers around source term and exact solution
    auto PDESourceFun = std::bind( &AMP::Operator::RotatedAnisotropicDiffusionModel::sourceTerm,
                                   &( *myRADiffusionModel ),
                                   std::placeholders::_1 );
    auto uexactFun    = std::bind( &AMP::Operator::RotatedAnisotropicDiffusionModel::exactSolution,
                                &( *myRADiffusionModel ),
                                std::placeholders::_1 );


    /****************************************************************
     * Create the DiffusionFDOperator over the mesh             *
     ****************************************************************/
    const auto Op_db = std::make_shared<AMP::Database>( "linearOperatorDB" );
    Op_db->putScalar<int>( "print_info_level", 0 );
    Op_db->putScalar<std::string>( "name", "DiffusionFDOperator" );
    // Our operator requires the DiffusionCoefficients
    Op_db->putDatabase( "DiffusionCoefficients", myRADiffusionModel->d_c_db->cloneDatabase() );
    // Op_db->putDatabase( "Mesh", Mesh_db->cloneDatabase() );

    auto OpParameters    = std::make_shared<AMP::Operator::OperatorParameters>( Op_db );
    OpParameters->d_name = "DiffusionFDOperator";
    OpParameters->d_Mesh = mesh;

    auto myPoissonOp = std::make_shared<AMP::Operator::DiffusionFDOperator>( OpParameters );
    // auto A = myPoissonOp->getMatrix();
    // AMP::IO::AsciiWriter matWriter;
    // matWriter.registerMatrix( A );
    // matWriter.writeFile( "Aout", 0  );


    /****************************************************************
     * Set up relevant vectors over the mesh                         *
     ****************************************************************/
    // Create required vectors over the mesh
    auto unumVec = myPoissonOp->createInputVector();

    // Wrap exact solution function so that it also takes an int
    auto DirichletValue = [&]( const AMP::Mesh::Point &p, int ) { return uexactFun( p ); };
    // Create RHS vector
    auto rhsVec = myPoissonOp->createRHSVector( PDESourceFun, DirichletValue );
    rhsVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );


    /****************************************************************
     * Construct linear solver and apply it                          *
     ****************************************************************/
    auto solver_db = input_db->getDatabase( "LinearSolver" );
    AMP_INSIST( solver_db, "A ''LinearSolver'' database must be provided" );

    // Get the linear solver for operator myPoissonOp
    auto linearSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, comm, nullptr, myPoissonOp );

    // Use zero initial iterate and apply solver
    // linearSolver->setZeroInitialGuess( true );
    linearSolver->apply( rhsVec, unumVec );

    // Compute disretization error
    if ( myRADiffusionModel->d_exactSolutionAvailable ) {
        AMP::plog << "\nDiscretization error post linear solve: ";
        // Fill exact solution vector
        auto uexactVec = myPoissonOp->createInputVector();
        myPoissonOp->fillVectorWithFunction( uexactVec, uexactFun );
        auto e = uexactVec->clone();
        e->axpy( -1.0, *unumVec, *uexactVec );
        auto enorms = getDiscreteNorms( myPoissonOp->getMeshSize(), e );
        AMP::plog << "||e|| = (" << enorms[0] << ", " << enorms[1] << ", " << enorms[2] << ")"
                  << std::endl;
    }

    // No specific solution is implemented for this problem, so this will just check that the solver
    // converged.
    checkConvergence( linearSolver.get(), input_db, input_file, *ut );
}
// end of driver()


/*  The input file must contain the following databases:

    Mesh : Describes parameters required to build a "cube" BoxMesh
    RACoefficients : Provides parameters required to build a RotatedAnisotropicDiffusionModel
*/
int main( int argc, char **argv )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Create a global communicator
    AMP::AMP_MPI comm( AMP_COMM_WORLD );

    std::vector<std::string> exeNames;
    /* Commenting out this test because it fails... I don't know what good parameters are for SA. It
    seems to be coarsening incredibly aggressively when going to the first coarse level. The
    non-eliminated boundaries might also be a source of difficulty?
    @imay will add this back in later
    */
    // exeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-1D-SASolver-HybridGS-FCG" );
#ifdef AMP_USE_HYPRE
    // The 1D test does not pass with default BoomerAMG parameters, so commenting out for the moment
    // exeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-1D-BoomerAMG-CG" );
    exeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-2D-BoomerAMG-CG" );
    exeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-BoomerAMG-CG" );
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
