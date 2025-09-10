#include "AMP/operators/diffusionFD/DiffusionFD.h"
#include "AMP/operators/diffusionFD/DiffusionRotatedAnisotropicModel.h"
#include "AMP/operators/testHelpers/FDHelper.h"

#include "AMP/IO/AsciiWriter.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"

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
     * Create the DiffusionFDOperator over the mesh                  *
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
    // matWriter.writeFile( "Aout", 0 );


    /****************************************************************
     * Create RHS vector                                             *
     ****************************************************************/
    // Wrap exact solution function so that it also takes an int
    auto DirichletValue = [&]( AMP::Mesh::MeshElement &node, int ) { return uexactFun( node ); };
    // Create RHS vector
    auto rhsVec = myPoissonOp->createRHSVector( PDESourceFun, DirichletValue );
    rhsVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );


    /***************************************************************************
     * Compute discrete residual norm on continuous solution (truncation error) *
     ***************************************************************************/
    if ( myRADiffusionModel->d_exactSolutionAvailable ) {
        auto uexactVec = myPoissonOp->createInputVector();
        auto rexactVec = myPoissonOp->createOutputVector();
        // Set exact solution
        myPoissonOp->fillVectorWithFunction( uexactVec, uexactFun );
        uexactVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        AMP::plog << "\nDiscrete residual of continuous manufactured solution: ";
        myPoissonOp->residual( rhsVec, uexactVec, rexactVec );
        auto rnorms = getDiscreteNorms( myPoissonOp->getMeshSize(), rexactVec );
        // Print residual norms
        AMP::plog << "||r|| = (" << rnorms[0] << ", " << rnorms[1] << ", " << rnorms[2] << ")"
                  << std::endl
                  << std::endl;

// Debugging: If convergence rate of truncation error is off, it can be useful to look at the
// individual components of f, u, r to see where things may be going wrong.
#if 0
        for ( size_t row = 0; row < myPoissonOp->getMatrix()->numLocalRows(); row++ ) {
            auto f = rhsVec->getValueByLocalID( row );
            auto u = uexactVec->getValueByLocalID( row );
            auto r = rexactVec->getValueByLocalID( row );
            std::cout << "i=" << row << ": f,u,r=" << f << "," << u << "," << r << std::endl;
        }
#endif

        ut->passes( inputFileName + ": truncation error calculation" );
    }
}
// end of driver()


/*  The input file must contain the following databases:

    Mesh           : Describes parameters required to build a "cube" BoxMesh
    RACoefficients : Provides parameters required to build a RotatedAnisotropicDiffusionModel

    TODO in a future MR: At the moment this test assembles the operator and computes a truncation
   error, but doesn't do anything with it. A more insightful test would be to compute the truncation
   error across a sequence of finer and finer meshes, and to check that the truncation error is
   converging at the correct rate (second order)
*/
int main( int argc, char **argv )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Create a global communicator
    AMP::AMP_MPI comm( AMP_COMM_WORLD );

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "input_testDiffusionFDOperator-1D" );
    exeNames.emplace_back( "input_testDiffusionFDOperator-2D" );
    exeNames.emplace_back( "input_testDiffusionFDOperator-3D" );

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