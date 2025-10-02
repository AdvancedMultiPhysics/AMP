#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDBDFWrappers.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionModel.h"
#include "AMP/operators/testHelpers/FDHelper.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>


/** This is a test of a RadDifOp, which is a finite-difference discretization of a radiation
 * diffusion operator. There are two tests:
 * 
 * 1. A manufactured solution is provided, and this is used to compute a truncation error for a BDF
 * step. This tests that apply() of the operator performs as expected, and it also provides some 
 * type of a consistency check on the discretization that it converges (without further study it's 
 * not completely clear how this truncation error should decrease w.r.t. dt and h)
 * 
 * 2. The associated linearized operator is constructed, and its apply() is tested.
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
    auto PDE_basic_db    = input_db->getDatabase( "PDE" );
    auto mesh_db         = input_db->getDatabase( "Mesh" );
    auto trunc_db        = input_db->getDatabase( "TruncationError" );
    auto manufactured_db = input_db->getDatabase( "Manufactured_Parameters" );

    // Basic error check the input has required things
    AMP_INSIST( PDE_basic_db, "PDE is null" );
    AMP_INSIST( mesh_db, "Mesh is null" );
    AMP_INSIST( trunc_db, "TruncationError is null" );
    AMP_INSIST( manufactured_db, "Manufactured_Parameters is null" );

    // Push problem dimension into PDE_basic_db
    PDE_basic_db->putScalar<int>( "dim", mesh_db->getScalar<int>( "dim" ) );

    /****************************************************************
     * Create a manufactured radiation-diffusion model               *
     ****************************************************************/
    auto myRadDifModel =
        std::make_shared<AMP::Operator::Manufactured_RadDifModel>( PDE_basic_db, manufactured_db );
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
    AMP::pout << "Input database to RadDifOp is" << std::endl;
    AMP::pout << "------------------------------" << std::endl;
    RadDifOp_db->print( AMP::pout );
    AMP::pout << "------------------------------" << std::endl;

    // Create an OperatorParameters object, from a Database.
    auto Op_db       = std::make_shared<AMP::Database>( "Op_db" );
    auto OpParams    = std::make_shared<AMP::Operator::OperatorParameters>( Op_db );
    OpParams->d_Mesh = mesh;        // Set mesh of parameters
    OpParams->d_db   = RadDifOp_db; // Set DataBase of parameters.

    // Create BDFRadDifOp
    auto myBDFRadDifOp = std::make_shared<AMP::Operator::BDFRadDifOp>( OpParams );
    // Extract the underlying RadDifOp
    auto myRadDifOp = myBDFRadDifOp->d_RadDifOp;

    // Create an OperatorFactory and register Jacobian of BDFRadDifOp in it
    auto &operatorFactory = AMP::Operator::OperatorFactory::getFactory();
    operatorFactory.registerFactory( "BDFRadDifOpPJac", AMP::Operator::BDFRadDifOpPJac::create );

    // Create hassle-free wrappers around source term and exact solution
    auto PDESourceFun            = std::bind( &AMP::Operator::RadDifModel::sourceTerm,
                                   &( *myRadDifModel ),
                                   std::placeholders::_1,
                                   std::placeholders::_2 );
    auto PDEManufacturedSolution = std::bind( &AMP::Operator::RadDifModel::exactSolution,
                                              &( *myRadDifModel ),
                                              std::placeholders::_1,
                                              std::placeholders::_2 );

    // Overwrite the default RadDifOp boundary condition functions to point to those of the
    // Manufactured model
    myRadDifOp->setBoundaryFunctionE(
        std::bind( &AMP::Operator::Manufactured_RadDifModel::getBoundaryFunctionValueE,
                   &( *myRadDifModel ),
                   std::placeholders::_1,
                   std::placeholders::_2 ) );

    // Point the pseudo Neumann T BC values in the radDifOp to those given by the manufactured
    // problem
    myRadDifOp->setBoundaryFunctionT(
        std::bind( &AMP::Operator::Manufactured_RadDifModel::getBoundaryFunctionValueT,
                   &( *myRadDifModel ),
                   std::placeholders::_1,
                   std::placeholders::_2 ) );


    /****************************************************************
     * Set up relevant vectors                                       *
     ****************************************************************/
    // Create required vectors over the mesh
    auto manSolVecOldOld    = myRadDifOp->createInputVector();
    auto manSolVecOld       = myRadDifOp->createInputVector();
    auto manSolVecNew       = myRadDifOp->createInputVector();
    auto truncationErrorVec = myRadDifOp->createInputVector();
    auto BDFSourceVec       = myRadDifOp->createInputVector();

    // We measure the truncation error in stepping from T - dt to T, so as to measure the truncation
    // error at the same time, irrespective of dt.
    double newTime    = trunc_db->getScalar<double>( "time" );
    double dt         = trunc_db->getScalar<double>( "dt" );
    double oldTime    = newTime - dt;
    double oldOldTime = oldTime - dt;
    // Populate vectors with manufactured solution
    myRadDifModel->setCurrentTime( newTime );
    fillMultiVectorWithFunction( myRadDifOp->getMesh(),
                                 myRadDifOp->getGeomType(),
                                 myRadDifOp->getScalarDOFManager(),
                                 manSolVecNew,
                                 PDEManufacturedSolution );
    myRadDifModel->setCurrentTime( oldTime );
    fillMultiVectorWithFunction( myRadDifOp->getMesh(),
                                 myRadDifOp->getGeomType(),
                                 myRadDifOp->getScalarDOFManager(),
                                 manSolVecOld,
                                 PDEManufacturedSolution );
    myRadDifModel->setCurrentTime( oldOldTime );
    fillMultiVectorWithFunction( myRadDifOp->getMesh(),
                                 myRadDifOp->getGeomType(),
                                 myRadDifOp->getScalarDOFManager(),
                                 manSolVecOldOld,
                                 PDEManufacturedSolution );


    /****************************************************************
     * Compute the truncation error                                 *
     ****************************************************************/
    /**  Given the ODEs u'(t) + L(u, t) = s(t), a BDF step from t_old -> t_new == t_old + dt is
     * given by solving the nonlinear system
     *      [u_new + gamma*L(u_new, t_new)] = RHS,
     * where RHS == [{4*u_old - u_oldOld}/3 + gamma*s(t_new)]
     * for u_new, with gamma=2/3*dt.
     *
     * As such, the truncation error is
     *      e_new = [ u_new + gamma*L(u_new, t_new) ] - RHS
     */
    // Compute RHS vector
    myRadDifModel->setCurrentTime( newTime ); // Set model to new time to ensure source term and
                                              // Robin values are sampled at this time.
    fillMultiVectorWithFunction( myRadDifOp->getMesh(),
                                 myRadDifOp->getGeomType(),
                                 myRadDifOp->getScalarDOFManager(),
                                 BDFSourceVec,
                                 PDESourceFun ); // BDFSourceVec <- s
    double gamma = 2.0 / 3.0 * dt;
    BDFSourceVec->axpby( 4.0 / 3.0, gamma, *manSolVecOld );   // this <- gamma*this + 4/3*u_old
    BDFSourceVec->axpby( -1.0 / 3.0, 1.0, *manSolVecOldOld ); // this <- 1.0*this - 1/3*u_oldOld
    // Compute LHS vector
    myBDFRadDifOp->setGamma( gamma );
    myBDFRadDifOp->apply( manSolVecNew, truncationErrorVec ); // e <- u_new + dt*L(u_new)
    // Subtract over RHS vector
    truncationErrorVec->axpby( -1.0, 1.0, *BDFSourceVec ); // e <- e - BDFSourceVec

    /* Get discrete norms of truncation vector and display them */
    AMP::pout << "----------------------------------------" << std::endl;
    AMP::pout << "Manufactured truncation error norms:" << std::endl;
    auto enorms = getDiscreteNorms( myRadDifOp->getMeshSize(), truncationErrorVec );
    AMP::pout.precision( 3 );
    AMP::pout << "||e||=(" << enorms[0] << "," << enorms[1] << "," << enorms[2] << ")" << std::endl;
    AMP::pout << "----------------------------------------" << std::endl;

    ut->passes( inputFileName + ": truncation error calculation" );


    /****************************************************************
     * Test 2: Build linearized operator and test its apply         *
     ****************************************************************/
    // Get linearized parameters about manufactured solution 
    auto linearizedOpParams = myBDFRadDifOp->getParameters( "Jacobian", manSolVecNew );
    auto myBDFRadDifOpPJac = std::make_shared<AMP::Operator::BDFRadDifOpPJac>( linearizedOpParams );
    // Extract underlying RadDifOpPJac
    auto myRadDifOpPJac = myBDFRadDifOpPJac->d_RadDifOpPJac;
    // Create input and output vectors
    auto inVec  = myRadDifOpPJac->createInputVector();
    inVec->setRandomValues();
    auto outVec = inVec->clone();
    // Apply 
    myBDFRadDifOpPJac->apply( inVec, outVec );

    ut->passes( inputFileName + ": apply of linearized operator" );
    

}
// end of driver()


int main( int argc, char **argv )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Create a global communicator
    AMP::AMP_MPI comm( AMP_COMM_WORLD );

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "input_testTruncation-RadiationDiffusionFD-1D" );
    exeNames.emplace_back( "input_testTruncation-RadiationDiffusionFD-2D" );
    exeNames.emplace_back( "input_testTruncation-RadiationDiffusionFD-3D" );

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