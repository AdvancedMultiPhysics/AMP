// #if 0
// #include "AMP/IO/PIO.h"
// #include "AMP/IO/AsciiWriter.h"
// #include "AMP/utils/AMPManager.h"

// #include "AMP/vectors/CommunicationList.h"
// #include "AMP/matrices/petsc/NativePetscMatrix.h"
// #include "AMP/vectors/VectorBuilder.h"
// #include "AMP/vectors/Vector.h"
// #include "AMP/vectors/MultiVector.h"
// #include "AMP/vectors/MultiVariable.h"
// #include "AMP/vectors/data/VectorData.h"
// #include "AMP/vectors/data/VectorDataNull.h"
// #include "AMP/vectors/operations/default/VectorOperationsDefault.h"
// #include "AMP/vectors/VectorBuilder.h"

// #include "AMP/discretization/boxMeshDOFManager.h"
// #include "AMP/discretization/MultiDOF_Manager.h"
// #include "AMP/mesh/Mesh.h"
// #include "AMP/mesh/MeshID.h"
// #include "AMP/mesh/MeshParameters.h"
// #include "AMP/mesh/MeshElement.h"
// #include "AMP/mesh/structured/BoxMesh.h"

// #include "AMP/matrices/CSRMatrix.h"
// #include "AMP/matrices/MatrixBuilder.h"

// #include "AMP/operators/Operator.h"
// #include "AMP/operators/OperatorParameters.h"
// #include "AMP/operators/LinearOperator.h"
// #include "AMP/operators/petsc/PetscMatrixShellOperator.h"
// #include "AMP/operators/OperatorFactory.h"
// #include "AMP/operators/NullOperator.h"

// #include "AMP/solvers/SolverFactory.h"
// #include "AMP/solvers/SolverStrategy.h"
// #include "AMP/solvers/testHelpers/SolverTestParameters.h"
// #include "AMP/solvers/SolverStrategyParameters.h"
// #include "AMP/solvers/SolverStrategy.h"
// #include "AMP/solvers/SolverFactory.h"
// #include "AMP/solvers/petsc/PetscSNESSolver.h"
//#endif

//#include "AMP/time_integrators/TimeOperator.h"
//#include "AMP/time_integrators/TimeIntegratorFactory.h"

///// which of the below are needed?
#include "AMP/operators/OperatorFactory.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
//#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/BDFIntegrator.h"
///////
#include "AMP/IO/AsciiWriter.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionModel.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDBEWrappers.h"
#include "AMP/operators/radiationDiffusionFD/RDUtils.h"

#include "AMP/operators/testHelpers/testDiffusionFDHelper.h"


/** This is a test of a RadDifOp, which is a finite-difference discretization of a radiation 
 * diffusion operator.
 * A manufactured solution is provided, and this is used to compute a truncation error for a BE/BDF1
 * step. This tests that the operator performs as expected, and it provides a consistency check on
 * the discretization that it converges with the correct order of accuracy. 
 * 
 * Given the ODEs u'(t) + L(u, t) = s(t), a BE step from t_old -> t_new == t_old + dt is given by solving the nonlinear system 
 *      [u_new + dt*L(u_new, t_new)] = [u_old + dt*s(t_new)]
 * for u_new.
 * 
 * As such, the truncation error is
 *      e_new = [ u_new + dt*L(u_new, t_new) ] - [u_old + dt*s(t_new)]
 * 
 * Note: The local truncation error of BE is second order, i.e. O(dt^2)
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
    auto mesh_db      = input_db->getDatabase( "Mesh" )->cloneDatabase();
    auto trunc_db        = input_db->getDatabase( "TruncationError" );
    auto manufactured_db = input_db->getDatabase( "Manufactured_Parameters" );
    
    // Basic error check the input has required things
    AMP_INSIST( PDE_basic_db, "PDE is null" );
    AMP_INSIST( mesh_db, "Mesh is null" );
    AMP_INSIST( trunc_db, "TruncationError is null" );
    AMP_INSIST( manufactured_db, "Manufactured_Parameters is null" );


    /****************************************************************
    * Create a manufactured radiation-diffusion model               *
    ****************************************************************/
    auto myRadDifModel = std::make_shared<AMP::Operator::Manufactured_RadDifModel>( PDE_basic_db, manufactured_db );
    // Get parameters needed to build the RadDifOp
    auto RadDifOp_db   = myRadDifModel->getRadiationDiffusionFD_input_db( );

    /****************************************************************
    * Create a mesh                                                 *
    ****************************************************************/
    // Put variable "dim" into mesh Database 
    mesh_db->putScalar<int>( "dim", RadDifOp_db->getScalar<int>( "dim" ) );
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh = createBoxMesh( comm, mesh_db );

    AMP::pout << "The RadDifOp database is" << std::endl;
    AMP::pout << "------------------------------" << std::endl;
    RadDifOp_db->print( AMP::pout );
    AMP::pout << "------------------------------" << std::endl;
    

    /****************************************************************
    * Create a BERadDifOperator                                     *
    ****************************************************************/
    // Create an OperatorParameters object, from a Database.
    auto Op_db = std::make_shared<AMP::Database>( "Op_db" );
    auto OpParams = std::make_shared<AMP::Operator::OperatorParameters>( Op_db );
    OpParams->d_Mesh = mesh; // Set mesh of parameters
    OpParams->d_db   = RadDifOp_db; // Set DataBase of parameters.

    // Create BERadDifOp 
    auto myBERadDifOp = std::make_shared<AMP::Operator::BERadDifOp>( OpParams );  
    // Extract the underlying RadDifOp
    auto myRadDifOp = myBERadDifOp->d_RadDifOp; 

    // Create an OperatorFactory and register Jacobian of BERadDifOp in it 
    auto & operatorFactory = AMP::Operator::OperatorFactory::getFactory();
    operatorFactory.registerFactory( "BERadDifOpPJac", AMP::Operator::BERadDifOpPJac::create );

    // Create hassle-free wrappers around source term and exact solution
    auto PDESourceFun = std::bind( &AMP::Operator::RadDifModel::sourceTerm, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 );
    auto PDEManufacturedSolution = std::bind( &AMP::Operator::RadDifModel::exactSolution, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 );

    // Overwrite the default RadDifOp boundary condition functions to point to those of the Manufactured model
    myRadDifOp->setRobinFunctionE( std::bind( &AMP::Operator::Manufactured_RadDifModel::getRobinValueE, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4 ) );

    // Point the pseudo Neumann T BC values in the radDifOp to those given by the manufactured problem
    myRadDifOp->setPseudoNeumannFunctionT( std::bind( &AMP::Operator::Manufactured_RadDifModel::getPseudoNeumannValueT, &( *myRadDifModel ), std::placeholders::_1, std::placeholders::_2 ) );


    /****************************************************************
    * Set up relevant vectors                                       *
    ****************************************************************/
    // Create required vectors over the mesh
    auto manSolVecOldOld = myRadDifOp->createInputVector();
    auto manSolVecOld = myRadDifOp->createInputVector();
    auto manSolVecNew = myRadDifOp->createInputVector();
    auto truncationErrorVec = myRadDifOp->createInputVector();
    auto BDFSourceVec = myRadDifOp->createInputVector();

    // We measure the truncation error in stepping from T - dt to T, so as to measure the truncation error at the same time, irrespective of dt. 
    double newTime = trunc_db->getScalar<double>( "time" );
    double h       = myRadDifOp->getMeshSize()[0];
    double dt      = h * trunc_db->getScalar<double>( "CFL" );
    double oldTime = newTime - dt;
    double oldOldTime = oldTime - dt;
    // Populate vectors with manufactured solution
    myRadDifModel->setCurrentTime( newTime );
    myRadDifOp->fillMultiVectorWithFunction( manSolVecNew, PDEManufacturedSolution );
    myRadDifModel->setCurrentTime( oldTime );
    myRadDifOp->fillMultiVectorWithFunction( manSolVecOld, PDEManufacturedSolution );
    myRadDifModel->setCurrentTime( oldOldTime );
    myRadDifOp->fillMultiVectorWithFunction( manSolVecOldOld, PDEManufacturedSolution );
    

    /****************************************************************
    * Compute the truncation error                                  *
    ****************************************************************/
    /**  Given the ODEs u'(t) + L(u, t) = s(t), a BDF step from t_old -> t_new == t_old + dt is 
     * given by solving the nonlinear system 
     *      [u_new + gamma*L(u_new, t_new)] = RHS,
     * where RHS == [{4*u_old - u_oldOld}/3 + gamma*s(t_new)]
     * for u_new, with gamma=2/3*dt.
     * 
     * As such, the truncation error is
     *      e_new = [ u_new + dt*L(u_new, t_new) ] - RHS
     */
    // Compute RHS vector
    myRadDifModel->setCurrentTime( newTime ); // Set model to new time to ensure source term and Robin values are sampled at this time.
    myRadDifOp->fillMultiVectorWithFunction( BDFSourceVec, PDESourceFun ); // BDFSourceVec <- s
    double gamma = 2.0/3.0 * dt;
    BDFSourceVec->axpby(  4.0/3.0, gamma, *manSolVecOld    ); // this <- gamma*this + 4/3*u_old 
    BDFSourceVec->axpby( -1.0/3.0,   1.0, *manSolVecOldOld ); // this <- 1.0*this - 1/3*u_oldOld
    // Compute LHS vector
    myBERadDifOp->setGamma( gamma );
    myBERadDifOp->apply( manSolVecNew, truncationErrorVec ); // e <- u_new + dt*L(u_new)
    // Subtract over RHS vector
    truncationErrorVec->axpby( -1.0, 1.0, *BDFSourceVec ); // e <- e - BDFSourceVec

    /* Get discrete norms of truncation vector and display them */
    AMP::pout << "----------------------------------------" << std::endl;
    AMP::pout << "Manufactured truncation error norms:" << std::endl;
    auto enorms = getDiscreteNorms( myRadDifOp->getMeshSize(), truncationErrorVec );
    AMP::pout << "||e||=(" << enorms[0] << "," << enorms[1] << "," << enorms[2] << ")" << std::endl;
    AMP::pout << "----------------------------------------" << std::endl;


    ut->passes( inputFileName + ": truncation error calculation" );
}
// end of driver()




/** Input usage is e.g., 
 *  >> mpirun -n 1 radiationDiffusionFD-truncationTest
 */
int main( int argc, char **argv )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Create a global communicator
    AMP::AMP_MPI comm( AMP_COMM_WORLD );

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "input_testRadiationDiffusionFD-truncation" );
    //exeNames.emplace_back( "input_testRadiationDiffusionFD-1D" );
    //exeNames.emplace_back( "input_testRadiationDiffusionFD-2D" );

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