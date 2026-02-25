// This test checks the verification problem in SubChannelFlow.tex
#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/structuredFaceDOFManager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelTwoEqLinearOperator.h"
#include "AMP/operators/subchannel/SubchannelTwoEqNonlinearOperator.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/solvers/trilinos/nox/TrilinosNOXSolver.h"
#include "AMP/solvers/trilinos/nox/TrilinosNOXSolverParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <memory>
#include <string>


// Function to get the enthalpy solution
// Note: this is only an approximation that assumes incompressible water and no friction
static double getSolutionEnthalpy( double Q, double H, double m, double hin, double z )
{
    const double pi = 3.141592653589793;
    return hin + 0.5 * Q / m * ( 1.0 - cos( pi * z / H ) );
}


// Function to get the pressure solution
// Note: this is only an approximation for an incompressible fluid with a fixed density
static double
getSolutionPressure( std::shared_ptr<AMP::Database> db, double H, double Pout, double p, double z )
{
    if ( db->keyExists( "Inlet_Pressure" ) )
        return Pout + ( 1. - z / H ) * ( db->getScalar<double>( "Inlet_Pressure" ) - Pout );
    else
        return Pout + ( H - z ) * 9.80665 * p;
}


static void flowTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logAllNodes( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( globalComm );

    // Create the meshes from the input database
    auto subchannelMesh = AMP::Mesh::MeshFactory::create( meshParams );

    // get subchannel physics model
    auto subchannelPhysics_db = input_db->getDatabase( "SubchannelPhysicsModel" );
    auto params =
        std::make_shared<AMP::Operator::ElementPhysicsModelParameters>( subchannelPhysics_db );
    auto subchannelPhysicsModel = std::make_shared<AMP::Operator::SubchannelPhysicsModel>( params );

    // Create the SubchannelOperatorParameters
    auto nonlinearOperator_db = input_db->getDatabase( "SubchannelTwoEqNonlinearOperator" );
    auto subchannelOpParams =
        std::make_shared<AMP::Operator::SubchannelOperatorParameters>( nonlinearOperator_db );
    subchannelOpParams->d_Mesh                   = subchannelMesh;
    subchannelOpParams->d_subchannelPhysicsModel = subchannelPhysicsModel;
    subchannelOpParams->clad_x =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "x" );
    subchannelOpParams->clad_y =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "y" );
    subchannelOpParams->clad_d =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "d" );

    // create nonlinear operator
    auto nonlinearOperator =
        std::dynamic_pointer_cast<AMP::Operator::SubchannelTwoEqNonlinearOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                subchannelMesh, "SubchannelTwoEqNonlinearOperator", input_db ) );

    // pass creation test
    ut->passes( exeName + ": creation" );
    std::cout.flush();

    // get input and output variables
    auto inputVariable  = nonlinearOperator->getInputVariable();
    auto outputVariable = nonlinearOperator->getOutputVariable();

    // get dof manager
    int DOFsPerFace[3]  = { 0, 0, 2 };
    auto faceDOFManager = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
        subchannelMesh, DOFsPerFace, 0 );

    // create solution, rhs, and residual vectors
    auto manufacturedVec = AMP::LinearAlgebra::createVector( faceDOFManager, inputVariable, true );
    auto solVec          = AMP::LinearAlgebra::createVector( faceDOFManager, inputVariable, true );
    auto rhsVec          = AMP::LinearAlgebra::createVector( faceDOFManager, outputVariable, true );
    auto resVec          = AMP::LinearAlgebra::createVector( faceDOFManager, outputVariable, true );
    rhsVec->zero();

    // Get the problem parameters
    auto box = subchannelMesh->getBoundingBox();
    AMP_ASSERT( box[4] == 0.0 );
    double H  = box[5] - box[4];
    auto m    = nonlinearOperator_db->getScalar<double>( "Inlet_Mass_Flow_Rate" );
    auto Q    = nonlinearOperator_db->getScalar<double>( "Rod_Power" );
    auto Pout = nonlinearOperator_db->getScalar<double>( "Exit_Pressure" );
    auto Tin  = nonlinearOperator_db->getScalar<double>( "Inlet_Temperature" );

    // compute inlet enthalpy
    double Pin = Pout;
    double hin = 0.0;
    double rho = 1000;
    for ( int i = 0; i < 3; i++ ) {
        std::map<std::string, std::shared_ptr<std::vector<double>>> enthalpyArgMap;
        enthalpyArgMap.insert(
            std::make_pair( "temperature", std::make_shared<std::vector<double>>( 1, Tin ) ) );
        enthalpyArgMap.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, Pin ) ) );
        std::vector<double> enthalpyResult( 1 );
        subchannelPhysicsModel->getProperty( "Enthalpy", enthalpyResult, enthalpyArgMap );
        hin = enthalpyResult[0];
        std::map<std::string, std::shared_ptr<std::vector<double>>> volumeArgMap_plus;
        volumeArgMap_plus.insert(
            std::make_pair( "enthalpy", std::make_shared<std::vector<double>>( 1, hin ) ) );
        volumeArgMap_plus.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, Pin ) ) );
        std::vector<double> volumeResult_plus( 1 );
        subchannelPhysicsModel->getProperty(
            "SpecificVolume", volumeResult_plus, volumeArgMap_plus );
        rho = 1.0 / volumeResult_plus[0];
        Pin = getSolutionPressure( input_db, H, Pout, rho, 0 );
    }
    std::cout << "Inlet density:" << rho << std::endl;
    std::cout << "Enthalpy Solution:" << hin << std::endl;

    // Compute the manufactured solution
    auto xyFaceMesh = subchannelMesh->Subset(
        AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( subchannelMesh, 0 ) );
    auto face = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    std::vector<size_t> dofs;
    const double h_scale = 1.0 / AMP::Operator::Subchannel::scaleEnthalpy; // Scale to change the
                                                                           // input vector back to
                                                                           // correct units
    const double P_scale = 1.0 / AMP::Operator::Subchannel::scalePressure; // Scale to change the
                                                                           // input vector back to
                                                                           // correct units
    for ( int i = 0; i < (int) face.size(); i++ ) {
        faceDOFManager->getDOFs( face->globalID(), dofs );
        auto coord = face->centroid();
        double z   = coord[2];
        double h   = getSolutionEnthalpy( Q, H, m, hin, z );
        double P   = getSolutionPressure( input_db, H, Pout, rho, z );
        h *= AMP::Operator::Subchannel::scaleEnthalpy;
        P *= AMP::Operator::Subchannel::scalePressure;
        manufacturedVec->setValuesByGlobalID( 1, &dofs[0], &h );
        manufacturedVec->setValuesByGlobalID( 1, &dofs[1], &P );
        ++face;
    }

    // Compute the initial guess solution
    face = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    for ( int i = 0; i < (int) face.size(); i++ ) {
        faceDOFManager->getDOFs( face->globalID(), dofs );
        double val = AMP::Operator::Subchannel::scaleEnthalpy * hin;
        solVec->setValuesByGlobalID( 1, &dofs[0], &val );
        val = AMP::Operator::Subchannel::scalePressure * Pout;
        solVec->setValuesByGlobalID( 1, &dofs[1], &val );
        ++face;
    }
    solVec->copyVector( manufacturedVec );
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // get nonlinear solver database
    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );

    // put manufactured RHS into resVec
    nonlinearOperator->reset( subchannelOpParams );

    // create nonlinear solver parameters
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::TrilinosNOXSolverParameters>( nonlinearSolver_db );

    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nonlinearOperator;
    nonlinearSolverParams->d_pInitialGuess = solVec;

    // create nonlinear solver
    auto nonlinearSolver =
        std::make_shared<AMP::Solver::TrilinosNOXSolver>( nonlinearSolverParams );

    // don't use zero initial guess
    nonlinearSolver->setZeroInitialGuess( false );

    // solve
    nonlinearSolver->apply( rhsVec, solVec );
    nonlinearOperator->residual( rhsVec, solVec, resVec );

    // Compute the flow temperature
    int tempDOFsPerFace[3] = { 0, 0, 1 };
    auto tempDOFManager    = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
        subchannelMesh, tempDOFsPerFace, 0 );
    auto tempVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );
    auto tempVec      = AMP::LinearAlgebra::createVector( tempDOFManager, tempVariable, true );
    face              = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    std::vector<size_t> tdofs;
    bool pass = true;
    for ( int i = 0; i < (int) face.size(); i++ ) {
        faceDOFManager->getDOFs( face->globalID(), dofs );
        tempDOFManager->getDOFs( face->globalID(), tdofs );
        double h = h_scale * solVec->getValueByGlobalID( dofs[0] );
        double P = P_scale * solVec->getValueByGlobalID( dofs[1] );
        std::map<std::string, std::shared_ptr<std::vector<double>>> temperatureArgMap;
        temperatureArgMap.insert(
            std::make_pair( "enthalpy", std::make_shared<std::vector<double>>( 1, h ) ) );
        temperatureArgMap.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, P ) ) );
        std::vector<double> temperatureResult( 1 );
        subchannelPhysicsModel->getProperty( "Temperature", temperatureResult, temperatureArgMap );
        tempVec->setValuesByGlobalID( 1, &tdofs[0], &temperatureResult[0] );
        // Check that we recover the enthalapy from the temperature
        std::map<std::string, std::shared_ptr<std::vector<double>>> enthalpyArgMap;
        enthalpyArgMap.insert( std::make_pair(
            "temperature", std::make_shared<std::vector<double>>( 1, temperatureResult[0] ) ) );
        enthalpyArgMap.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, P ) ) );
        std::vector<double> enthalpyResult( 1 );
        subchannelPhysicsModel->getProperty( "Enthalpy", enthalpyResult, enthalpyArgMap );
        double h2 = enthalpyResult[0];
        if ( !AMP::Utilities::approx_equal( h, h2, 1e-7 ) )
            pass = false;
        ++face;
    }
    if ( !pass )
        ut->failure( "failed to recover h" );

    // Print the Inlet/Outlet properties
    std::cout << std::endl << std::endl;
    face = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    faceDOFManager->getDOFs( face->globalID(), dofs );
    tempDOFManager->getDOFs( face->globalID(), tdofs );
    double TinSol = tempVec->getValueByGlobalID( tdofs[0] );
    std::cout << "Inlet Computed Enthalpy = " << h_scale * solVec->getValueByGlobalID( dofs[0] )
              << std::endl;
    std::cout << "Inlet Computed Pressure = " << P_scale * solVec->getValueByGlobalID( dofs[1] )
              << std::endl;
    std::cout << "Inlet Computed Temperature = " << TinSol << std::endl;
    std::cout << std::endl;
    face = --( ( xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 ) ).end() );
    faceDOFManager->getDOFs( face->globalID(), dofs );
    tempDOFManager->getDOFs( face->globalID(), tdofs );
    double ToutSol = tempVec->getValueByGlobalID( tdofs[0] );
    std::cout << "Outlet Computed Enthalpy = " << h_scale * solVec->getValueByGlobalID( dofs[0] )
              << std::endl;
    std::cout << "Outlet Computed Pressure = " << P_scale * solVec->getValueByGlobalID( dofs[1] )
              << std::endl;
    std::cout << "Outlet Computed Temperature = " << ToutSol << std::endl;

    // Compute the error
    auto absErrorVec = solVec->clone();
    absErrorVec->axpy( -1.0, *solVec, *manufacturedVec );
    auto relErrorVec = solVec->clone();
    relErrorVec->divide( *absErrorVec, *manufacturedVec );
    double absErrorNorm = static_cast<double>( absErrorVec->L2Norm() );
    double relErrorNorm = static_cast<double>( relErrorVec->L2Norm() );

    // check that norm of relative error is less than tolerance
    auto tol = input_db->getWithDefault<double>( "TOLERANCE", 1e-6 );
    if ( relErrorNorm <= tol && fabs( Tin - TinSol ) < tol ) {
        ut->passes( exeName + ": manufactured solution test" );
    } else {
        ut->failure( exeName + ": manufactured solution test" );
    }

    // Print final solution
    face        = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    int N_print = std::max( 1, (int) face.size() / 10 );
    for ( int i = 0; i < (int) face.size(); i++ ) {
        if ( i % N_print == 0 ) {
            faceDOFManager->getDOFs( face->globalID(), dofs );
            std::cout << "Computed Enthalpy[" << i
                      << "] = " << h_scale * solVec->getValueByGlobalID( dofs[0] ) << std::endl;
            std::cout << "Solution Enthalpy[" << i
                      << "] = " << h_scale * manufacturedVec->getValueByGlobalID( dofs[0] )
                      << std::endl;
            std::cout << "Computed Pressure[" << i
                      << "] = " << P_scale * solVec->getValueByGlobalID( dofs[1] ) << std::endl;
            std::cout << "Solution Pressure[" << i
                      << "] = " << P_scale * manufacturedVec->getValueByGlobalID( dofs[1] )
                      << std::endl;
            std::cout << std::endl;
        }
        ++face;
    }
    std::cout << "Delta T: " << ToutSol - TinSol << std::endl << std::endl;
    std::cout << "L2 Norm of Absolute Error: " << absErrorNorm << std::endl;
    std::cout << "L2 Norm of Relative Error: " << relErrorNorm << std::endl;
}

int testSubchannelSolutionNOX( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files( 2 );
    files[0] = "testSubchannelSolution-1";
    files[1] = "testSubchannelSolution-2";

    for ( const auto &file : files )
        flowTest( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
