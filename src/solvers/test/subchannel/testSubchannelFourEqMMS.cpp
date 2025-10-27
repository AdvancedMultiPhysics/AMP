#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/structuredFaceDOFManager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelFourEqLinearOperator.h"
#include "AMP/operators/subchannel/SubchannelFourEqNonlinearOperator.h"
#include "AMP/operators/subchannel/SubchannelHelpers.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
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


// Function to get the linear heat generation rate
double getLinearHeatGeneration( double Q, double H, double z )
{
    const double pi = 3.141592653589793;
    return 0.5 * pi * Q / H * sin( pi * z / H );
}


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

    //=============================================================================
    // mesh and dof manager
    //=============================================================================

    // Get the Mesh database and create the mesh parameters
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( globalComm );

    // Create the meshes from the input database
    auto subchannelMesh = AMP::Mesh::MeshFactory::create( meshParams );

    // get dof manager
    int DOFsPerFace[3]        = { 1, 1, 3 };
    auto subchannelDOFManager = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
        subchannelMesh, DOFsPerFace, 1 );

    //=============================================================================
    // physics model, parameters, and operator creation
    //=============================================================================
    // get input and output variables
    auto inputVariable  = std::make_shared<AMP::LinearAlgebra::Variable>( "flow" );
    auto outputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "flow" );

    // create solution, rhs, and residual vectors
    auto manufacturedVec =
        AMP::LinearAlgebra::createVector( subchannelDOFManager, inputVariable, true );
    auto solVec = AMP::LinearAlgebra::createVector( subchannelDOFManager, inputVariable, true );
    auto rhsVec = AMP::LinearAlgebra::createVector( subchannelDOFManager, outputVariable, true );
    auto resVec = AMP::LinearAlgebra::createVector( subchannelDOFManager, outputVariable, true );
    rhsVec->zero();

    // get subchannel physics model
    auto subchannelPhysics_db = input_db->getDatabase( "SubchannelPhysicsModel" );
    auto params =
        std::make_shared<AMP::Operator::ElementPhysicsModelParameters>( subchannelPhysics_db );
    auto subchannelPhysicsModel = std::make_shared<AMP::Operator::SubchannelPhysicsModel>( params );

    // Create the SubchannelOperatorParameters
    auto nonlinearOperator_db = input_db->getDatabase( "SubchannelFourEqNonlinearOperator" );
    auto linearOperator_db    = input_db->getDatabase( "SubchannelFourEqLinearOperator" );
    auto nonlinearOpParams =
        std::make_shared<AMP::Operator::SubchannelOperatorParameters>( nonlinearOperator_db );
    auto linearOpParams =
        std::make_shared<AMP::Operator::SubchannelOperatorParameters>( linearOperator_db );
    nonlinearOpParams->d_Mesh                   = subchannelMesh;
    nonlinearOpParams->d_subchannelPhysicsModel = subchannelPhysicsModel;
    nonlinearOpParams->clad_x = input_db->getDatabase( "CladProperties" )->getVector<double>( "x" );
    nonlinearOpParams->clad_y = input_db->getDatabase( "CladProperties" )->getVector<double>( "y" );
    nonlinearOpParams->clad_d = input_db->getDatabase( "CladProperties" )->getVector<double>( "d" );

    // create nonlinear operator
    auto nonlinearOperator =
        std::make_shared<AMP::Operator::SubchannelFourEqNonlinearOperator>( nonlinearOpParams );
    // reset the nonlinear operator
    nonlinearOperator->reset( nonlinearOpParams );

    // pass creation test
    ut->passes( exeName + ": creation" );
    std::cout.flush();

    //=============================================================================
    // compute manufactured solution
    //=============================================================================

    // Get the problem parameters
    auto box = subchannelMesh->getBoundingBox();
    AMP_ASSERT( box[4] == 0.0 );
    double H             = box[5] - box[4];
    auto m_in            = nonlinearOperator_db->getScalar<double>( "Inlet_Mass_Flow_Rate" );
    auto w_in            = nonlinearOperator_db->getScalar<double>( "Inlet_Lateral_Flow_Rate" );
    auto Q               = nonlinearOperator_db->getScalar<double>( "Max_Rod_Power" );
    auto Pout            = nonlinearOperator_db->getScalar<double>( "Exit_Pressure" );
    auto Tin             = nonlinearOperator_db->getScalar<double>( "Inlet_Temperature" );
    size_t N_subchannels = AMP::Operator::Subchannel::getNumberOfSubchannels( subchannelMesh );
    m_in                 = m_in / N_subchannels;

    // compute inlet enthalpy
    double Pin    = Pout;
    double hin    = 0.0;
    double rho_in = 1000;
    // iterate to find inlet pressure and inlet enthalpy
    for ( int i = 0; i < 3; i++ ) {
        // compute inlet enthalpy using inlet temperature and outlet pressure
        std::map<std::string, std::shared_ptr<std::vector<double>>> enthalpyArgMap;
        enthalpyArgMap.insert(
            std::make_pair( "temperature", std::make_shared<std::vector<double>>( 1, Tin ) ) );
        enthalpyArgMap.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, Pin ) ) );
        std::vector<double> enthalpyResult( 1 );
        subchannelPhysicsModel->getProperty( "Enthalpy", enthalpyResult, enthalpyArgMap );
        hin = enthalpyResult[0];
        // compute inlet density using computed inlet enthalpy and outlet pressure
        std::map<std::string, std::shared_ptr<std::vector<double>>> volumeArgMap_plus;
        volumeArgMap_plus.insert(
            std::make_pair( "enthalpy", std::make_shared<std::vector<double>>( 1, hin ) ) );
        volumeArgMap_plus.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, Pin ) ) );
        std::vector<double> volumeResult_plus( 1 );
        subchannelPhysicsModel->getProperty(
            "SpecificVolume", volumeResult_plus, volumeArgMap_plus );
        rho_in = 1.0 / volumeResult_plus[0];
        // compute inlet pressure
        Pin = getSolutionPressure( input_db, H, Pout, rho_in, 0 );
    }
    std::cout << "Inlet density:" << rho_in << std::endl;
    std::cout << "Enthalpy Solution:" << hin << std::endl;

    // Compute the manufactured solution
    auto xyFaceMesh = subchannelMesh->Subset(
        AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( subchannelMesh, 0 ) );
    auto face = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    std::vector<size_t> axialDofs;
    // Scale to change the input vector back to correct units
    const double h_scale = 1.0 / AMP::Operator::Subchannel::scaleEnthalpy;
    const double P_scale = 1.0 / AMP::Operator::Subchannel::scalePressure;
    const double m_scale = 1.0 / AMP::Operator::Subchannel::scaleAxialMassFlowRate;
    const double w_scale = 1.0 / AMP::Operator::Subchannel::scaleLateralMassFlowRate;
    // loop over axial faces
    double val;
    for ( int i = 0; i < (int) face.size(); i++ ) {
        subchannelDOFManager->getDOFs( face->globalID(), axialDofs );
        auto coord = face->centroid();
        double z   = coord[2];
        double h   = getSolutionEnthalpy( Q, H, m_in, hin, z );
        double P   = getSolutionPressure( input_db, H, Pout, rho_in, z );
        val        = m_in / m_scale;
        manufacturedVec->setValuesByGlobalID( 1, &axialDofs[0], &val );
        val = h / h_scale;
        manufacturedVec->setValuesByGlobalID( 1, &axialDofs[1], &val );
        val = P / P_scale;
        manufacturedVec->setValuesByGlobalID( 1, &axialDofs[2], &val );
        ++face;
    }
    // get lateral face map
    std::map<AMP::Mesh::Point, std::unique_ptr<AMP::Mesh::MeshElement>> interiorLateralFaceMap;
    std::map<AMP::Mesh::Point, std::unique_ptr<AMP::Mesh::MeshElement>> exteriorLateralFaceMap;
    nonlinearOperator->getLateralFaces(
        nonlinearOpParams->d_Mesh, interiorLateralFaceMap, exteriorLateralFaceMap );
    // loop over lateral faces
    for ( face = face.begin(); face != face.end(); ++face ) {
        auto faceCentroid        = face->centroid();
        auto lateralFaceIterator = interiorLateralFaceMap.find( faceCentroid );
        if ( lateralFaceIterator != interiorLateralFaceMap.end() ) {
            // get lateral face
            auto &lateralFace = lateralFaceIterator->second;
            // get crossflow from solution vector
            std::vector<size_t> gapDofs;
            subchannelDOFManager->getDOFs( lateralFace->globalID(), gapDofs );
            double w = 0.0;
            val      = w / w_scale;
            manufacturedVec->setValuesByGlobalID( 1, &gapDofs[0], &val );
        }
    }

    //=============================================================================
    // compute initial guess
    //=============================================================================

    // Compute the initial guess solution
    face = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    // loop over axial faces
    for ( int i = 0; i < (int) face.size(); i++ ) {
        subchannelDOFManager->getDOFs( face->globalID(), axialDofs );
        val = m_in / m_scale;
        solVec->setValuesByGlobalID( 1, &axialDofs[0], &val );
        val = hin / h_scale;
        solVec->setValuesByGlobalID( 1, &axialDofs[1], &val );
        val = Pout / P_scale;
        solVec->setValuesByGlobalID( 1, &axialDofs[2], &val );
        ++face;
    }
    // loop over lateral faces
    for ( face = face.begin(); face != face.end(); ++face ) {
        auto faceCentroid        = face->centroid();
        auto lateralFaceIterator = interiorLateralFaceMap.find( faceCentroid );
        if ( lateralFaceIterator != interiorLateralFaceMap.end() ) {
            // get lateral face
            auto &lateralFace = lateralFaceIterator->second;
            // get crossflow from solution vector
            std::vector<size_t> gapDofs;
            subchannelDOFManager->getDOFs( lateralFace->globalID(), gapDofs );
            val = w_in / w_scale;
            solVec->setValuesByGlobalID( 1, &gapDofs[0], &val );
        }
    }
    solVec->copyVector( manufacturedVec );
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    //=============================================================================
    // solve
    //=============================================================================

    // put manufactured RHS into resVec
    nonlinearOperator->reset( nonlinearOpParams );

    // Create the solver
    auto nonlinearSolver = AMP::Solver::Test::buildSolver(
        "NonlinearSolver", input_db, globalComm, solVec, nonlinearOperator );

    // don't use zero initial guess
    nonlinearSolver->setZeroInitialGuess( false );

    // solve
    nonlinearOperator->residual( rhsVec, solVec, resVec );
    nonlinearSolver->apply( rhsVec, solVec );
    nonlinearOperator->residual( rhsVec, solVec, resVec );

    // Compute the flow temperature
    auto tempDOFManager = AMP::Discretization::simpleDOFManager::create(
        subchannelMesh,
        AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( subchannelMesh, 1 ),
        AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( subchannelMesh, 0 ),
        1 );
    auto tempVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );
    auto tempVec      = AMP::LinearAlgebra::createVector( tempDOFManager, tempVariable, true );
    face              = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    std::vector<size_t> tdofs;
    bool pass = true;
    for ( int i = 0; i < (int) face.size(); i++ ) {
        subchannelDOFManager->getDOFs( face->globalID(), axialDofs );
        tempDOFManager->getDOFs( face->globalID(), tdofs );
        double h = h_scale * solVec->getValueByGlobalID( axialDofs[1] );
        double P = P_scale * solVec->getValueByGlobalID( axialDofs[2] );
        std::map<std::string, std::shared_ptr<std::vector<double>>> temperatureArgMap;
        temperatureArgMap.insert(
            std::make_pair( "enthalpy", std::make_shared<std::vector<double>>( 1, h ) ) );
        temperatureArgMap.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, P ) ) );
        std::vector<double> temperatureResult( 1 );
        subchannelPhysicsModel->getProperty( "Temperature", temperatureResult, temperatureArgMap );
        tempVec->setValuesByGlobalID( 1, &tdofs[0], &temperatureResult[0] );
        // Check that we recover the enthalpy from the temperature
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
    subchannelDOFManager->getDOFs( face->globalID(), axialDofs );
    tempDOFManager->getDOFs( face->globalID(), tdofs );
    double TinSol = tempVec->getValueByGlobalID( tdofs[0] );
    std::cout << "Inlet Computed Mass Flow Rate = "
              << m_scale * solVec->getValueByGlobalID( axialDofs[0] ) << std::endl;
    std::cout << "Inlet Computed Enthalpy = "
              << h_scale * solVec->getValueByGlobalID( axialDofs[1] ) << std::endl;
    std::cout << "Inlet Computed Pressure = "
              << P_scale * solVec->getValueByGlobalID( axialDofs[2] ) << std::endl;
    std::cout << "Inlet Computed Temperature = " << TinSol << std::endl;
    std::cout << std::endl;
    face = --( ( xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 ) ).end() );
    subchannelDOFManager->getDOFs( face->globalID(), axialDofs );
    tempDOFManager->getDOFs( face->globalID(), tdofs );
    double ToutSol = tempVec->getValueByGlobalID( tdofs[0] );
    std::cout << "Outlet Computed Mass Flow Rate = "
              << m_scale * solVec->getValueByGlobalID( axialDofs[0] ) << std::endl;
    std::cout << "Outlet Computed Enthalpy = "
              << h_scale * solVec->getValueByGlobalID( axialDofs[1] ) << std::endl;
    std::cout << "Outlet Computed Pressure = "
              << P_scale * solVec->getValueByGlobalID( axialDofs[2] ) << std::endl;
    std::cout << "Outlet Computed Temperature = " << ToutSol << std::endl;

    // Compute the error
    auto absErrorVec = solVec->clone();
    absErrorVec->axpy( -1.0, *solVec, *manufacturedVec );
    auto relErrorVec = solVec->clone();
    relErrorVec->divide( *absErrorVec, *manufacturedVec );
    for ( size_t i = 0; i < solVec->getLocalSize(); i++ ) {
        if ( manufacturedVec->getValueByLocalID( i ) == 0 ) {
            val = fabs( solVec->getValueByLocalID( i ) );
            relErrorVec->setValuesByLocalID( 1, &i, &val );
        }
    }
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
    auto channel0 = AMP::Operator::Subchannel::subsetForSubchannel( subchannelMesh, 0, 0 );
    face          = AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( channel0, 0 );
    int N_print   = std::max( 1, (int) face.size() / 10 );
    for ( int i = 0; i < (int) face.size(); i++ ) {
        if ( i % N_print == 0 ) {
            subchannelDOFManager->getDOFs( face->globalID(), axialDofs );
            std::cout << "Computed Mass Flow Rate[" << i
                      << "] = " << m_scale * solVec->getValueByGlobalID( axialDofs[0] )
                      << std::endl;
            std::cout << "Solution Mass Flow Rate[" << i
                      << "] = " << m_scale * manufacturedVec->getValueByGlobalID( axialDofs[0] )
                      << std::endl;
            std::cout << "Computed Enthalpy[" << i
                      << "] = " << h_scale * solVec->getValueByGlobalID( axialDofs[1] )
                      << std::endl;
            std::cout << "Solution Enthalpy[" << i
                      << "] = " << h_scale * manufacturedVec->getValueByGlobalID( axialDofs[1] )
                      << std::endl;
            std::cout << "Computed Pressure[" << i
                      << "] = " << P_scale * solVec->getValueByGlobalID( axialDofs[2] )
                      << std::endl;
            std::cout << "Solution Pressure[" << i
                      << "] = " << P_scale * manufacturedVec->getValueByGlobalID( axialDofs[2] )
                      << std::endl;
            std::cout << std::endl;
        }
        ++face;
    }
    std::cout << "Delta T: " << ToutSol - TinSol << std::endl << std::endl;
    std::cout << "L2 Norm of Absolute Error: " << absErrorNorm << std::endl;
    std::cout << "L2 Norm of Relative Error: " << relErrorNorm << std::endl;

    input_db.reset();
}

int testSubchannelFourEqMMS( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files;
    if ( argc >= 2 ) {
        files.resize( argc - 1 );
        for ( int i = 0; i < argc - 1; i++ )
            files[i] = std::string( argv[i + 1] );
    } else {
        files.resize( 2 );
        files[0] = "testSubchannelFourEqMMS-1";
        files[1] = "testSubchannelFourEqMMS-2";
    }

    for ( auto &file : files )
        flowTest( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
