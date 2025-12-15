#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/FickSoretNonlinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"

#include "applyTests.h"

#include <iostream>
#include <memory>
#include <string>


static void bvpTest1( AMP::UnitTest *ut, const std::string &exeName )
{
    // Tests FickSoret Dirchlet BVP operator for temperature

    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logOnlyNodeZero( log_file );

    // Input database
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Create the Mesh.
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create nonlinear FickSoret BVP operator and access volume nonlinear FickSoret operator
    auto nlinBVPOperator = AMP::Operator::OperatorBuilder::createOperator(
        mesh, "testFickSoretBVPOperator", input_db );
    auto nlinBVPOp =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>( nlinBVPOperator );
    auto nlinOp = std::dynamic_pointer_cast<AMP::Operator::FickSoretNonlinearFEOperator>(
        nlinBVPOp->getVolumeOperator() );
    auto fickOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nlinOp->getFickOperator() );
    auto soretOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nlinOp->getSoretOperator() );

    // use the linear BVP operator to create a Fick linear operator with bc's
    auto linBVPOperator = AMP::Operator::OperatorBuilder::createOperator(
        mesh, "testLinearFickBVPOperator", input_db );
    auto linBVPOp = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>( linBVPOperator );

    ut->passes( exeName + ": creation" );
    std::cout.flush();

    // set shift, scale for applyTests
    std::map<std::string, std::pair<double, double>> adjustment;
    auto fickTransportModel = fickOp->getTransportModel();
    std::vector<double> defaults( 2, 0 );
    auto fmat = fickTransportModel->getMaterial();
    // the Soret has a principal variable of temperature
    if ( soretOp->getPrincipalVariable() == "temperature" ) {
        std::string property = "ThermalDiffusionCoefficient";
        if ( fmat->property( property )->is_argument( "temperature" ) ) {
            auto range                = fmat->property( property )->get_arg_range( "temperature" );
            double scale              = 0.999 * ( range[1] - range[0] );
            double shift              = range[0] + 0.001 * ( range[1] - range[0] );
            adjustment["temperature"] = std::pair<int, int>( scale, shift );
            defaults                  = fmat->property( property )->get_defaults();
        }
    }
    // the Fick has a principal variable of temperature
    if ( fickOp->getPrincipalVariable() == "concentration" ) {
        std::string property = "FickCoefficient";
        if ( fmat->property( property )->is_argument( "concentration" ) ) {
            auto range   = fmat->property( property )->get_arg_range( "concentration" );
            double scale = 0.999 * ( range[1] - range[0] );
            double shift = range[0] + 0.001 * ( range[1] - range[0] );
            adjustment["concentration"] = std::pair<int, int>( scale, shift );
            defaults                    = fmat->property( property )->get_defaults();
        }
    }

    // Set up input and output vectors
    auto cVar     = std::make_shared<AMP::LinearAlgebra::Variable>( "concentration" );
    auto tVar     = std::make_shared<AMP::LinearAlgebra::Variable>( "temperature" );
    auto fsInpVar = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "fsInput" );
    fsInpVar->add( cVar );
    fsInpVar->add( tVar );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // create solution, rhs, and residual vectors
    auto multivector = AMP::LinearAlgebra::MultiVector::create( "mulitvector", mesh->getComm() );
    multivector->addVector( AMP::LinearAlgebra::createVector( nodalDofMap, cVar ) );
    multivector->addVector( AMP::LinearAlgebra::createVector( nodalDofMap, tVar ) );
    auto solVec = std::dynamic_pointer_cast<AMP::LinearAlgebra::Vector>( multivector );
    auto rhsVec = solVec->clone();
    auto resVec = solVec->clone();

    // set default values of input variables
    auto inConcVec = solVec->subsetVectorForVariable( cVar );
    auto inTempVec = solVec->subsetVectorForVariable( tVar );
    inConcVec->setToScalar( defaults[1] );
    inTempVec->setToScalar( defaults[0] );
    rhsVec->setToScalar( 0. );

    AMP_INSIST( nlinOp->isValidVector( solVec ), "input variable not set up correctly" );

    // Test apply
    std::string msgPrefix = exeName + ": apply";
    {
        applyTests( ut, msgPrefix, nlinBVPOperator, rhsVec, solVec, resVec, adjustment );
    }
    std::cout.flush();

    // Test linear reset from getJacobianParameters
    for ( int i = 0; i < 3; i++ ) {
        inConcVec->setRandomValues();
        inTempVec->setRandomValues();
        adjust( solVec, adjustment );
        rhsVec->setRandomValues();
        resVec->setRandomValues();
        auto jacparams = nlinBVPOp->getParameters( "Jacobian", solVec );
        linBVPOp->reset( jacparams );
    } // end for i

    ut->passes( exeName + ": getJacobianParameters" );
    std::cout.flush();
}


int testFickSoretBVP( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    std::vector<std::string> files = { "FickSoret-BVP-TUI-1", "FickSoret-BVP-UO2MSRZC09-1" };
    for ( auto &file : files )
        bvpTest1( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
