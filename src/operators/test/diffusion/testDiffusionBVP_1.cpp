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
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearElement.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include "../applyTests.h"

#include <iostream>
#include <memory>
#include <string>


static void bvpTest1( AMP::UnitTest *ut, const std::string &exeName )
{
    // Tests diffusion Dirchlet BVP operator for temperature

    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logOnlyNodeZero( log_file );

    // Input database
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    //   Create the Mesh.
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create nonlinear Diffusion BVP operator and access volume nonlinear Diffusion operator
    auto nbvp_db         = input_db->getDatabase( "ThermalNonlinearBVPOperator" );
    auto nlinBVPOperator = AMP::Operator::OperatorBuilder::createOperator(
        mesh, "ThermalNonlinearBVPOperator", input_db );
    auto nlinBVPOp =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>( nlinBVPOperator );
    auto nlinOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nlinBVPOp->getVolumeOperator() );
    auto elementPhysicsModel = nlinOp->getTransportModel();

    // use the linear BVP operator to create a thermal linear operator with bc's
    auto linBVPOp = std::make_shared<AMP::Operator::LinearBVPOperator>(
        nlinBVPOp->getParameters( "Jacobian", nullptr ) );
    ut->passes( exeName + ": creation" );
    std::cout.flush();

    // Set up input and output vectors
    auto bvpSolVar = nlinOp->getOutputVariable();
    auto bvpRhsVar = nlinOp->getOutputVariable();
    auto bvpResVar = nlinOp->getOutputVariable();

    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // create solution, rhs, and residual vectors
    auto bvpSolVec = AMP::LinearAlgebra::createVector( nodalDofMap, bvpSolVar );
    auto bvpRhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, bvpRhsVar );
    auto bvpResVec = AMP::LinearAlgebra::createVector( nodalDofMap, bvpResVar );

    bvpRhsVec->setToScalar( 0.0 );

    auto volOp_db = input_db->getDatabase( nbvp_db->getString( "VolumeOperator" ) );
    auto model_db = volOp_db->getDatabase( "LocalModel" );
    auto property = model_db->getString( "Property" );

    // set shift, scale for applyTests
    std::map<std::string, std::pair<double, double>> adjustment;
    auto transportModel =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( elementPhysicsModel );
    auto mat = transportModel->getMaterial();
    if ( nlinOp->getPrincipalVariable() == "temperature" ) {
        if ( mat->property( property )->is_argument( "temperature" ) ) {
            auto range = mat->property( property )->get_arg_range( "temperature" ); // Compile error
            double scale              = 0.999 * ( range[1] - range[0] );
            double shift              = range[0] + 0.001 * ( range[1] - range[0] );
            adjustment["temperature"] = std::pair<int, int>( scale, shift );
        }
    }
    if ( nlinOp->getPrincipalVariable() == "concentration" ) {
        if ( mat->property( property )->is_argument( "concentration" ) ) {
            auto range =
                mat->property( property )->get_arg_range( "concentration" ); // Compile error
            double scale                = 0.999 * ( range[1] - range[0] );
            double shift                = range[0] + 0.001 * ( range[1] - range[0] );
            adjustment["concentration"] = std::pair<int, int>( scale, shift );
        }
    }
    if ( nlinOp->getPrincipalVariable() == "burnup" ) {
        AMP_INSIST( false, "do not know what to do" );
    }

    // Test apply
    std::string msgPrefix = exeName + ": apply";
    {
        applyTests( ut, msgPrefix, nlinBVPOperator, bvpRhsVec, bvpSolVec, bvpResVec, adjustment );
    }
    std::cout.flush();

    // Test linear reset from getJacobianParameters
    for ( int i = 0; i < 3; i++ ) {
        bvpSolVec->setRandomValues();
        adjust( bvpSolVec, adjustment );
        bvpRhsVec->setRandomValues();
        bvpResVec->setRandomValues();
        auto jacparams = nlinBVPOp->getParameters( "Jacobian", bvpSolVec );
        linBVPOp->reset( jacparams );
    } // end for i

    ut->passes( exeName + ": getJacobianParameters" );
    std::cout.flush();
}

int testDiffusionBVP_1( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    const int NUMFILES          = 6;
    std::string files[NUMFILES] = {
        "Diffusion-TUI-Thermal-2",        "Diffusion-TUI-Fick-2",        "Diffusion-TUI-Soret-2",
        "Diffusion-UO2MSRZC09-Thermal-2", "Diffusion-UO2MSRZC09-Fick-2",
        "Diffusion-UO2MSRZC09-Soret-2" //,
        //"Diffusion-TUI-Fick-3", "Diffusion-UO2MSRZC09-Fick-3"
    };

    for ( auto &file : files )
        bvpTest1( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
