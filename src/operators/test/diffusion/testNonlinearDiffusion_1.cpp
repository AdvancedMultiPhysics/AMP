#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/ElementPhysicsModelParameters.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperatorParameters.h"
#include "AMP/operators/diffusion/DiffusionNonlinearElement.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperatorParameters.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "../applyTests.h"

#include <iostream>
#include <memory>
#include <string>


static void nonlinearTest( AMP::UnitTest *ut, const std::string &exeName )
{
    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    AMP::pout << "testing with input file " << input_file << std::endl;

    // Test create
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // nonlinear operator
    auto diffFEOp_db = input_db->getDatabase( "NonlinearDiffusionOp" );
    auto nonlinearOperator =
        AMP::Operator::OperatorBuilder::createOperator( mesh, "NonlinearDiffusionOp", input_db );
    auto diffOp =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>( nonlinearOperator );

    // linear operator
    auto linearOperator =
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearDiffusionOp", input_db );
    auto linOp =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>( linearOperator );

    ut->passes( exeName + ": create" );

    // set up defaults for materials arguments and create transport model
    auto transportModel_db = input_db->getDatabase( "DiffusionTransportModel" );
    auto elementPhysicsModel =
        AMP::Operator::ElementPhysicsModelFactory::createElementPhysicsModel( transportModel_db );
    auto transportModel =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( elementPhysicsModel );

    auto defTemp = transportModel_db->getWithDefault<double>( "Default_Temperature", 400.0 );
    auto defConc = transportModel_db->getWithDefault<double>( "Default_Concentration", .33 );
    auto defBurn = transportModel_db->getWithDefault<double>( "Default_Burnup", .5 );

    std::string property = transportModel_db->getString( "Property" );

    // create parameters for reset test
    auto diffOpParams =
        std::make_shared<AMP::Operator::DiffusionNonlinearFEOperatorParameters>( diffFEOp_db );

    // nullify vectors in parameters
    diffOpParams->d_FrozenVecs.clear();

    // create vectors for parameters
    auto active_db = diffFEOp_db->getDatabase( "ActiveInputVariables" );
    auto tVar      = std::make_shared<AMP::LinearAlgebra::Variable>(
        active_db->getWithDefault<std::string>( "temperature", "not_specified" ) );
    auto cVar = std::make_shared<AMP::LinearAlgebra::Variable>(
        active_db->getWithDefault<std::string>( "concentration", "not_specified" ) );
    auto bVar = std::make_shared<AMP::LinearAlgebra::Variable>(
        active_db->getWithDefault<std::string>( "burnup", "not_specified" ) );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // create solution, rhs, and residual vectors
    auto tVec = AMP::LinearAlgebra::createVector( nodalDofMap, tVar );
    auto cVec = AMP::LinearAlgebra::createVector( nodalDofMap, cVar );
    auto bVec = AMP::LinearAlgebra::createVector( nodalDofMap, bVar );
    tVec->setToScalar( defTemp );
    cVec->setToScalar( defConc );
    bVec->setToScalar( defBurn );

    // set principal variable vector and shift for applyTests
    std::map<std::string, std::pair<double, double>> adjustment;
    auto mat = transportModel->getMaterial();
    if ( diffOp->getPrincipalVariable() == "temperature" ) {
        diffOpParams->d_FrozenVecs["temperature"] = tVec;
        if ( mat->property( property )->is_argument( "temperature" ) ) {
            auto range = mat->property( property )->get_arg_range( "temperature" ); // Compile error
            double scale              = 0.999 * ( range[1] - range[0] );
            double shift              = range[0] + 0.001 * ( range[1] - range[0] );
            adjustment["temperature"] = std::pair<int, int>( scale, shift );
        }
    }
    if ( diffOp->getPrincipalVariable() == "concentration" ) {
        diffOpParams->d_FrozenVecs["concentration"] = cVec;
        if ( mat->property( property )->is_argument( "concentration" ) ) {
            auto range =
                mat->property( property )->get_arg_range( "concentration" ); // Compile error
            double scale                = 0.999 * ( range[1] - range[0] );
            double shift                = range[0] + 0.001 * ( range[1] - range[0] );
            adjustment["concentration"] = std::pair<int, int>( scale, shift );
        }
    }
    if ( diffOp->getPrincipalVariable() == "burnup" ) {
        AMP_ERROR( "do not know what to do" );
    }

    // set frozen vectors in parameters
    if ( diffFEOp_db->getWithDefault<bool>( "Freezetemperature", false ) )
        diffOpParams->d_FrozenVecs["temperature"] = tVec;
    if ( diffFEOp_db->getWithDefault<bool>( "Freezeconcentration", false ) )
        diffOpParams->d_FrozenVecs["concentration"] = cVec;
    if ( diffFEOp_db->getWithDefault<bool>( "Freezeburnup", false ) )
        diffOpParams->d_FrozenVecs["burnup"] = bVec;

    // set transport model
    diffOpParams->d_transportModel = transportModel;

    // Test reset
    {
        diffOp->reset( diffOpParams );
        ut->passes( exeName + ": reset" );
    }

    // set up variables for apply tests
    // auto diffSolVar = diffOp->getInputVariable(diffOp->getPrincipalVariableId());
    auto diffSolVar  = diffOp->getOutputVariable();
    auto diffRhsVar  = diffOp->getOutputVariable();
    auto diffResVar  = diffOp->getOutputVariable();
    auto workVar     = std::make_shared<AMP::LinearAlgebra::Variable>( "work" );
    auto nonPrincIds = diffOp->getNonPrincipalVariableIds();
    std::vector<std::shared_ptr<AMP::LinearAlgebra::Variable>> nonPrincVars( nonPrincIds.size() );
    for ( size_t i = 0; i < nonPrincIds.size(); i++ )
        nonPrincVars[i] = std::make_shared<AMP::LinearAlgebra::Variable>( nonPrincIds[i] );

    // Test apply
    {
        auto msgPrefix  = exeName + ": apply";
        auto diffSolVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffSolVar );
        auto diffRhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffRhsVar );
        auto diffResVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffResVar );
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> nonPrincVecs( nonPrincIds.size() );
        for ( size_t i = 0; i < nonPrincIds.size(); i++ ) {
            nonPrincVecs[i] = AMP::LinearAlgebra::createVector( nodalDofMap, nonPrincVars[i] );
            if ( nonPrincIds[i] == "temperature" )
                nonPrincVecs[i]->setToScalar( defTemp );
            if ( nonPrincIds[i] == "concentration" )
                nonPrincVecs[i]->setToScalar( defConc );
            if ( nonPrincIds[i] == "burnup" )
                nonPrincVecs[i]->setToScalar( defBurn );
        }
        diffRhsVec->setToScalar( 0.0 );
        applyTests(
            ut, msgPrefix, nonlinearOperator, diffRhsVec, diffSolVec, diffResVec, adjustment );

        // Test getJacobianParameters and linear operator creation
        {
            diffSolVec->setRandomValues();
            adjust( diffSolVec, adjustment );
            auto jacParams = diffOp->getParameters( "Jacobian", diffSolVec );
            linOp->reset(
                std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperatorParameters>(
                    jacParams ) );
            ut->passes( exeName + ": getJacobianParameters" );
        }
    }

    // now run apply tests with multi-vectors
    auto auxInpVar  = diffSolVar->clone( "NonlinearDiffusionOperator-auxInpVar" );
    auto auxOutVar  = diffResVar->clone( "NonlinearDiffusionOperator-auxOutVar" );
    auto auxWorkVar = diffSolVar->clone( "NonlinearDiffusionOperator-auxWorkVar" );

    auto myMultiInpVar =
        std::make_shared<AMP::LinearAlgebra::MultiVariable>( "MultiInputVariable" );
    myMultiInpVar->add( diffSolVar );
    myMultiInpVar->add( auxInpVar );

    auto myMultiOutVar =
        std::make_shared<AMP::LinearAlgebra::MultiVariable>( "MultiOutputVariable" );
    myMultiOutVar->add( diffResVar );
    myMultiOutVar->add( auxOutVar );

    auto myMultiWorkVar =
        std::make_shared<AMP::LinearAlgebra::MultiVariable>( "MultiWorkVariable" );
    myMultiWorkVar->add( workVar );
    myMultiWorkVar->add( auxWorkVar );

    {
        std::string msgPrefix = exeName + ": apply MultiVector ";
        auto solVec           = AMP::LinearAlgebra::createVector( nodalDofMap, myMultiInpVar );
        auto rhsVec           = AMP::LinearAlgebra::createVector( nodalDofMap, myMultiOutVar );
        auto resVec           = AMP::LinearAlgebra::createVector( nodalDofMap, myMultiOutVar );

        // test apply with single variable vectors
        applyTests( ut, msgPrefix, nonlinearOperator, rhsVec, solVec, resVec, adjustment );
    }

    // Test isValidVector function
    {
        auto testVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffSolVar );

        testVec->setToScalar( -1000. );
        if ( !diffOp->isValidVector( testVec ) )
            ut->passes( exeName + ": validInput-1" );
        else {
            if ( ( diffOp->getPrincipalVariable() == "temperature" ) &&
                 ( ( mat->property( property ) )->is_argument( "temperature" ) ) ) {
                ut->failure( exeName + ": validInput-1" );
            } else if ( ( diffOp->getPrincipalVariable() == "concentration" ) &&
                        ( ( mat->property( property ) )->is_argument( "concentration" ) ) ) {
                ut->failure( exeName + ": validInput-1" );
            }
        }
        testVec->setToScalar( 1.e99 );
        if ( !diffOp->isValidVector( testVec ) )
            ut->passes( exeName + ": validInput-2" );
        else {
            if ( ( diffOp->getPrincipalVariable() == "temperature" ) &&
                 ( ( mat->property( property ) )->is_argument( "temperature" ) ) ) {
                ut->failure( exeName + ": validInput-2" );
            } else if ( ( diffOp->getPrincipalVariable() == "concentration" ) &&
                        ( ( mat->property( property ) )->is_argument( "concentration" ) ) ) {
                ut->failure( exeName + ": validInput-2" );
            }
        }
        testVec->setToScalar( 1.e99 );
    }
}

int testNonlinearDiffusion_1( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    const int NUMFILES          = 14;
    std::string files[NUMFILES] = {
        "Diffusion-CylindricalFick-1",
        "Diffusion-TUI-Thermal-1",
        "Diffusion-TUI-Fick-1",
        "Diffusion-TUI-Soret-1",
        "Diffusion-UO2MSRZC09-Thermal-1",
        "Diffusion-UO2MSRZC09-Fick-1",
        "Diffusion-UO2MSRZC09-Soret-1",
        "Diffusion-TUI-Thermal-ActiveTemperatureAndConcentration-1",
        "Diffusion-TUI-Fick-ActiveTemperatureAndConcentration-1",
        "Diffusion-TUI-Soret-ActiveTemperatureAndConcentration-1",
        "Diffusion-UO2MSRZC09-Thermal-ActiveTemperatureAndConcentration-1",
        "Diffusion-UO2MSRZC09-Fick-ActiveTemperatureAndConcentration-1",
        "Diffusion-UO2MSRZC09-Soret-ActiveTemperatureAndConcentration-1",
        "Diffusion-TUI-TensorFick-1"
    };

    for ( auto &file : files )
        nonlinearTest( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
