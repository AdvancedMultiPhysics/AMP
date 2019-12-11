#include "AMP/ampmesh/Mesh.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/ElementPhysicsModelParameters.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/diffusion/DiffusionConstants.h"
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
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include <memory>

#include "../applyTests.h"

#include <iostream>
#include <string>


static void nonlinearTest( AMP::UnitTest *ut, const std::string &exeName )
{
    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    std::cout << "testing with input file " << input_file << std::endl;
    std::cout.flush();

    // Test create

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    std::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
    std::shared_ptr<AMP::Mesh::MeshParameters> params( new AMP::Mesh::MeshParameters( database ) );
    params->setComm( globalComm );

    // Create the meshes from the input database
    std::shared_ptr<AMP::Mesh::Mesh> meshAdapter = AMP::Mesh::Mesh::buildMesh( params );

    // nonlinear operator
    std::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> diffOp;
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> elementModel;
    std::shared_ptr<AMP::Database> diffFEOp_db =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "NonlinearDiffusionOp" ) );
    std::shared_ptr<AMP::Operator::Operator> nonlinearOperator =
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "NonlinearDiffusionOp", input_db, elementModel );
    diffOp =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>( nonlinearOperator );

    // linear operator
    std::shared_ptr<AMP::Operator::DiffusionLinearFEOperator> linOp;
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> linElementModel;
    std::shared_ptr<AMP::Operator::Operator> linearOperator =
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "LinearDiffusionOp", input_db, linElementModel );
    linOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>( linearOperator );

    ut->passes( exeName + ": create" );
    std::cout.flush();

    // set up defaults for materials arguments and create transport model
    std::shared_ptr<AMP::Database> transportModel_db;
    if ( input_db->keyExists( "DiffusionTransportModel" ) )
        transportModel_db = input_db->getDatabase( "DiffusionTransportModel" );
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel =
        AMP::Operator::ElementPhysicsModelFactory::createElementPhysicsModel( transportModel_db );
    std::shared_ptr<AMP::Operator::DiffusionTransportModel> transportModel =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( elementPhysicsModel );

    double defTemp = transportModel_db->getWithDefault<double>( "Default_Temperature", 400.0 );
    double defConc = transportModel_db->getWithDefault<double>( "Default_Concentration", .33 );
    double defBurn = transportModel_db->getWithDefault<double>( "Default_Burnup", .5 );

    std::string property = transportModel_db->getString( "Property" );

    // create parameters for reset test
    std::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperatorParameters> diffOpParams(
        new AMP::Operator::DiffusionNonlinearFEOperatorParameters( diffFEOp_db ) );

    // nullify vectors in parameters
    diffOpParams->d_FrozenTemperature.reset();
    diffOpParams->d_FrozenConcentration.reset();
    diffOpParams->d_FrozenBurnup.reset();

    // create vectors for parameters
    std::shared_ptr<AMP::Database> active_db = diffFEOp_db->getDatabase( "ActiveInputVariables" );
    AMP::LinearAlgebra::Variable::shared_ptr tVar( new AMP::LinearAlgebra::Variable(
        active_db->getWithDefault<std::string>( "temperature", "not_specified" ) ) );
    AMP::LinearAlgebra::Variable::shared_ptr cVar( new AMP::LinearAlgebra::Variable(
        active_db->getWithDefault<std::string>( "concentration", "not_specified" ) ) );
    AMP::LinearAlgebra::Variable::shared_ptr bVar( new AMP::LinearAlgebra::Variable(
        active_db->getWithDefault<std::string>( "burnup", "not_specified" ) ) );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap =
        AMP::Discretization::simpleDOFManager::create(
            meshAdapter, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    //----------------------------------------------------------------------------------------------------------------------------------------------//

    // create solution, rhs, and residual vectors
    AMP::LinearAlgebra::Vector::shared_ptr tVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, tVar );
    AMP::LinearAlgebra::Vector::shared_ptr cVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, cVar );
    AMP::LinearAlgebra::Vector::shared_ptr bVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, bVar );
    tVec->setToScalar( defTemp );
    cVec->setToScalar( defConc );
    bVec->setToScalar( defBurn );

    // set principal variable vector and shift for applyTests
    double shift = 0., scale = 1.;
    std::vector<double> range( 2 );
    AMP::Materials::Material::shared_ptr mat = transportModel->getMaterial();
    if ( diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::TEMPERATURE ) {
        diffOpParams->d_FrozenTemperature = tVec;
        if ( ( mat->property( property ) )->is_argument( "temperature" ) ) {
            range = ( mat->property( property ) )->get_arg_range( "temperature" ); // Compile error
            scale = range[1] - range[0];
            shift = range[0] + 0.001 * scale;
            scale *= 0.999;
        }
    }
    if ( diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::CONCENTRATION ) {
        diffOpParams->d_FrozenConcentration = cVec;
        if ( ( mat->property( property ) )->is_argument( "concentration" ) ) {
            range =
                ( mat->property( property ) )->get_arg_range( "concentration" ); // Compile error
            scale = range[1] - range[0];
            shift = range[0] + 0.001 * scale;
            scale *= 0.999;
        }
    }
    if ( diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::BURNUP ) {
        AMP_INSIST( false, "do not know what to do" );
    }

    // set frozen vectors in parameters
    if ( diffFEOp_db->getWithDefault( "Freezetemperature", false ) )
        diffOpParams->d_FrozenTemperature = tVec;
    if ( diffFEOp_db->getWithDefault( "Freezeconcentration", false ) )
        diffOpParams->d_FrozenConcentration = cVec;
    if ( diffFEOp_db->getWithDefault( "Freezeburnup", false ) )
        diffOpParams->d_FrozenBurnup = bVec;

    // set transport model
    diffOpParams->d_transportModel = transportModel;

    // Test reset
    {
        diffOp->reset( diffOpParams );
        ut->passes( exeName + ": reset" );
        std::cout.flush();
    }

    // set up variables for apply tests
    // AMP::LinearAlgebra::Variable::shared_ptr diffSolVar =
    // diffOp->getInputVariable(diffOp->getPrincipalVariableId());
    AMP::LinearAlgebra::Variable::shared_ptr diffSolVar = diffOp->getOutputVariable();

    AMP::LinearAlgebra::Variable::shared_ptr diffRhsVar = diffOp->getOutputVariable();
    AMP::LinearAlgebra::Variable::shared_ptr diffResVar = diffOp->getOutputVariable();
    AMP::LinearAlgebra::Variable::shared_ptr workVar( new AMP::LinearAlgebra::Variable( "work" ) );
    std::vector<unsigned int> nonPrincIds = diffOp->getNonPrincipalVariableIds();
    unsigned int numNonPrincIds           = nonPrincIds.size();
    std::vector<AMP::LinearAlgebra::Variable::shared_ptr> nonPrincVars( numNonPrincIds );
    AMP::LinearAlgebra::Variable::shared_ptr inputVar = diffOp->getInputVariable();
    for ( size_t i = 0; i < numNonPrincIds; i++ ) {
        // nonPrincVars[i] = diffOp->getInputVariable(nonPrincIds[i]);
        nonPrincVars[i] = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>( inputVar )
                              ->getVariable( i );
    }

    // Test apply
    {
        std::string msgPrefix = exeName + ": apply";
        AMP::LinearAlgebra::Vector::shared_ptr diffSolVec =
            AMP::LinearAlgebra::createVector( nodalDofMap, diffSolVar );
        AMP::LinearAlgebra::Vector::shared_ptr diffRhsVec =
            AMP::LinearAlgebra::createVector( nodalDofMap, diffRhsVar );
        AMP::LinearAlgebra::Vector::shared_ptr diffResVec =
            AMP::LinearAlgebra::createVector( nodalDofMap, diffResVar );
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> nonPrincVecs( numNonPrincIds );
        for ( unsigned int i = 0; i < numNonPrincIds; i++ ) {
            nonPrincVecs[i] = AMP::LinearAlgebra::createVector( nodalDofMap, nonPrincVars[i] );
            if ( nonPrincIds[i] == AMP::Operator::Diffusion::TEMPERATURE )
                nonPrincVecs[i]->setToScalar( defTemp );
            if ( nonPrincIds[i] == AMP::Operator::Diffusion::CONCENTRATION )
                nonPrincVecs[i]->setToScalar( defConc );
            if ( nonPrincIds[i] == AMP::Operator::Diffusion::BURNUP )
                nonPrincVecs[i]->setToScalar( defBurn );
        }
        diffRhsVec->setToScalar( 0.0 );
        applyTests(
            ut, msgPrefix, nonlinearOperator, diffRhsVec, diffSolVec, diffResVec, shift, scale );
        std::cout.flush();

        // Test getJacobianParameters and linear operator creation
        {
            diffSolVec->setRandomValues();
            adjust( diffSolVec, shift, scale );
            std::shared_ptr<AMP::Operator::OperatorParameters> jacParams =
                diffOp->getParameters( "Jacobian", diffSolVec );
            linOp->reset(
                std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperatorParameters>(
                    jacParams ) );
            ut->passes( exeName + ": getJacobianParameters" );
            std::cout.flush();
        }
    }

    // now run apply tests with multi-vectors
    AMP::LinearAlgebra::Variable::shared_ptr auxInpVar =
        diffSolVar->cloneVariable( "NonlinearDiffusionOperator-auxInpVar" );
    AMP::LinearAlgebra::Variable::shared_ptr auxOutVar =
        diffResVar->cloneVariable( "NonlinearDiffusionOperator-auxOutVar" );
    AMP::LinearAlgebra::Variable::shared_ptr auxWorkVar =
        diffSolVar->cloneVariable( "NonlinearDiffusionOperator-auxWorkVar" );

    std::shared_ptr<AMP::LinearAlgebra::MultiVariable> myMultiInpVar(
        new AMP::LinearAlgebra::MultiVariable( "MultiInputVariable" ) );
    myMultiInpVar->add( diffSolVar );
    myMultiInpVar->add( auxInpVar );

    std::shared_ptr<AMP::LinearAlgebra::MultiVariable> myMultiOutVar(
        new AMP::LinearAlgebra::MultiVariable( "MultiOutputVariable" ) );
    myMultiOutVar->add( diffResVar );
    myMultiOutVar->add( auxOutVar );

    std::shared_ptr<AMP::LinearAlgebra::MultiVariable> myMultiWorkVar(
        new AMP::LinearAlgebra::MultiVariable( "MultiWorkVariable" ) );
    myMultiWorkVar->add( workVar );
    myMultiWorkVar->add( auxWorkVar );

    {
        std::string msgPrefix = exeName + ": apply MultiVector ";
        AMP::LinearAlgebra::Vector::shared_ptr solVec =
            AMP::LinearAlgebra::createVector( nodalDofMap, myMultiInpVar );
        AMP::LinearAlgebra::Vector::shared_ptr rhsVec =
            AMP::LinearAlgebra::createVector( nodalDofMap, myMultiOutVar );
        AMP::LinearAlgebra::Vector::shared_ptr resVec =
            AMP::LinearAlgebra::createVector( nodalDofMap, myMultiOutVar );

        // test apply with single variable vectors
        applyTests( ut, msgPrefix, nonlinearOperator, rhsVec, solVec, resVec, shift, scale );
        std::cout.flush();
    }

    // Test isValidInput function
    {
        AMP::LinearAlgebra::Vector::shared_ptr testVec =
            AMP::LinearAlgebra::createVector( nodalDofMap, diffSolVar );

        testVec->setToScalar( -1000. );
        if ( not diffOp->isValidInput( testVec ) )
            ut->passes( exeName + ": validInput-1" );
        else {
            if ( ( diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::TEMPERATURE ) &&
                 ( ( mat->property( property ) )->is_argument( "temperature" ) ) ) {
                ut->failure( exeName + ": validInput-1" );
            } else if ( ( diffOp->getPrincipalVariableId() ==
                          AMP::Operator::Diffusion::CONCENTRATION ) &&
                        ( ( mat->property( property ) )->is_argument( "concentration" ) ) ) {
                ut->failure( exeName + ": validInput-1" );
            }
        }
        testVec->setToScalar( 1.e99 );
        if ( not diffOp->isValidInput( testVec ) )
            ut->passes( exeName + ": validInput-2" );
        else {
            if ( ( diffOp->getPrincipalVariableId() == AMP::Operator::Diffusion::TEMPERATURE ) &&
                 ( ( mat->property( property ) )->is_argument( "temperature" ) ) ) {
                ut->failure( exeName + ": validInput-2" );
            } else if ( ( diffOp->getPrincipalVariableId() ==
                          AMP::Operator::Diffusion::CONCENTRATION ) &&
                        ( ( mat->property( property ) )->is_argument( "concentration" ) ) ) {
                ut->failure( exeName + ": validInput-2" );
            }
        }
        testVec->setToScalar( 1.e99 );
        std::cout.flush();
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
