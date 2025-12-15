#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "../applyTests.h"

#include <iostream>
#include <memory>
#include <string>


static void thermalOxygenDiffusionTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database   = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( database );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto manager = AMP::Mesh::MeshFactory::create( meshParams );
    auto mesh    = manager->Subset( "brick" );

    // create a nonlinear BVP operator for nonlinear thermal
    AMP_INSIST( input_db->keyExists( "testNonlinearThermalOperator" ), "key missing!" );

    auto nonlinearThermalOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testNonlinearThermalOperator", input_db ) );
    auto nonlinearThermalVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator->getVolumeOperator() );
    auto thermalMaterialModel = nonlinearThermalVolumeOperator->getTransportModel();

    // create a nonlinear BVP operator for nonlinear oxygen diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearOxygenOperator" ), "key missing!" );

    auto nonlinearOxygenOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testNonlinearOxygenOperator", input_db ) );
    auto nonlinearOxygenVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearOxygenOperator->getVolumeOperator() );
    auto oxygenTransportModel = nonlinearOxygenVolumeOperator->getTransportModel();

    auto fickOperator = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nonlinearOxygenOperator->getVolumeOperator() );

    // create a column operator object for nonlinear thermothermal
    auto nonlinearThermalOxygenOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    nonlinearThermalOxygenOperator->append( nonlinearThermalOperator );
    nonlinearThermalOxygenOperator->append( nonlinearOxygenOperator );

    // initialize the input multi-variable
    auto volumeOperator = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nonlinearThermalOperator->getVolumeOperator() );
    auto inputVariable = volumeOperator->getInputVariable();

    // initialize the output multi-variable
    auto outputVariable = nonlinearThermalOxygenOperator->getOutputVariable();

    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // create solution, rhs, and residual vectors
    auto solVec = AMP::LinearAlgebra::createVector( nodalDofMap, inputVariable );
    auto rhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable );
    auto resVec = AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable );

    // set up the frozen variables for each operator
    // first get defaults
    auto thermalTransportModel =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( thermalMaterialModel );
    auto oxyModel =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( oxygenTransportModel );
    double defTemp = thermalTransportModel->getProperty()->get_default( "temperature" );
    double defConc = thermalTransportModel->getProperty()->get_default( "concentration" );

    // next get vectors
    auto tempVec = solVec->subsetVectorForVariable( "temperature" );
    auto concVec = solVec->subsetVectorForVariable( "concentration" );
    tempVec->setToScalar( defTemp );
    concVec->setToScalar( defConc );

    // set up the shift and scale parameters
    std::map<std::string, std::pair<double, double>> adjustment;
    auto matt = thermalTransportModel->getMaterial();
    auto mato = oxyModel->getMaterial();
    if ( volumeOperator->getPrincipalVariable() == "temperature" ) {
        std::string property = "ThermalConductivity";
        if ( matt->property( property )->is_argument( "temperature" ) ) {
            auto range                = matt->property( property )->get_arg_range( "temperature" );
            double scale              = 0.999 * ( range[1] - range[0] );
            double shift              = range[0] + 0.001 * ( range[1] - range[0] );
            adjustment["temperature"] = std::pair<int, int>( scale, shift );
        }
    }
    // the Fick has a principal variable of oxygen
    if ( fickOperator->getPrincipalVariable() == "concentration" ) {
        std::string property = "FickCoefficient";
        if ( mato->property( property )->is_argument( "concentration" ) ) {
            auto range   = mato->property( property )->get_arg_range( "concentration" );
            double scale = 0.999 * ( range[1] - range[0] );
            double shift = range[0] + 0.001 * ( range[1] - range[0] );
            adjustment["concentration"] = std::pair<int, int>( scale, shift );
        }
    }

    // construct the linear BVP operatora
    auto linearThermalOperator = std::make_shared<AMP::Operator::LinearBVPOperator>(
        nonlinearThermalOperator->getParameters( "Jacobian", nullptr ) );
    auto linearOxygenOperator = std::make_shared<AMP::Operator::LinearBVPOperator>(
        nonlinearOxygenOperator->getParameters( "Jacobian", nullptr ) );

    // create a column operator object for linear thermomechanics
    auto linearThermalOxygenOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    linearThermalOxygenOperator->append( linearThermalOperator );
    linearThermalOxygenOperator->append( linearOxygenOperator );

    ut->passes( exeName + " : create" );

    // test apply
    std::string msgPrefix = exeName + " : apply";
    auto testOperator =
        std::dynamic_pointer_cast<AMP::Operator::Operator>( nonlinearThermalOxygenOperator );
    applyTests( ut, msgPrefix, testOperator, rhsVec, solVec, resVec, adjustment );

    ut->passes( msgPrefix );

    auto resetParams = nonlinearThermalOxygenOperator->getParameters( "Jacobian", solVec );

    ut->passes( exeName + " : getJacobianParameters" );

    linearThermalOxygenOperator->reset( resetParams );

    ut->passes( exeName + " : Linear::reset" );
}


int testNonlinearThermalOxygenDiffusion_1( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "nonlinearBVP-Thermal-Oxygen-UO2MSRZC09-1" );

    for ( auto &exeName : exeNames )
        thermalOxygenDiffusionTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
