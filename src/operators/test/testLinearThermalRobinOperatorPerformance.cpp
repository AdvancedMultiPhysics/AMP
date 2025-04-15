#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <iomanip>
#include <memory>
#include <string>

namespace AMP::Operator {

class DerivedTestDiffusionLinearFEOperator : public DiffusionLinearFEOperator
{
public:
    explicit DerivedTestDiffusionLinearFEOperator(
        std::shared_ptr<const OperatorParameters> params )
        : DiffusionLinearFEOperator( params )
    {
    }
    //! Subset output vector
    std::shared_ptr<AMP::LinearAlgebra::Vector>
    subsetOutputVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec ) override
    {
        return vec;
    }

    //! Subset output vector
    std::shared_ptr<const AMP::LinearAlgebra::Vector>
    subsetOutputVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> vec ) override
    {
        return vec;
    }

    //! Subset input vector
    std::shared_ptr<AMP::LinearAlgebra::Vector>
    subsetInputVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec ) override
    {
        return vec;
    }

    //! Subset input vector
    std::shared_ptr<const AMP::LinearAlgebra::Vector>
    subsetInputVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> vec ) override
    {
        return vec;
    }
};

std::shared_ptr<Operator> createOperator( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                          const std::string &operatorName,
                                          std::shared_ptr<AMP::Database> input_db )
{
    std::shared_ptr<ElementPhysicsModel> elementPhysicsModel;
    auto operator_db = input_db->getDatabase( operatorName );
    AMP_INSIST( operator_db,
                "Error:: createOperator(): No operator database entry with "
                "given name exists in input database: " +
                    operatorName );

    // we create the element physics model if a database entry exists
    // and the incoming element physics model pointer is NULL
    if ( operator_db->keyExists( "LocalModel" ) ) {
        // extract the name of the local model from the operator database
        auto localModelName = operator_db->getString( "LocalModel" );
        // check whether a database exists in the global database
        // (NOTE: not the operator database) with the given name
        AMP_INSIST( input_db->keyExists( localModelName ),
                    "Error::createOperator(): No local model "
                    "database entry with given name exists in input database" );

        auto localModel_db = input_db->getDatabase( localModelName );
        AMP_INSIST( localModel_db,
                    "Error:: OperatorBuilder::createOperator(): No local model database "
                    "entry with given name exists in input databaseot" );

        // If a non-NULL factory is being supplied through the argument list
        // use it, else call the AMP ElementPhysicsModelFactory interface
        elementPhysicsModel =
            ElementPhysicsModelFactory::createElementPhysicsModel( localModel_db );

        AMP_INSIST( elementPhysicsModel, "Error::createOperator(): local model creation failed" );
    }
    PROFILE( "OperatorBuilder::createLinearDiffusionOperator" );

    // first create a DiffusionTransportModel
    std::shared_ptr<DiffusionTransportModel> transportModel;
    if ( elementPhysicsModel ) {
        transportModel = std::dynamic_pointer_cast<DiffusionTransportModel>( elementPhysicsModel );
    } else {
        std::shared_ptr<AMP::Database> transportModel_db;
        if ( input_db->keyExists( "DiffusionTransportModel" ) ) {
            transportModel_db = input_db->getDatabase( "DiffusionTransportModel" );
        } else {
            AMP_INSIST( false, "Key ''DiffusionTransportModel'' is missing!" );
        }
        elementPhysicsModel =
            ElementPhysicsModelFactory::createElementPhysicsModel( transportModel_db );
        transportModel = std::dynamic_pointer_cast<DiffusionTransportModel>( elementPhysicsModel );
    }
    AMP_INSIST( transportModel, "NULL transport model" );

    // next create a ElementOperation object
    AMP_INSIST( operator_db->keyExists( "DiffusionElement" ),
                "Key ''DiffusionElement'' is missing!" );
    std::shared_ptr<ElementOperation> diffusionLinElem =
        ElementOperationFactory::createElementOperation(
            operator_db->getDatabase( "DiffusionElement" ) );

    // now create the linear diffusion operator
    std::shared_ptr<AMP::Database> diffusionLinFEOp_db;
    if ( operator_db->getString( "name" ) == "DiffusionLinearFEOperator" ) {
        diffusionLinFEOp_db = operator_db;
    } else {
        AMP_INSIST( operator_db->keyExists( "name" ), "Key ''name'' is missing!" );
    }

    AMP_INSIST( diffusionLinFEOp_db,
                "Error: The database object for DiffusionLinearFEOperator is NULL" );

    auto diffusionOpParams =
        std::make_shared<DiffusionLinearFEOperatorParameters>( diffusionLinFEOp_db );
    diffusionOpParams->d_transportModel = transportModel;
    diffusionOpParams->d_elemOp         = diffusionLinElem;
    diffusionOpParams->d_Mesh           = mesh;
    diffusionOpParams->d_inDofMap       = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    diffusionOpParams->d_outDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto diffusionOp = std::make_shared<DerivedTestDiffusionLinearFEOperator>( diffusionOpParams );

    auto matrix = diffusionOp->getMatrix();
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    return diffusionOp;
}
} // namespace AMP::Operator

void linearThermalTest( AMP::UnitTest *ut, const std::string &inputFileName )
{
    PROFILE( "DRIVER::testLinearThermalRobinOperatorPerformanceTest" );

    // Input and output file names
    std::string input_file = inputFileName;
    std::ostringstream ss;
    ss << "output_testLinearThermalRobinOperatorPerformance_r" << std::setw( 3 )
       << std::setfill( '0' ) << AMP::AMPManager::getCommWorld().getSize();

    AMP::pout << "Running linearThermalTest with input " << input_file << std::endl;

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    AMP::logAllNodes( ss.str() );

    // Create the Mesh
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    auto comm      = AMP::AMP_MPI( AMP_COMM_WORLD );
    mgrParams->setComm( comm );
    auto meshAdapter = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode          = 1;
    int DOFsPerElement       = 8;
    int nodalGhostWidth      = 1;
    int gaussPointGhostWidth = 1;
    bool split               = true;
    auto nodalDofMap         = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Cell, gaussPointGhostWidth, DOFsPerElement, split );

    // CREATE THE THERMAL BVP OPERATOR
    auto linearOperator = AMP::Operator::OperatorBuilder::createOperator(
        meshAdapter, "DiffusionLinearFEOperator", input_db );

    auto TemperatureInKelvinVec =
        AMP::LinearAlgebra::createVector( nodalDofMap,
                                          linearOperator->getInputVariable(),
                                          true,
                                          linearOperator->getMemoryLocation() );
    auto RightHandSideVec = AMP::LinearAlgebra::createVector( nodalDofMap,
                                                              linearOperator->getOutputVariable(),
                                                              true,
                                                              linearOperator->getMemoryLocation() );

    auto testOperator =
        AMP::Operator::createOperator( meshAdapter, "DiffusionLinearFEOperator", input_db );

    const int N = 100;

    for ( auto i = 0; i < N; ++i ) {
        PROFILE( "DiffusionOperator::apply" );
        linearOperator->apply( TemperatureInKelvinVec, RightHandSideVec );
    }
    for ( auto i = 0; i < N; ++i ) {
        PROFILE( "TestOperator::apply" );
        testOperator->apply( TemperatureInKelvinVec, RightHandSideVec );
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files;

    PROFILE_ENABLE();

    if ( argc > 1 ) {

        files.emplace_back( argv[1] );

    } else {
        files.emplace_back( "input_testLinearThermalRobinOperatorPerformance" );
    }

    {
        PROFILE( "DRIVER::main(test loop)" );
        for ( auto &file : files ) {
            linearThermalTest( &ut, file );
        }
    }

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testLinearThermalRobinOperatorPerformance_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize();

    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
