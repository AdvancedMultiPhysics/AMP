#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );


    AMP::AMP_MPI globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );
    auto input_db           = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database   = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( database );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto manager = AMP::Mesh::MeshFactory::create( meshParams );
    auto mesh    = manager->Subset( "cylinder" );

    AMP::pout << "Constructing Nonlinear Thermal Operator..." << std::endl;

    // create a nonlinear BVP operator for nonlinear thermal diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearThermalOperator" ), "key missing!" );

    auto nonlinearThermalOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testNonlinearThermalOperator", input_db ) );

    // initialize the input variable
    auto thermalVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator->getVolumeOperator() );

    auto thermalVariable = thermalVolumeOperator->getOutputVariable();

    // create solution, rhs, and residual vectors
    auto NodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto solVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, thermalVariable, true );
    auto rhsVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, thermalVariable, true );
    auto resVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, thermalVariable, true );

    // create the following shared pointers for ease of use
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    AMP::pout << "Constructing Linear Thermal Operator..." << std::endl;

    // now construct the linear BVP operator for thermal
    AMP_INSIST( input_db->keyExists( "testLinearThermalOperator" ), "key missing!" );
    auto linearThermalOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testLinearThermalOperator", input_db ) );

    // CREATE THE NEUTRONICS SOURCE
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db  = input_db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams = std::make_shared<AMP::Operator::OperatorParameters>( neutronicsOp_db );
    neutronicsParams->d_Mesh = mesh;
    auto neutronicsOperator  = std::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );

    // Create a DOF manager for a gauss point vector
    int DOFsPerNode    = 8;
    int ghostWidth     = 1;
    bool split         = true;
    auto gauss_dof_map = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, ghostWidth, DOFsPerNode, split );

    auto SpecificPowerVar = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec =
        AMP::LinearAlgebra::createVector( gauss_dof_map, SpecificPowerVar, split );

    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    // Integrate Nuclear Rhs over Desnity * GeomType::Cell
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "VolumeIntegralOperator", input_db ) );

    // Create the power (heat source) vector.
    auto PowerInWattsVar = sourceOperator->getOutputVariable();
    auto PowerInWattsVec =
        AMP::LinearAlgebra::createVector( NodalScalarDOF, PowerInWattsVar, true );
    PowerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, PowerInWattsVec );

    rhsVec->copyVector( PowerInWattsVec );

    AMP::pout << "RHS L2 norm before corrections = " << ( rhsVec->L2Norm() ) << "\n";
    AMP::pout << "RHS max before corrections = " << ( rhsVec->max() ) << "\n";
    AMP::pout << "RHS min before corrections = " << ( rhsVec->min() ) << "\n";

    nonlinearThermalOperator->modifyRHSvector( rhsVec );

    AMP::pout << "RHS L2 norm after corrections = " << ( rhsVec->L2Norm() ) << "\n";
    AMP::pout << "RHS max after corrections = " << ( rhsVec->max() ) << "\n";
    AMP::pout << "RHS min after corrections = " << ( rhsVec->min() ) << "\n";

    //---------------------------------------------------------------------------------------------//
    // Initial guess

    auto initGuess = input_db->getWithDefault<double>( "InitialGuess", 400.0 );
    solVec->setToScalar( initGuess );

    AMP::pout << "initial guess L2 norm before corrections = " << ( solVec->L2Norm() ) << "\n";
    AMP::pout << "initial guess max before corrections = " << ( solVec->max() ) << "\n";
    AMP::pout << "initial guess min before corrections = " << ( solVec->min() ) << "\n";

    nonlinearThermalOperator->modifyInitialSolutionVector( solVec );

    AMP::pout << "initial guess L2 norm after corrections = " << ( solVec->L2Norm() ) << "\n";
    AMP::pout << "initial guess max after corrections = " << ( solVec->max() ) << "\n";
    AMP::pout << "initial guess min after corrections = " << ( solVec->min() ) << "\n";

    //---------------------------------------------------------------------------------------------/

    nonlinearThermalOperator->modifyInitialSolutionVector( solVec );
    linearThermalOperator->reset( nonlinearThermalOperator->getParameters( "Jacobian", solVec ) );

    AMP::pout << "Finished reseting the jacobian." << std::endl;

    nonlinearThermalOperator->residual( rhsVec, solVec, resVec );

    double initialResidualNorm = static_cast<double>( resVec->L2Norm() );
    AMP::pout << "Initial Residual Norm: " << initialResidualNorm << std::endl;

    if ( initialResidualNorm > 1.0e-08 ) {
        ut->failure( "Nonlinear Diffusion Operator with stand alone Robin BC " );
    } else {
        ut->passes( "Nonlinear Diffusion Operator with stand alone Robin BC " );
    }
    ut->passes( exeName );
}

int testNonlinearRobin( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testNonlinearRobin-1" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
