#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
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
#include "AMP/vectors/VectorBuilder.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>


/**
 * This test is patch test for the diffusion operator.
 */
static void nonlinearTest( AMP::UnitTest *ut,
                           const std::string &exeName,
                           std::function<double( double, double, double )> fun )
{
    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    std::cout << "testing with input file " << input_file << std::endl;
    std::cout.flush();

    // Test create
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    auto diffFEOp_db = input_db->getDatabase( "NonlinearDiffusionOp" );
    auto nonlinearOperator =
        AMP::Operator::OperatorBuilder::createOperator( mesh, "NonlinearDiffusionOp", input_db );
    auto diffOp =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>( nonlinearOperator );

    ut->passes( exeName + ": create" );
    std::cout.flush();

    // set up defaults for materials arguments and create transport model
    auto transportModel_db = input_db->getDatabase( "DiffusionTransportModel" );
    auto elementPhysicsModel =
        AMP::Operator::ElementPhysicsModelFactory::createElementPhysicsModel( transportModel_db );
    auto transportModel =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( elementPhysicsModel );

    auto defTemp = transportModel_db->getWithDefault<double>( "Default_Temperature", 400.0 );
    auto defConc = transportModel_db->getWithDefault<double>( "Default_Concentration", .33 );
    auto defBurn = transportModel_db->getWithDefault<double>( "Default_Burnup", .5 );

    // create parameters
    auto diffOpParams =
        std::make_shared<AMP::Operator::DiffusionNonlinearFEOperatorParameters>( diffFEOp_db );

    // nullify vectors in parameters
    diffOpParams->d_FrozenVecs.clear();

    // create vectors for parameters
    auto active_db = diffFEOp_db->getDatabase( "ActiveInputVariables" );
    auto tVar      = std::make_shared<AMP::LinearAlgebra::Variable>(
        active_db->getWithDefault<std::string>( "Temperature", "not_specified" ) );
    auto cVar = std::make_shared<AMP::LinearAlgebra::Variable>(
        active_db->getWithDefault<std::string>( "Concentration", "not_specified" ) );
    auto bVar = std::make_shared<AMP::LinearAlgebra::Variable>(
        active_db->getWithDefault<std::string>( "Burnup", "not_specified" ) );

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

    // set principal variable vector
    if ( diffOp->getPrincipalVariable() == "temperature" )
        diffOpParams->d_FrozenVecs["temperature"] = tVec;
    if ( diffOp->getPrincipalVariable() == "concentration" )
        diffOpParams->d_FrozenVecs["concentration"] = cVec;
    if ( diffOp->getPrincipalVariable() == "burnup" )
        diffOpParams->d_FrozenVecs["burnup"] = bVec;

    // set frozen vectors in parameters
    if ( diffFEOp_db->getWithDefault<bool>( "FreezeTemperature", false ) )
        diffOpParams->d_FrozenVecs["temperature"] = tVec;
    if ( diffFEOp_db->getWithDefault<bool>( "FreezeConcentration", false ) )
        diffOpParams->d_FrozenVecs["concentration"] = cVec;
    if ( diffFEOp_db->getWithDefault<bool>( "FreezeBurnup", false ) )
        diffOpParams->d_FrozenVecs["burnup"] = bVec;

    // set transport model
    diffOpParams->d_transportModel = transportModel;

    // Reset with new parameters
    diffOp->reset( diffOpParams );

    // set  up variables for apply tests
    // auto diffOp->getInputVariable(diffOp->getPrincipalVariableId());
    auto diffSolVar = diffOp->getOutputVariable();

    auto diffRhsVar  = diffOp->getOutputVariable();
    auto diffResVar  = diffOp->getOutputVariable();
    auto nonPrincIds = diffOp->getNonPrincipalVariableIds();
    std::vector<std::shared_ptr<AMP::LinearAlgebra::Variable>> nonPrincVars( nonPrincIds.size() );
    auto inputVar = diffOp->getInputVariable();
    for ( size_t i = 0; i < nonPrincIds.size(); i++ )
        nonPrincVars[i] = std::make_shared<AMP::LinearAlgebra::Variable>( nonPrincIds[i] );

    // set up vectors for apply tests
    // std::string msgPrefix=exeName+": apply";
    auto diffSolVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffSolVar );
    auto diffRhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffRhsVar );
    auto diffResVec = AMP::LinearAlgebra::createVector( nodalDofMap, diffResVar );

    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> nonPrincVecs( nonPrincIds.size() );
    for ( unsigned int i = 0; i < nonPrincIds.size(); i++ ) {
        nonPrincVecs[i] = AMP::LinearAlgebra::createVector( nodalDofMap, nonPrincVars[i] );
        if ( nonPrincIds[i] == "temperature" )
            nonPrincVecs[i]->setToScalar( defTemp );
        if ( nonPrincIds[i] == "concentration" )
            nonPrincVecs[i]->setToScalar( defConc );
        if ( nonPrincIds[i] == "burnup" )
            nonPrincVecs[i]->setToScalar( defBurn );
    }
    diffRhsVec->setToScalar( 0.0 );

    auto curNode = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto endNode = curNode.end();

    for ( curNode = curNode.begin(); curNode != endNode; ++curNode ) {
        // double x = curNode->x();
        double x = curNode->coord()[0];
        double y = curNode->coord()[1];
        double z = curNode->coord()[2];
        std::vector<size_t> dofs;
        nodalDofMap->getDOFs( curNode->globalID(), dofs );
        double fval = fun( x, y, z );
        diffSolVec->setValuesByGlobalID( 1, &dofs[0], &fval );
    }
    diffSolVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Compute finite element operator
    diffOp->apply( diffSolVec, diffResVec );

    // Check that interior values are zero.
    double totalBnd = 0.;
    for ( size_t face = 1; face < 64; face++ ) {
        for ( curNode = curNode.begin(); curNode != endNode; ++curNode ) {
            std::vector<size_t> dofs;
            nodalDofMap->getDOFs( curNode->globalID(), dofs );
            double fval = diffResVec->getValueByGlobalID( dofs[0] );
            totalBnd += fabs( fval );
        }
    }
    double totalAll = 0.;
    for ( curNode = curNode.begin(); curNode != endNode; ++curNode ) {
        std::vector<size_t> dofs;
        nodalDofMap->getDOFs( curNode->globalID(), dofs );
        double fval = diffResVec->getValueByGlobalID( dofs[0] );
        totalAll += fabs( fval );
    }
    totalBnd = globalComm.sumReduce( totalBnd );
    totalAll = globalComm.sumReduce( totalAll );
    int rank = globalComm.getRank();
    if ( rank == 0 ) {
        std::cout << "******** All = " << totalAll << ", Bnd = " << totalBnd
                  << ", Err = " << totalAll - totalBnd << "********" << std::endl;
    }

    ut->passes( "values-" + exeName );
}

int testNonlinearDiffusion_2( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    const int NUMFILES          = 1;
    std::string files[NUMFILES] = { "Diffusion-TUI-Thermal-1" };

    auto x_linear = []( double x, double, double ) { return x; };
    for ( auto &file : files )
        nonlinearTest( &ut, file, x_linear );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
