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
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>


/**
 * This test is designed to allow the programmer to set up a function on a mesh and compute the
 * finite
 * element discretization to check correctness of the operator discretization.
 */
static void linearTest( AMP::UnitTest *ut,
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

    auto diffFEOp_db = input_db->getDatabase( "LinearDiffusionOp" );
    auto linearOperator =
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearDiffusionOp", input_db );
    auto diffOp =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>( linearOperator );

    // create parameters
    auto diffOpParams =
        std::make_shared<AMP::Operator::DiffusionLinearFEOperatorParameters>( diffFEOp_db );

    // set up defaults for materials arguments and create transport model
    auto transportModel_db = input_db->getDatabase( "DiffusionTransportModel" );
    auto defTemp = transportModel_db->getWithDefault<double>( "Default_Temperature", 400.0 );
    auto defConc = transportModel_db->getWithDefault<double>( "Default_Concentration", .33 );
    auto defBurn = transportModel_db->getWithDefault<double>( "Default_Burnup", .5 );
    auto elementPhysicsModel =
        AMP::Operator::ElementPhysicsModelFactory::createElementPhysicsModel( transportModel_db );
    auto transportModel =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( elementPhysicsModel );

    // create vectors for parameters
    auto NodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto tempVar = std::make_shared<AMP::LinearAlgebra::Variable>( "testTempVar" );
    auto concVar = std::make_shared<AMP::LinearAlgebra::Variable>( "testConcVar" );
    auto burnVar = std::make_shared<AMP::LinearAlgebra::Variable>( "testBurnVar" );
    std::shared_ptr<AMP::LinearAlgebra::Vector> tempVec, concVec, burnVec;
    if ( !diffFEOp_db->getWithDefault<bool>( "FixedTemperature", false ) ) {
        tempVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, tempVar, true );
        tempVec->setToScalar( defTemp );
    }
    if ( !diffFEOp_db->getWithDefault<bool>( "FixedConcentration", false ) ) {
        concVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, concVar, true );
        concVec->setToScalar( defConc );
    }
    if ( !diffFEOp_db->getWithDefault<bool>( "FixedBurnup", false ) ) {
        burnVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, burnVar, true );
        burnVec->setToScalar( defBurn );
    }
    diffOpParams->d_transportModel             = transportModel;
    diffOpParams->d_inputVecs["temperature"]   = tempVec;
    diffOpParams->d_inputVecs["concentration"] = concVec;
    diffOpParams->d_inputVecs["burnup"]        = burnVec;

    // reset with parameters
    diffOp->reset( diffOpParams );

    // set  up variables for apply
    auto diffSolVar = diffOp->getInputVariable();
    auto diffRhsVar = diffOp->getOutputVariable();
    auto diffResVar = diffOp->getOutputVariable();

    // set up vectors for apply tests
    auto diffSolVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, diffSolVar, true );
    auto diffRhsVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, diffRhsVar, true );
    auto diffResVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, diffResVar, true );
    diffRhsVec->setToScalar( 0.0 );

    auto curNode = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto endNode = curNode.end();
    std::vector<size_t> dofs;
    while ( curNode != endNode ) {
        auto pos = curNode->coord();
        double x = pos[0];
        double y = pos[1];
        double z = pos[2];
        NodalScalarDOF->getDOFs( curNode->globalID(), dofs );
        size_t i    = dofs[0];
        double fval = fun( x, y, z );
        diffSolVec->setValuesByGlobalID( 1, &i, &fval );
        ++curNode;
    }
    diffSolVec->makeConsistent();

    // Compute finite element operator
    diffOp->apply( diffSolVec, diffResVec );

    // write values in mathematica form
    int nranks = globalComm.getSize();
    if ( nranks == 1 ) {
        size_t nnodes        = mesh->numLocalElements( AMP::Mesh::GeomType::Vertex );
        int proc             = globalComm.getRank();
        int nproc            = globalComm.getSize();
        std::string filename = "values-" + exeName;
        std::ofstream file( filename.c_str() );
        if ( proc == 0 ) {
            file << "values={"
                 << "\n";
        }
        curNode = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
        for ( size_t i = 0; i < nnodes; i++ ) {
            auto pos = curNode->coord();
            double x = pos[0];
            double y = pos[1];
            double z = pos[2];

            int ii      = i;
            double rval = diffResVec->getValueByLocalID( ii );
            double fval = fun( x, y, z );
            file << "{" << x << "," << y << "," << z << "," << rval << "," << fval << "}";
            if ( i < nnodes - 1 )
                file << ",\n";
            ++curNode;
        }
        if ( proc < nproc - 1 ) {
            file << ",\n";
        } else {
            file << "}\n";
        }
    }

    ut->passes( "values-" + exeName );
}

int testLinearDiffusion_2( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    const int NUMFILES          = 1;
    std::string files[NUMFILES] = {
        "Diffusion-TUI-Thermal-1" /*, "Diffusion-TUI-Fick-1", "Diffusion-TUI-Soret-1",
      "Diffusion-UO2MSRZC09-Thermal-1", "Diffusion-UO2MSRZC09-Fick-1",
      "Diffusion-UO2MSRZC09-Soret-1" */
    };

    auto x_linear = []( double x, double, double ) { return x; };
    for ( auto &file : files )
        linearTest( &ut, file, x_linear );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
