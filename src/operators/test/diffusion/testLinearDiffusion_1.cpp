#include "AMP/ampmesh/Mesh.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/diffusion/DiffusionConstants.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperatorParameters.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
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

#include <iostream>
#include <memory>
#include <string>


static void linearTest1( AMP::UnitTest *ut, const std::string &exeName )
{
    // this tests creation from database and usage

    // Test create
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto meshAdapter = AMP::Mesh::Mesh::buildMesh( params );


    auto diffLinFEOp_db =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "LinearDiffusionOp" ) );
    auto diffOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "LinearDiffusionOp", input_db ) );
    auto elementModel = diffOp->getTransportModel();

    auto diffSolVar = diffOp->getInputVariable();
    auto diffRhsVar = diffOp->getOutputVariable();
    auto diffResVar = diffOp->getOutputVariable();

    auto NodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Vertex, 1, 1, true );

    auto diffSolVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, diffSolVar, true );
    auto diffRhsVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, diffRhsVar, true );
    auto diffResVec = AMP::LinearAlgebra::createVector( NodalScalarDOF, diffResVar, true );

    ut->passes( exeName );

    // Test apply
    for ( int i = 0; i < 10; i++ ) {
        diffSolVec->setRandomValues();
        diffRhsVec->setRandomValues();
        diffResVec->setRandomValues();
        diffOp->residual( diffRhsVec, diffSolVec, diffResVec );
    } // end for i

    ut->passes( exeName );

    // Test reset
    auto diffOpParams =
        std::make_shared<AMP::Operator::DiffusionLinearFEOperatorParameters>( diffLinFEOp_db );
    diffOpParams->d_transportModel =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( elementModel );
    diffOp->reset( diffOpParams );

    ut->passes( exeName );

    // Test eigenvalues (run output through mathematica)
    auto diffMat  = diffOp->getMatrix();
    int nranks    = globalComm.getSize();
    size_t matdim = 24;
    if ( nranks == 1 ) {
        std::cout << "cols={" << std::endl;
        for ( size_t i = 0; i < matdim; i++ ) {
            std::vector<size_t> matCols;
            std::vector<double> matVals;
            diffMat->getRowByGlobalID( i, matCols, matVals );
            std::cout << "{";
            for ( size_t j = 0; j < matCols.size(); j++ ) {
                std::cout << matCols[j];
                if ( j < matCols.size() - 1 )
                    std::cout << ",";
            }
            std::cout << "}";
            if ( i < matdim - 1 )
                std::cout << ",";
            std::cout << std::endl;
        }
        std::cout << "};" << std::endl;

        std::cout << "matrix = {" << std::endl;

        for ( size_t i = 0; i < matdim; i++ ) {
            std::vector<size_t> matCols;
            std::vector<double> matVals;
            diffMat->getRowByGlobalID( i, matCols, matVals );
            std::cout << "{";
            size_t col = 0;
            for ( size_t j = 0; j < matCols.size(); j++ ) {
                while ( col < matCols[j] ) {
                    std::cout << "0.";
                    std::cout << ",";
                    col++;
                }
                std::cout << matVals[j];
                if ( matCols[j] < matdim - 1 )
                    std::cout << ",";
                col++;
            } // end for j
            while ( col < matdim ) {
                std::cout << "0";
                if ( col < matdim - 1 )
                    std::cout << ",";
                col++;
            }
            std::cout << "}";
            if ( i < matdim - 1 )
                std::cout << "," << std::endl;
        } // end for i

        std::cout << "};" << std::endl;
    }

    ut->passes( exeName );
}


int testLinearDiffusion_1( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    const int NUMFILES          = 8;
    std::string files[NUMFILES] = { "Diffusion-TUI-Thermal-1",     "Diffusion-TUI-Fick-1",
                                    "Diffusion-TUI-Soret-1",       "Diffusion-UO2MSRZC09-Thermal-1",
                                    "Diffusion-UO2MSRZC09-Fick-1", "Diffusion-UO2MSRZC09-Soret-1",
                                    "Diffusion-TUI-TensorFick-1",  "Diffusion-CylindricalFick-1" };

    for ( auto &file : files )
        linearTest1( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}