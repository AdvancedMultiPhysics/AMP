#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"

#include <iostream>
#include <string>

#include "utils/shared_ptr.h"

#include "utils/AMPManager.h"
#include "utils/AMP_MPI.h"
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/PIO.h"

#include "ampmesh/Mesh.h"

#include "operators/OperatorBuilder.h"
#include "operators/OperatorParameters.h"
#include "operators/diffusion/DiffusionConstants.h"
#include "operators/diffusion/DiffusionLinearElement.h"
#include "operators/diffusion/DiffusionLinearFEOperator.h"
#include "operators/diffusion/DiffusionLinearFEOperatorParameters.h"
#include "operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "operators/diffusion/DiffusionTransportModel.h"

#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Variable.h"
#include "vectors/Vector.h"
#include "vectors/VectorBuilder.h"


void linearTest1( AMP::UnitTest *ut, std::string exeName )
{
    // this tests creation from database and usage

    // Test create
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    AMP::shared_ptr<AMP::InputDatabase> input_db( new AMP::InputDatabase( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile( input_file, input_db );
    input_db->printClassData( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    AMP::shared_ptr<AMP::Database> database = input_db->getDatabase( "Mesh" );
    AMP::shared_ptr<AMP::Mesh::MeshParameters> params( new AMP::Mesh::MeshParameters( database ) );
    params->setComm( globalComm );

    // Create the meshes from the input database
    AMP::shared_ptr<AMP::Mesh::Mesh> meshAdapter = AMP::Mesh::Mesh::buildMesh( params );


    AMP::shared_ptr<AMP::Operator::DiffusionLinearFEOperator> diffOp;
    AMP::shared_ptr<AMP::InputDatabase> diffLinFEOp_db =
        AMP::dynamic_pointer_cast<AMP::InputDatabase>(
            input_db->getDatabase( "LinearDiffusionOp" ) );

    diffOp = AMP::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "LinearDiffusionOp", input_db ) );
    AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> elementModel = diffOp->getTransportModel();

    AMP::LinearAlgebra::Variable::shared_ptr diffSolVar = diffOp->getInputVariable();
    AMP::LinearAlgebra::Variable::shared_ptr diffRhsVar = diffOp->getOutputVariable();
    AMP::LinearAlgebra::Variable::shared_ptr diffResVar = diffOp->getOutputVariable();

    AMP::Discretization::DOFManager::shared_ptr NodalScalarDOF =
        AMP::Discretization::simpleDOFManager::create( meshAdapter, AMP::Mesh::Vertex, 1, 1, true );

    AMP::LinearAlgebra::Vector::shared_ptr diffSolVec =
        AMP::LinearAlgebra::createVector( NodalScalarDOF, diffSolVar, true );
    AMP::LinearAlgebra::Vector::shared_ptr diffRhsVec =
        AMP::LinearAlgebra::createVector( NodalScalarDOF, diffRhsVar, true );
    AMP::LinearAlgebra::Vector::shared_ptr diffResVec =
        AMP::LinearAlgebra::createVector( NodalScalarDOF, diffResVar, true );


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
    AMP::shared_ptr<AMP::Operator::DiffusionLinearFEOperatorParameters> diffOpParams(
        new AMP::Operator::DiffusionLinearFEOperatorParameters( diffLinFEOp_db ) );
    diffOpParams->d_transportModel =
        AMP::dynamic_pointer_cast<AMP::Operator::DiffusionTransportModel>( elementModel );
    diffOp->reset( diffOpParams );

    ut->passes( exeName );

    // Test eigenvalues (run output through mathematica)
    AMP::shared_ptr<AMP::LinearAlgebra::Matrix> diffMat = diffOp->getMatrix();

    int nranks = globalComm.getSize();

    size_t matdim = 24;
    if ( nranks == 1 ) {
        std::cout << "cols={" << std::endl;
        for ( size_t i = 0; i < matdim; i++ ) {
            std::vector<unsigned int> matCols;
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
            std::vector<unsigned int> matCols;
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

// void linearTestReset(AMP::UnitTest *ut, std::string exeName)
// {
//   // this tests creation from database and usage
//
//   // Test create
//   std::string input_file = "input_" + exeName;
//   std::string log_file = "output_" + exeName;
//
//   AMP::PIO::logOnlyNodeZero(log_file);
//
//   AMP::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
//   AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
//   input_db->printClassData(AMP::plog);
//
//   AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
//   std::string mesh_file = input_db->getString("Mesh");
//
//   //create a mesh adaptor
//   AMP::MeshManager::Adapter::shared_ptr meshAdapter =
//               AMP::MeshManager::Adapter::shared_ptr ( new AMP::MeshManager::Adapter () );
//   meshAdapter->readExodusIIFile ( mesh_file.c_str() );
//
//   // create the linear diffusion operator
//   AMP::shared_ptr<AMP::Operator::DiffusionLinearFEOperator> diffOp;
//   AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> elementModel;
//   AMP::shared_ptr<AMP::InputDatabase> diffLinFEOp_db =
//           AMP::dynamic_pointer_cast<AMP::InputDatabase>(input_db->getDatabase("LinearDiffusionOp"));
//   diffOp = AMP::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(
//                                        AMP::Operator::OperatorBuilder::createOperator(meshAdapter,
//                                        diffLinFEOp_db,
//                                        elementModel));
//
//   // first test: reset with a NULL parameter object
//   AMP::shared_ptr<OperatorParameters> resetParameters;
//   bool passed=false;
//   try
//     {
//       diffOp->reset(resetParameters);
//     }
//   catch(std::exceptions)
//     {
//       passed=true;
//     }
//   if(passed)
//     {
//       ut.passes(exeName+": DiffusionLinearFEOperator::reset with NULL parameter object");
//     }
//   else
//     {
//       ut.numFails++;
//     }
//
//   // second test: create a non null parameter object but dont initialize it fully  passed=false;
//   try
//     {
//       diffOp->reset(resetParameters);
//     }
//   catch(std::exceptions)
//     {
//       passed=true;
//     }
//   if(passed)
//     {
//       ut.passes(exeName+": DiffusionLinearFEOperator::reset with parameter object having no valid
//       physics and
//       operation objects");
//     }
//   else
//     {
//       ut.numFails++;
//     }
//
//
//   AMP::shared_ptr<AMP::Operator::DiffusionLinearFEOperatorParameters> diffusionOpParams(new
//   AMP::Operator::DiffusionLinearFEOperatorParameters( diffLinFEOp_db ));
//   resetParameters = AMP::dynamic_pointer_cast<OperatorParameters>(diffusionOpParams);
//   AMP_INSIST(resetParameters.get()!=NULL, "unable to create parameters");
//
//   passed=false;
//   try
//     {
//       diffOp->reset(resetParameters);
//     }
//   catch(std::exceptions)
//     {
//       passed=true;
//     }
//   if(passed)
//     {
//       ut.passes(exeName+": DiffusionLinearFEOperator::reset with parameter object having no valid
//       physics and
//       operation objects");
//     }
//   else
//     {
//       ut.numFails++;
//     }
//
//   // third test: use a parameter object without a element operation object
//   AMP::shared_ptr<AMP::Database> transportModel_db =
//   input_db->getDatabase("DiffusionTransportModel");
//   AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel =
//   ElementPhysicsModelFactory::createElementPhysicsModel(transportModel_db);
//   AMP::shared_ptr<AMP::Operator::DiffusionTransportModel> transportModel =
//   AMP::dynamic_pointer_cast<DiffusionTransportModel>(elementPhysicsModel);
//   AMP_INSIST(transportModel.get()!=NULL, "unable to create transport model");
//   diffusionOpParams->d_transportModel = transportModel;
//
//   passed=false;
//   try
//     {
//       diffOp->reset(resetParameters);
//     }
//   catch(std::exceptions)
//     {
//       passed=true;
//     }
//   if(passed)
//     {
//       ut.passes(exeName+": DiffusionLinearFEOperator::reset with parameter object having no valid
//       element operation
//       objects");
//     }
//   else
//     {
//       ut.numFails++;
//     }
//
//   // next create a ElementOperation object
//   AMP_INSIST(input_db->keyExists("DiffusionElement"), "Key ''DiffusionElement'' is missing!");
//   AMP::shared_ptr<AMP::Operator::ElementOperation> diffusionLinElem =
//   ElementOperationFactory::createElementOperation(input_db->getDatabase("DiffusionElement"));
//
// }

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    const int NUMFILES          = 8;
    std::string files[NUMFILES] = { "Diffusion-TUI-Thermal-1",     "Diffusion-TUI-Fick-1",
                                    "Diffusion-TUI-Soret-1",       "Diffusion-UO2MSRZC09-Thermal-1",
                                    "Diffusion-UO2MSRZC09-Fick-1", "Diffusion-UO2MSRZC09-Soret-1",
                                    "Diffusion-TUI-TensorFick-1",  "Diffusion-CylindricalFick-1" };

    try {
        for ( int i = 0; i < NUMFILES; i++ ) {
            linearTest1( &ut, files[i] );
        }
    } catch ( std::exception &err ) {
        std::cout << "ERROR: While testing " << argv[0] << err.what() << std::endl;
        ut.failure( "ERROR: While testing" );
    } catch ( ... ) {
        std::cout << "ERROR: While testing " << argv[0] << "An unknown exception was thrown."
                  << std::endl;
        ut.failure( "ERROR: While testing" );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
