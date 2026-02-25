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
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearElement.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/libmesh/MassDensityModel.h"
#include "AMP/operators/libmesh/MassLinearElement.h"
#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/ManufacturedSolution.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include "../applyTests.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
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
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    //   Create the Mesh.
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create nonlinear diffusion BVP operator and access volume nonlinear Diffusion operator
    auto nlinBVPOperator = AMP::Operator::OperatorBuilder::createOperator(
        mesh, "ThermalNonlinearBVPOperator", input_db );
    auto nlinBVPOp =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>( nlinBVPOperator );
    auto nlinOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nlinBVPOp->getVolumeOperator() );

    // Get source mass operator
    auto sourceOperator = AMP::Operator::OperatorBuilder::createOperator(
        mesh, "ManufacturedSourceOperator", input_db );
    auto sourceOp =
        std::dynamic_pointer_cast<AMP::Operator::MassLinearFEOperator>( sourceOperator );

    auto densityModel = sourceOp->getDensityModel();
    auto mfgSolution  = densityModel->getManufacturedSolution();

    // Set up input and output vectors
    // auto solVar = nlinOp->getInputVariable(nlinOp->getPrincipalVariableId());
    auto solVar    = nlinOp->getOutputVariable();
    auto rhsVar    = nlinOp->getOutputVariable();
    auto resVar    = nlinOp->getOutputVariable();
    auto sourceVar = sourceOp->getOutputVariable();
    auto workVar   = sourceOp->getOutputVariable();

    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // create solution, rhs, and residual vectors
    auto solVec    = AMP::LinearAlgebra::createVector( nodalDofMap, solVar );
    auto rhsVec    = AMP::LinearAlgebra::createVector( nodalDofMap, rhsVar );
    auto resVec    = AMP::LinearAlgebra::createVector( nodalDofMap, resVar );
    auto sourceVec = AMP::LinearAlgebra::createVector( nodalDofMap, sourceVar );
    auto workVec   = AMP::LinearAlgebra::createVector( nodalDofMap, workVar );

    rhsVec->setToScalar( 0.0 );

    // Fill in manufactured solution
    auto iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    for ( ; iterator != iterator.end(); ++iterator ) {
        double x  = ( iterator->coord() )[0];
        double y  = ( iterator->coord() )[1];
        double z  = ( iterator->coord() )[2];
        auto poly = mfgSolution->evaluate( x, y, z );
        std::vector<size_t> i;
        nodalDofMap->getDOFs( iterator->globalID(), i );
        solVec->setValuesByGlobalID( 1, &i[0], &poly[0] );
    }
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Evaluate manufactured solution as an FE source
    sourceOp->apply( solVec, sourceVec );

    // Evaluate action of diffusion operator
    nlinBVPOp->residual( sourceVec, solVec, resVec );
    resVec->scale( -1.0 );

    // Output Mathematica form (requires serial execution)
    for ( int i = 0; i < globalComm.getSize(); i++ ) {
        if ( globalComm.getRank() == i ) {
            std::string filename          = "data_" + exeName;
            int rank                      = globalComm.getRank();
            int nranks                    = globalComm.getSize();
            std::ios_base::openmode omode = std::ios_base::out;
            if ( rank > 0 )
                omode |= std::ios_base::app;
            std::ofstream file( filename.c_str(), omode );
            if ( rank == 0 ) {
                file << "(* x y z solution solution fe-source fe-operator error *)" << std::endl;
                file << "results={" << std::endl;
            }

            size_t numNodes = iterator.size();
            size_t iNode    = 0;
            double l2err    = 0.;
            auto myIterator = iterator.begin();
            for ( ; myIterator != iterator.end(); ++myIterator ) {
                double x = ( myIterator->coord() )[0];
                double y = ( myIterator->coord() )[1];
                double z = ( myIterator->coord() )[2];
                std::vector<size_t> gid;
                nodalDofMap->getDOFs( myIterator->globalID(), gid );
                double val, res, sol, src, err;
                res       = resVec->getValueByGlobalID( gid[0] );
                sol       = solVec->getValueByGlobalID( gid[0] );
                src       = sourceVec->getValueByGlobalID( gid[0] );
                err       = res / ( src + .5 * res + std::numeric_limits<double>::epsilon() );
                auto poly = mfgSolution->evaluate( x, y, z );
                val       = poly[0];
                workVec->setValuesByGlobalID( 1, &gid[0], &err );

                file << "{" << x << "," << y << "," << z << "," << val << "," << sol << "," << src
                     << "," << res + src << "," << err << "}";
                if ( iNode < numNodes - 1 )
                    file << "," << std::endl;

                l2err += ( res * res );
                iNode++;
            }

            if ( rank == nranks - 1 ) {
                file << "};" << std::endl;
                file << "nodes = " << numNodes << "; l2err = " << l2err << ";" << std::endl;
            }

            file.close();
        }
        globalComm.barrier();
    }

    ut->passes( exeName );
    std::cout.flush();
}

int testDiffusionManufacturedSolution_1( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::string> files;
    files.emplace_back( "Diffusion-Fick-TUI-MMS-1" );
    // files.push_back("Diffusion-Fick-OxMSRZC09-MMS-1");

    for ( auto &file : files )
        bvpTest1( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
