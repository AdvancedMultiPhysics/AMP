#include "AMP/ampmesh/Mesh.h"
#include "AMP/ampmesh/libmesh/libmeshMesh.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/IsotropicElasticModel.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include <memory>

#include "libmesh/mesh_generation.h"

#include <iostream>
#include <string>


static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testLinearMechanics-eigenValues-1" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    int npes = globalComm.getSize();

    if ( npes == 1 ) {

        auto input_db = AMP::Database::parseInputFile( input_file );
        input_db->print( AMP::plog );

        AMP_INSIST( input_db->keyExists( "OutputFileName" ), "Key ''OutputFileName'' is missing!" );
        std::string outFileName = input_db->getString( "OutputFileName" );

        FILE *fp;
        fp = fopen( outFileName.c_str(), "w" );
        fprintf( fp, "clc; \n clear; \n A = zeros(24, 24); \n \n" );

        AMP_INSIST( input_db->keyExists( "DISTORT_ELEMENT" ),
                    "Key ''DISTORT_ELEMENT'' is missing!" );
        bool distortElement = input_db->getScalar<bool>( "DISTORT_ELEMENT" );

        libMesh::Parallel::Communicator comm( globalComm.getCommunicator() );
        auto mesh = std::make_shared<libMesh::Mesh>( comm, 3 );
        libMesh::MeshTools::Generation::build_cube(
            ( *( mesh.get() ) ), 1, 1, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, libMesh::HEX8, false );

        if ( distortElement ) {
            libMesh::Elem *elemPtr = mesh->elem( 0 );

            ( elemPtr->point( 0 ) )( 0 ) -= 0.1;
            ( elemPtr->point( 0 ) )( 1 ) -= 0.2;
            ( elemPtr->point( 0 ) )( 2 ) -= 0.3;

            ( elemPtr->point( 3 ) )( 0 ) -= 0.2;
            ( elemPtr->point( 3 ) )( 1 ) += 0.2;
            ( elemPtr->point( 3 ) )( 2 ) -= 0.1;

            ( elemPtr->point( 6 ) )( 0 ) += 0.3;
            ( elemPtr->point( 6 ) )( 1 ) += 0.2;
            ( elemPtr->point( 6 ) )( 2 ) += 0.1;
        }

        auto meshAdapter = std::make_shared<AMP::Mesh::libmeshMesh>( mesh, "TestMesh" );

        AMP_INSIST( input_db->keyExists( "Isotropic_Model" ),
                    "Key ''Isotropic_Model'' is missing!" );
        auto matModel_db = input_db->getDatabase( "Isotropic_Model" );
        auto matModelParams =
            std::make_shared<AMP::Operator::MechanicsMaterialModelParameters>( matModel_db );
        auto isotropicModel =
            std::make_shared<AMP::Operator::IsotropicElasticModel>( matModelParams );

        AMP_INSIST( input_db->keyExists( "Mechanics_Linear_Element" ),
                    "Key ''Mechanics_Linear_Element'' is missing!" );
        auto elemOp_db = input_db->getDatabase( "Mechanics_Linear_Element" );
        auto elemOpParams =
            std::make_shared<AMP::Operator::ElementOperationParameters>( elemOp_db );
        auto mechLinElem = std::make_shared<AMP::Operator::MechanicsLinearElement>( elemOpParams );

        std::shared_ptr<AMP::Discretization::DOFManager> dofMap =
            AMP::Discretization::simpleDOFManager::create(
                meshAdapter, AMP::Mesh::GeomType::Vertex, 1, 3, true );

        AMP_INSIST( input_db->keyExists( "Mechanics_Assembly" ),
                    "Key ''Mechanics_Assembly'' is missing!" );
        auto mechAssembly_db = input_db->getDatabase( "Mechanics_Assembly" );
        auto mechOpParams =
            std::make_shared<AMP::Operator::MechanicsLinearFEOperatorParameters>( mechAssembly_db );
        mechOpParams->d_materialModel = isotropicModel;
        mechOpParams->d_elemOp        = mechLinElem;
        mechOpParams->d_Mesh          = meshAdapter;
        mechOpParams->d_inDofMap      = dofMap;
        mechOpParams->d_outDofMap     = dofMap;
        auto mechOp = std::make_shared<AMP::Operator::MechanicsLinearFEOperator>( mechOpParams );

        auto mechMat = mechOp->getMatrix();

        for ( int i = 0; i < 24; ++i ) {
            std::vector<size_t> matCols;
            std::vector<double> matVals;
            mechMat->getRowByGlobalID( i, matCols, matVals );
            for ( size_t j = 0; j < matCols.size(); j++ ) {
                fprintf(
                    fp, "A(%d, %d) = %.15f ; \n", ( i + 1 ), (int) ( matCols[j] + 1 ), matVals[j] );
            } // end for j
            fprintf( fp, "\n" );
        } // end for i

        auto nd     = meshAdapter->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
        auto end_nd = nd.end();

        for ( int i = 0; nd != end_nd; ++nd, ++i ) {
            auto pt = nd->coord();
            fprintf( fp, "nd = %d, x = %.15f, y = %.15f, z = %.15f \n", i, pt[0], pt[1], pt[2] );
            std::vector<size_t> globalIds;
            dofMap->getDOFs( nd->globalID(), globalIds );
            fprintf( fp,
                     "nd = %d, d0 = %d, d1 = %d, d2 = %d \n",
                     i,
                     (int) globalIds[0],
                     (int) globalIds[1],
                     (int) globalIds[2] );
        }

        fprintf( fp, "format long e; \n\n" );
        fprintf( fp, "sort(eig(A)) \n\n" );
        fclose( fp );
    }

    ut->passes( exeName );
}

int testLinearMechanics_eigenValues( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    auto libmeshInit = std::make_shared<AMP::Mesh::initializeLibMesh>( AMP_COMM_WORLD );

    AMP::UnitTest ut;
    myTest( &ut );
    ut.report();
    int num_failed = ut.NumFailGlobal();

    libmeshInit.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
