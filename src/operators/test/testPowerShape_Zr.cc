#include "AMP/ampmesh/Mesh.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/libmesh/PowerShape.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/InputDatabase.h"
#include "AMP/utils/InputManager.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/Writer.h"
#include "AMP/utils/shared_ptr.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <string>


static void test_with_shape( AMP::UnitTest *ut )
{
    //  Read Input File
    auto input_db = AMP::make_shared<AMP::InputDatabase>( "input_db" );
    auto mesh_db  = input_db->putDatabase( "Mesh" );
    mesh_db->putString( "FileName", "cylinder270.e" );
    mesh_db->putString( "MeshType", "libMesh" );
    mesh_db->putString( "MeshName", "fuel" );
    mesh_db->putInteger( "dim", 3 );
    mesh_db->putDouble( "x_offset", 0. );
    mesh_db->putDouble( "y_offset", 0. );
    mesh_db->putDouble( "z_offset", 0. );

    //   Create the Mesh
    auto mgrParams = AMP::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto meshAdapter = AMP::Mesh::Mesh::buildMesh( mgrParams );

    //  Construct PowerShape for a radial only term.
    auto shape_db = input_db->putDatabase( "shape_db" );
    shape_db->putString( "coordinateSystem", "cylindrical" );
    shape_db->putString( "type", "zernikeRadial" );
    shape_db->putInteger( "print_info_level", 1 );

    // Create a DOF manager for a gauss point vector
    int DOFsPerNode = 8;
    int ghostWidth  = 0;
    bool split      = true;
    auto dof_map    = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Volume, ghostWidth, DOFsPerNode, split );

#ifdef USE_EXT_SILO
    auto shapeVar   = AMP::make_shared<AMP::LinearAlgebra::Variable>( "PowerShape" );
    auto shapeVec   = AMP::LinearAlgebra::createVector( dof_map, shapeVar, split );
    auto siloWriter = AMP::Utilities::Writer::buildWriter( "Silo" );
    siloWriter->registerMesh( meshAdapter );
    siloWriter->registerVector( shapeVec, meshAdapter, AMP::Mesh::GeomType::Volume, "PowerShape" );
#endif

    for ( int nMoments = 0; nMoments < 3; nMoments++ ) {
        shape_db->putInteger( "numMoments", nMoments );
        if ( nMoments > 0 ) {
            std::vector<double> moments( nMoments, 0. );
            moments[nMoments - 1] = -1.;
            shape_db->putDoubleArray( "Moments", moments );
        }
        auto shape_params    = AMP::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
        shape_params->d_Mesh = meshAdapter;
        auto shape           = AMP::make_shared<AMP::Operator::PowerShape>( shape_params );

        // Create a shared pointer to a Variable - Power - Output because it will be used in the
        // "residual" location of apply.
        auto SpecificPowerShapeVar =
            AMP::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerInWattsPerKg" );

        // Create a vector associated with the Variable.
        auto SpecificPowerShapeVec =
            AMP::LinearAlgebra::createVector( dof_map, SpecificPowerShapeVar, split );
        auto SpecificPowerMagnitudeVec = SpecificPowerShapeVec->cloneVector();
        SpecificPowerMagnitudeVec->setToScalar( 1.0 );

        // Set the initial value for all nodes of SpecificPowerVec to zero.
        SpecificPowerShapeVec->setToScalar( 0.0 );
        AMP::LinearAlgebra::Vector::shared_ptr nullVec;
        try {
            shape->apply( SpecificPowerMagnitudeVec, SpecificPowerShapeVec );
        } catch ( std::exception const &a ) {
            std::cout << a.what() << std::endl;
            ut->failure( "error" );
        }
        if ( nMoments == 0 ) {
            if ( !AMP::Utilities::approx_equal( SpecificPowerShapeVec->max(), 1.0, 1e-9 ) ) {
                ut->failure( "flat solution is not really flat (max)." );
                printf( "This %.9e is not 1.0. \n", SpecificPowerShapeVec->max() );
            }
            if ( !AMP::Utilities::approx_equal( SpecificPowerShapeVec->min(), 1.0, 1e-9 ) ) {
                ut->failure( "flat solution is not really flat (min)." );
                printf( "This %.9e is not 1.0. \n", SpecificPowerShapeVec->min() );
            }
        }

#ifdef USE_EXT_SILO
        shapeVec->copyVector( SpecificPowerShapeVec );
        // SpecificPowerShapeVec->copyVector(shapeVec);
        siloWriter->writeFile( "PowerShape-Zr", nMoments );
#endif

    } // end for elements

    ut->passes( "PowerShape produces a non-negative power shape." );

    //  Construct PowerShape for a full Zernike basis. -
    shape_db->putString( "type", "zernike" );
    int i = 0;
    for ( int nMoments = 0; nMoments < 9; nMoments++ ) {
        //    for ( int nMoments = 0; nMoments < 5; nMoments++ ) {
        for ( int n = -nMoments; n <= nMoments; i++ ) {
            shape_db->putInteger( "numMoments", nMoments );
            int nMNmoments = ( nMoments + 2 ) * ( nMoments + 1 ) / 2 - 1;
            if ( nMoments > 0 ) {
                std::vector<double> moments( nMNmoments, 0. );
                moments[i - 1] = -1.;
                shape_db->putDoubleArray( "Moments", moments );
            }
            auto shape_params = AMP::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
            shape_params->d_Mesh = meshAdapter;
            auto shape           = AMP::make_shared<AMP::Operator::PowerShape>( shape_params );
            auto SpecificPowerShapeVar =
                AMP::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerShape" );

            // Create a vector associated with the Variable.
            auto SpecificPowerShapeVec =
                AMP::LinearAlgebra::createVector( dof_map, SpecificPowerShapeVar, split );
            auto SpecificPowerMagnitudeVec = SpecificPowerShapeVec->cloneVector();
            SpecificPowerMagnitudeVec->setToScalar( 1.0 );

            // Set the initial value for all nodes of SpecificPowerVec to zero.
            SpecificPowerShapeVec->setToScalar( 0.0 );
            AMP::LinearAlgebra::Vector::shared_ptr nullVec;
            try {
                shape->apply( SpecificPowerMagnitudeVec, SpecificPowerShapeVec );
            } catch ( std::exception const &a ) {
                std::cout << a.what() << std::endl;
                ut->failure( "PowerShape error" );
            }
            if ( nMoments == 0 ) {
                if ( !AMP::Utilities::approx_equal( SpecificPowerShapeVec->max(), 1.0, 1e-9 ) ) {
                    ut->failure( "flat solution is not flat (max)." );
                    printf( "This %.9e is not 1.0. \n", SpecificPowerShapeVec->max() );
                }
                if ( !AMP::Utilities::approx_equal( SpecificPowerShapeVec->min(), 1.0, 1e-9 ) ) {
                    ut->failure( "flat solution is not flat (min)." );
                    printf( "This %.9e is not 1.0. \n", SpecificPowerShapeVec->min() );
                }
            }

#ifdef USE_EXT_SILO
            shapeVec->copyVector( SpecificPowerShapeVec );
            // SpecificPowerShapeVec->copyVector(shapeVec);
            siloWriter->writeFile( "PowerShape-Zr", i + 3 );
#endif

            n += 2;
        }
    }

    ut->passes( "PowerShape produces a non-negative power shape." );
}

int testPowerShape_Zr( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    test_with_shape( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}