#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/libmesh/PowerShape.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void test_with_shape( AMP::UnitTest *ut )
{
    //  Read Input File
    auto input_db = std::make_shared<AMP::Database>();
    auto mesh_db  = input_db->putDatabase( "Mesh" );
    mesh_db->putScalar( "MeshName", "fuel" );
    mesh_db->putScalar( "MeshType", "AMP" );
    mesh_db->putScalar( "Generator", "cylinder" );
    mesh_db->putVector<int>( "Size", { 1, 8 } );
    mesh_db->putVector<double>( "Range", { 1, -2, 2 } );
    mesh_db->putVector<int>( "surfaceIds", { 1, 1, 1, 1, 2, 3 } );

    //   Create the Mesh
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    //  Construct PowerShape for a radial only term.
    auto shape_db = input_db->putDatabase( "shape_db" );
    shape_db->putScalar( "coordinateSystem", "cylindrical" );
    shape_db->putScalar( "type", "zernikeRadial" );
    shape_db->putScalar( "print_info_level", 1 );

    // Create a DOF manager for a gauss point vector
    int DOFsPerNode = 8;
    int ghostWidth  = 0;
    bool split      = true;
    auto dof_map    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, ghostWidth, DOFsPerNode, split );

    auto shapeVar = std::make_shared<AMP::LinearAlgebra::Variable>( "PowerShape" );
    auto shapeVec = AMP::LinearAlgebra::createVector( dof_map, shapeVar, split );

    for ( int nMoments = 0; nMoments < 3; nMoments++ ) {
        shape_db->putScalar( "numMoments", nMoments );
        if ( nMoments > 0 ) {
            std::vector<double> moments( nMoments, 0. );
            moments[nMoments - 1] = -1.;
            shape_db->putVector( "Moments", moments );
        }
        auto shape_params    = std::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
        shape_params->d_Mesh = mesh;
        auto shape           = std::make_shared<AMP::Operator::PowerShape>( shape_params );

        // Create a shared pointer to a Variable - Power - Output because it will be used in the
        // "residual" location of apply.
        auto SpecificPowerShapeVar =
            std::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerInWattsPerKg" );

        // Create a vector associated with the Variable.
        auto SpecificPowerShapeVec =
            AMP::LinearAlgebra::createVector( dof_map, SpecificPowerShapeVar, split );
        auto SpecificPowerMagnitudeVec = SpecificPowerShapeVec->clone();
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
            double max( SpecificPowerShapeVec->max() );
            double min( SpecificPowerShapeVec->min() );
            if ( !AMP::Utilities::approx_equal( max, 1.0, 1e-9 ) ) {
                ut->failure( "flat solution is not really flat (max)." );
                printf( "This %.9e is not 1.0. \n", max );
            }
            if ( !AMP::Utilities::approx_equal( min, 1.0, 1e-9 ) ) {
                ut->failure( "flat solution is not really flat (min)." );
                printf( "This %.9e is not 1.0. \n", min );
            }
        }

        shapeVec->copyVector( SpecificPowerShapeVec );
        // SpecificPowerShapeVec->copyVector(shapeVec);

    } // end for elements

    ut->passes( "PowerShape produces a non-negative power shape." );

    //  Construct PowerShape for a full Zernike basis. -
    shape_db->putScalar( "type", "zernike" );
    int i = 0;
    for ( int nMoments = 0; nMoments < 9; nMoments++ ) {
        //    for ( int nMoments = 0; nMoments < 5; nMoments++ ) {
        for ( int n = -nMoments; n <= nMoments; i++ ) {
            shape_db->putScalar( "numMoments", nMoments );
            int nMNmoments = ( nMoments + 2 ) * ( nMoments + 1 ) / 2 - 1;
            if ( nMoments > 0 ) {
                std::vector<double> moments( nMNmoments, 0. );
                moments[i - 1] = -1.;
                shape_db->putVector( "Moments", moments );
            }
            auto shape_params = std::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
            shape_params->d_Mesh = mesh;
            auto shape           = std::make_shared<AMP::Operator::PowerShape>( shape_params );
            auto SpecificPowerShapeVar =
                std::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerShape" );

            // Create a vector associated with the Variable.
            auto SpecificPowerShapeVec =
                AMP::LinearAlgebra::createVector( dof_map, SpecificPowerShapeVar, split );
            auto SpecificPowerMagnitudeVec = SpecificPowerShapeVec->clone();
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
                double max( SpecificPowerShapeVec->max() );
                double min( SpecificPowerShapeVec->min() );
                if ( !AMP::Utilities::approx_equal( max, 1.0, 1e-9 ) ) {
                    ut->failure( "flat solution is not flat (max)." );
                    printf( "This %.9e is not 1.0. \n", max );
                }
                if ( !AMP::Utilities::approx_equal( min, 1.0, 1e-9 ) ) {
                    ut->failure( "flat solution is not flat (min)." );
                    printf( "This %.9e is not 1.0. \n", min );
                }
            }

            shapeVec->copyVector( SpecificPowerShapeVec );
            // SpecificPowerShapeVec->copyVector(shapeVec);


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
