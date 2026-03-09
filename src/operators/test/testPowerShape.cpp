#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/libmesh/PowerShape.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static auto Cell   = AMP::Mesh::GeomType::Cell;
static auto Vertex = AMP::Mesh::GeomType::Vertex;


static void test_with_shape( AMP::UnitTest &ut, const std::string &input )
{
    //  Read Input File
    AMP::pout << "Testing " << input << std::endl;
    auto input_db = AMP::Database::parseInputFile( input );

    // Create the Mesh
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP_COMM_WORLD );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    //  Construct PowerShape.
    AMP_INSIST( input_db->keyExists( "MyPowerShape" ), "Key ''MyPowerShape'' is missing!" );
    auto shape_db        = input_db->getDatabase( "MyPowerShape" );
    auto shape_params    = std::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
    shape_params->d_Mesh = mesh;
    auto shape           = std::make_shared<AMP::Operator::PowerShape>( shape_params );

    // Create a DOF manager for a gauss point vector
    auto dof_map = AMP::Discretization::simpleDOFManager::create( mesh, Cell, 1, 8 );

    // Create vectors
    auto SpecificPowerShapeVar =
        std::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerInWattsPerKg" );
    auto SpecificPowerShapeVec = AMP::LinearAlgebra::createVector( dof_map, SpecificPowerShapeVar );
    auto SpecificPowerMagnitudeVec = SpecificPowerShapeVec->clone();
    SpecificPowerMagnitudeVec->setToScalar( 4157. );

    // Set the initial value for all nodes of SpecificSpecificPowerShapeVec to zero
    SpecificPowerShapeVec->setToScalar( 0.0 );
    shape->apply( SpecificPowerMagnitudeVec, SpecificPowerShapeVec );
    ut.passes( input + ": PowerShape gets past apply with a non-flat power shape." );

    AMP::pout << "SpecificPowerShapeVec->max()"
              << " : " << SpecificPowerShapeVec->min() << " : " << SpecificPowerShapeVec->max()
              << std::endl;
    // Check that the data is non-negative
    bool itpasses = true;
    for ( auto &elem : mesh->getIterator( Cell, 1 ) ) {
        for ( int i = 0; i < 8; i++ ) {
            std::vector<size_t> ndx;
            dof_map->getDOFs( elem.globalID(), ndx );
            int offset = ndx[i];
            if ( SpecificPowerShapeVec->getValueByGlobalID( offset ) < 0.0 ) {
                if ( !itpasses )
                    ut.failure( input + ": PowerShape error" );
                itpasses = false;
            }
        }
    }

    if ( itpasses )
        ut.passes( input + ": PowerShape produces a non-negative power shape." );

    //  Testing the new legendre function. valLegendre(int n, double x)
    double pn = shape->evalLegendre( 3, 2.0 );
    if ( pn != 17. )
        ut.failure( input + ": PowerShape error" );
}


static void test_with_shape_Zr( AMP::UnitTest &ut, const std::string &input )
{
    //  Read Input File
    AMP::pout << "Testing " << input << std::endl;
    auto input_db = AMP::Database::parseInputFile( input );

    // Create the Mesh
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP_COMM_WORLD );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Construct PowerShape for a radial only term.
    auto shape_db  = input_db->putDatabase( "shape_db" );
    auto Overwrite = AMP::Database::Check::Overwrite;
    shape_db->putScalar( "coordinateSystem", "cylindrical" );
    shape_db->putScalar( "type", "zernikeRadial" );
    shape_db->putScalar( "print_info_level", 1 );

    // Create a DOF manager for a gauss point vector
    auto dof_map  = AMP::Discretization::simpleDOFManager::create( mesh, Cell, 0, 8 );
    auto shapeVar = std::make_shared<AMP::LinearAlgebra::Variable>( "PowerShape" );
    auto shapeVec = AMP::LinearAlgebra::createVector( dof_map, shapeVar );
    for ( int nMoments = 0; nMoments < 3; nMoments++ ) {
        shape_db->putScalar( "numMoments", nMoments, {}, Overwrite );
        if ( nMoments > 0 ) {
            std::vector<double> moments( nMoments, 0. );
            moments[nMoments - 1] = -1.;
            shape_db->putVector( "Moments", moments, {}, Overwrite );
        }
        auto shape_params    = std::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
        shape_params->d_Mesh = mesh;
        auto shape           = std::make_shared<AMP::Operator::PowerShape>( shape_params );

        // Create vectors
        auto SpecificPowerShapeVar =
            std::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerInWattsPerKg" );
        auto SpecificPowerShapeVec =
            AMP::LinearAlgebra::createVector( dof_map, SpecificPowerShapeVar );
        auto SpecificPowerMagnitudeVec = SpecificPowerShapeVec->clone();
        SpecificPowerMagnitudeVec->setToScalar( 1.0 );

        // Set the initial value for all nodes of SpecificSpecificPowerShapeVec to zero.
        SpecificPowerShapeVec->setToScalar( 0.0 );
        shape->apply( SpecificPowerMagnitudeVec, SpecificPowerShapeVec );
        if ( nMoments == 0 ) {
            double max( SpecificPowerShapeVec->max() );
            double min( SpecificPowerShapeVec->min() );
            if ( !AMP::Utilities::approx_equal( max, 1.0, 1e-9 ) ) {
                ut.failure( input + ": flat solution is not really flat (max)." );
                printf( "This %.9e is not 1.0. \n", max );
            }
            if ( !AMP::Utilities::approx_equal( min, 1.0, 1e-9 ) ) {
                ut.failure( input + ": flat solution is not really flat (min)." );
                printf( "This %.9e is not 1.0. \n", min );
            }
        }
        shapeVec->copyVector( SpecificPowerShapeVec );
    }

    ut.passes( input + ": PowerShape produces a non-negative power shape." );

    //  Construct PowerShape for a full Zernike basis. -
    shape_db->putScalar( "type", "zernike", {}, Overwrite );
    int i = 0;
    for ( int nMoments = 0; nMoments < 9; nMoments++ ) {
        for ( int n = -nMoments; n <= nMoments; i++ ) {
            shape_db->putScalar( "numMoments", nMoments, {}, Overwrite );
            int nMNmoments = ( nMoments + 2 ) * ( nMoments + 1 ) / 2 - 1;
            if ( nMoments > 0 ) {
                std::vector<double> moments( nMNmoments, 0. );
                moments[i - 1] = -1.;
                shape_db->putVector( "Moments", moments, {}, Overwrite );
            }
            auto shape_params = std::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
            shape_params->d_Mesh = mesh;
            auto shape           = std::make_shared<AMP::Operator::PowerShape>( shape_params );
            auto SpecificPowerShapeVar =
                std::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerShape" );

            // Create a vector associated with the Variable.
            auto SpecificPowerShapeVec =
                AMP::LinearAlgebra::createVector( dof_map, SpecificPowerShapeVar );
            auto SpecificPowerMagnitudeVec = SpecificPowerShapeVec->clone();
            SpecificPowerMagnitudeVec->setToScalar( 1.0 );

            // Set the initial value for all nodes of SpecificSpecificPowerShapeVec to zero.
            SpecificPowerShapeVec->setToScalar( 0.0 );
            shape->apply( SpecificPowerMagnitudeVec, SpecificPowerShapeVec );
            if ( nMoments == 0 ) {
                double max( SpecificPowerShapeVec->max() );
                double min( SpecificPowerShapeVec->min() );
                if ( !AMP::Utilities::approx_equal( max, 1.0, 1e-9 ) ) {
                    ut.failure( input + ": flat solution is not flat (max)." );
                    printf( "This %.9e is not 1.0. \n", max );
                }
                if ( !AMP::Utilities::approx_equal( min, 1.0, 1e-9 ) ) {
                    ut.failure( input + ": flat solution is not flat (min)." );
                    printf( "This %.9e is not 1.0. \n", min );
                }
            }
            shapeVec->copyVector( SpecificPowerShapeVec );
            n += 2;
        }
    }
    ut.passes( input + ": PowerShape produces a non-negative power shape." );
}


static void test_with_shape_volint( AMP::UnitTest &ut, const std::string &input )
{
    //  Read Input File
    AMP::pout << "Testing " << input << std::endl;
    auto input_db = AMP::Database::parseInputFile( input );

    // Create the Mesh.
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP_COMM_WORLD );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Construct PowerShape
    AMP_INSIST( input_db->keyExists( "PowerShape" ), "Key ''PowerShape'' is missing!" );
    auto shape_db        = input_db->getDatabase( "PowerShape" );
    auto shape_params    = std::make_shared<AMP::Operator::PowerShapeParameters>( shape_db );
    shape_params->d_Mesh = mesh;
    auto shape           = std::make_shared<AMP::Operator::PowerShape>( shape_params );

    // Create a DOF manager for a gauss point vector
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create( mesh, Cell, 0, 8 );
    auto nodalDofMap      = AMP::Discretization::simpleDOFManager::create( mesh, Vertex, 1, 1 );

    // Create input and output vectors
    auto shapeVar    = std::make_shared<AMP::LinearAlgebra::Variable>( "interVar" );
    auto shapeInpVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, shapeVar );
    auto shapeOutVec = shapeInpVec->clone();

    shapeInpVec->setToScalar( 1. );

    // CREATE THE VOLUME INTEGRAL OPERATOR
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    auto volumeDatabase = input_db->getDatabase( "VolumeIntegralOperator" );
    auto inputVarDB     = volumeDatabase->getDatabase( "ActiveInputVariables" );
    inputVarDB->putScalar( "ActiveVariable_0", "interVar" );
    auto volumeOp = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "VolumeIntegralOperator", input_db ) );

    auto outputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "heatsource" );
    auto resVec         = AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable );

    shape->apply( shapeInpVec, shapeOutVec );

    AMP::pout << "shapeOutVec->max/min"
              << " : " << shapeOutVec->min() << " : " << shapeOutVec->max() << std::endl;
    ut.passes( input + ": PowerShape didn't crash the system" );

    volumeOp->apply( shapeOutVec, resVec );

    ut.passes( input + ": VolumeIntegralOperator didn't either" );
}


int testPowerShape( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    test_with_shape( ut, "input_testPowerShape-1" );
    test_with_shape( ut, "input_testPowerShape-2" );
    test_with_shape( ut, "input_testPowerShape-3" );
    test_with_shape( ut, "input_testPowerShape-5" );
    test_with_shape( ut, "input_testPowerShape-diffusion" );

    test_with_shape_Zr( ut, "input_testPowerShape_Zr" );

    test_with_shape_volint( ut, "input_testPowerShapeToVolIntOperator" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
