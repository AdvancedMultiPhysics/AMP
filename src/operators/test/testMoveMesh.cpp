#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/MoveMeshOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void runTest( AMP::UnitTest *ut, const std::string &input )
{
    //  Read Input File
    AMP::pout << "Testing " << input << std::endl;
    auto input_db = AMP::Database::parseInputFile( input );

    //   Create the Mesh
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Get the initial coordinates
    auto x0 = mesh->getPositionVector( "Position", 0 );

    // Create the operator
    auto op_db       = std::make_shared<AMP::Database>( "" );
    auto opParams    = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
    opParams->d_Mesh = mesh;
    auto op          = std::make_shared<AMP::Operator::MoveMeshOperator>( opParams );
    op->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "Position" ) );

    // Displace the mesh
    bool pass         = true;
    auto d            = { 0.1, 2.3, 1.0, -0.5 };
    auto displacement = x0->clone();
    for ( auto d2 : d ) {
        displacement->copy( *x0 );
        displacement->scale( d2 );
        op->apply( displacement, nullptr );
        auto x1  = mesh->getPositionVector( "Position", 0 );
        auto ans = x0->clone();
        ans->axpy( -( 1.0 + d2 ), *x0, *x1 );
        auto err = static_cast<double>( ans->L2Norm() / x0->L2Norm() );
        pass     = pass && err < 1e-12;
    }
    if ( pass )
        ut->passes( input );
    else
        ut->failure( input );
}


int testMoveMesh( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    if ( argc == 1 ) {
        auto files = { "input_MoveMesh_1D", "input_MoveMesh_2D", "input_MoveMesh_3D" };
        for ( const auto &input : files )
            runTest( &ut, input );
    } else {
        for ( int i = 1; i < argc; i++ )
            runTest( &ut, argv[i] );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
