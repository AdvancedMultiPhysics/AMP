#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ParameterFactory.h"
#include "AMP/operators/boundary/DirichletMatrixCorrectionParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>


static void ParameterFactoryTest( AMP::UnitTest *ut )
{
    std::string exeName( "testParameterFactory-1" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    AMP_INSIST( input_db->keyExists( "Parameter" ), "Key ''Parameter'' is missing!" );
    auto elemOp_db          = input_db->getDatabase( "Parameter" );
    auto operatorParameters = AMP::Operator::ParameterFactory::createParameter( elemOp_db, mesh );

    if ( elemOp_db->getString( "name" ) == "DirichletMatrixCorrection" ) {
        auto operatorParams =
            std::dynamic_pointer_cast<AMP::Operator::DirichletMatrixCorrectionParameters>(
                operatorParameters );

        if ( operatorParams )
            ut->passes( exeName );
        else
            ut->failure( exeName );
    } else {
        ut->failure( exeName );
    }
}


int testParameterFactory( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    ParameterFactoryTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
