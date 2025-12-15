#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/libmesh/SourceNonlinearElement.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void sourceTest( AMP::UnitTest *ut, const std::string &exeName )
{
    // Initialization
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logAllNodes( log_file );
    std::cout << "testing with input file " << input_file << std::endl;

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    //   CREATE THE VOLUME INTEGRAL OPERATOR
    AMP_INSIST( input_db->keyExists( "NeutronicsRhs" ), "key missing!" );
    auto ntx_db = input_db->getDatabase( "NeutronicsRhs" );

    // Construct stand-alone.
    if ( input_db->getWithDefault<bool>( "ConstructStandAlone", true ) ) {
        // construct it.
        auto ntxPrm = std::make_shared<AMP::Operator::OperatorParameters>( ntx_db );
        auto ntxRhs = std::make_shared<AMP::Operator::NeutronicsRhs>( ntxPrm );
        ut->passes( "NeutronicsRhs was constructed stand-alone for: " + input_file );
        // set the time.
        ntxRhs->setTimeStep( 0 );
        ut->passes( "NeutronicsRhs, constructed stand-alone, set the time for: " + input_file );
        // set the time.
        // ntxRhs->setTimeInSeconds(8000000.);
        // ut.passes( "NeutronicsRhs, constructed stand-alone, set the time in seconds for: " +
        // input_file);
        // Create a DOF manager for a gauss point vector
        int DOFsPerNode = 8;
        int ghostWidth  = 1;
        bool split      = true;
        auto dof_map    = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Cell, ghostWidth, DOFsPerNode, split );
        // create a variable/vector combo.
        AMP::LinearAlgebra::Vector::shared_ptr nullVec;
        // AMP::Operator::NeutronicsRhs::SP_HexGaussPointVariable outVar(new
        // AMP::Operator::NeutronicsRhs::HexGaussPointVariable("outpower") );
        auto outVar = ntxRhs->getOutputVariable();
        auto outVec = AMP::LinearAlgebra::createVector( dof_map, outVar, split );
        ntxRhs->apply( nullVec, outVec );
    }

    // Construct with OperatorBuilder
    {
        auto ntxBld = std::dynamic_pointer_cast<AMP::Operator::NeutronicsRhs>(
            AMP::Operator::OperatorBuilder::createOperator( mesh, "NeutronicsRhs", input_db ) );
        AMP_INSIST( ntxBld, "NULL rhs out of OperatorBuilder" );
        ut->passes( "NeutronicsRhs was constructed by OperatorBuilder for: " + input_file );
        // ntxBld->setTimeStep(0);
        // ut->passes( "NeutronicsRhs, constructed by OperatorBuilder, set the time for: " +
        // input_file);
    }
}


// Input and output file names
int testNeutronicsRhs( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    auto files = { "testNeutronicsRhs-1",
                   "testNeutronicsRhs-2",
                   "testNeutronicsRhs-3",
                   "testNeutronicsRhs-4",
                   "testNeutronicsRhs-db" };

    for ( auto &file : files )
        sourceTest( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
