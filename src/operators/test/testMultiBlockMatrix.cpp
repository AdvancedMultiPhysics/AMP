#include "AMP/IO/AsciiWriter.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/Writer.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


#define ITFAILS ut.failure( __LINE__ );
#define UNIT_TEST( a ) \
    if ( !( a ) )      \
        ut.failure( __LINE__ );

static void LinearTimeOperatorTest( AMP::UnitTest *ut )
{
    std::string input_file = "input_testMultiBlockMatrix";
    std::string log_file   = "log_testMultiBlockMatrix";

    AMP::logOnlyNodeZero( log_file );


    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // create a linear BVP operator
    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearOperator", input_db ) );

    // create a mass linear BVP operator
    auto massOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MassLinearOperator", input_db ) );

    auto fullMat = linearOperator->getMatrix();
    AMP::IO::AsciiWriter fullMatWriter;
    fullMatWriter.registerMatrix( fullMat );
    fullMatWriter.writeFile( "FullMat", 0 );

    auto massMat = massOperator->getMatrix();
    auto diffMat = linearOperator->getMatrix();
    auto sinMat  = diffMat->clone();
    sinMat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    sinMat->zero();

    sinMat->axpy( 1.0, diffMat );
    sinMat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    AMP::IO::AsciiWriter sinMatWriter;
    sinMatWriter.registerMatrix( sinMat );
    sinMatWriter.writeFile( "SinMat", 0 );

    ut->passes( "Ran to completion" );
}


//---------------------------------------------------------------------------//

int testMultiBlockMatrix( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    LinearTimeOperatorTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
