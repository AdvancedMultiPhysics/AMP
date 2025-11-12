#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <string>


void myTest( AMP::UnitTest *ut, const std::string &input_file )
{

    std::string log_file = "output_testMatCreate";
    AMP::logOnlyNodeZero( log_file );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create the DOF manager
    auto scalarDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );
    auto vectorDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 3 );

    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );
    auto inVec  = AMP::LinearAlgebra::createVector( vectorDOFs, inVar );
    auto outVec = AMP::LinearAlgebra::createVector( scalarDOFs, outVar );

    // Create the matrix
    auto mat1 = AMP::LinearAlgebra::createMatrix( inVec, outVec );
    if ( mat1 ) {
        ut->passes( "Able to create a non-square matrices" );
    } else {
        ut->failure( "Unable to create a non-square matrices" );
    }

    auto scalarVar   = std::make_shared<AMP::LinearAlgebra::Variable>( "scalarVar" );
    auto vectorVar   = std::make_shared<AMP::LinearAlgebra::Variable>( "multiVar" );
    auto multiVarVec = AMP::LinearAlgebra::MultiVector::create( "MultiVec", mesh->getComm() );
    multiVarVec->addVector( AMP::LinearAlgebra::createVector( vectorDOFs, vectorVar ) );
    multiVarVec->addVector( AMP::LinearAlgebra::createVector( scalarDOFs, scalarVar ) );

    // Create the matrix
    auto mat2 = AMP::LinearAlgebra::createMatrix( multiVarVec, multiVarVec );
    if ( mat2 ) {
        ut->passes( "Able to create a mutli-var matrices" );
    } else {
        ut->failure( "Unable to create a multi-var matrices" );
    }
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    myTest( &ut, "input_testMatCreate-1" );

    ut.report();

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
