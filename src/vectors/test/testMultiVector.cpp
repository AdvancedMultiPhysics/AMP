#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>

inline bool approx_equal( const AMP::Scalar &x, const AMP::Scalar &y, double tol )
{
    return AMP::Utilities::approx_equal( static_cast<double>( x ), static_cast<double>( y ), tol );
}

static void myTest( AMP::UnitTest *ut, const std::string &input )
{
    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Construct Variables
    auto Variable1 = std::make_shared<AMP::LinearAlgebra::Variable>( "Var1" );
    auto Variable2 = std::make_shared<AMP::LinearAlgebra::Variable>( "Var2" );
    auto Variable3 = std::make_shared<AMP::LinearAlgebra::Variable>( "Var3" );
    auto Variable4 = std::make_shared<AMP::LinearAlgebra::Variable>( "Var4" );
    auto dummyVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "dummy" );

    auto subVariable = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "subVar" );
    subVariable->add( Variable1 );
    subVariable->add( Variable2 );

    auto fullVariable = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "fullVariable" );
    fullVariable->add( Variable1 );
    fullVariable->add( Variable2 );
    fullVariable->add( Variable3 );
    fullVariable->add( Variable4 );

    // Create the DOF manager
    auto DOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );

    // Create the vectors
    auto multiVector = AMP::LinearAlgebra::createVector( DOFs, fullVariable, true );
    auto singleVector =
        AMP::LinearAlgebra::createVector( multiVector->getDOFManager(), dummyVar, false );
    auto subVector = multiVector->subsetVectorForVariable( subVariable );
    if ( singleVector->getGlobalSize() == multiVector->getGlobalSize() )
        ut->passes( "single and multivector are the right size" );
    else
        ut->failure( "Sub Vector is the right size" );
    if ( subVector ) {
        ut->passes( "Sub Vector is not NULL" );
        if ( multiVector->getGlobalSize() == 2 * subVector->getGlobalSize() )
            ut->passes( "Sub Vector is the right size" );
        else
            ut->failure( "Sub Vector is the right size" );
    } else {
        ut->failure( "Sub Vector is not NULL" );
    }

    // Try to copy data between the single vector and multivector
    singleVector->setRandomValues();
    multiVector->copyVector( singleVector );
    if ( approx_equal( singleVector->L2Norm(), multiVector->L2Norm(), 1e-12 ) )
        ut->passes( "Data copied from single vector to multivector" );
    else
        ut->failure( "Data copied from single vector to multivector" );
    singleVector->zero();
    singleVector->copyVector( multiVector );
    if ( approx_equal( singleVector->L2Norm(), multiVector->L2Norm(), 1e-12 ) )
        ut->passes( "Data copied from multivector to single vector" );
    else
        ut->failure( "Data copied from multivector to single vector" );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "input_testMultiVector" );
    for ( auto &exeName : exeNames ) {
        myTest( &ut, exeName );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
