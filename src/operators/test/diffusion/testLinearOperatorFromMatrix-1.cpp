#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>


void userLinearOperatorTest( AMP::UnitTest *const ut, const std::string &exeName )
{
    // Test create
    const std::string input_file = "input_" + exeName;
    const std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // read the input file into a database
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // extract the Mesh database and create the mesh parameters
    auto meshDB = input_db->getDatabase( "Mesh" );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( meshDB );
    params->setComm( globalComm );

    // create the mesh
    const auto mesh = AMP::Mesh::MeshFactory::create( params );

    // create a linear diffusion operator
    auto linearOperator =
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearDiffusionOp", input_db );
    auto diffOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( linearOperator );

    // extract the internal matrix
    const auto &userMat = diffOp->getMatrix();

    AMP_INSIST( userMat->numGlobalColumns() == userMat->numGlobalRows(), "matrix is not square" );

    // extract the right vector
    const auto userVector = userMat->getInputVector();

    // concludes creation of a native linear operator
    // ************************************************************************************************

    // extract information about the local size and mpi comm
    const auto localSize = userVector->getLocalSize();
    const auto ampComm   = userVector->getComm();

    // construct a dof manager
    const auto userDM     = userVector->getDOFManager();
    const auto dofManager = std::make_shared<AMP::Discretization::DOFManager>(
        localSize, ampComm, userDM->getRemoteDOFs() );
    const auto copyVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "copyVariable" );

    // create a vector based on the dofs and variable
    auto ampVector = AMP::LinearAlgebra::createVector( dofManager, copyVariable );
    AMP_INSIST( ampVector != nullptr, "ampVector is null" );

    // copy values from one vector to another
    std::copy( userVector->begin(), userVector->end(), ampVector->begin() );
    ampVector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    // concludes demonstrating how to initialize an AMP vector from a user vector
    // ************************************************************************************************

    // create a lambda that returns non zero column ids given a global row id
    auto getColumnIDS = [userMat]( size_t row ) { return userMat->getColumnIDs( row ); };

    // create a matrix based on the dimensions of the copied vector
    auto ampMat = AMP::LinearAlgebra::createMatrix( ampVector, ampVector, "auto", getColumnIDS );


    // construct a LinearOperator and set its matrix
    const auto linearOpDB = std::make_shared<AMP::Database>( "linearOperatorDB" );
    linearOpDB->putScalar<int>( "print_info_level", 0 );
    auto linearOpParameters = std::make_shared<AMP::Operator::OperatorParameters>( linearOpDB );
    auto linearOp           = std::make_shared<AMP::Operator::LinearOperator>( linearOpParameters );

    ampMat->axpy( 1.0, *userMat );
    ampMat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    // concludes demonstrating how to initialize an AMP linear operator from a user matrix
    // ************************************************************************************************

    auto u = userVector->clone();
    auto v = userVector->clone();

    u->setRandomValues();
    v->setRandomValues();

    linearOp->setMatrix( ampMat );
    linearOp->setVariables( u->getVariable(), v->getVariable() );

    // form the difference of the matrices
    // COMMENT: simple add, subtract routines would be nice for matrices
    ampMat->axpy( -1.0, userMat );
    ampMat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    const auto matLinfNorm = static_cast<double>( ampMat->LinfNorm() );
    auto passed            = ( matLinfNorm <= std::numeric_limits<double>::min() );

    if ( passed ) {
        ut->passes( exeName );
    } else {
        ut->failure( "difference of matrix and copy is not zero (LinfNorm of difference): " +
                     std::to_string( matLinfNorm ) );
    }

    linearOp->apply( u, v );

    const auto maxNorm = static_cast<double>( v->maxNorm() );

    passed = ( maxNorm <= std::numeric_limits<double>::min() );

    if ( passed ) {
        ut->passes( exeName );
    } else {
        ut->failure( "unable to create a copy of a linear operator, difference" +
                     std::to_string( maxNorm ) );
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files = { "Diffusion-TUI-Thermal-1",
                                       "Diffusion-UO2MSRZC09-Thermal-1" };

    for ( const auto &file : files )
        userLinearOperatorTest( &ut, file );

    ut.report();

    const int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
