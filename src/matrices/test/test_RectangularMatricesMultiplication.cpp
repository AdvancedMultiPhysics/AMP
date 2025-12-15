#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>


void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
#if !defined( AMP_USE_PETSC ) && !defined( AMP_USE_TRILINOS )
    if ( AMP::AMP_MPI( AMP_COMM_WORLD ).getSize() > 1 ) {
        ut->expected_failure( "No parallel matrix to test" );
        return;
    }
#endif
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    // build mesh database
    auto database = std::make_shared<AMP::Database>( "Mesh" );
    database->putScalar( "MeshName", "cube" );
    database->putScalar( "MeshType", "AMP" );
    database->putScalar( "dim", 3 );
    database->putScalar( "Generator", "cube" );
    database->putScalar( "x_offset", 0.0 );
    database->putScalar( "y_offset", 0.0 );
    database->putScalar( "z_offset", 0.0 );
    std::vector<int> size( 3, 8 );
    database->putVector<int>( "Size", size );
    std::vector<double> range( 6, 0.0 );
    range[1] = range[3] = range[5] = 1.0;
    database->putVector( "Range", range );
    // create mesh
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );
    auto mesh = AMP::Mesh::MeshFactory::create( params );
    // create two different dof managers
    auto firstDofManager = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto secondDofManager = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, 0, 2, true );
    size_t n = firstDofManager->numGlobalDOF();
    size_t m = secondDofManager->numGlobalDOF();
    AMP_ASSERT( n != m );
    // create the two corresponding vectors
    auto var       = std::make_shared<AMP::LinearAlgebra::Variable>( "var" );
    auto firstVec  = AMP::LinearAlgebra::createVector( firstDofManager, var, true );  // n
    auto secondVec = AMP::LinearAlgebra::createVector( secondDofManager, var, true ); // m
    firstVec->zero();
    secondVec->zero();
    // Loop through the valid matrix build types
    std::vector<std::string> types;
    if ( globalComm.getSize() == 1 )
        types.emplace_back( "DenseSerialMatrix" );
#ifdef AMP_USE_TRILINOS
    types.emplace_back( "ManagedEpetraMatrix" );
#endif
    for ( auto &type : types ) {
        auto tmp = exeName + ": " + type;
        // create four matrices
        auto firstMat  = AMP::LinearAlgebra::createMatrix( firstVec, secondVec, type ); // mxn
        auto secondMat = AMP::LinearAlgebra::createMatrix( secondVec, firstVec, type ); // nxm
        AMP_ASSERT( firstMat->numGlobalRows() == m && firstMat->numGlobalColumns() == n );
        firstMat->setScalar( 1.0 );
        secondMat->setScalar( 2.0 );
        // do matrix multiplications
        try {
            auto thirdMat = AMP::LinearAlgebra::Matrix::matMatMult( secondMat, firstMat );
            AMP_ASSERT( thirdMat->numGlobalRows() == n && thirdMat->numGlobalColumns() == n );
            ut->passes( tmp );
        } catch ( ... ) {
            ut->failure( tmp );
        }
        try {
            auto fourthMat = AMP::LinearAlgebra::Matrix::matMatMult( firstMat, secondMat );
            AMP_ASSERT( fourthMat->numGlobalRows() == m && fourthMat->numGlobalColumns() == m );
            ut->passes( tmp );
        } catch ( ... ) {
            ut->failure( tmp );
        }
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "test_RectangularMatricesMultiplication";

    myTest( &ut, exeName );

    ut.report();

    int numFailed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return numFailed;
}
