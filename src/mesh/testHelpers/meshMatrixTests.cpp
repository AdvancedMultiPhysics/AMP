#ifndef included_AMP_MeshMatrixTests
#define included_AMP_MeshMatrixTests

#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/testHelpers/meshTests.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"

#include "ProfilerApp.h"


namespace AMP::Mesh {


void meshTests::VerifyGetMatrixTrivialTest( AMP::UnitTest &ut,
                                            std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                            int dofsPerNode,
                                            bool split )
{
    PROFILE( "VerifyGetMatrixTrivialTest", 1 );

    // Create the DOF_Manager
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, dofsPerNode );

    // Create a nodal variable
    auto variable = std::make_shared<AMP::LinearAlgebra::Variable>( "test vector" );

    // Create the matrix and vectors
    auto vector1 = AMP::LinearAlgebra::createVector( DOFs, variable, split );
    auto vector2 = AMP::LinearAlgebra::createVector( DOFs, variable, split );
    auto matrixa = AMP::LinearAlgebra::createMatrix( vector1, vector2 );

    // Currently there is a bug with multivectors
    bool isMultiVector =
        std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( vector1 ) != nullptr;
    if ( isMultiVector ) {
        ut.expected_failure( "VerifyGetMatrixTrivialTest with split=true" );
        return;
    }

    // Run some tests
    vector1->setRandomValues();
    matrixa->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    matrixa->mult( vector1, vector2 );
    if ( vector2->L1Norm() < 0.00000001 )
        ut.passes( "obtained 0 matrix from mesh" );
    else
        ut.failure( "did not obtain 0 matrix from mesh" );

    // Need to get another matrix to store data due to Epetra insert/replace idiom.
    // Matrixa is fixed with no entires.
    auto matrixb = AMP::LinearAlgebra::createMatrix( vector1, vector2 );

    vector2->setToScalar( 1. );
    matrixb->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    matrixb->setDiagonal( vector2 );
    matrixb->mult( vector1, vector2 );
    vector1->subtract( *vector1, *vector2 );

    if ( vector1->L1Norm() < 0.0000001 )
        ut.passes( "created identity matrix from mesh" );
    else
        ut.failure( "created identity matrix from mesh" );
}


void meshTests::RowWriteTest( AMP::UnitTest &ut,
                              std::shared_ptr<AMP::Mesh::Mesh> mesh,
                              int dofsPerNode,
                              bool split )
{
    PROFILE( "RowWriteTest", 1 );

    // Create the DOF_Manager
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, dofsPerNode );

    // Create a nodal variable
    auto variable = std::make_shared<AMP::LinearAlgebra::Variable>( "test vector" );

    // Create the matrix and vectors
    auto vector1 = AMP::LinearAlgebra::createVector( DOFs, variable, split );
    auto vector2 = AMP::LinearAlgebra::createVector( DOFs, variable, split );
    auto matrix  = AMP::LinearAlgebra::createMatrix( vector1, vector2 );

    // For each processor, make sure it can write to all entries in the row
    auto comm = mesh->getComm();
    double p  = comm.getRank();
    matrix->setScalar( -1.0 );
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    std::vector<size_t> rows, columns, dofs, cols2;
    std::vector<double> procs, values;
    // Loop through the owned nodes
    // The row is "owned" by the local owner of the node
    bool pass  = true;
    auto nodes = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    for ( auto &node : nodes ) {
        // Get the DOFs for the node and it's neighbors
        DOFs->getDOFs( node.globalID(), rows );
        columns.clear();
        for ( auto r : rows )
            columns.push_back( r );
        for ( const auto &neighbor : node.getNeighbors() ) {
            if ( neighbor.isNull() )
                continue;
            DOFs->getDOFs( neighbor.globalID(), dofs );
            for ( auto dof : dofs )
                columns.push_back( dof );
        }
        // For each local DOF, set all matrix elements involving the current DOF
        for ( auto row : rows ) {
            procs.resize( columns.size(), p );
            matrix->setRowByGlobalID( row, columns, procs );
            matrix->getRowByGlobalID( row, cols2, values );
            for ( size_t i1 = 0; i1 < cols2.size(); i1++ ) {
                bool found = true;
                for ( size_t i2 = 0; i2 < columns.size(); i2++ ) {
                    if ( cols2[i1] == columns[i2] ) {
                        found = true;
                        pass  = pass && values[i1] == p;
                    }
                }
                if ( !found )
                    pass = false;
            }
        }
    }
    if ( pass )
        ut.passes( "Able to write to ghost entries in matrix" );
    else
        ut.failure( "Able to write to ghost entries in matrix" );

    // Apply make consistent
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


} // namespace AMP::Mesh

#endif
