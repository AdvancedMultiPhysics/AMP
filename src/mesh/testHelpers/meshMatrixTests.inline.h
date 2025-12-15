#ifndef included_AMP_MeshMatrixTests
#define included_AMP_MeshMatrixTests

#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"

#include "ProfilerApp.h"


namespace AMP::Mesh {


template<int DOF_PER_NODE, bool SPLIT>
void meshTests::VerifyGetMatrixTrivialTest( AMP::UnitTest &ut,
                                            std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "VerifyGetMatrixTrivialTest", 1 );

    // Create the DOF_Manager
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, DOF_PER_NODE );

    // Create a nodal variable
    auto variable = std::make_shared<AMP::LinearAlgebra::Variable>( "test vector" );

    // Create the matrix and vectors
    auto vector1 = AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
    auto vector2 = AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
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


template<int DOF_PER_NODE, bool SPLIT>
void meshTests::GhostWriteTest( AMP::UnitTest &ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    PROFILE( "GhostWriteTest", 1 );

    // Create the DOF_Manager
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, DOF_PER_NODE );

    // Create a nodal variable
    auto variable = std::make_shared<AMP::LinearAlgebra::Variable>( "test vector" );

    // Create the matrix and vectors
    auto vector1 = AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
    auto vector2 = AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
    auto matrix  = AMP::LinearAlgebra::createMatrix( vector1, vector2 );

    // For each processor, make sure it can write to all entries
    auto comm = mesh->getComm();
    for ( int p = 0; p < comm.getSize(); p++ ) {
        matrix->setScalar( -1.0 );
        matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
        // Processor p should fill the values
        if ( p == comm.getRank() ) {
            try {
                double proc = mesh->getComm().getRank();
                bool passes = true;
                // Loop through the owned nodes
                auto nodes = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
                for ( auto &node : nodes ) {
                    // Get the DOFs for the node and it's neighbors
                    std::vector<size_t> localDOFs;
                    DOFs->getDOFs( node.globalID(), localDOFs );
                    std::vector<size_t> neighborDOFs, dofs;
                    auto neighbors = node.getNeighbors();
                    for ( const auto &neighbor : neighbors ) {
                        if ( neighbor == nullptr )
                            continue;
                        DOFs->getDOFs( neighbor->globalID(), dofs );
                        for ( size_t j = 0; j < dofs.size(); j++ )
                            neighborDOFs.push_back( dofs[j] );
                    }
                    // For each local DOF, set all matrix elements involving the current DOF
                    for ( size_t j = 0; j < localDOFs.size(); j++ ) {
                        for ( size_t k = 0; k < localDOFs.size(); k++ ) {
                            matrix->setValueByGlobalID( localDOFs[k], localDOFs[j], proc );
                            matrix->setValueByGlobalID( localDOFs[j], localDOFs[k], proc );
                        }
                        for ( size_t k = 0; k < neighborDOFs.size(); k++ ) {
                            matrix->setValueByGlobalID( localDOFs[j], neighborDOFs[k], proc );
                            matrix->setValueByGlobalID( neighborDOFs[k], localDOFs[j], proc );
                        }
                        std::vector<size_t> cols;
                        std::vector<double> values;
                        matrix->getRowByGlobalID( localDOFs[j], cols, values );
                        for ( size_t i1 = 0; i1 < cols.size(); i1++ ) {
                            for ( size_t i2 = 0; i2 < localDOFs.size(); i2++ ) {
                                if ( cols[i1] == localDOFs[i2] ) {
                                    if ( values[i1] != proc )
                                        passes = false;
                                }
                            }
                            for ( size_t i2 = 0; i2 < neighborDOFs.size(); i2++ ) {
                                if ( cols[i1] == neighborDOFs[i2] ) {
                                    if ( values[i1] != proc )
                                        passes = false;
                                }
                            }
                        }
                    }
                }
                if ( passes )
                    ut.passes( "Able to write to ghost entries in matrix" );
                else
                    ut.failure( "Able to write to ghost entries in matrix" );
            } catch ( ... ) {
                ut.failure( "Able to write to ghost entries in matrix (exception)" );
            }
        }

        // Apply make consistent
        matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

        /*// Test that each matrix entry has the proper value
        bool passes = true;
        // Get a list of all nodes owned by the given processor p
        std::set<AMP::Mesh::MeshElementID> nodes_p;
        auto it = mesh->getIterator(AMP::Mesh::GeomType::Vertex,1);
        for (size_t i=0; i<it.size(); i++, ++it) {
            auto id = it->globalID();
            const auto &map = proc_map.find(id.meshID())->second;
            int rank = map[id.owner_rank()];
            if ( rank == p )
                nodes_p.insert( it->globalID() );
        }
        // Get a list of all DOFs associated with the given nodes
        std::vector<size_t> dofs_p;
        std::vector<AMP::Mesh::MeshElementID> tmp(nodes_p.begin(),nodes_p.end());
        DOFs->getDOFs( tmp, dofs_p );
        if ( dofs_p.size()==0 )
            dofs_p.push_back( (size_t)-1 );
        AMP::Utilities::quicksort( dofs_p );
        it = mesh->getIterator(AMP::Mesh::GeomType::Vertex,0);
        double proc = p;
        for (size_t i=0; i<it.size(); i++, ++it) {
            std::vector<size_t> dofs;
            DOFs->getDOFs( it->globalID(), dofs );
            for (size_t j=0; j<dofs.size(); j++) {
                size_t row = dofs[j];
                size_t index = AMP::Utilities::findfirst( dofs_p, row );
                if ( index==dofs_p.size() ) { index--; }
                bool row_found = row==dofs_p[index];
                std::vector<unsigned int> cols;
                std::vector<double> values;
                matrix->getRowByGlobalID( row, cols, values );
                for (size_t k=0; k<cols.size(); k++) {
                    index = AMP::Utilities::findfirst( dofs_p, (size_t) cols[k] );
                    if ( index==dofs_p.size() ) { index--; }
                    bool col_found = cols[k]==dofs_p[index];
                    if ( row_found || col_found ) {
                        if ( values[k] != proc )
                            passes = false;
                    } else {
                        if ( values[k] != -1.0 )
                            passes = false;
                    }
                }
            }
        }
        auto msg = AMP::Utilities::stringf(
            "Matrix entries set by processor %i read correctly on processor %i",
            p,comm.getRank());
        if ( passes )
            ut.passes( msg );
        else
            ut.failure( msg );*/
    }
}


} // namespace AMP::Mesh

#endif
