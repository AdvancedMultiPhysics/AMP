#ifndef included_AMP_MeshMatrixTests
#define included_AMP_MeshMatrixTests

#include "AMP/ampmesh/Mesh.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"

#include "ProfilerApp.h"


namespace AMP {
namespace Mesh {


template<int DOF_PER_NODE, bool SPLIT>
void meshTests::VerifyGetMatrixTrivialTest( AMP::UnitTest *utils, AMP::Mesh::Mesh::shared_ptr mesh )
{
    PROFILE_START( "VerifyGetMatrixTrivialTest", 1 );

    // Create the DOF_Manager
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, DOF_PER_NODE );

    // Create a nodal variable
    auto variable = AMP::make_shared<AMP::LinearAlgebra::Variable>( "test vector" );

    // Create the matrix and vectors
    auto vector1 = AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
    auto vector2 = AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
    auto matrixa = AMP::LinearAlgebra::createMatrix( vector1, vector2 );

    // Currently there is a bug with multivectors
    bool isMultiVector =
        dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( vector1 ) != nullptr;
    if ( isMultiVector ) {
        utils->expected_failure( "VerifyGetMatrixTrivialTest with split=true" );
        PROFILE_STOP2( "VerifyGetMatrixTrivialTest", 1 );
        return;
    }

    // Run some tests
    vector1->setRandomValues();
    matrixa->makeConsistent();
    matrixa->mult( vector1, vector2 );
    if ( vector2->L1Norm() < 0.00000001 )
        utils->passes( "obtained 0 matrix from mesh" );
    else
        utils->failure( "did not obtain 0 matrix from mesh" );

    // Need to get another matrix to store data due to Epetra insert/replace idiom.
    // Matrixa is fixed with no entires.
    auto matrixb = AMP::LinearAlgebra::createMatrix( vector1, vector2 );

    vector2->setToScalar( 1. );
    matrixb->makeConsistent();
    matrixb->setDiagonal( vector2 );
    matrixb->mult( vector1, vector2 );
    vector1->subtract( vector1, vector2 );

    if ( vector1->L1Norm() < 0.0000001 )
        utils->passes( "created identity matrix from mesh" );
    else
        utils->failure( "created identity matrix from mesh" );
    PROFILE_STOP( "VerifyGetMatrixTrivialTest", 1 );
}


template<int DOF_PER_NODE, bool SPLIT>
void meshTests::GhostWriteTest( AMP::UnitTest *utils, AMP::Mesh::Mesh::shared_ptr mesh )
{
    PROFILE_START( "GhostWriteTest", 1 );

    // Create the DOF_Manager
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, DOF_PER_NODE );

    // Create a nodal variable
    auto variable = AMP::make_shared<AMP::LinearAlgebra::Variable>( "test vector" );

    // Create the matrix and vectors
    auto vector1 = AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
    auto vector2 = AMP::LinearAlgebra::createVector( DOFs, variable, SPLIT );
    auto matrix  = AMP::LinearAlgebra::createMatrix( vector1, vector2 );

    // For each mesh, get a mapping of it's processor id's to the comm of the mesh
    auto proc_map = createRankMap( mesh );
    NULL_USE( proc_map );

    // For each processor, make sure it can write to all entries
    auto comm = mesh->getComm();
    for ( int p = 0; p < comm.getSize(); p++ ) {
        matrix->setScalar( -1.0 );
        matrix->makeConsistent();
        // Processor p should fill the values
        if ( p == comm.getRank() ) {
            try {
                double proc = mesh->getComm().getRank();
                bool passes = true;
                // Loop through the owned nodes
                auto it = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
                for ( size_t i = 0; i < it.size(); i++, ++it ) {
                    // Get the DOFs for the node and it's neighbors
                    std::vector<size_t> localDOFs;
                    DOFs->getDOFs( it->globalID(), localDOFs );
                    std::vector<size_t> neighborDOFs, dofs;
                    auto neighbors = it->getNeighbors();
                    for ( const auto &neighbor : neighbors ) {
                        if ( neighbor == nullptr )
                            continue;
                        DOFs->getDOFs( neighbor->globalID(), dofs );
                        for ( size_t j = 0; j < dofs.size(); j++ )
                            neighborDOFs.push_back( dofs[j] );
                    }
                    // For each local DOF, set all matrix elements involving the current DOF
                    for ( size_t j = 0; j < localDOFs.size(); j++ ) {
                        for ( size_t i = 0; i < localDOFs.size(); i++ ) {
                            matrix->setValueByGlobalID( localDOFs[i], localDOFs[j], proc );
                            matrix->setValueByGlobalID( localDOFs[j], localDOFs[i], proc );
                        }
                        for ( size_t i = 0; i < neighborDOFs.size(); i++ ) {
                            matrix->setValueByGlobalID( localDOFs[j], neighborDOFs[i], proc );
                            matrix->setValueByGlobalID( neighborDOFs[i], localDOFs[j], proc );
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
                    utils->passes( "Able to write to ghost entries in matrix" );
                else
                    utils->failure( "Able to write to ghost entries in matrix" );
            } catch ( ... ) {
                utils->failure( "Able to write to ghost entries in matrix (exception)" );
            }
        }

        // Apply make consistent
        matrix->makeConsistent();

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
        char msg[100];
        sprintf(msg,"Matrix entries set by processor %i read correctly on processor
        %i",p,comm.getRank());
        if ( passes )
            utils->passes( msg );
        else
            utils->failure( msg );*/
    }
    PROFILE_STOP( "GhostWriteTest", 1 );
}


} // namespace Mesh
} // namespace AMP

#endif
