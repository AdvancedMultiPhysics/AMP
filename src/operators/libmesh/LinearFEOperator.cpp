#include "AMP/operators/libmesh/LinearFEOperator.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include "libmesh/cell_hex8.h"
#include "libmesh/node.h"

namespace AMP::Operator {


LinearFEOperator::LinearFEOperator( std::shared_ptr<const OperatorParameters> inParams )
    : LinearOperator( inParams ),
      d_currElemPtr( nullptr ),
      d_elemOp( nullptr ),
      d_inDofMap( nullptr ),
      d_outDofMap( nullptr )
{
    auto params = std::dynamic_pointer_cast<const LinearFEOperatorParameters>( inParams );
    AMP_ASSERT( params );
    d_elemOp    = params->d_elemOp;
    d_inDofMap  = params->d_inDofMap;
    d_outDofMap = params->d_outDofMap;
    if ( !d_inDofMap )
        d_inDofMap = d_outDofMap;
    if ( !d_outDofMap )
        d_outDofMap = d_inDofMap;
}


void LinearFEOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    PROFILE( "reset" );
    AMP_INSIST( params, "NULL parameter" );
    AMP_INSIST( params->d_db, "NULL database" );

    const bool reuse_matrix = params->d_db->getWithDefault<bool>( "reset_reuses_matrix", true );
    auto lfeparams          = std::dynamic_pointer_cast<const LinearFEOperatorParameters>( params );

    if ( !d_matrix || !reuse_matrix ) {
        auto matrix_type = params->d_db->getWithDefault<std::string>( "matrix_type", "auto" );
        auto inVec       = AMP::LinearAlgebra::createVector(
            d_inDofMap, getInputVariable(), true, d_memory_location );
        auto outVec = AMP::LinearAlgebra::createVector(
            d_outDofMap, getOutputVariable(), true, d_memory_location );
        d_matrix = AMP::LinearAlgebra::createMatrix( inVec, outVec, matrix_type );
        d_matrix->zero();
        d_matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    }
    // AMP_ASSERT( ( *d_inDofMap ) == ( *d_matrix->getLeftDOFManager() ) );
    // AMP_ASSERT( ( *d_inDofMap ) == ( *d_matrix->getRightDOFManager() ) );

    this->preAssembly( params );

    for ( const auto &elem : d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 ) ) {
        this->preElementOperation( elem );
        d_elemOp->apply();
        this->postElementOperation();
    }

    this->postAssembly();
}


void LinearFEOperator::createCurrentLibMeshElement()
{
    d_currElemPtr = new libMesh::Hex8;
    for ( size_t j = 0; j < d_currNodes.size(); ++j ) {
        auto pt                      = d_currNodes[j].coord();
        d_currElemPtr->set_node( j ) = new libMesh::Node( pt[0], pt[1], pt[2], j );
    } // end for j
}


void LinearFEOperator::destroyCurrentLibMeshElement()
{
    for ( size_t j = 0; j < d_currElemPtr->n_nodes(); ++j ) {
        delete ( d_currElemPtr->node_ptr( j ) );
        d_currElemPtr->set_node( j ) = nullptr;
    } // end for j
    delete d_currElemPtr;
    d_currElemPtr = nullptr;
}
} // namespace AMP::Operator
