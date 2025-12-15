#include "AMP/operators/libmesh/NonlinearFEOperator.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include "libmesh/cell_hex8.h"
#include "libmesh/node.h"

namespace AMP::Operator {


NonlinearFEOperator::NonlinearFEOperator( std::shared_ptr<const OperatorParameters> in )
    : Operator( in )
{
    auto params = std::dynamic_pointer_cast<const FEOperatorParameters>( in );
    AMP_INSIST( params, "NULL parameter!" );
    d_elemOp = params->d_elemOp;
    createLibMeshElementList();
    d_currElemIdx = static_cast<unsigned int>( -1 );
}


NonlinearFEOperator::~NonlinearFEOperator() { destroyLibMeshElementList(); }


void NonlinearFEOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                 AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "apply" );

    AMP_INSIST( ( r != nullptr ), "NULL Residual/Output Vector" );
    auto rInternal = this->subsetOutputVector( r );
    AMP_INSIST( ( rInternal != nullptr ), "NULL Residual/Output Vector" );

    if ( u )
        AMP_ASSERT( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    d_currElemIdx = static_cast<unsigned int>( -1 );
    this->preAssembly( u, rInternal );

    d_currElemIdx = 0;
    for ( auto &el : d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 ) ) {
        this->preElementOperation( el );
        d_elemOp->apply();
        this->postElementOperation();
        d_currElemIdx++;
    }

    d_currElemIdx = static_cast<unsigned int>( -1 );
    this->postAssembly();

    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_iDebugPrintInfoLevel > 2 )
        AMP::pout << "L2 norm of result of NonlinearFEOperator::apply is: " << rInternal->L2Norm()
                  << std::endl;
    if ( d_iDebugPrintInfoLevel > 5 )
        std::cout << rInternal << std::endl;
}


void NonlinearFEOperator::createLibMeshElementList()
{
    auto el = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    d_currElemPtrs.resize( d_Mesh->numLocalElements( AMP::Mesh::GeomType::Cell ) );
    for ( size_t i = 0; i < el.size(); ++el, ++i ) {
        auto currNodes    = el->getElements( AMP::Mesh::GeomType::Vertex );
        d_currElemPtrs[i] = new libMesh::Hex8;
        for ( size_t j = 0; j < currNodes.size(); ++j ) {
            auto pt                          = currNodes[j]->coord();
            d_currElemPtrs[i]->set_node( j ) = new libMesh::Node( pt[0], pt[1], pt[2], j );
        } // end for j
    }     // end for i
}


void NonlinearFEOperator::destroyLibMeshElementList()
{
    for ( auto &elem : d_currElemPtrs ) {
        for ( size_t j = 0; j < elem->n_nodes(); ++j ) {
            delete ( elem->node_ptr( j ) );
            elem->set_node( j ) = nullptr;
        } // end for j
        delete ( elem );
        elem = nullptr;
    } // end for i
    d_currElemPtrs.clear();
}
} // namespace AMP::Operator
