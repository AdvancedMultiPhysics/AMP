#include "AMP/operators/boundary/libmesh/TractionBoundaryOperator.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/VectorSelector.h"

// Libmesh includes
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/auto_ptr.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/node.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


namespace AMP::Operator {

TractionBoundaryOperator::TractionBoundaryOperator(
    std::shared_ptr<const TractionBoundaryOperatorParameters> params )
    : BoundaryOperator( params ), d_residualMode( false )
{
    AMP_INSIST( params->d_db->keyExists( "Variable" ), "key not found" );
    std::string varName = params->d_db->getString( "Variable" );
    d_var.reset( new AMP::LinearAlgebra::Variable( varName ) );
    AMP_INSIST( params->d_db->keyExists( "ResidualMode" ), "key not found" );
    d_residualMode   = params->d_db->getScalar<bool>( "ResidualMode" );
    d_traction       = params->d_traction;
    d_volumeElements = params->d_volumeElements;
    d_sideNumbers    = params->d_sideNumbers;
    d_nodeID         = params->d_nodeID;
}

void TractionBoundaryOperator::addRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    if ( !d_residualMode ) {
        auto myRhs = mySubsetVector( rhs, d_var );
        if ( d_correction == nullptr ) {
            d_correction = myRhs->clone();
        }
        computeCorrection();
        myRhs->add( *myRhs, *d_correction );
        myRhs->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
}

void TractionBoundaryOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                      AMP::LinearAlgebra::Vector::shared_ptr r )
{
    if ( d_residualMode ) {
        auto rInternal = mySubsetVector( r, d_var );
        if ( d_correction == nullptr ) {
            d_correction = rInternal->clone();
        }
        computeCorrection();
        rInternal->subtract( *rInternal, *d_correction );
        rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
}

void TractionBoundaryOperator::computeCorrection()
{
    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );
    auto qruleType   = libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( "QGAUSS" );
    auto feType      = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
    auto qruleOrder  = feType->default_quadrature_order();
    std::shared_ptr<libMesh::QBase> qrule(
        ( libMesh::QBase::build( qruleType, 2, qruleOrder ) ).release() );

    auto dofMap = d_correction->getDOFManager();

    d_correction->zero();
    std::vector<size_t> dofIndices[8];
    for ( size_t b = 0; b < d_sideNumbers.size(); ++b ) {
        libMesh::Elem *elem = new libMesh::Hex8;
        for ( int j = 0; j < 8; ++j ) {
            auto ptr            = &d_volumeElements[( 24 * b ) + ( 3 * j )];
            elem->set_node( j ) = new libMesh::Node( ptr[0], ptr[1], ptr[2], j );
        }

        std::shared_ptr<libMesh::FEBase> fe( ( libMesh::FEBase::build( 3, *feType ) ).release() );
        fe->attach_quadrature_rule( qrule.get() );
        fe->reinit( elem, d_sideNumbers[b] );

        const auto &phi  = fe->get_phi();
        const auto &djxw = fe->get_JxW();

        AMP_ASSERT( phi.size() == 8 );
        AMP_ASSERT( djxw.size() == 4 );
        AMP_ASSERT( qrule->n_points() == 4 );

        bool found = true;
        for ( int i = 0; i < 8; ++i ) {
            dofMap->getDOFs( d_nodeID[( 8 * b ) + i], dofIndices[i] );
            found = found && !dofIndices[0].empty();
        }
        if ( !found )
            continue;

        for ( size_t i = 0; i < 8; ++i ) {
            double res[3] = { 0, 0, 0 };
            for ( int d = 0; d < 3; ++d ) {
                for ( size_t qp = 0; qp < qrule->n_points(); ++qp ) {
                    double val = d_traction[( 12 * b ) + ( 3 * qp ) + d];
                    res[d] += djxw[qp] * phi[i][qp] * val;
                }
            }
            d_correction->addValuesByGlobalID( 3, dofIndices[i].data(), res );
        }

        for ( size_t j = 0; j < elem->n_nodes(); ++j ) {
            delete elem->node_ptr( j );
            elem->set_node( j ) = nullptr;
        }
        delete elem;
        elem = nullptr;
    }
    d_correction->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

AMP::LinearAlgebra::Vector::shared_ptr
TractionBoundaryOperator::mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                                          std::shared_ptr<AMP::LinearAlgebra::Variable> var )
{
    if ( vec != nullptr ) {
        if ( d_Mesh ) {
            AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
            auto meshSubsetVec = vec->select( meshSelector );
            return meshSubsetVec->subsetVectorForVariable( var );
        } else {
            return vec->subsetVectorForVariable( var );
        }
    } else {
        return vec;
    }
}
} // namespace AMP::Operator
