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
        AMP::LinearAlgebra::Vector::shared_ptr myRhs = mySubsetVector( rhs, d_var );
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
        AMP::LinearAlgebra::Vector::shared_ptr rInternal = mySubsetVector( r, d_var );
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
    std::shared_ptr<libMesh::FEType> feType( new libMesh::FEType( feTypeOrder, feFamily ) );
    libMeshEnums::Order qruleOrder = feType->default_quadrature_order();
    std::shared_ptr<libMesh::QBase> qrule(
        ( libMesh::QBase::build( qruleType, 2, qruleOrder ) ).release() );

    std::shared_ptr<AMP::Discretization::DOFManager> dofMap = d_correction->getDOFManager();

    d_correction->zero();
    for ( size_t b = 0; b < d_sideNumbers.size(); ++b ) {
        libMesh::Elem *elem = new libMesh::Hex8;
        for ( int j = 0; j < 8; ++j ) {
            elem->set_node( j ) = new libMesh::Node( d_volumeElements[( 24 * b ) + ( 3 * j ) + 0],
                                                     d_volumeElements[( 24 * b ) + ( 3 * j ) + 1],
                                                     d_volumeElements[( 24 * b ) + ( 3 * j ) + 2],
                                                     j );
        } // end j

        std::shared_ptr<libMesh::FEBase> fe(
            ( libMesh::FEBase::build( 3, ( *feType ) ) ).release() );
        fe->attach_quadrature_rule( qrule.get() );
        fe->reinit( elem, d_sideNumbers[b] );

        const std::vector<std::vector<libMesh::Real>> &phi = fe->get_phi();
        const std::vector<libMesh::Real> &djxw             = fe->get_JxW();

        AMP_ASSERT( phi.size() == 8 );
        AMP_ASSERT( djxw.size() == 4 );
        AMP_ASSERT( qrule->n_points() == 4 );

        std::vector<std::vector<size_t>> dofIndices( 8 );
        for ( int i = 0; i < 8; ++i ) {
            dofMap->getDOFs( d_nodeID[( 8 * b ) + i], dofIndices[i] );
        } // end i

        for ( size_t i = 0; i < 8; ++i ) {
            for ( int d = 0; d < 3; ++d ) {
                double res = 0;
                for ( size_t qp = 0; qp < qrule->n_points(); ++qp ) {
                    double val = d_traction[( 12 * b ) + ( 3 * qp ) + d];
                    res += djxw[qp] * phi[i][qp] * val;
                } // end qp
                d_correction->addValuesByGlobalID( 1, &dofIndices[i][d], &res );
            } // end d
        }     // end i

        for ( size_t j = 0; j < elem->n_nodes(); ++j ) {
            delete ( elem->node_ptr( j ) );
            elem->set_node( j ) = nullptr;
        } // end for j
        delete elem;
        elem = nullptr;
    } // end b
    d_correction->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

AMP::LinearAlgebra::Vector::shared_ptr
TractionBoundaryOperator::mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                                          std::shared_ptr<AMP::LinearAlgebra::Variable> var )
{
    if ( vec != nullptr ) {
        if ( d_Mesh ) {
            AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
            AMP::LinearAlgebra::Vector::shared_ptr meshSubsetVec =
                vec->select( meshSelector, ( ( vec->getVariable() )->getName() ) );
            return meshSubsetVec->subsetVectorForVariable( var );
        } else {
            return vec->subsetVectorForVariable( var );
        }
    } else {
        return vec;
    }
}
} // namespace AMP::Operator
