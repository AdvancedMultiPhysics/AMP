#include "AMP/operators/libmesh/NodeToGaussPointOperator.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"

#include "ProfilerApp.h"

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/auto_ptr.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS

#include <cstring>


namespace AMP::Operator {


// Constructor
NodeToGaussPointOperator::NodeToGaussPointOperator(
    std::shared_ptr<const OperatorParameters> params )
    : Operator( params )
{
    PROFILE( "NodeToGaussPointOperator" );
    d_NodalVariable.reset(
        new AMP::LinearAlgebra::Variable( params->d_db->getString( "InputVariable" ) ) );
    d_GaussPtVariable.reset(
        new AMP::LinearAlgebra::Variable( params->d_db->getString( "OutputVariable" ) ) );
    d_UseSurfaceElements = params->d_db->getWithDefault<bool>( "UseSurfaceElements", true );
    // Get the iterator for the mesh
    d_dim = 0;
    if ( d_UseSurfaceElements ) {
        d_dim      = 2;
        d_iterator = d_Mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    } else {
        d_dim      = 3;
        d_iterator = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    }
    // Initialize some libmesh variables
    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
    auto qruleOrder  = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "SECOND" );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );
    auto qtype       = libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( "QGAUSS" );
    auto feType( new libMesh::FEType( feTypeOrder, feFamily ) );
    auto qrule  = libMesh::QBase::build( qtype, d_dim, qruleOrder );
    auto febase = libMesh::FEBase::build( d_dim, *feType );
    febase->attach_quadrature_rule( qrule.get() );
    // Cache data for all elements (improves performance)
    d_nodes.resize( d_iterator.size() );
    d_N_quad.resize( d_iterator.size(), 0 );
    d_phi.resize( d_iterator.size() );
    AMP::Mesh::MeshIterator iterator = d_iterator.begin();
    for ( size_t i = 0; i < iterator.size(); ++i, ++iterator ) {
        // Cache the nodes for all elements
        auto nodes = iterator->getElements( AMP::Mesh::GeomType::Vertex );
        d_nodes[i].resize( nodes.size() );
        for ( size_t j = 0; j < nodes.size(); j++ )
            d_nodes[i][j] = nodes[j]->globalID();
        size_t N_nodes = d_nodes[i].size();
        // Cache the shape functions for all elements
        libMesh::Elem *elem =
            AMP::Discretization::createLibmeshElements::createElement( *iterator );
        febase->reinit( elem );
        const auto &phi = febase->get_phi();
        AMP_ASSERT( d_nodes[i].size() == N_nodes );
        d_N_quad[i] = phi[0].size();
        d_phi[i].resize( d_N_quad[i] * N_nodes, 0 );
        for ( size_t j = 0; j < phi.size(); j++ ) {
            AMP_ASSERT( phi[j].size() == d_N_quad[i] );
            for ( size_t k = 0; k < phi[j].size(); k++ )
                d_phi[i][j + k * N_nodes] = phi[j][k];
        }
        delete elem;
    }
}


// Apply operator
void NodeToGaussPointOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                      AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "apply" );

    auto nodalVec   = subsetInputVector( u );
    auto gaussPtVec = subsetOutputVector( r );
    AMP_ASSERT( nodalVec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    auto dof_map         = nodalVec->getDOFManager();
    auto gaussPt_dof_map = gaussPtVec->getDOFManager();

    AMP::Mesh::MeshIterator iterator = d_iterator.begin();
    std::vector<size_t> gaussPtIndices, bndGlobalIds;
    for ( size_t i = 0; i < iterator.size(); ++i, ++iterator ) {

        // Get the dofs for the gauss points
        gaussPt_dof_map->getDOFs( iterator->globalID(), gaussPtIndices );

        // Check if we need to set any gauss points for the current element
        if ( gaussPtIndices.size() == 0 )
            continue;
        unsigned int N_quad = d_N_quad[i];
        AMP_ASSERT( gaussPtIndices.size() == N_quad );

        // Get the dofs for the nodes
        size_t N_nodes = d_nodes[i].size();
        dof_map->getDOFs( d_nodes[i], bndGlobalIds );
        AMP_ASSERT( bndGlobalIds.size() == N_nodes );

        // Get the values at the nodes
        double nodeVals[16];
        nodalVec->getValuesByGlobalID( N_nodes, &bndGlobalIds[0], nodeVals );

        // Set the values at the gauss points
        double computedAtGauss[27];
        memset( computedAtGauss, 0, 27 * sizeof( double ) );
        for ( unsigned int j = 0; j < N_nodes; ++j ) {
            double computedAtNode = nodeVals[j];
            for ( unsigned int qp = 0; qp < N_quad; ++qp ) {
                computedAtGauss[qp] += ( computedAtNode * d_phi[i][j + qp * N_nodes] );
            } // end for qp
        } // end for j
        gaussPtVec->setValuesByGlobalID( N_quad, &gaussPtIndices[0], computedAtGauss );

    } // end for
} // end apply
} // namespace AMP::Operator
