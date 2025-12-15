#include "AMP/operators/libmesh/SourceNonlinearElement.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/auto_ptr.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS

#include <string>

namespace AMP::Operator {

SourceNonlinearElement::SourceNonlinearElement(
    std::shared_ptr<const ElementOperationParameters> params )
    : ElementOperation( params ), d_elementOutputVector( nullptr ), d_elem( nullptr )
{

    AMP_INSIST( ( params ), "''params'' is NULL" );

    AMP_INSIST( ( ( ( params->d_db ).get() ) != nullptr ), "NULL database" );

    auto feTypeOrderName = params->d_db->getWithDefault<std::string>( "FE_ORDER", "FIRST" );
    auto feTypeOrder     = libMesh::Utility::string_to_enum<libMeshEnums::Order>( feTypeOrderName );

    auto feFamilyName = params->d_db->getWithDefault<std::string>( "FE_FAMILY", "LAGRANGE" );
    auto feFamily     = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( feFamilyName );

    auto qruleTypeName = params->d_db->getWithDefault<std::string>( "QRULE_TYPE", "QGAUSS" );
    auto qruleType =
        libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( qruleTypeName );

    d_integrateVolume = params->d_db->getWithDefault<bool>( "INTEGRATEVOLUME", true );

    const unsigned int dimension = 3;

    d_feType.reset( new libMesh::FEType( feTypeOrder, feFamily ) );

    d_fe.reset( ( libMesh::FEBase::build( dimension, ( *d_feType ) ) ).release() );
    d_fe->get_xyz();

    auto qruleOrderName = params->d_db->getWithDefault<std::string>( "QRULE_ORDER", "DEFAULT" );

    libMeshEnums::Order qruleOrder;

    if ( qruleOrderName == "DEFAULT" ) {
        qruleOrder = d_feType->default_quadrature_order();
    } else {
        qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( qruleOrderName );
    }

    d_qrule.reset( ( libMesh::QBase::build( qruleType, dimension, qruleOrder ) ).release() );

    d_fe->attach_quadrature_rule( d_qrule.get() );

    //        d_isInputType = (params->d_db)->getWithDefault<std::string>("InputVariableType",
    //        "NodalScalar");

    d_JxW = &( d_fe->get_JxW() );

    d_phi = &( d_fe->get_phi() );

    d_dphi = &( d_fe->get_dphi() );
}


void SourceNonlinearElement::initializeForCurrentElement(
    const libMesh::Elem *elem, std::shared_ptr<SourcePhysicsModel> sourcePhysicsModel )
{
    d_elem = elem;

    if ( sourcePhysicsModel ) {
        d_sourcePhysicsModel = sourcePhysicsModel;
    }
}


void SourceNonlinearElement::apply()
{
    PROFILE( "apply", 5 );

    d_fe->reinit( d_elem );
    const unsigned int n_nodes                         = d_elem->n_nodes();
    const unsigned int n_points                        = d_qrule->n_points();
    const std::vector<libMesh::Real> &JxW              = ( *d_JxW );
    const std::vector<std::vector<libMesh::Real>> &phi = ( *d_phi );
    // const std::vector<std::vector<libMesh::RealGradient> > & dphi = (*d_dphi);
    // std::vector<std::vector<double> > & elementInputVector = d_elementInputVector;
    std::vector<double> &elementOutputVector = ( *d_elementOutputVector );
    std::vector<double> source_physics( static_cast<int>( n_points ) );
    std::vector<std::vector<double>> source_vectors( d_elementInputVector.size() );
    std::vector<std::vector<double>> auxillary_vectors( d_elementAuxVector.size() );
    const auto &coordinates = d_fe->get_xyz();

    if ( d_isInputType == "IntegrationPointScalar" ) {

        for ( unsigned int var = 0; var < d_elementInputVector.size(); var++ ) {
            source_vectors[var] = d_elementInputVector[var];
            if ( d_elementAuxVector.size() > 0 ) {
                auxillary_vectors[var] = d_elementAuxVector[var];
            }
        }

        if ( d_sourcePhysicsModel ) {
            d_sourcePhysicsModel->getConstitutiveProperty(
                source_physics, source_vectors, auxillary_vectors, coordinates );
        }
    } else if ( d_isInputType == "NodalScalar" ) {

        for ( unsigned int var = 0; var < d_elementInputVector.size(); var++ ) {
            source_vectors[var].resize( n_points );
            for ( unsigned int qp = 0; qp < n_points; qp++ ) {
                source_vectors[var][qp] = 0.0;
                for ( unsigned int j = 0; j < n_nodes; j++ ) {
                    source_vectors[var][qp] += d_elementInputVector[var][j] * phi[j][qp];
                } // end for j
            }     // end for qp
            if ( d_elementAuxVector.size() > 0 ) {
                auxillary_vectors[var].resize( n_points );
                for ( unsigned int qp = 0; qp < n_points; qp++ ) {
                    auxillary_vectors[var][qp] = 0.0;
                    for ( unsigned int j = 0; j < n_nodes; j++ ) {
                        auxillary_vectors[var][qp] += d_elementAuxVector[var][j] * phi[j][qp];
                    } // end for j
                }
            }
        }

        if ( d_sourcePhysicsModel ) {
            d_sourcePhysicsModel->getConstitutiveProperty(
                source_physics, source_vectors, auxillary_vectors, coordinates );
        }
    }

    if ( d_sourcePhysicsModel.get() == nullptr ) {
        AMP_INSIST(
            ( source_vectors.size() == 1 ),
            "In absence of SourcePhysicsModel Element Operation Expects only one source vector" );
    }
    for ( unsigned int j = 0; j < n_nodes; j++ ) {
        for ( unsigned int qp = 0; qp < n_points; qp++ ) {
            if ( d_sourcePhysicsModel ) {
                elementOutputVector[j] += ( JxW[qp] * source_physics[qp] * phi[j][qp] );
            } else {
                if ( d_integrateVolume ) {
                    elementOutputVector[j] += ( JxW[qp] * source_vectors[0][qp] * phi[j][qp] );
                } else {
                    elementOutputVector[j] += ( source_vectors[0][qp] * phi[j][qp] ) / 8;
                }
            }
        } // end for j
    }     // end for qp
}
} // namespace AMP::Operator
