#include "AMP/operators/boundary/libmesh/RobinVectorCorrection.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/boundary/libmesh/RobinMatrixCorrectionParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/VectorSelector.h"

#include "ProfilerApp.h"

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS

#include <string>

namespace AMP::Operator {


/****************************************************************
 * Create the appropriate parameters                             *
 ****************************************************************/
static std::shared_ptr<const NeumannVectorCorrectionParameters>
convert( std::shared_ptr<const OperatorParameters> inParams )
{
    AMP_ASSERT( inParams );
    if ( std::dynamic_pointer_cast<const NeumannVectorCorrectionParameters>( inParams ) )
        return std::dynamic_pointer_cast<const NeumannVectorCorrectionParameters>( inParams );
    auto bndParams = std::dynamic_pointer_cast<const BoundaryOperatorParameters>( inParams );
    AMP_ASSERT( bndParams );
    auto params        = std::make_shared<NeumannVectorCorrectionParameters>( inParams->d_db );
    params->d_Mesh     = inParams->d_Mesh;
    params->d_variable = bndParams->d_volumeOperator->getOutputVariable();
    if ( params->d_db->keyExists( "LocalModel" ) ) {
        auto model_db = params->d_db->getDatabase( "LocalModel" );
        auto model    = ElementPhysicsModelFactory::createElementPhysicsModel( model_db );
        params->d_robinPhysicsModel = std::dynamic_pointer_cast<RobinPhysicsModel>( model );
    }
    return params;
}


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
RobinVectorCorrection::RobinVectorCorrection( std::shared_ptr<const OperatorParameters> inParams )
    : NeumannVectorCorrection( inParams )
{
    auto params = convert( inParams );
    AMP_ASSERT( params );
    d_hef        = 0;
    d_alpha      = 0;
    d_beta       = 0;
    d_gamma      = 0;
    d_skipParams = false;
    reset( params );
    d_InstanceID = d_iInstance_id;
}


void RobinVectorCorrection::reset( std::shared_ptr<const OperatorParameters> params )
{
    NeumannVectorCorrection::reset( params );

    AMP_INSIST( params, "NULL parameters" );
    AMP_INSIST( params->d_db, "NULL database" );

    d_skipParams = params->d_db->getWithDefault<bool>( "skip_params", false );

    AMP_INSIST( params->d_db->keyExists( "alpha" ), "Missing key: prefactor alpha" );
    d_alpha = params->d_db->getScalar<double>( "alpha" );

    AMP_INSIST( params->d_db->keyExists( "beta" ), "Missing key: prefactor beta" );
    d_beta = params->d_db->getScalar<double>( "beta" );

    AMP_INSIST( params->d_db->keyExists( "gamma" ), "Missing key: total prefactor gamma" );
    d_gamma = params->d_db->getScalar<double>( "gamma" );
}


void RobinVectorCorrection::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                   AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "apply" );
    AMP_INSIST( r, "NULL Residual Vector" );
    AMP_INSIST( u, "NULL Solution Vector" );

    auto rInternal = this->subsetInputVector( r );
    auto uInternal = this->subsetInputVector( u );

    AMP_ASSERT( uInternal->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
    // rInternal->makeConsistent ( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    std::vector<std::string> variableNames;
    size_t numVar = 0;
    if ( d_robinPhysicsModel ) {
        variableNames = d_robinPhysicsModel->getVariableName();
        numVar        = variableNames.size();
    }

    d_elementInputVec.resize( numVar + 1 );
    d_elementInputVec[0] = d_variableFlux;

    if ( d_robinPhysicsModel ) {
        for ( size_t i = 0; i < variableNames.size(); i++ ) {
            std::string cview = variableNames[i] + " view";
            if ( d_Frozen ) {
                if ( d_Frozen->select(
                         AMP::LinearAlgebra::VS_ByVariableName( variableNames[i] ) ) ) {
                    d_elementInputVec[i + 1] = d_Frozen->select(
                        AMP::LinearAlgebra::VS_ByVariableName( variableNames[i] ) );
                } else {
                    d_elementInputVec[i + 1] = uInternal->select(
                        AMP::LinearAlgebra::VS_ByVariableName( variableNames[i] ) );
                }
            } else {
                d_elementInputVec[i + 1] =
                    uInternal->select( AMP::LinearAlgebra::VS_ByVariableName( variableNames[i] ) );
            }
            AMP_INSIST( d_elementInputVec[i + 1],
                        "Did not find vector '" + variableNames[i] + "'" );
            AMP_ASSERT( d_elementInputVec[i + 1]->getUpdateStatus() ==
                        AMP::LinearAlgebra::UpdateState::UNCHANGED );
        }

        if ( d_iDebugPrintInfoLevel == 100 ) {
            std::cout << "processing robin boundary operator " << d_InstanceID << "\n";
        }
    }

    // Get the DOF managers
    auto dofManager = rInternal->getDOFManager();
    std::shared_ptr<AMP::Discretization::DOFManager> gpDOFManager;
    if ( d_isFluxGaussPtVector && d_variableFlux != nullptr )
        gpDOFManager = d_variableFlux->getDOFManager();

    // Check that the DOF managers match for the different vectors
    AMP_ASSERT( *dofManager == *( uInternal->getDOFManager() ) );
    AMP_ASSERT( *dofManager == *( rInternal->getDOFManager() ) );
    if ( !d_isFluxGaussPtVector ) {
        if ( d_variableFlux )
            AMP_ASSERT( *dofManager == *( d_variableFlux->getDOFManager() ) );
    }

    auto numIds = d_boundaryIds.size();
    std::vector<size_t> gpDofs;
    std::vector<size_t> dofs;
    std::vector<std::vector<size_t>> dofIndices;
    std::vector<size_t> dofsElementVec;
    PROFILE( "integration loop" );
    for ( unsigned int nid = 0; nid < numIds; nid++ ) {
        unsigned int numDofIds = d_dofIds[nid].size();

        for ( unsigned int k = 0; k < numDofIds; k++ ) {

            auto bnd1 =
                d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, d_boundaryIds[nid], 0 );

            for ( const auto &elem : bnd1 ) {
                // Get the nodes for the current element
                d_currNodes             = elem.getElements( AMP::Mesh::GeomType::Vertex );
                auto numNodesInCurrElem = d_currNodes.size();

                dofIndices.resize( numNodesInCurrElem );
                // Get the dofs for the vectors
                std::vector<AMP::Mesh::MeshElementID> ids( d_currNodes.size() );
                for ( size_t i = 0; i < d_currNodes.size(); i++ )
                    ids[i] = d_currNodes[i].globalID();

                for ( unsigned int i = 0; i < numNodesInCurrElem; i++ )
                    dofManager->getDOFs( d_currNodes[i].globalID(), dofIndices[i] );

                dofs.resize( numNodesInCurrElem );
                for ( size_t n = 0; n < dofIndices.size(); n++ )
                    dofs[n] = dofIndices[n][d_dofIds[nid][k]];

                AMP_ASSERT( dofs.size() == numNodesInCurrElem );

                if ( d_isFluxGaussPtVector && d_IsCoupledBoundary[nid] ) {
                    gpDOFManager->getDOFs( elem.globalID(), gpDofs );
                    AMP_ASSERT( gpDofs.size() > 0 );
                }

                // Get the current libmesh element
                auto fe   = d_libmeshElements.getFEBase( elem.globalID() );
                auto rule = d_libmeshElements.getQBase( elem.globalID() );
                AMP_ASSERT( fe );
                AMP_ASSERT( rule );
                auto numGaussPts = rule->n_points();

                auto JxW = fe->get_JxW();
                auto phi = fe->get_phi();

                std::vector<std::vector<double>> inputArgs(
                    d_elementInputVec.size(), std::vector<double>( numNodesInCurrElem ) );
                std::vector<std::vector<double>> inputArgsAtGpts(
                    d_elementInputVec.size(), std::vector<double>( numGaussPts ) );
                std::vector<double> beta( numGaussPts, d_beta );
                std::vector<double> gamma( numGaussPts, d_gamma );
                if ( d_robinPhysicsModel ) {
                    unsigned int startIdx = 0;
                    if ( d_isFluxGaussPtVector && d_IsCoupledBoundary[nid] ) {
                        d_variableFlux->getValuesByGlobalID(
                            gpDofs.size(), &gpDofs[0], &inputArgsAtGpts[0][0] );
                        startIdx = 1;
                    }
                    for ( unsigned int m = startIdx; m < d_elementInputVec.size(); m++ ) {
                        // Note: elementInputVecs may use different DOFManagers from u and r
                        // internal
                        d_elementInputVec[m]->getDOFManager()->getDOFs( ids, dofsElementVec );
                        AMP_ASSERT( dofsElementVec.size() == dofs.size() );
                        d_elementInputVec[m]->getValuesByGlobalID(
                            dofs.size(), &dofsElementVec[0], &inputArgs[m][0] );
                        for ( size_t qp = 0; qp < numGaussPts; qp++ ) {
                            for ( size_t n = 0; n < numNodesInCurrElem; n++ ) {
                                inputArgsAtGpts[m][qp] += phi[n][qp] * inputArgs[m][n];
                            }
                        }
                    }

                    d_robinPhysicsModel->getConductance( beta, gamma, inputArgsAtGpts );
                }
                std::vector<double> values( dofs.size(), 0.0 );
                std::vector<double> gpValues( gpDofs.size(), 0.0 );
                std::vector<double> addValues( dofs.size(), 0.0 );
                uInternal->getValuesByGlobalID( dofs.size(), &dofs[0], &values[0] );
                for ( unsigned int qp = 0; qp < numGaussPts; qp++ ) {
                    libMesh::Real phi_val = 0.0;
                    for ( unsigned int l = 0; l < numNodesInCurrElem; l++ )
                        phi_val += phi[l][qp] * values[l];
                    for ( unsigned int j = 0; j < numNodesInCurrElem; j++ )
                        addValues[j] += ( JxW[qp] * phi[j][qp] * beta[qp] * phi_val );
                } // end for qp
                if ( d_IsCoupledBoundary[nid] ) {
                    if ( !d_isFluxGaussPtVector ) {
                        d_variableFlux->getValuesByGlobalID( dofs.size(), &dofs[0], &values[0] );
                    } else {
                        d_variableFlux->getValuesByGlobalID(
                            gpDofs.size(), &gpDofs[0], &gpValues[0] );
                    }
                    for ( unsigned int qp = 0; qp < numGaussPts; qp++ ) {
                        libMesh::Real phi_val = 0.0;
                        if ( !d_isFluxGaussPtVector ) {
                            for ( unsigned int l = 0; l < numNodesInCurrElem; l++ )
                                phi_val += phi[l][qp] * values[l];
                        } else {
                            phi_val = gpValues[qp];
                        }
                        for ( unsigned int j = 0; j < numNodesInCurrElem; j++ )
                            addValues[j] -= JxW[qp] * phi[j][qp] * gamma[qp] * phi_val;
                    } // end for qp
                } // coupled
                rInternal->addValuesByGlobalID( dofs.size(), &dofs[0], &addValues[0] );

            } // end for bnd
        } // end for dof ids
    } // end for nid

    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}


std::shared_ptr<OperatorParameters>
RobinVectorCorrection::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr vec )
{
    auto db = NeumannVectorCorrection::getJacobianParameters( vec )->d_db;
    db->putScalar( "name", "RobinMatrixCorrection", {}, AMP::Database::Check::Overwrite );
    db->putScalar( "skip_params", d_skipParams, {}, AMP::Database::Check::Overwrite );
    db->putScalar( "skip_rhs_correction", true );
    db->putScalar( "skip_matrix_correction", false );
    db->putScalar( "IsFluxGaussPtVector", d_isFluxGaussPtVector );
    db->putScalar( "alpha", d_alpha );
    db->putScalar( "beta", d_beta );
    db->putScalar( "gamma", d_gamma );
    auto outParams                 = std::make_shared<RobinMatrixCorrectionParameters>( db );
    outParams->d_robinPhysicsModel = d_robinPhysicsModel;
    outParams->d_elementInputVec   = d_elementInputVec;
    outParams->d_variableFlux      = d_variableFlux;
    outParams->d_Mesh              = d_Mesh;
    // outParams->d_variable = linearDiffusion->getOutputVariable();
    // outParams->d_inputMatrix = linearDiffusion->getMatrix();
    return outParams;
}


} // namespace AMP::Operator
